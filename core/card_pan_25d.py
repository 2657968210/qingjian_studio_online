# -*- coding: utf-8 -*-
"""Product Spotlight 的 2.5D 运镜渲染（纯本地，零 AI 调用）。

2026-11-04 用户需求：Product Spotlight 不再调用火山识图（不耗算力/额度），
直接沿用 Decor Showcase 的 2.5D 运镜引擎（core.decor_cam）：水平平移 /
垂直平移 / 对角平移 / 缓推近 / 拉远展开，深度视差由本地浅浮雕深度图算出，
全部像素搬移在本地完成（cv2.remap，无任何网络请求）。

编排规则
--------
· 单图：固定 5 镜、每镜 ~3s、整片 ~15s（card_pan.MAX_TOTAL 红线），
  运镜序列严格按用户 2026-11-05 指定、互不重复：左下→右上、右下→左上、
  左→右、右→左、结尾拉远；整体舒缓不抢戏；
· 多图(1~8)：每图 ~8s（3 个不重复运镜），从同一固定池按偏移取值，
  相邻图起始不重样，硬切衔接（与 2.5D 参考片一致）；
· 平移行程按画布余量自动收敛（decor_cam.make_move 内置），绝不顶边界卡住；
  对角两种朝向由 make_move(diag_cross=True) 支持。

接口
----
render_25d(images, out_path, ratio="9x16", fps=30, target_secs=None,
           per_image_secs=None, progress=None)
  images : 图片路径列表（1~8 张）
  返回 dict：{video, secs, images, moves:[{move,dir,dur,tag}]}
（只出画面，不含配乐；BGM/字幕等后处理仍由 server 完成）
"""
import os

from core import card_pan as cp
from core import decor_cam as dc
from core.decor_show import _encode_frames

# 多图模式下每张图默认展示时长（与旧 card_pan 多图「每张目标 8s」对齐）
PER_IMAGE_SECS = 8.0
# 单图默认总时长（card_pan.MAX_TOTAL=15s 红线）
SINGLE_SECS = 15.0

_TAG = {"h": "pan-h", "v": "pan-v", "diag": "pan-diag",
        "push": "push-in", "pull": "pull-out"}


# 用户 2026-11-05 指定的「固定运镜序列」：运镜不重复 + 明确方向 + 结尾拉远。
# 每个元素 = (family, direction, diag_cross)：
#   diag + direction>0 + diag_cross → 左下→右上 (BL→TR)
#   diag + direction<0 + diag_cross → 右下→左上 (BR→TL)
#   h    + direction>0              → 左→右 (L→R)
#   h    + direction<0              → 右→左 (R→L)
#   pull                            → 拉远（结尾揭示全貌）
_PRODUCT_POOL = [
    ("diag", 1, True),
    ("diag", -1, True),
    ("h", 1, False),
    ("h", -1, False),
    ("pull", 1, False),
]


def plan_single(meta, vw, vh, n, avoid=None):
    """Product Spotlight 固定运镜序列（用户 2026-11-05）。

    不再按图片特征打分，直接取 _PRODUCT_POOL 里的前 n 个不重复运镜，
    保证整段运镜不重复、方向明确、结尾拉远。avoid 仅用于多图模式让相邻图
    起始不重样（见 render_25d）。
    返回 [(family, direction, diag_cross), ...]。
    """
    offset = 0
    if avoid is not None:
        try:
            offset = (_PRODUCT_POOL.index(avoid) + 1) % len(_PRODUCT_POOL)
        except ValueError:
            offset = 0
    n = max(1, min(int(n), len(_PRODUCT_POOL)))
    return [_PRODUCT_POOL[(offset + k) % len(_PRODUCT_POOL)] for k in range(n)]


def render_25d(images, out_path, ratio="9x16", fps=30, target_secs=None,
               per_image_secs=None, progress=None):
    """N(1~8) 张产品图 → 2.5D 运镜展示成片（仅画面，无音轨，零 AI）。"""
    paths = [p for p in (images or []) if p and os.path.isfile(p)]
    if not paths:
        raise ValueError("Product Spotlight needs at least 1 photo")
    if len(paths) > 8:
        paths = paths[:8]

    OW, OH = cp.ratio_size(ratio)
    vw, vh = int(OW), int(OH)

    # 1) 编排：[(path, family, direction, diag_cross, dur), ...]
    seq = []
    if len(paths) == 1:
        meta = dc.probe(paths[0], vw, vh)
        target = float(target_secs or SINGLE_SECS)
        target = min(target, cp.MAX_TOTAL)
        # 单图固定 5 镜（与 _PRODUCT_POOL 一致），每镜 ~3s → 整片 ~15s，运镜舒缓
        n = min(5, len(_PRODUCT_POOL))
        dur = target / n
        for fam, d, cross in plan_single(meta, vw, vh, n):
            seq.append((paths[0], fam, d, cross, dur))
    else:
        per = float(per_image_secs or PER_IMAGE_SECS)
        n = max(2, min(4, int(round(per / 3.0))))
        dur = per / n
        avoid = None
        for p in paths:
            meta = dc.probe(p, vw, vh)
            fams = plan_single(meta, vw, vh, n, avoid=avoid)
            for fam, d, cross in fams:
                seq.append((p, fam, d, cross, dur))
            avoid = fams[-1][0]

    total_secs = sum(m[3] for m in seq)
    n_frames = int(round(total_secs * fps))

    if progress:
        progress(8, 100, "Rendering 2.5D moves")

    # 2) 渲染：同一张图的连续镜头复用同一块画布（省一次解码+深度图）
    def frames():
        done = 0
        cur_path = None
        cv_ = None
        for idx, (p, fam, direction, cross, dur) in enumerate(seq):
            if p != cur_path:
                cur_path = p
                cv_ = dc.build_canvas(p, vh, vw)
            cam_fn = dc.make_move(fam, cv_, vw, vh, direction=direction,
                                  diag_cross=cross)
            cut_frames = max(2, int(round(dur * fps)))
            for i in range(cut_frames):
                t = i / max(cut_frames - 1, 1)
                done += 1
                yield dc.warp(cv_, cam_fn(t), vw, vh)
                if progress and done % 40 == 0:
                    progress(8 + int(88 * done / max(n_frames, 1)), 100,
                             "Rendering 2.5D moves")

    rc = _encode_frames(out_path, frames(), vw, vh, fps)
    if rc != 0:
        raise RuntimeError("ffmpeg failed rc=%s" % rc)

    if progress:
        progress(100, 100, "Done")
    # 运镜标签：明确写出方向（左下→右上 / 右下→左上 / 左→右 / 右→左 / 拉远）
    def _move_tag(fam, d, cross):
        if fam == "pull":
            return "pull-out (拉远)"
        if fam == "diag":
            return "pan-diag BL→TR" if (d > 0) else "pan-diag BR→TL"
        return "pan-h L→R" if d > 0 else "pan-h R→L"
    moves = [{"move": fam, "dir": d, "dur": round(dur, 2),
              "tag": _move_tag(fam, d, cross)}
             for (_, fam, d, cross, dur) in seq]
    return {"video": out_path, "secs": round(total_secs, 2),
            "images": len(paths), "ratio": ratio, "moves": moves}
