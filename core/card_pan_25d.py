# -*- coding: utf-8 -*-
"""Product Spotlight 的 2.5D 运镜渲染（纯本地，零 AI 调用）。

2026-11-04 用户需求：Product Spotlight 不再调用火山识图（不耗算力/额度），
直接沿用 Decor Showcase 的 2.5D 运镜引擎（core.decor_cam）：水平平移 /
垂直平移 / 对角平移 / 缓推近 / 拉远展开，深度视差由本地浅浮雕深度图算出，
全部像素搬移在本地完成（cv2.remap，无任何网络请求）。

编排规则
--------
· 单图：默认凑满 ~15s（card_pan.MAX_TOTAL 红线）——按图片特征选 5 个不重复
  家族（横幅先水平、竖幅先垂直、细节多先推近、开阔先拉远），方向交替，
  每镜 ~3.0s；
· 多图(1~8)：每图 ~8s（3 个不重复运镜），相邻图之间首家族不重样，
  硬切衔接（与 2.5D 参考片一致）；
· 平移行程按画布余量自动收敛（decor_cam.make_move 内置），绝不顶边界卡住。

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


def plan_single(meta, vw, vh, n, avoid=None):
    """按单张图特征选 n 个不重复的 2.5D 运镜家族。

    评分口径与 decor_cam.plan_moves 一致（内容契合 + 行程可得性），
    额外支持 avoid：上一张图的末家族强惩罚，保证相邻图不重样。
    行程不足的平移家族直接出局（避免「画面卡住」）；可用家族不足 n 时
    自动收缩 n、拉长单镜时长，宁少勿滥。
    返回 [(family, direction), ...]。
    """
    a, edge = meta.aspect, meta.edge
    sx, sy = meta.slack(1.12, vw, vh)
    score = {
        "h": 1.0 + 0.8 * min(1.0, sx / 240.0) + (0.5 if a >= 1.12 else 0.0),
        "v": 1.0 + 0.8 * min(1.0, sy / 240.0) + (0.5 if a <= 0.85 else 0.0),
        "diag": 0.9 + 0.5 * min(1.0, sx / 240.0) + 0.5 * min(1.0, sy / 240.0),
        "push": 1.0 + 1.3 * edge + (0.2 if a >= 0.9 else 0.0),
        "pull": 1.0 + 1.3 * (1.0 - edge),
    }
    # 硬门槛：没有行程的平移家族不用
    if sx < 55.0:
        score["h"] = -99.0
    if sy < 50.0:
        score["v"] = -99.0
    if sx < 55.0 or sy < 50.0:
        score["diag"] = -99.0
    if avoid:
        score[avoid] = score.get(avoid, 0.0) - 3.0

    avail = [f for f in dc.FAMILIES if score[f] > -50.0]
    if not avail:                      # 极端兜底：全没行程时至少给个推近
        avail = ["push"]
    n = max(1, min(int(n), len(avail)))
    order = sorted(avail, key=lambda f: score[f], reverse=True)[:n]

    plan = []
    for k, fam in enumerate(order):
        if fam in ("push", "pull"):
            plan.append((fam, 1))
        else:
            plan.append((fam, 1 if k % 2 == 0 else -1))
    return plan


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

    # 1) 编排：[(path, family, direction, dur), ...]
    seq = []
    if len(paths) == 1:
        meta = dc.probe(paths[0], vw, vh)
        target = float(target_secs or SINGLE_SECS)
        target = min(target, cp.MAX_TOTAL)
        n = max(3, min(5, int(round(target / 3.0))))
        dur = target / n
        for fam, d in plan_single(meta, vw, vh, n):
            seq.append((paths[0], fam, d, dur))
    else:
        per = float(per_image_secs or PER_IMAGE_SECS)
        n = max(2, min(4, int(round(per / 3.0))))
        dur = per / n
        avoid = None
        for p in paths:
            meta = dc.probe(p, vw, vh)
            fams = plan_single(meta, vw, vh, n, avoid=avoid)
            for fam, d in fams:
                seq.append((p, fam, d, dur))
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
        for idx, (p, fam, direction, dur) in enumerate(seq):
            if p != cur_path:
                cur_path = p
                cv_ = dc.build_canvas(p, vh, vw)
            cam_fn = dc.make_move(fam, cv_, vw, vh, direction=direction)
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
    moves = [{"move": fam, "dir": d, "dur": round(dur, 2),
              "tag": _TAG[fam] + ("" if fam in ("push", "pull")
                                  else (" →" if d > 0 else " ←"))}
             for (_, fam, d, dur) in seq]
    return {"video": out_path, "secs": round(total_secs, 2),
            "images": len(paths), "ratio": ratio, "moves": moves}
