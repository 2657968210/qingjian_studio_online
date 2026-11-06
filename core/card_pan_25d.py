# -*- coding: utf-8 -*-
"""Product Spotlight 的 2.5D 运镜渲染（纯本地，零 AI 调用）。

2026-11-04 用户需求：Product Spotlight 不再调用火山识图（不耗算力/额度），
直接沿用 Decor Showcase 的 2.5D 运镜引擎（core.decor_cam）：水平平移 /
垂直平移 / 对角平移 / 缓推近 / 拉远展开，深度视差由本地浅浮雕深度图算出，
全部像素搬移在本地完成（cv2.remap，无任何网络请求）。

2026-11-06 重做运镜规划（用户反馈「运镜不好、要按图匹配、要近景特写、不要重复」）：
· 不再用固定 5 镜序列，改为**按图片特征匹配**运镜：
    宽幅图→水平平移、竖幅图→垂直平移、方图→对角平移、细节强（纸品/文字/花艺
    近景）→缓推近特写、开阔少细节→拉远；与 decor_cam.plan_moves 同口径；
· **近景特写**：细节强的卡片优先插入一次 push-in，焦点对准细节热区（高浮雕
    区质心），让运镜「切近景、重点卡片细节」；
· **不连续重复**：近期 3 镜 + 上一镜同家族重罚，保证相邻两镜不重样；多图时
    相邻图首镜也不与上一图末镜同家族；
· 单图固定 5 镜（~3s/镜，整片≤15s），结尾拉远揭示全貌；多图每图 2~4 镜（每张
    ~8s），硬切衔接。

接口
----
render_25d(images, out_path, ratio="9x16", fps=30, target_secs=None,
           per_image_secs=None, progress=None)
  images : 图片路径列表（1~8 张）
  返回 dict：{video, secs, images, moves:[{move,dir,dur,tag}]}
（只出画面，不含配乐；BGM/字幕等后处理仍由 server 完成）
"""
import os

import numpy as np
from core import card_pan as cp
from core import decor_cam as dc
from core.decor_show import _encode_frames

# 多图模式下每张图默认展示时长（与旧 card_pan 多图「每张目标 8s」对齐）
PER_IMAGE_SECS = 8.0
# 单图默认总时长（card_pan.MAX_TOTAL=15s 红线）
SINGLE_SECS = 15.0


# ---------------------------------------------------------------------------
# 细节热区：高浮雕区加权质心 → push 运镜的焦点（让近景对准卡片重点细节）
# ---------------------------------------------------------------------------
def _detail_anchor(canvas):
    """返回 (ax, ay) 画布像素坐标：细节最强区域的加权质心。

    取深度图绝对值前 15% 像素的加权质心；无显著细节（平面图）时回落画面中心。
    坐标与 decor_cam.make_move 的 ax/ay 同坐标系（画布 px）；make_move 内部会
    再把焦点夹到安全区，推近时不露画布黑边。
    """
    d = canvas.depth
    mag = np.abs(d).astype(np.float64)
    if mag.size == 0 or mag.max() <= 1e-6:
        return (canvas.w * 0.5, canvas.h * 0.5)
    thr = np.percentile(mag, 85)
    ys, xs = np.where(mag >= thr)
    if xs.size == 0:
        return (canvas.w * 0.5, canvas.h * 0.5)
    w = mag[ys, xs]
    cx = float((xs * w).sum() / w.sum())
    cy = float((ys * w).sum() / w.sum())
    return (cx, cy)


# ---------------------------------------------------------------------------
# 特征打分 + 单图运镜规划
# ---------------------------------------------------------------------------
def _feature_scores(c, vw, vh):
    """单图各运镜家族契合分（与 decor_cam.plan_moves 同口径）：
      横幅→h、竖幅→v、方图→diag、细节强→push、开阔→pull；缺平移余量直接禁用。"""
    a, edge = c.aspect, c.edge
    sx, sy = c.slack(1.12, vw, vh)
    score = {
        "h": 1.0 + 0.8 * min(1.0, sx / 240.0) + (0.5 if a >= 1.12 else 0.0),
        "v": 1.0 + 0.8 * min(1.0, sy / 240.0) + (0.5 if a <= 0.85 else 0.0),
        "diag": 0.9 + 0.5 * min(1.0, sx / 240.0) + 0.5 * min(1.0, sy / 240.0),
        "push": 1.0 + 1.3 * edge + (0.2 if a >= 0.9 else 0.0),
        "pull": 0.9 + 0.8 * (1.0 - edge),
    }
    if sx < 55.0:           # 没有水平行程 → 禁用 h / diag
        score["h"] = -99.0
        score["diag"] = -99.0
    if sy < 50.0:           # 没有垂直行程 → 禁用 v / diag
        score["v"] = -99.0
        score["diag"] = -99.0
    return score


def _best(score):
    return max(score, key=lambda f: score[f])


def _plan_image(c, vw, vh, n, force_push=True, force_last_pull=False,
                prev_family=None):
    """为一帧图规划 n 个运镜（feature-aware，不连续重复，含近景特写）。

    返回 [(family, direction, diag_cross), ...]。
      · 近期 3 镜 + 上一镜同家族重罚 → 相邻两镜不重样；
      · force_push：中段（非首非尾）强制插入一次推近特写（近景/重点细节）；
      · force_last_pull：末镜拉远揭示全貌（单图整片收尾用）；
      · 首镜不拉远（拉远留给收尾）；
      · 同家族连用时反向；否则按序交替方向，避免整片同向扫。
    """
    recent, seq = [], []
    used_push = False
    for k in range(n):
        score = _feature_scores(c, vw, vh)
        for f in recent[-3:]:
            score[f] -= 2.2
        if seq:
            score[seq[-1][0]] -= 3.0
        if prev_family is not None and k == 0:
            score[prev_family] -= 2.0          # 与上一图末镜不重样
        if force_last_pull and k == n - 1:
            fam = "pull" if score["pull"] > -90 else _best(score)
        elif force_push and not used_push and 1 <= k < n - 1:
            fam = "push" if score["push"] > -90 else _best(score)
        elif k == 0:
            score["pull"] -= 6.0               # 开场不拉远
            fam = _best(score)
        else:
            fam = _best(score)
        if fam == "push":
            used_push = True
        direction = 1
        if seq and seq[-1][0] == fam:
            direction = -seq[-1][1]
        elif fam in ("h", "v", "diag"):
            direction = -1 if (len(seq) % 2 == 1) else 1
        seq.append((fam, direction, fam == "diag"))
        recent.append(fam)
    return seq


def _move_tag(fam, d, cross):
    if fam == "pull":
        return "pull-out (拉远揭示)"
    if fam == "push":
        return "push-in (近景特写)"
    if fam == "diag":
        return "pan-diag BL→TR" if (d > 0) else "pan-diag BR→TL"
    return "pan-h L→R" if d > 0 else "pan-h R→L"


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
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

    # 1) 逐图建画布 + 规划运镜（特征匹配 + 近景特写 + 不连续重复）
    seq = []
    prev_family = None
    for p in paths:
        cv_ = dc.build_canvas(p, vh, vw)
        is_single = (len(paths) == 1)
        if is_single:
            target = float(target_secs or SINGLE_SECS)
            target = min(target, cp.MAX_TOTAL)
            n = 5
            dur = target / n
            plan = _plan_image(cv_, vw, vh, n, force_push=True,
                               force_last_pull=True, prev_family=prev_family)
        else:
            per = float(per_image_secs or PER_IMAGE_SECS)
            n = max(2, min(4, int(round(per / 3.0))))
            dur = per / n
            # 多图：只在细节强的卡片插入近景特写（按图匹配，避免每张都推近的雷同）
            force_push = cv_.edge >= 0.22
            plan = _plan_image(cv_, vw, vh, n, force_push=force_push,
                               force_last_pull=False, prev_family=prev_family)
        anchor = _detail_anchor(cv_)
        for fam, d, cross in plan:
            seq.append((p, fam, d, cross, dur,
                        anchor if fam == "push" else None, cv_))
        prev_family = plan[-1][0]

    total_secs = sum(m[4] for m in seq)
    n_frames = int(round(total_secs * fps))

    if progress:
        progress(8, 100, "Rendering 2.5D moves")

    # 2) 渲染：同一张图的连续镜头复用同一块画布（省一次解码+深度图）
    def frames():
        done = 0
        cur_path = None
        cv_ = None
        for (p, fam, direction, cross, dur, anchor, cv_in) in seq:
            if p != cur_path:
                cur_path = p
                cv_ = cv_in
            cam_fn = dc.make_move(fam, cv_, vw, vh, direction=direction,
                                  diag_cross=cross, anchor=anchor)
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
              "tag": _move_tag(fam, d, cross)}
             for (_, fam, d, cross, dur, anchor, cv_) in seq]
    return {"video": out_path, "secs": round(total_secs, 2),
            "images": len(paths), "ratio": ratio, "moves": moves}
