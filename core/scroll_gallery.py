# -*- coding: utf-8 -*-
"""滚动画廊模版（N 张请柬图 → 一条左右两列反向慢滚动成片）。

灵感来源：E:\\推广素材\\案例\\多种效果\\document_810w-dGMncCA8-us.mp4
（2 列瀑布流、纯向上滚动、无文字）。用户 2026-10-19 反复打磨后的成品口径：

  * 每张请柬图裁成统一 **3:4 竖版**，圆角 + 浅灰描边，无文字叠加；
  * **左右两列反向滚动**：左列向上、右列向下，速度相同（视觉上差速是因为方向相反）；
  * 左右两列用**不同**的照片：左列取前半、右列取后半，每列只滚自己的图；
  * 两列**不对齐**（右列初始半格偏移），相邻图之间始终保留均匀白线间距；
  * 左右两列各留 SIDE_MARGIN 边距，**不靠屏边**；
  * **不加任何滤镜**，保留原始图片颜色（纯白背景，无阴影）。
  * 2026-10-26 改版：段一左右两列反向滚动固定 SCROLL_SECS（默认 5s）后即切入段二——
    全部 N 张图依次「单图全屏 + 按图匹配合适运镜」逐一展示（每张 SINGLE_DUR 秒，
    smoothstep 缓动；运镜由 image_motion 逐张判定：纸品/竖幅→推近，横幅场景→横扫，
    平铺→微旋，避免每张都用推近），整片时长 = SCROLL_SECS + SINGLE_DUR×N，随图片数增长；
    单图全屏无圆角、无描边。

与 Collection Showcase（multi_suite）完全独立：它做「逐款分段 + 名牌」，
本模块只做「两列反向无限滚动」。两条模版互不影响。

接口
----
render_scroll(images, out_path, ratio="9x16", fps=30,
              scroll_secs=10.0, freeze_secs=2.0, progress=None)
  images : 图片路径列表（≥2 张；前半给左列、后半给右列）
  返回 dict：{video, secs, images}
（只出画面，不含配乐；配乐由调用方混，与本工作台其它模版一致）
"""
import os
import sys
import time
import subprocess
import tempfile

import numpy as np
import cv2
from PIL import Image, ImageDraw

from core import card_pan as cp
from core import image_motion as imn

# ---------------------------------------------------------------------------
# 几何 / 风格常量（与用户 2026-10-19 确认的成品一致）
# ---------------------------------------------------------------------------
GAP = 18                            # 相邻卡片之间的白线间距（px，均匀保留）
SIDE_MARGIN = 30                    # 每列距屏边的留白（px）
COLS = 2
RADIUS = 22                         # 圆角半径（px）
BORDER_RGBA = (224, 224, 224, 255)  # 浅灰描边（2026-10-20 调浅一档：205→224），避免白底图与白 gap 糊成一片
BORDER_W = 3
BG_RGB = (255, 255, 255)            # 纯白背景，无任何调色

SPEED = 140.0                       # 滚动速度 px/s（左右相同，方向相反；2026-10-19 用户定档 140）
CRF = 22
PRESET = "medium"

# ---------------------------------------------------------------------------
# 段二：单图全屏按图匹配运镜展示（2026-10-26 改版：滚动 5s 后逐张单图运镜；
#       运镜类型由 image_motion 逐张判定，不再固定推近）
# ---------------------------------------------------------------------------
SCROLL_SECS = 5.0        # 段一滚动时长（s）
SINGLE_DUR = 2.5         # 段二每张单图展示时长（s）；2026-10-29 用户定稿：速度
                         # 参考 Collection Showcase（其 TARGET_SEG=2.5s/段），
                         # 原 2.8s 偏拖。整片 = SCROLL_SECS + SINGLE_DUR×N
# 段二运镜幅度（2026-10-29 参考 Collection Showcase：横扫 V_SWEEP=0.40、
# 推近 close 0.70-0.84 即 16%~30%）。**只在_colour Inspiration 生效**——
# 不改 image_motion.FS_*（那套共享常量还被 Grid Showcase 复用，动了会串台）。
SGL_PAN = 0.32           # 横扫行程 = 裁窗宽的 32%（原 FS_PAN 0.13 偏近静止）
SGL_PUSH = 0.15          # 推近幅度（原 FS_PUSH 0.07 偏弱）
SGL_PULL = 1.18          # 拉远起幅（CS 的 zoom|out 家族，起幅足以框住重点）


# ---------------------------------------------------------------------------
# 图片读取（与 multi_suite._load_work 同款：cv2 读 BGR→RGBA，保留 alpha）
# ---------------------------------------------------------------------------
def _read_rgba(p):
    arr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("cannot read %s" % p)
    # 2026-10-29 用户红线：Colour Inspiration 保留原图色调光线，**不加任何亮度/
    # 色调滤镜**（撤销 2026-10-27 串入的 normalize_luma 自适应 gamma，它把整图
    # 平均亮度拉到统一档，会改变光影观感，被用户识别为「一层滤镜」）。
    if arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        a = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)
        arr = np.concatenate([arr, a], axis=2)
    elif arr.shape[2] == 4:
        b, g, r, a = cv2.split(arr)
        arr = cv2.merge([r, g, b, a])
    return arr


def _cover_crop(im, tw, th):
    """PIL RGBA → 中心裁到 tw×th（cover，不拉伸）。"""
    iw, ih = im.size
    scale = max(tw / iw, th / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    im = im.resize((nw, nh), Image.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return im.crop((left, top, left + tw, top + th))


def _add_rounded_corners(img, radius):
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
    img.putalpha(mask)
    return img


def _make_cell(im):
    c = _add_rounded_corners(im.copy(), RADIUS)
    d = ImageDraw.Draw(c)
    d.rounded_rectangle((1, 1, c.size[0] - 2, c.size[1] - 2),
                        radius=max(RADIUS - 2, 2),
                        outline=BORDER_RGBA, width=BORDER_W)
    return c


# ---------------------------------------------------------------------------
# 构造一列可无缝循环的竖向内容画布
# ---------------------------------------------------------------------------
def _build_column(order, CELL_W, CELL_H, GAP):
    """order: 该列的 PIL RGBA 图片列表。

    复制两份首尾相接，使滚动到接缝处也保留 GAP（否则末图与首图会紧贴）。
    若只有 1 张，复制成 2 张以便滚动有运动感。
    """
    if len(order) == 1:
        order = order * 2
    n = len(order)
    group_h = n * (CELL_H + GAP)          # 含尾部 GAP，保证循环缝也有间距
    canvas_h = group_h * 2
    col = Image.new("RGBA", (CELL_W, canvas_h), BG_RGB + (255,))
    for rep in range(2):
        for k in range(n):
            im = _cover_crop(order[k], CELL_W, CELL_H)
            y = rep * group_h + k * (CELL_H + GAP)
            cell = _make_cell(im)
            col.paste(cell, (0, y), cell)
    return np.array(col)[:, :, :3], group_h   # RGB uint8（与 multi_suite 的 BGR 不同，本模块统一 RGB）


def _strip(col_np, group_h, off, VIEW_H):
    y0 = int(round(off)) % group_h
    y1 = y0 + VIEW_H
    if y1 <= col_np.shape[0]:
        return col_np[y0:y1, :, :]
    part1 = col_np[y0:, :, :]
    part2 = col_np[:y1 - col_np.shape[0], :, :]
    return np.vstack([part1, part2])


def render_scroll(images, out_path, ratio="9x16", fps=30,
                  scroll_secs=SCROLL_SECS, single_dur=SINGLE_DUR, progress=None):
    """N 张请柬图 → 段一左右两列反向滚动 SCROLL_SECS 秒 + 段二逐张单图全屏按图匹配运镜（仅画面，无音轨）。

    整片时长 = SCROLL_SECS + SINGLE_DUR×N，随图片数增长。"""
    paths = [p for p in (images or []) if p]
    if len(paths) < 2:
        raise ValueError("Scrolling gallery needs at least 2 photos")

    OW, OH = cp.ratio_size(ratio)
    VIEW_W, VIEW_H = OW, OH

    CELL_W = (VIEW_W - SIDE_MARGIN * 2 - GAP * (COLS - 1)) // COLS
    CELL_H = int(round(CELL_W * 4 / 3))            # 3:4 统一比例
    # 右列初始半格偏移，使左右不对齐
    INIT_R_OFFSET = (CELL_H + GAP) // 2

    raw = [_read_rgba(p) for p in paths]
    # 左列前半、右列后半
    half = (len(raw) + 1) // 2
    left_imgs = [Image.fromarray(a) for a in raw[:half]]
    right_imgs = [Image.fromarray(a) for a in raw[half:]]

    if progress:
        progress(8, 100, "Preparing photos")

    col_l_np, group_l = _build_column(left_imgs, CELL_W, CELL_H, GAP)
    col_r_np, group_r = _build_column(right_imgs, CELL_W, CELL_H, GAP)

    # 单图段：加载原图（BGR 超分）+ 逐张图像感知运镜决策（不固定推近）
    ASP_F = VIEW_W / float(VIEW_H)
    imgs_bgr = [imn._upscale_bgr(cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGB2BGR)) for a in raw]
    ANS = []
    for k, im in enumerate(imgs_bgr):
        a = imn.analyze(im, "S%d" % k, aspect=ASP_F)
        a["_ih"], a["_iw"] = im.shape[:2]
        ANS.append(a)
    PLAN = [imn.decide_fullscreen(a, ASP_F) for a in ANS]

    def _alt_moves(an):
        """CS 式备选序列（2026-10-29 运镜方式参考 Collection Showcase）：
        pan（向余量足的一侧）/ pull / rot —— 相邻两镜撞家族时依序轮换，
        避免整组图全是同一个推近。只影响本模版，不动共享 FS_* 常量。"""
        ih, iw = an["_ih"], an["_iw"]
        out = []
        x0, _, w0, _ = imn.focal_window(ih, iw, ASP_F, 1.0, an["focal_n"])
        lm, rm = x0, iw - w0 - x0
        if rm >= w0 * 0.04 and rm >= lm:
            out.append(("panR", SGL_PAN))
        if lm >= w0 * 0.04:
            out.append(("panL", SGL_PAN))
        out.append(("pull", SGL_PULL))
        out.append(("rot", imn.FS_ROT))
        return out

    # 幅度覆盖为本模块定稿值（push/pan 一律用 SGL_*）+ 相邻两镜不撞家族
    # （撞了轮换 pan/pull/rot，CS 式防重）；pan 起幅补 min_zoom_centre，
    # 让「先切重点、重点居中」对 pan 也成立（2026-10-29）。
    SHOTS = []
    _prev = None
    for k in range(len(ANS)):
        mot, par = PLAN[k]
        focal = ANS[k]["focal_n"]
        pzoom = None
        if mot in ("panL", "panR"):
            par = SGL_PAN
            pzoom = max(1.0, imn.min_zoom_centre(ANS[k]["_ih"], ANS[k]["_iw"],
                                                 ASP_F, focal))
        elif mot == "push":
            par = SGL_PUSH
        if mot == _prev:
            for am, apar in _alt_moves(ANS[k]):
                if am != _prev:
                    mot, par = am, apar
                    pzoom = None
                    if mot in ("panL", "panR"):
                        pzoom = max(1.0, imn.min_zoom_centre(
                            ANS[k]["_ih"], ANS[k]["_iw"], ASP_F, focal))
                    break
        _prev = mot
        SHOTS.append((mot, focal, par, pzoom))

    scroll_frames = int(round(scroll_secs * fps))
    single_frames = int(round(single_dur * fps))
    total_frames = scroll_frames + single_frames * len(raw)
    total_secs = total_frames / fps

    # 滚动结束处的偏移（仅用于滚动段计算）
    _end_off_l = (scroll_frames / fps * SPEED) % group_l
    _end_off_r = (INIT_R_OFFSET - scroll_frames / fps * SPEED) % group_r

    # 输出 RGB（与读入一致），ffmpeg 收 rgb24
    cmd = [cp.FFMPEG, "-y", "-hide_banner",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (VIEW_W, VIEW_H), "-r", str(fps), "-i", "-",
           "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", str(CRF), "-preset", PRESET, "-movflags", "+faststart",
           out_path]
    err_f = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=err_f, stderr=err_f)

    x_r0 = SIDE_MARGIN + CELL_W + GAP
    x_r1 = VIEW_W - SIDE_MARGIN
    try:
        for i in range(total_frames):
            if i < scroll_frames:
                off_l = (i / fps * SPEED) % group_l                  # 左：向上
                off_r = (INIT_R_OFFSET - i / fps * SPEED) % group_r  # 右：向下
                frame = np.full((VIEW_H, VIEW_W, 3), BG_RGB, dtype=np.uint8)
                frame[:, SIDE_MARGIN:SIDE_MARGIN + CELL_W, :] = _strip(col_l_np, group_l, off_l, VIEW_H)
                frame[:, x_r0:x_r1, :] = _strip(col_r_np, group_r, off_r, VIEW_H)
            else:
                # 段二：逐张单图全屏 + 慢推近
                k = i - scroll_frames
                img_i = k // single_frames
                pp = (k % single_frames) / max(single_frames - 1, 1)
                e = pp * pp * (3 - 2 * pp)                           # smoothstep 缓动
                mot, focal, par, pzoom = SHOTS[img_i]
                _kw = ({"push": par} if mot == "push"
                       else {"pan": par} if mot in ("panL", "panR")
                       else {"rot": par} if mot == "rot" else {})
                frame = imn.render_single_shot(imgs_bgr[img_i], mot, focal, e,
                                              VIEW_W, VIEW_H,
                                              pan_zoom=pzoom, **_kw)
            proc.stdin.write(frame.tobytes())
            if progress and (i % 30 == 0):
                progress(int(10 + 80 * i / max(total_frames, 1)), 100, "Rendering")
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    if progress:
        progress(100, 100, "Done")
    return {"video": out_path, "secs": round(total_secs, 2), "images": len(paths),
            "ratio": ratio}
