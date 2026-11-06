# -*- coding: utf-8 -*-
"""黑白→彩色 切换揭示模版（B&W -> Color Reveal）
=================================================

N(2~12) 张请柬/纸品图 -> 每张先以**黑白**呈现，再**同图彩色揭示**；黑白镜与彩色镜是
同一段连续运镜的**前半程**与**后半程**（phase 0->0.5 / 0.5->1.0），切换发生在运镜滑行
途中、**不回位、不重置**；彩色 reveal 在运镜继续推进时自然浮现。

开头 1 秒斜体衬线「开场标题」可编辑、可关（由调用方传 text / use_text 控制）。

与 fast_cut / four_grid 等完全独立：本模块只做「黑白->彩色 连续运镜揭示 + 可选开场标题」，
只出画面（无音轨），配乐由调用方（server）混 heartwarming 等、单次播放随片长截断（不循环、不卡顿）。

接口
----
render_bw_reveal(images, out_path, ratio="9x16", fps=30,
                text="details ...incoming", use_text=True, progress=None, font=None)
  images   : 图片路径列表（2~12 张）
  text     : 开场标题（可选；空则回退默认句）
  use_text : False 则完全不叠开场标题（即便 text 非空）
  font     : 客户自选字体键（core.fonts；None = 斜体 Cormorant Garamond 默认）
  返回 dict：{video, secs, images, ratio}
（只出画面，不含配乐）
"""
import os
import subprocess
import tempfile

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from core import card_pan as cp

# ---------------------------------------------------------------------------
# 字体（随包 static/fonts/，Linux 部署不依赖系统字体）
# ---------------------------------------------------------------------------
_FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "fonts")
ITALIC = os.path.join(_FONT_DIR, "CormorantGaramond-SemiBoldItalic.ttf")

# ---------------------------------------------------------------------------
# 节奏 / 风格常量
# ---------------------------------------------------------------------------
FPS = 30
KH = 3                      # 镜间交叉溶解帧数
SHOT_TARGET = 0.55          # 每镜目标时长（秒）
TOTAL_CAP = 12.37           # 总长上限（秒）—— 对齐原曲卡点 BGM（~12.4s），视频恒 ≤ 原曲，
                              # server 用 mix_bgm 单次播放随片长截断、不循环、不卡顿
PUSH_MAX = 0.12             # 整段运镜最大推近倍率
DRIFT_AMP = 0.05            # 漂移幅度（占帧宽/高的比例）
TITLE_SECS = 1.0            # 开场标题显示时长（秒）
TITLE_FADE_IN = 0.15
TITLE_FADE_OUT = 0.25
TITLE_W_RATIO = 0.62        # 标题字宽上限（占屏宽）
TITLE_CY = 0.42             # 标题垂直中心（占屏高）
DEFAULT_TEXT = "details ...incoming"

CRF = 18
PRESET = "medium"

# 每张图运镜方向循环（轻漂移 + 轻微推近），相邻图方向不同、避免重复感
_DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1)]

MIN_PHOTOS, MAX_PHOTOS = 2, 12


# ---------------------------------------------------------------------------
# 图片读取
# ---------------------------------------------------------------------------
def _imread_bgr(p):
    """cv2 中文路径安全读取（红线：np.fromfile + imdecode）。"""
    arr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("cannot read %s" % p)
    return arr


# ---------------------------------------------------------------------------
# 缓动 + 取景
# ---------------------------------------------------------------------------
def _ease(t):
    """平滑缓动（起点/终点速度连续，便于黑白->彩色 衔接无顿挫）。"""
    return 0.55 * t + 0.45 * (3 * t * t - 2 * t * t * t)


def _crop_box(im, z, dx, dy, W, H):
    """按 zoom z(>1 收紧) + 漂移 (dx,dy) 算取景框（cover 裁切，居中起步）。"""
    h, w = im.shape[:2]
    asp_t = W / float(H)
    if w / h > asp_t:                 # 源图更宽 -> 高度取满，宽度居中裁
        base_ch, base_cw = h, h * asp_t
    else:                             # 源图更高 -> 宽度取满，高度居中裁
        base_cw, base_ch = w, w / asp_t
    cw, ch = base_cw / z, base_ch / z
    mx = (w - base_cw) / 2.0
    my = (h - base_ch) / 2.0
    ox = dx * (mx + (base_cw - cw) / 2.0)
    oy = dy * (my + (base_ch - ch) / 2.0)
    cx = w / 2.0 + ox
    cy = h / 2.0 + oy
    x0 = max(0, min(w - int(cw), int(round(cx - cw / 2))))
    y0 = max(0, min(h - int(ch), int(round(cy - ch / 2))))
    return x0, y0, int(cw), int(ch)


def _frame_for(img, phase, dirx, diry, W, H):
    """phase 0->1 跨整张图的连续运镜；phase=0 居中、phase=1 推近+漂移到位。"""
    z = 1.0 + PUSH_MAX * _ease(phase)
    dx = dirx * DRIFT_AMP * _ease(phase)
    dy = diry * DRIFT_AMP * _ease(phase)
    x0, y0, cw, ch = _crop_box(img, z, dx, dy, W, H)
    crop = img[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------------------
# 开场标题 overlay（PIL：斜体衬线白字、无阴影，仿参考片风格）
# ---------------------------------------------------------------------------
def _build_title(text, W, H, font_path=None):
    text = (text or "").strip()
    if not text:
        return None
    fpath = font_path or ITALIC
    target_w = int(W * TITLE_W_RATIO)
    size = 120
    while size > 12:
        f = ImageFont.truetype(fpath, size)
        d = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
        bb = d.textbbox((0, 0), text, font=f)
        if bb[2] - bb[0] <= target_w:
            break
        size -= 2
    f = ImageFont.truetype(fpath, size)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    bb = d.textbbox((0, 0), text, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x = (W - tw) // 2 - bb[0]
    y = int(H * TITLE_CY) - th // 2 - bb[1]
    d.text((x, y), text, font=f, fill=(255, 255, 255, 255))
    arr = np.array(layer)             # RGBA uint8
    rgb = arr[:, :, :3][:, :, ::-1].astype(np.float32)      # RGB -> BGR
    a = arr[:, :, 3:4].astype(np.float32) / 255.0
    return rgb, a


# ---------------------------------------------------------------------------
# 主渲染
# ---------------------------------------------------------------------------
def render_bw_reveal(images, out_path, ratio="9x16", fps=30,
                     text=DEFAULT_TEXT, use_text=True, progress=None, font=None):
    """N(2~12) 张图 -> 黑白->彩色 连续运镜揭示（仅画面，无音轨）。

    时长随张数自动调整：总镜头 = 2N（每图 黑白镜+彩色镜），总长 = min(12.5, 2N*0.55)。
    font: 客户自选字体键（core.fonts；None = 斜体 Cormorant Garamond 默认）。
    """
    from core import fonts as _fonts
    paths = [p for p in (images or []) if p and os.path.isfile(p)]
    if len(paths) < MIN_PHOTOS:
        raise ValueError("B&W->Color Reveal needs at least %d photos" % MIN_PHOTOS)
    if len(paths) > MAX_PHOTOS:
        paths = paths[:MAX_PHOTOS]

    OW, OH = cp.ratio_size(ratio)
    W, H = int(OW), int(OH)

    if progress:
        progress(8, 100, "Preparing photos")

    photos = [_imread_bgr(p) for p in paths]
    n = len(photos)

    # 每张图 2 镜（黑白 / 彩色），连续运镜：黑白 phase 0->0.5，彩色 phase 0.5->1.0
    total_target = min(TOTAL_CAP, 2 * n * SHOT_TARGET)
    per_shot = total_target / (2.0 * n)
    shot_frames = max(1, int(round(per_shot * fps)))
    total_frames = 2 * n * shot_frames

    dirs = [_DIRS[i % len(_DIRS)] for i in range(n)]
    title_rgb = None
    title_a = None
    if use_text:
        tt = _build_title(text, W, H, font_path=_fonts.ttf(font))
        if tt is not None:
            title_rgb, title_a = tt

    if progress:
        progress(12, 100, "Rendering")

    # 1) 逐镜渲染（每镜内部按 intra 推进 phase）
    shots = []  # list[list[frame BGR uint8]]
    for k in range(2 * n):
        iidx = k // 2
        half = k % 2
        img = photos[iidx]
        dx, dy = dirs[iidx]
        frs = []
        for fi in range(shot_frames):
            intra = (fi + 0.5) / shot_frames
            phase = (half + intra) / 2.0          # 0->0.5 (B&W) / 0.5->1.0 (color)
            fr = _frame_for(img, phase, dx, dy, W, H)
            if half == 0:                          # 黑白镜
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                fr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # 强制中性色度，消除 yuv420p 编码在灰阶上引入的色度噪点（否则均值饱和偏高）
                yuv = cv2.cvtColor(fr, cv2.COLOR_BGR2YUV)
                yuv[:, :, 1:] = 128
                fr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            frs.append(fr)
        shots.append(frs)
        if progress and (k % 4 == 0):
            progress(int(15 + 75 * k / (2 * n)), 100, "Rendering")

    # 2) 镜间交叉溶解（KH 帧）；溶解源取上一镜最后一帧（无标题），避免标题残影
    out_frames = []
    prev_last = None
    for k in range(2 * n):
        frs = shots[k]
        if prev_last is not None and k > 0:
            for j in range(KH):
                a = (j + 1) / (KH + 1)
                blended = cv2.addWeighted(prev_last, 1.0 - a, frs[j].astype(np.float32), a, 0.0)
                out_frames.append(np.clip(blended, 0, 255).astype(np.uint8))
        start = KH if (prev_last is not None and k > 0) else 0
        for fi in range(start, len(frs)):
            out_frames.append(frs[fi])
        prev_last = frs[-1].astype(np.float32)

    # 3) 开场标题（仅最前 TITLE_SECS，渐入渐出；溶解源为无标题帧，标题不渗入后续镜）
    if title_rgb is not None and title_a is not None:
        title_frames = int(round(TITLE_SECS * fps))
        for f in range(min(title_frames, len(out_frames))):
            ft = f / float(fps)
            fade = 1.0
            if ft < TITLE_FADE_IN:
                fade = ft / TITLE_FADE_IN
            elif ft > TITLE_SECS - TITLE_FADE_OUT:
                fade = max(0.0, (TITLE_SECS - ft) / TITLE_FADE_OUT)
            alpha = (title_a * fade)
            out_frames[f] = np.clip(
                out_frames[f].astype(np.float32) * (1.0 - alpha) + title_rgb * alpha,
                0, 255).astype(np.uint8)

    if progress:
        progress(92, 100, "Encoding")

    cmd = [cp.FFMPEG, "-y", "-hide_banner",
           "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", "%dx%d" % (W, H), "-r", str(fps), "-i", "-",
           "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", str(CRF), "-preset", PRESET, "-movflags", "+faststart",
           out_path]
    err_f = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=err_f, stderr=err_f)
    try:
        for fr in out_frames:
            proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed rc=%s" % proc.returncode)

    total_secs = len(out_frames) / float(fps)
    if progress:
        progress(100, 100, "Done")
    return {"video": out_path, "secs": round(total_secs, 2),
            "images": n, "ratio": ratio}
