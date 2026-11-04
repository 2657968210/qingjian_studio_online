# -*- coding: utf-8 -*-
"""四宫格开场 + 单图快切模版（Grid Showcase）。

参考 document_810w-QwngsntVQZc.mp4 的样式，用户 2026-10-20 确认的成品口径：

  * 段一：上传图的前 4 张做 2×2 无缝铺满整屏，四格错峰依次淡入出现
    （每格间隔 CELL_STAG，慢浮现），全部出现后再定格 1s 转单图切换；
    中央可选叠文字：TITLE（衬线单行，如 WEDDING TREND）+
    SUB（花体，如 Dusty Blue），白色 + 柔和阴影；
    show_text=False 或两者皆空则不叠任何文字；不叠 logo/网址；
    图片不加任何暗化遮罩、不做调色（用户红线：无滤镜调色）。
  * 段二：全部 N 张图依次单图展示，硬切，**不使用运镜**（2026-11-02 用户定：
    Grid Showcase 单图显示只做切换）；每张满幅停留 1.2~1.8s（按序交替，落在 1~2s
    区间内，制造轻微节奏）；单图上不叠任何文字；总时长 = 四宫格定格 +
    各图停留时长之和，随图片数增长。
  * 仅出画面，不含配乐；配乐由调用方（server）混，与本工作台其它模版一致。

与 scroll_gallery / multi_suite 完全独立：本模块只做「四宫格 + 快切」。

接口
----
render_four_grid(images, out_path, ratio="9x16", fps=30,
                 title="", subtitle="", show_text=True, progress=None)
  images   : 图片路径列表（≥4 张；前 4 张做四宫格，全部 N 张做单图快切）
  title    : 衬线单行中央文字（可选）
  subtitle : 花体中央文字（可选）
  show_text: False 则完全不叠文字（即便 title/subtitle 非空）
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
from core import image_motion as imn

# ---------------------------------------------------------------------------
# 字体（随包 static/fonts/，Linux 部署不依赖系统字体）
# ---------------------------------------------------------------------------
_FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "fonts")
SERIF = os.path.join(_FONT_DIR, "CormorantGaramond-SemiBold.ttf")
SCRIPT = os.path.join(_FONT_DIR, "GreatVibes-Regular.ttf")

# ---------------------------------------------------------------------------
# 节奏 / 风格常量（与用户 2026-10-20 确认一致）
# ---------------------------------------------------------------------------
TITLE_RATIO = 1.5       # 衬线（方体）标题字号 = 花体副标题 × 1.5（方体字明显大于花体字）

CELL_FADE = 0.5         # 每格淡入时长（s）
CELL_STAG = 0.5         # 四格依次出现的错峰间隔（s）—— 稍微慢一点
GRID_APPEAR = CELL_STAG * 3 + CELL_FADE   # 四格全部出现的时间点
GRID_DUR = GRID_APPEAR + 1.0              # 出现后再定格 1s 转单图切换
# 文字依次显现：标题先浮、副标题随后（均在第2张出现后启动，慢浮现）
TXT_T0, TXT_T1 = CELL_STAG + 0.15, CELL_STAG + 1.25   # 标题淡入窗口
TXT_S0, TXT_S1 = CELL_STAG + 0.80, CELL_STAG + 1.95   # 副标题淡入窗口（标题约6成后启动）
CUT_DUR_MIN = 1.3       # 单图展示每张最短停留（s）—— 2026-11-02 用户定：不运镜，仅切换，每张 1~2s
CUT_DUR_MAX = 1.7       # 单图展示每张最长停留（s）—— 与 MIN 交替，落在 1~2s 区间内

CRF = 17
PRESET = "slow"


# ---------------------------------------------------------------------------
# 图片读取 / 裁切
# ---------------------------------------------------------------------------
def _imread_bgr(p):
    """cv2 中文路径安全读取（红线：np.fromfile + imdecode）。"""
    arr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("cannot read %s" % p)
    from core.imgproc import normalize_luma   # 2026-10-27 亮度归一（用户需求）
    return normalize_luma(arr)


def _cover_resize(bgr, tw, th):
    ih, iw = bgr.shape[:2]
    scale = max(tw / iw, th / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    r = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return r[top:top + th, left:left + tw]


# ---------------------------------------------------------------------------
# 文字 overlay（一次生成 RGBA，PIL）
# ---------------------------------------------------------------------------
def _draw_tracked(d, cx, cy, text, font, tracking, fill):
    widths = [d.textlength(ch, font=font) for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    asc, desc = font.getmetrics()
    y = cy - (asc + desc) // 2
    x = cx - total / 2
    for ch, w in zip(text, widths):
        d.text((x, y), ch, font=font, fill=fill)
        x += w + tracking


def _measure(text, font, trk):
    d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    return sum(d.textlength(c, font=font) for c in text) + trk * (len(text) - 1)


def _fit_font(path, text, trk_ratio, target_w, start=100):
    """按目标像素宽自适应字号，返回 (font, tracking)。"""
    f0 = ImageFont.truetype(path, start)
    w0 = _measure(text, f0, trk_ratio * start)
    size = max(20, int(round(start * target_w / w0)))
    return ImageFont.truetype(path, size), trk_ratio * size


def _make_line_overlay(text, font, trk, cy, W, H):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if not text:
        return np.zeros((H, W, 4), np.uint8)   # 空 overlay
    d = ImageDraw.Draw(ov)
    shadow = (0, 0, 0, 120)
    white = (255, 255, 255, 255)
    _draw_tracked(d, W // 2 + 2, cy + 4, text, font, trk, shadow)   # 柔和阴影
    _draw_tracked(d, W // 2, cy, text, font, trk, white)
    return np.array(ov)   # RGBA uint8


def _build_text_overlay(title, subtitle, W, H, font_key=None):
    """返回 (标题 overlay, 副标题 overlay) 两张独立 RGBA，便于依次显现。

    只叠有效（非空）的行：两行都在 → 标题上、副标题下；只有一行 → 居中。
    font_key: 客户自选字体键（2026-10-26 core.fonts；None = 模版默认衬线）——
    只替换标题字体；副标题保持花体 GreatVibes（若客户选的就是花体，
    副标题改用细斜体，避免两行同花体失去层级）。
    """
    from core import fonts as _fonts
    title = (title or "").strip()
    subtitle = (subtitle or "").strip()
    tpath = _fonts.ttf(font_key) or SERIF
    spath = SCRIPT
    try:
        if os.path.abspath(tpath) == os.path.abspath(SCRIPT):
            spath = _fonts.ttf("light") or SERIF       # 标题=花体时副标题用细斜体
    except Exception:
        pass
    both = bool(title) and bool(subtitle)
    if both:
        title_cy, sub_cy = int(0.458 * H), int(0.568 * H)
    else:
        mid = int(0.513 * H)
        title_cy, sub_cy = mid, mid
    # 花体副标题：适配至屏宽 56%（2026-11-03 用户：第二行花体字号太小，加大）
    title_basis = 0
    if subtitle:
        f_script, _ = _fit_font(spath, subtitle, 0.0, 0.56 * W)
        sub_px = f_script.size
        # 方体标题仍按旧花体基准（0.46W）取字号——2026-11-03 用户：方体大小不变
        f_basis, _ = _fit_font(spath, subtitle, 0.0, 0.46 * W)
        title_basis = f_basis.size
    else:
        sub_px = 0
    # 衬线（方体）标题：明确大于花体副标题（× TITLE_RATIO，基准=旧花体字号），
    # 限最大屏宽 86% 防出屏
    if title:
        title_px = (int(round(title_basis * TITLE_RATIO)) if title_basis
                    else int(0.11 * W))
        trk = 0.14 * title_px
        while _measure(title, ImageFont.truetype(tpath, title_px), trk) > 0.86 * W and title_px > 40:
            title_px -= 2
        f_serif = ImageFont.truetype(tpath, title_px)
        trk = 0.14 * title_px
    else:
        f_serif, trk = None, 0
    title_ov = _make_line_overlay(title, f_serif, trk, title_cy, W, H) if title else np.zeros((H, W, 4), np.uint8)
    sub_ov = _make_line_overlay(subtitle, f_script, 0, sub_cy, W, H) if subtitle else np.zeros((H, W, 4), np.uint8)
    return title_ov, sub_ov


def _ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)   # smoothstep 缓动：慢慢浮现


def render_four_grid(images, out_path, ratio="9x16", fps=30,
                     title="", subtitle="", show_text=True, progress=None,
                     font=None):
    """N(≥4) 张请柬图 → 四宫格(前4) + 全图快切成片（仅画面，无音轨）。

    font: 客户自选字体键（core.fonts；只替换标题字体，None = 模版默认衬线）。
    """
    from core import fonts as _fonts
    paths = [p for p in (images or []) if p and os.path.isfile(p)]
    if len(paths) < 4:
        raise ValueError("Grid Showcase needs at least 4 photos")

    OW, OH = cp.ratio_size(ratio)
    W, H = int(OW), int(OH)
    cw, ch = W // 2, H // 2

    if progress:
        progress(8, 100, "Preparing photos")

    cells_bgr = [_cover_resize(_imread_bgr(p), cw, ch) for p in paths[:4]]
    cell_pos = [(0, 0), (cw, 0), (0, ch), (cw, ch)]

    # 单图段：加载原图并裁成满幅，每张硬切展示（2026-11-02 用户定：不运镜，仅切换）
    ASP_F = W / float(H)
    imgs_full = [_cover_resize(_imread_bgr(p), W, H) for p in paths]   # BGR 满幅

    # 文字 overlay（可选）
    if show_text:
        title_ov, sub_ov = _build_text_overlay(title, subtitle, W, H, font_key=font)
    else:
        title_ov, sub_ov = np.zeros((H, W, 4), np.uint8), np.zeros((H, W, 4), np.uint8)
    title_rgb = title_ov[:, :, :3].astype(np.float32)
    title_a = title_ov[:, :, 3].astype(np.float32)[..., None] / 255.0
    sub_rgb = sub_ov[:, :, :3].astype(np.float32)
    sub_a = sub_ov[:, :, 3].astype(np.float32)[..., None] / 255.0

    grid_frames = int(round(GRID_DUR * fps))
    # 每张停留时长在 [CUT_DUR_MIN, CUT_DUR_MAX] 内按序交替；逐张硬切，无运镜
    durs = [CUT_DUR_MIN + (CUT_DUR_MAX - CUT_DUR_MIN) * (k % 2)
            for k in range(len(paths))]
    cut_frames_list = [int(round(d * fps)) for d in durs]
    seg_starts = []
    _s = grid_frames
    for cf in cut_frames_list:
        seg_starts.append(_s)
        _s += cf
    total = _s
    total_secs = total / fps

    if progress:
        progress(12, 100, "Rendering")

    cmd = [cp.FFMPEG, "-y", "-hide_banner",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (W, H), "-r", str(fps), "-i", "-",
           "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", str(CRF), "-preset", PRESET, "-movflags", "+faststart",
           out_path]
    err_f = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=err_f, stderr=err_f)
    try:
        for i in range(total):
            if i < grid_frames:
                t = i / fps
                frame = np.zeros((H, W, 3), np.float32)
                for ci, (x, y) in enumerate(cell_pos):
                    a = min(1.0, max(0.0, (t - ci * CELL_STAG) / CELL_FADE))
                    cell = cells_bgr[ci][..., ::-1].astype(np.float32)  # BGR->RGB
                    frame[y:y + ch, x:x + cw] += cell * a
                ta = _ease((t - TXT_T0) / (TXT_T1 - TXT_T0))   # 标题先浮
                if ta > 0:
                    frame = frame * (1 - title_a * ta) + title_rgb * (title_a * ta)
                ts = _ease((t - TXT_S0) / (TXT_S1 - TXT_S0))   # 副标题随后
                if ts > 0:
                    frame = frame * (1 - sub_a * ts) + sub_rgb * (sub_a * ts)
            else:
                # 单图段：满幅硬切展示，不使用运镜
                img_i = len(seg_starts) - 1
                for _j, _st in enumerate(seg_starts):
                    if i >= _st:
                        img_i = _j
                frame = imgs_full[img_i][..., ::-1].astype(np.float32)   # BGR->RGB
            proc.stdin.write(frame.astype(np.uint8).tobytes())
            if progress and (i % 60 == 0):
                progress(int(15 + 80 * i / max(total, 1)), 100, "Rendering")
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed rc=%s" % proc.returncode)

    if progress:
        progress(100, 100, "Done")
    return {"video": out_path, "secs": round(total_secs, 2),
            "images": len(paths), "ratio": ratio}
