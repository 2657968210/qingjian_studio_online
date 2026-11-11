# -*- coding: utf-8 -*-
"""Whip-Zoom Showcase（zoom_punch）—— N 张产品图 → 甩镜变焦转场 + 白条深色衬线标题。

复刻参考视频 韩梦瑶分享的视频.mp4（2026-11-10 用户提供，16.7s 竖屏）的成品口径：

  * 每张图出场 1 次、按顺序播放、不循环；每镜 ~1.8s 慢推/慢拉 Ken Burns
    （居中取景，奇偶镜推拉交替，产品恒在画面中央不被裁切）；
  * 镜与镜之间 = **逐个不同的快速转场**（参考片每个接点手法都不一样，绝不全片一个样）。
    7 种手法全部从参考片逐帧拆解得到，按原片出现顺序轮换、相邻永不重复：
      punch   放射缩放模糊 zoom punch          —— 原片 3.8s（平铺→镂空特写）
      slices  竖向条带挤压擦除                —— 原片 5.6s
      radial  强放射扇形模糊 + 旋转           —— 原片 7.6s
      soft    柔和交叉淡化 + 轻推             —— 原片 9.6s
      cut     干脆硬切 + 轻推                 —— 原片 11.6s
      defocus 失焦模糊淡出到白再清晰         —— 原片 13.2s
      hwhip   水平条纹运动模糊甩切           —— 原片 15.4s
    每个接点 0.4s，节奏统一；循环到第二圈时按 variant 改变方向 / 条带数 / 幅度，
    即使重复出现也不重样。可用 transitions= 显式指定序列覆盖默认轮换。
  * 标题文字 = 参考片开头 "2027 New Collection · Laser cut invites" 的样式：
    **深色衬线字压白色标签条**（两行各一条白 bar）+ 首行两侧四角星装饰；
    开场 0.3s 淡入，约 5.5s（跨前两镜）淡出；字体随包 Cormorant SemiBold；
  * 原图直出，不加滤镜调色；仅出画面不含配乐（BGM 由 server 混 canon_in_d）。

与 fastcut / decor_show / four_grid 完全独立：本模块只做「Ken Burns + 甩镜 + 标题」。

接口
----
render_zoom_punch(images, out_path, ratio="9x16", fps=30,
                  title=DEFAULT_TITLE, show_title=True, progress=None,
                  font=None, transitions=None)
  images     : 图片路径列表（3~99 张）
  title      : 标题文字（"·" 或换行分行；空则回退默认句）
  show_title : False 则完全不叠标题
  font       : 字体键（core.fonts；None = 衬线正体 CG，即参考片观感）
  transitions: 可选，转场序列（长度 >= n-1）；None = 按参考片顺序自动轮换
  返回 dict：{video, secs, images, ratio, transitions}
"""
import os
import math
import subprocess
import tempfile

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from core import card_pan as cp

# ---------------------------------------------------------------------------
# 字体
# ---------------------------------------------------------------------------
_FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "fonts")
SERIF = os.path.join(_FONT_DIR, "CormorantGaramond-SemiBold.ttf")

# ---------------------------------------------------------------------------
# 节奏 / 风格常量（照参考片量得）
# ---------------------------------------------------------------------------
PER_SCENE = 1.8          # 每镜时长（s）
TRANS = 0.4              # 甩镜转场时长（s）
ZB_NEAR, ZB_FAR = 1.00, 1.10     # Ken Burns 收紧范围（居中，不裁产品）
WHIP_ZOOM_MAX = 0.80     # 转场中点最大放大倍数（1.0 -> 1.40，参考片的 punch 幅度）
WHIP_BLUR_SPREAD = 0.12  # 径向模糊最大扩散（占缩放比例）
WHIP_STEPS = 6           # 径向模糊采样层数
WHIP_XFADE = 0.16        # 换镜点附近交叉淡化半宽（p 空间）

# 参考片逐接点拆解出的 7 种转场，按原片出现顺序排列。
# 相邻永远不同；超过 7 个接点后循环，但用 variant 改方向/条带数/幅度避免重样。
TRANSITION_SEQUENCE = ("punch", "slices", "radial", "soft", "cut", "defocus", "hwhip")
HWhip_STEPS = 6          # 水平/放射模糊采样层数
SLICE_BANDS = 9          # 竖条带数量（variant 会改成 7 / 11）

# 标题（参考片原文）
DEFAULT_TITLE = "2027 New Collection \u00b7 Laser cut invites"
TITLE_W_RATIO = 0.70     # 每行字宽上限（占屏宽）
TITLE_BLOCK_CY = 0.105   # 标题块中心（占屏高；参考片两行块 ~7%-14%）
TITLE_BAR_PADX = 0.030   # 白条水平内边距（占屏宽）
TITLE_BAR_PADY = 0.009   # 白条垂直内边距（占屏高；参考片白条扁、贴字）
TITLE_GAP = 0.001        # 两行白条间隙（占屏高；参考片两行几乎贴合）
TITLE_ALPHA_MAX = 0.96   # 白条不透明度（参考片近乎实心）
TITLE_COLOR = (34, 30, 28)      # 深色字（近黑暖调）
TITLE_STAR_RATIO = 0.024        # 四角星外径（占屏宽）
TITLE_T0 = 0.30          # 标题淡入起点（s）
TITLE_FADE = 0.40        # 淡入/淡出时长（s）
TITLE_HOLD = 5.2         # 标题从 T0 起持续显示时长（s，参考片 ~0.3-5.5s）

MIN_PHOTOS, MAX_PHOTOS = 3, 99
CRF = 17
PRESET = "medium"


# ---------------------------------------------------------------------------
# 图片读取（中文路径安全 + 亮度归一，与 fastcut 同口径）
# ---------------------------------------------------------------------------
def _imread_bgr(p):
    arr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("cannot read %s" % p)
    from core.imgproc import normalize_luma
    return normalize_luma(arr)


# ---------------------------------------------------------------------------
# 取景：cover 裁切 + 缩放（全部居中，不裁产品）
# ---------------------------------------------------------------------------
def _cover_crop(im, z, W, H):
    """im: BGR 源图；z: 取景收紧倍率（1.0=全景）。返回 W×H 帧。"""
    h, w = im.shape[:2]
    asp_t = W / float(H)
    if w / h > asp_t:
        base_ch, base_cw = h, h * asp_t
    else:
        base_cw, base_ch = w, w / asp_t
    cw, ch = base_cw / z, base_ch / z
    x0 = max(0, min(w - int(cw), int(round(w / 2 - cw / 2))))
    y0 = max(0, min(h - int(ch), int(round(h / 2 - ch / 2))))
    crop = im[y0:y0 + int(ch), x0:x0 + int(cw)]
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_LANCZOS4)


def _kb_frame(im, z0, z1, p, W, H):
    """Ken Burns：p∈[0,1]，z 从 z0 平滑过渡到 z1（居中）。"""
    z = z0 + (z1 - z0) * cp.smoothstep(p)
    return _cover_crop(im, z, W, H)


# ---------------------------------------------------------------------------
# 甩镜变焦转场
# ---------------------------------------------------------------------------
def _zoom_blur(im, amt, W, H):
    """径向（缩放）动态模糊：amt∈[0,1]，1 = 最大扩散。中心不动。"""
    if amt <= 0.02:
        return im
    h, w = im.shape[:2]
    c = (w / 2.0, h / 2.0)
    acc = np.zeros((h, w, 3), np.float32)
    for i in range(WHIP_STEPS):
        s = 1.0 + amt * WHIP_BLUR_SPREAD * (i / float(WHIP_STEPS - 1))
        M = cv2.getRotationMatrix2D(c, 0.0, s)
        acc += cv2.warpAffine(im, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    return np.clip(acc / WHIP_STEPS, 0, 255).astype(np.uint8)


def _whip_frame(im_out, im_in, p, W, H, variant=0):
    """转场帧：p∈[0,1]。参考片 3.8s 的放射缩放 punch。

    p<0.5：前镜由 1.0 放大到 1+WHIP_ZOOM_MAX/2，模糊渐强；
    p>0.5：后镜由 1+WHIP_ZOOM_MAX/2 回落到 1.0，模糊渐弱；
    中点 ±WHIP_XFADE/2 交叉淡化（两侧都处于重模糊，切换无痕）。
    variant 改变 punch 幅度，让第二圈出现时不是同一个力度。
    """
    amp = WHIP_ZOOM_MAX + (0.0, -0.14, 0.12)[variant % 3]
    zf = 1.0 + amp * 0.5
    if p <= 0.5:
        q = p / 0.5
        z = 1.0 + (zf - 1.0) * (q * q)                    # ease-in 加速放大
        base = _cover_crop(im_out, z, W, H)
        frame = _zoom_blur(base, q, W, H)
    else:
        q = (p - 0.5) / 0.5
        z = zf - (zf - 1.0) * (1.0 - (1.0 - q) * (1.0 - q))   # ease-out 回落
        base = _cover_crop(im_in, z, W, H)
        frame = _zoom_blur(base, 1.0 - q, W, H)
    # 中点附近交叉淡化
    lo, hi = 0.5 - WHIP_XFADE / 2.0, 0.5 + WHIP_XFADE / 2.0
    if lo <= p <= hi:
        a = (p - lo) / WHIP_XFADE                          # 0..1 = 前镜权重
        fo = _cover_crop(im_out, zf, W, H)
        fi = _cover_crop(im_in, zf, W, H)
        fo = _zoom_blur(fo, 1.0, W, H)
        fi = _zoom_blur(fi, 1.0, W, H)
        frame = (fo.astype(np.float32) * a + fi.astype(np.float32) * (1.0 - a)).astype(np.uint8)
    return frame


# ---------------------------------------------------------------------------
# 转场库（7 种，全部由参考片逐接点拆解得到）
# 统一签名：(im_out, im_in, p, W, H, variant=0) -> BGR 帧
# 相邻接点永不同款；循环第二圈靠 variant 改方向 / 条带数 / 幅度，避免重样。
# ---------------------------------------------------------------------------
def _blend(a, b, q):
    """q=0 全 a，q=1 全 b。"""
    q = max(0.0, min(1.0, float(q)))
    if q <= 0.0:
        return a
    if q >= 1.0:
        return b
    return (a.astype(np.float32) * (1.0 - q) +
            b.astype(np.float32) * q).astype(np.uint8)


def _radial_blur(im, amt, W, H, spin):
    """放射扇形条纹模糊：多层「旋转 + 缩放」叠加 → 参考片 7.6s 的绿色风车。"""
    if amt <= 0.02:
        return im
    c = (W / 2.0, H / 2.0)
    acc = np.zeros((H, W, 3), np.float32)
    for i in range(HWhip_STEPS):
        u = i / float(HWhip_STEPS - 1) - 0.5
        s = 1.0 + amt * WHIP_BLUR_SPREAD * u
        ang = spin * amt * 9.0 * u
        M = cv2.getRotationMatrix2D(c, ang, s)
        acc += cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    return np.clip(acc / HWhip_STEPS, 0, 255).astype(np.uint8)


def _radial_frame(im_out, im_in, p, W, H, variant=0):
    """radial：强放射扇形模糊 + 旋转（原片 7.6s）。"""
    spin = 1.0 if variant % 2 == 0 else -1.0
    zf = 1.0 + 0.34
    if p <= 0.5:
        q = p / 0.5
        z = 1.0 + (zf - 1.0) * (q * q)
        frame = _radial_blur(_cover_crop(im_out, z, W, H), q, W, H, spin)
    else:
        q = (p - 0.5) / 0.5
        z = zf - (zf - 1.0) * (1.0 - (1.0 - q) ** 2)
        frame = _radial_blur(_cover_crop(im_in, z, W, H), 1.0 - q, W, H, spin)
    lo, hi = 0.5 - WHIP_XFADE / 2.0, 0.5 + WHIP_XFADE / 2.0
    if lo <= p <= hi:
        a = (p - lo) / WHIP_XFADE
        fo = _radial_blur(_cover_crop(im_out, zf, W, H), 1.0, W, H, spin)
        fi = _radial_blur(_cover_crop(im_in, zf, W, H), 1.0, W, H, spin)
        frame = _blend(fi, fo, a)
    return frame


def _slice_frame(im_out, im_in, p, W, H, variant=0):
    """slices：竖向条带挤压擦除，交替上下错峰滑入（原片 5.6s）。"""
    nb = (SLICE_BANDS, 7, 11, 8)[variant % 4]
    sign = 1 if variant % 2 == 0 else -1
    squeeze = 1.0 + 0.07 * math.sin(math.pi * p)
    fo = _cover_crop(im_out, squeeze, W, H)
    fi = _cover_crop(im_in, squeeze * 1.02, W, H)
    out = np.empty_like(fo)
    bw = int(math.ceil(W / float(nb)))
    for b in range(nb):
        x0, x1 = b * bw, min(W, (b + 1) * bw)
        if x0 >= x1:
            break
        delay = 0.28 * ((b % 3) / 3.0)
        q = cp.smoothstep((p - delay) / (1.0 - delay))
        shift = int(round(sign * (1 if b % 2 == 0 else -1) * (1.0 - q) * H * 0.28))
        tile = np.roll(fi[:, x0:x1], shift, axis=0)
        out[:, x0:x1] = _blend(fo[:, x0:x1], tile, q)
    return out


def _hblur(im, amt, W, H, sign):
    """水平方向运动模糊：多层水平平移叠加 → 参考片 15.4s 的横向拖影。"""
    if amt <= 0.02:
        return im
    acc = np.zeros((H, W, 3), np.float32)
    for i in range(HWhip_STEPS):
        u = i / float(HWhip_STEPS - 1) - 0.5
        dx = sign * amt * 0.16 * W * u
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        acc += cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    return np.clip(acc / HWhip_STEPS, 0, 255).astype(np.uint8)


def _hwhip_frame(im_out, im_in, p, W, H, variant=0):
    """hwhip：水平条纹运动模糊甩切（原片 15.4s）。"""
    sign = 1.0 if variant % 2 == 0 else -1.0
    if p <= 0.5:
        q = p / 0.5
        frame = _hblur(_cover_crop(im_out, 1.0 + 0.10 * q, W, H), q, W, H, sign)
    else:
        q = (p - 0.5) / 0.5
        frame = _hblur(_cover_crop(im_in, 1.10 - 0.10 * q, W, H), 1.0 - q, W, H, sign)
    lo, hi = 0.5 - WHIP_XFADE / 2.0, 0.5 + WHIP_XFADE / 2.0
    if lo <= p <= hi:
        a = (p - lo) / WHIP_XFADE
        fo = _hblur(_cover_crop(im_out, 1.10, W, H), 1.0, W, H, sign)
        fi = _hblur(_cover_crop(im_in, 1.10, W, H), 1.0, W, H, sign)
        frame = _blend(fi, fo, a)
    return frame


def _defocus_frame(im_out, im_in, p, W, H, variant=0):
    """defocus：失焦模糊 → 泛白 → 重新清晰（原片 13.2s）。"""
    white = np.full((H, W, 3), (240, 246, 250) if variant % 2 else (252, 252, 252), np.uint8)
    tri = math.sin(math.pi * p)
    k = int(2 + 14 * tri)
    if k % 2 == 0:
        k += 1
    if p <= 0.5:
        base = _cover_crop(im_out, 1.0 + 0.04 * (p / 0.5), W, H)
    else:
        base = _cover_crop(im_in, 1.06 - 0.06 * ((p - 0.5) / 0.5), W, H)
    blurred = cv2.GaussianBlur(base, (k, k), 0) if k >= 3 else base
    # 原片是「失焦到泛白、仍隐约可见轮廓」，不是纯白闪，白混合封顶 0.55
    return _blend(_blend(base, blurred, tri * 0.9), white, 0.55 * tri)


def _soft_frame(im_out, im_in, p, W, H, variant=0):
    """soft：柔和交叉淡化 + 轻推，中点轻微提亮（原片 9.6s）。"""
    q = cp.smoothstep(p)
    amp = 0.05 + 0.02 * (variant % 2)
    fo = _cover_crop(im_out, 1.0 + amp * q, W, H)
    fi = _cover_crop(im_in, 1.0 + amp * (1.0 - q), W, H)
    frame = _blend(fo, fi, q)
    lift = 0.16 * math.sin(math.pi * p)
    if lift > 0.002:
        frame = _blend(frame, np.full((H, W, 3), 255, np.uint8), lift)
    return frame


def _cut_frame(im_out, im_in, p, W, H, variant=0):
    """cut：干脆硬切 + 轻推，无过渡帧（原片 11.6s）。"""
    push = 0.05 + 0.02 * (variant % 3)
    if p < 0.48:
        return _cover_crop(im_out, 1.0 + push * (p / 0.48), W, H)
    q = (p - 0.48) / 0.52
    return _cover_crop(im_in, 1.0 + push * (1.0 - q), W, H)


_TRANSITIONS = {
    "punch": _whip_frame,
    "slices": _slice_frame,
    "radial": _radial_frame,
    "soft": _soft_frame,
    "cut": _cut_frame,
    "defocus": _defocus_frame,
    "hwhip": _hwhip_frame,
}
VALID_TRANSITIONS = tuple(_TRANSITIONS.keys())


def resolve_transitions(n, seq=None):
    """返回长度 n-1 的转场名列表；None = 按参考片顺序轮换，相邻永不重复。"""
    count = max(0, n - 1)
    if seq:
        out = [str(x).lower() for x in seq if str(x).lower() in _TRANSITIONS]
        while len(out) < count:
            cand = TRANSITION_SEQUENCE[len(out) % len(TRANSITION_SEQUENCE)]
            if out and cand == out[-1]:
                cand = TRANSITION_SEQUENCE[(len(out) + 1) % len(TRANSITION_SEQUENCE)]
            out.append(cand)
        return out[:count]
    return [TRANSITION_SEQUENCE[i % len(TRANSITION_SEQUENCE)] for i in range(count)]


# ---------------------------------------------------------------------------
# 标题 overlay（白条 + 深色衬线字 + 四角星），一次生成 RGBA
# ---------------------------------------------------------------------------
def _star(draw, cx, cy, r, fill):
    """四角星（参考片首行两侧的装饰）。"""
    ir = r * 0.30
    pts = []
    for k in range(8):
        ang = k * np.pi / 4.0 - np.pi / 2.0
        rad = r if k % 2 == 0 else ir
        pts.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
    draw.polygon(pts, fill=fill)


def _split_title(text):
    """标题分行：优先手动换行，其次 "·" 分两行，否则单行。"""
    t = (text or "").strip()
    if not t:
        t = DEFAULT_TITLE
    if "\n" in t:
        return [s.strip() for s in t.split("\n") if s.strip()][:2]
    if "\u00b7" in t:
        a, b = t.split("\u00b7", 1)
        return [a.strip(), b.strip()]
    if " - " in t:
        a, b = t.split(" - ", 1)
        return [a.strip(), b.strip()]
    return [t]


def _build_title_overlay(text, W, H, font_path=None):
    """返回 (rgb, base_alpha) —— BGR 空间 overlay 与基础 alpha 图（含白条+字+星）。"""
    fpath = font_path or SERIF
    lines = _split_title(text)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # 每行独立按宽度上限取字号（参考片两行字号一致，取两行最小值统一）
    size = int(H * 0.052)
    fonts = []
    for ln in lines:
        s = size
        while s > 10:
            f = ImageFont.truetype(fpath, s)
            bb = d.textbbox((0, 0), ln, font=f)
            if bb[2] - bb[0] <= W * TITLE_W_RATIO:
                break
            s -= 2
        fonts.append((ln, ImageFont.truetype(fpath, s), d.textbbox((0, 0), ln, font=f)))
    size = min(f.size for _ln, f, _bb in fonts) if fonts else size
    fonts = []
    for ln in lines:
        f = ImageFont.truetype(fpath, size)
        fonts.append((ln, f, d.textbbox((0, 0), ln, font=f)))

    pad_x = int(W * TITLE_BAR_PADX)
    pad_y = int(H * TITLE_BAR_PADY)
    gap = int(H * TITLE_GAP)
    bars = []
    total_h = 0
    for ln, f, bb in fonts:
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        bw = tw + 2 * pad_x
        bh = th + 2 * pad_y
        bars.append((ln, f, bb, bw, bh))
        total_h += bh
    total_h += gap * (len(bars) - 1)
    y = int(H * TITLE_BLOCK_CY) - total_h // 2
    dark = TITLE_COLOR + (255,)
    white = (255, 255, 255, int(255 * TITLE_ALPHA_MAX))
    l1_box = None
    for i, (ln, f, bb, bw, bh) in enumerate(bars):
        x = (W - bw) // 2
        d.rectangle([x, y, x + bw, y + bh], fill=white)
        tx = x + pad_x - bb[0]
        ty = y + pad_y - bb[1]
        d.text((tx, ty), ln, font=f, fill=dark)
        if i == 0:
            l1_box = (x, y, x + bw, y + bh)
        y += bh + gap
    # 首行两侧四角星（压在白条左右边缘外侧）
    if l1_box:
        r = max(6, int(W * TITLE_STAR_RATIO))
        lx, ty1, rx, by1 = l1_box
        cy = (ty1 + by1) // 2
        _star(d, lx - int(r * 0.9), ty1 + int(r * 0.9), r, dark)
        _star(d, rx + int(r * 0.9), ty1 + int(r * 0.9), r, dark)
    arr = np.array(layer)
    rgb = arr[:, :, :3][:, :, ::-1].astype(np.float32)
    a = arr[:, :, 3:4].astype(np.float32) / 255.0
    return rgb, a


def _title_alpha(t, total_secs, n):
    """标题 alpha 随时间：0.3s 淡入，持 TITLE_HOLD（跨前两镜），淡出。"""
    if t < TITLE_T0:
        return 0.0
    tin = min(1.0, (t - TITLE_T0) / TITLE_FADE)
    # 持续窗口：默认 TITLE_HOLD，但至少到第 1 镜结束、至多到第 3 镜结束
    end = TITLE_T0 + TITLE_HOLD
    scene_end2 = 2 * PER_SCENE + TRANS
    scene_end3 = 3 * PER_SCENE + 2 * TRANS
    end = max(end, PER_SCENE + 0.2)
    end = min(end, scene_end3 if n >= 3 else scene_end2, total_secs - 0.2)
    tout = 1.0
    if t > end - TITLE_FADE:
        tout = max(0.0, (end - t) / TITLE_FADE)
    a = cp.smoothstep(min(1.0, tin)) * cp.smoothstep(min(1.0, tout))
    return float(max(0.0, min(1.0, a)))


# ---------------------------------------------------------------------------
# 主渲染
# ---------------------------------------------------------------------------
def render_zoom_punch(images, out_path, ratio="9x16", fps=30,
                      title=DEFAULT_TITLE, show_title=True,
                      progress=None, font=None, transitions=None):
    """N(3~99) 张产品图 → 逐个不同转场的成片（仅画面，无音轨）。"""
    from core import fonts as _fonts
    paths = [p for p in (images or []) if p and os.path.isfile(p)]
    if len(paths) < MIN_PHOTOS:
        raise ValueError("Whip-Zoom Showcase needs at least %d photos" % MIN_PHOTOS)
    if len(paths) > MAX_PHOTOS:
        paths = paths[:MAX_PHOTOS]

    W, H = cp.ratio_size(ratio)
    if progress:
        progress(8, 100, "Preparing photos")
    photos = [_imread_bgr(p) for p in paths]
    n = len(photos)

    if show_title:
        # 参考片标题是「正体衬线」。font 未指定时锁 SERIF(正体)，
        # 绝不能走 fonts.ttf(None)——norm(None) 会回落 DEFAULT_KEY="italic" 斜体。
        fpath = _fonts.ttf(font) if font else None
        ttl_rgb, ttl_a = _build_title_overlay(title, W, H, font_path=fpath)
    else:
        ttl_rgb, ttl_a = None, None

    # 时间轴：n 镜 + (n-1) 个转场
    scene_f = int(round(PER_SCENE * fps))
    trans_f = int(round(TRANS * fps))
    total = n * scene_f + (n - 1) * trans_f
    total_secs = total / float(fps)

    # 每镜推拉方向：偶数镜推近(1.0->1.10)、奇数镜拉远(1.10->1.0)，首镜推近同参考片
    dirs = []
    for i in range(n):
        dirs.append((ZB_NEAR, ZB_FAR) if i % 2 == 0 else (ZB_FAR, ZB_NEAR))

    # 每个接点的转场手法：按参考片顺序轮换，相邻永不重复
    trans_kinds = resolve_transitions(n, transitions)

    if progress:
        progress(14, 100, "Rendering")

    # 转场帧本身是高速模糊 / 运动态：除硬切外一律半分辨率合成再放大。
    # 源图常为 4000px 级，1080x1920 上多层 warpAffine 极慢；半分辨率成本约 1/4，
    # 且放大后与原片那种「甩镜拖影」观感一致（实测 3.7x 提速、无可见画质损失）。
    Wt, Ht = W // 2, H // 2

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
        frame_no = 0
        for si in range(n):
            f0 = si * (scene_f + trans_f)
            z0, z1 = dirs[si]
            for k in range(scene_f):
                frame = _kb_frame(photos[si], z0, z1, k / float(scene_f - 1), W, H)
                # 标题（淡入淡出窗口内叠加）
                if ttl_a is not None:
                    t = frame_no / float(fps)
                    a = _title_alpha(t, total_secs, n)
                    if a > 0.003:
                        frame = (frame.astype(np.float32) * (1 - ttl_a * a)
                                 + ttl_rgb * (ttl_a * a)).astype(np.uint8)
                proc.stdin.write(frame.tobytes())
                frame_no += 1
            if si < n - 1:
                fn = _TRANSITIONS[trans_kinds[si]]
                variant = si // len(TRANSITION_SEQUENCE)
                half = trans_kinds[si] != "cut"
                for k in range(trans_f):
                    p = (k + 0.5) / trans_f
                    if half:
                        small = fn(photos[si], photos[si + 1], p, Wt, Ht, variant)
                        frame = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
                    else:
                        frame = fn(photos[si], photos[si + 1], p, W, H, variant)
                    if ttl_a is not None:
                        t = frame_no / float(fps)
                        a = _title_alpha(t, total_secs, n)
                        if a > 0.003:
                            frame = (frame.astype(np.float32) * (1 - ttl_a * a)
                                     + ttl_rgb * (ttl_a * a)).astype(np.uint8)
                    proc.stdin.write(frame.tobytes())
                    frame_no += 1
            if progress and si % 3 == 0:
                progress(int(15 + 80 * frame_no / max(total, 1)), 100, "Rendering")
    finally:
        if proc.stdin:
            proc.stdin.close()
        try:
            proc.wait()
        except BaseException:
            pass

    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed rc=%s" % proc.returncode)

    if progress:
        progress(100, 100, "Done")
    return {"video": out_path, "secs": round(total_secs, 2),
            "images": n, "ratio": ratio,
            "transitions": trans_kinds}
