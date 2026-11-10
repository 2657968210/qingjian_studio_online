# -*- coding: utf-8 -*-
"""Whip-Zoom Showcase（zoom_punch）—— N 张产品图 → 甩镜变焦转场 + 白条深色衬线标题。

复刻参考视频 韩梦瑶分享的视频.mp4（2026-11-10 用户提供，16.7s 竖屏）的成品口径：

  * 每张图出场 1 次、按顺序播放、不循环；每镜 ~1.8s 慢推/慢拉 Ken Burns
    （居中取景，奇偶镜推拉交替，产品恒在画面中央不被裁切）；
  * 镜与镜之间 = **甩镜变焦转场（whip zoom）**：前镜快速放大 + 径向动态模糊，
    后镜从放大态快速回落，中点重模糊处换镜（0.4s）——参考片 3.7s / 7.6s / 9.5s
    处的标志性 "zoom punch"；
  * 标题文字 = 参考片开头 "2027 New Collection · Laser cut invites" 的样式：
    **深色衬线字压白色标签条**（两行各一条白 bar）+ 首行两侧四角星装饰；
    开场 0.3s 淡入，约 5.5s（跨前两镜）淡出；字体随包 Cormorant SemiBold；
  * 原图直出，不加滤镜调色；仅出画面不含配乐（BGM 由 server 混 canon_in_d）。

与 fastcut / decor_show / four_grid 完全独立：本模块只做「Ken Burns + 甩镜 + 标题」。

接口
----
render_zoom_punch(images, out_path, ratio="9x16", fps=30,
                  title=DEFAULT_TITLE, show_title=True, progress=None, font=None)
  images    : 图片路径列表（3~99 张）
  title     : 标题文字（"·" 或换行分行；空则回退默认句）
  show_title: False 则完全不叠标题
  font      : 字体键（core.fonts；None = 衬线正体 CG，即参考片观感）
  返回 dict：{video, secs, images, ratio}
"""
import os
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


def _whip_frame(im_out, im_in, p, W, H):
    """转场帧：p∈[0,1]。

    p<0.5：前镜由 1.0 放大到 1+WHIP_ZOOM_MAX/2，模糊渐强；
    p>0.5：后镜由 1+WHIP_ZOOM_MAX/2 回落到 1.0，模糊渐弱；
    中点 ±WHIP_XFADE/2 交叉淡化（两侧都处于重模糊，切换无痕）。
    """
    zf = 1.0 + WHIP_ZOOM_MAX * 0.5
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
                      progress=None, font=None):
    """N(3~99) 张产品图 → 甩镜变焦成片（仅画面，无音轨）。"""
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

    if progress:
        progress(14, 100, "Rendering")

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
                for k in range(trans_f):
                    p = (k + 0.5) / trans_f
                    frame = _whip_frame(photos[si], photos[si + 1], p, W, H)
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
            "images": n, "ratio": ratio}
