# -*- coding: utf-8 -*-
"""快剪模版（Fast Cut）—— N 张请柬图 → 节拍对齐的等间隔快剪 + 持久斜体字幕。

复刻用户 2026-10-21 确认的成品口径（参考 案例/快剪/document_810w-k5Y5ikVno0k.mp4，
 standalone 脚本 wed_fastcut_v2.py / v3 的既有取景）：

  * N(3~12) 张图，每张出场 **2~3 次（动态）**：≤6 张出场 3 次、≥7 张出场 2 次；
    取景只保留 **全景(1.00) + 中景(1.10) 两档**，且全部**居中裁切**
    （anchor 恒为 0.5），去掉旧版的上/下竖锚漂移与 1.16 特写推近，
    保证产品始终在画面中央、不被裁切；按「全员一轮」循环排序；
  * 每个镜头时长 = 1 个音乐拍长（cheery_monday BPM 101.3 → 0.592s），
    硬切、镜内完全静止（用户 2026-10-21 反馈：镜内漂移会被感知为晃动，已移除）；
  * 持久斜体衬线白字盖在 44% 屏高、水平居中、字宽上限 62%（可自定义，
    默认 "the details, the day"；show_text=False 或文字为空则不出字幕）；
  * 原图直出，不加任何滤镜 / 调色（用户红线）；不做运镜；
  * **时长随张数自动增加（appear 动态 2~3）**：总镜头数 = appear·N，
    总时长 = appear·N × 0.592s
    （3 张 ≈ 5.3s，6 张 ≈ 10.7s，12 张 ≈ 14.2s）；
  * 仅出画面，不含配乐；配乐由调用方（server）混 cheery_monday，与其它模版一致。

与 four_grid / scroll_gallery / multi_suite 完全独立：本模块只做「快剪 + 字幕」。

接口
----
render_fastcut(images, out_path, ratio="9x16", fps=30,
               text=DEFAULT_TEXT, show_text=True, progress=None)
  images   : 图片路径列表（3~12 张）
  text     : 持久字幕（可选；空则回退默认句）
  show_text: False 则完全不叠字幕（即便 text 非空）
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
# 节奏 / 风格常量（与 wed_fastcut_v2.py 既有成品一致）
# ---------------------------------------------------------------------------
BEAT = 0.592            # cheery_monday.mp3 BPM 101.3 → 拍长（s）；每镜 1 拍
# 出场次数动态：小批量 3 次（利落 + 推近节奏），大批量 2 次控总时长（用户 2026-10-21）
APPEAR_SMALL, APPEAR_BIG = 3, 2
APPEAR_BIG_THRESHOLD = 7     # N >= 该值 → 每图出场 2 次
# 只保留 全景 + 中景 两档取景，且全部居中（anchor 恒 0.5），去掉上/下漂移与特写。
ZOOM_FULL, ZOOM_MEDIUM = 1.00, 1.10
ANCHOR_CENTER = 0.50
TEXT_W_RATIO = 0.62     # 字幕字宽上限（占屏宽）
TEXT_CY = 0.44          # 字幕垂直中心（占屏高）
DEFAULT_TEXT = "the details, the day"
MIN_PHOTOS, MAX_PHOTOS = 3, 12

CRF = 17
PRESET = "slow"


# ---------------------------------------------------------------------------
# 图片读取
# ---------------------------------------------------------------------------
def _imread_bgr(p):
    """cv2 中文路径安全读取（红线：np.fromfile + imdecode）。"""
    arr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("cannot read %s" % p)
    from core.imgproc import normalize_luma   # 2026-10-27 亮度归一（用户需求）
    return normalize_luma(arr)


# ---------------------------------------------------------------------------
# 字幕 overlay（一次生成 RGBA，PIL；斜体衬线白字、无阴影，照搬参考模版）
# ---------------------------------------------------------------------------
def _build_text_overlay(text, W, H, font_path=None):
    """返回 (rgb, alpha) 两个 HxW 张量（BGR 空间，与画面同构便于叠加）。

    font_path: 客户自选字体（2026-10-26 core.fonts 注册表）；None = 斜体 CG 默认。
    """
    text = (text or "").strip()
    if not text:
        return np.zeros((H, W, 3), np.float32), np.zeros((H, W, 1), np.float32)
    fpath = font_path or ITALIC
    target_w = int(W * TEXT_W_RATIO)
    size = 100
    while size > 10:
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
    y = int(H * TEXT_CY) - th // 2 - bb[1]
    d.text((x, y), text, font=f, fill=(255, 255, 255, 255))
    arr = np.array(layer)   # RGBA uint8
    rgb = arr[:, :, :3][:, :, ::-1].astype(np.float32)      # RGB -> BGR
    a = arr[:, :, 3:4].astype(np.float32) / 255.0
    return rgb, a


# ---------------------------------------------------------------------------
# 单镜：静态取景（cover 裁切 + zoom + 竖锚），无镜内运动
# ---------------------------------------------------------------------------
def _static_shot(im, z, ay, W, H, txt_rgb, txt_a):
    """im: BGR 源图；z: 取景收紧倍率；ay: 竖锚(0=顶 0.5=中 1=底)。"""
    h, w = im.shape[:2]
    asp_t = W / float(H)          # 目标画幅宽高比
    if w / h > asp_t:             # 源图更宽 → 高度取满，宽度居中裁
        base_ch, base_cw = h, h * asp_t
    else:                         # 源图更高 → 宽度取满，高度居中裁
        base_cw, base_ch = w, w / asp_t
    cw, ch = base_cw / z, base_ch / z
    mx = (w - base_cw) / 2.0      # 基准裁切的水平余量
    my = (h - base_ch) / 2.0      # 基准裁切的垂直余量
    ax = (2 * ((ay * 3) % 1.0) - 1.0)                  # 竖锚派生出的水平偏移因子
    cx = w / 2 + ax * (mx + (base_cw - cw) / 2.0)
    cy = h / 2 + (2 * ay - 1.0) * (my + (base_ch - ch) / 2.0)
    x0 = max(0, min(w - int(cw), int(round(cx - cw / 2))))
    y0 = max(0, min(h - int(ch), int(round(cy - ch / 2))))
    crop = im[y0:y0 + int(ch), x0:x0 + int(cw)]
    frame = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LANCZOS4)
    if txt_a is not None and float(txt_a.max()) > 0:
        frame = (frame.astype(np.float32) * (1 - txt_a) + txt_rgb * txt_a).astype(np.uint8)
    return frame


def _appear_count(n):
    """每张图出场次数：小批量 3 次、大批量 2 次（控总时长）。"""
    return APPEAR_BIG if n >= APPEAR_BIG_THRESHOLD else APPEAR_SMALL


def _framings(appear):
    """取景序列：只有 全景(1.00)+中景(1.10) 两档、全部居中。
    出场 3 次 → [全景, 中景, 全景]（in-out 呼吸感）；出场 2 次 → [全景, 中景]。
    """
    base = [(ZOOM_FULL, ANCHOR_CENTER), (ZOOM_MEDIUM, ANCHOR_CENTER)]
    return [base[i % len(base)] for i in range(appear)]


def shot_plan(n):
    """返回 [(photo_idx, appear_idx), ...]：每张图出场 _appear_count(n) 次，按轮循环。"""
    appear = _appear_count(n)
    seq = []
    for r in range(appear):
        for p in range(n):
            seq.append((p, r))
    return seq


def cut_frames(n, fps):
    """节拍对齐的剪切点（帧）：每镜 1 拍，返回 (cuts, total_frames)。"""
    seq = shot_plan(n)
    cuts = [int(round(k * BEAT * fps)) for k in range(len(seq) + 1)]
    return cuts, cuts[-1]


def render_fastcut(images, out_path, ratio="9x16", fps=30,
                   text=DEFAULT_TEXT, show_text=True, progress=None, font=None):
    """N(3~12) 张请柬图 → 节拍快剪成片（仅画面，无音轨）。

    时长随张数自动增加：shots = 3N，secs = 3N × BEAT。
    font: 客户自选字体键（core.fonts；None = 斜体 CG 默认）。
    """
    from core import fonts as _fonts
    paths = [p for p in (images or []) if p and os.path.isfile(p)]
    if len(paths) < MIN_PHOTOS:
        raise ValueError("Fast Cut needs at least %d photos" % MIN_PHOTOS)
    if len(paths) > MAX_PHOTOS:
        paths = paths[:MAX_PHOTOS]

    OW, OH = cp.ratio_size(ratio)
    W, H = int(OW), int(OH)

    if progress:
        progress(8, 100, "Preparing photos")

    photos = [_imread_bgr(p) for p in paths]
    if show_text:
        txt_rgb, txt_a = _build_text_overlay(text, W, H, font_path=_fonts.ttf(font))
    else:
        txt_rgb, txt_a = None, None

    appear = _appear_count(len(paths))
    seq = shot_plan(len(paths))
    cuts, total = cut_frames(len(paths), fps)
    framings = _framings(appear)
    total_secs = total / float(fps)

    if progress:
        progress(12, 100, "Rendering")

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
        for si, (pi, ai) in enumerate(seq):
            f0, f1 = cuts[si], cuts[si + 1]
            z, ay = framings[ai % len(framings)]
            frame = _static_shot(photos[pi], z, ay, W, H, txt_rgb, txt_a)
            buf = frame.tobytes()
            for _ in range(f1 - f0):
                proc.stdin.write(buf)
                frame_no += 1
            if progress and (si % 6 == 0):
                progress(int(15 + 80 * si / max(len(seq), 1)), 100, "Rendering")
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed rc=%s" % proc.returncode)

    if progress:
        progress(100, 100, "Done")
    return {"video": out_path, "secs": round(total_secs, 2),
            "images": len(paths), "ratio": ratio, "appear": appear}
