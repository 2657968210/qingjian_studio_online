# -*- coding: utf-8 -*-
"""before_after.py —— Mock Up / Finished Invitation 前后对比模版（Before / After Reveal）。

口径（2026-11-24 用户定稿，取自工作台验证版 build_capcut_style）：

  * 第一张 = 设计稿（mockup），停留 0.7s（快速进入开场拉幕），标题横条 **Mock Up**；
  * 其余 = 实拍图，每张停留 1.2s（末张 1.5s），标题横条 **Finished Invitation**；
  * 文字样式：白色底纹（宽度贴合文字，左右 26px 留白）+ 黑色衬线大字，
    位置在画面上方约 6% 高度；字体用随包 Cormorant Garamond SemiBold
    （SIL OFL 可商用，本地/线上表现一致）；
  * 图片按所选画幅居中裁剪铺满（无留白、无背景填充、无拉伸），不做缩放动画；
  * 转场五段全不同且形态拉开：向右拉揭幕（1→2，复刻参考片 New Before &
    Snap 开头：一条竖直边界匀速从左向右推进，左侧逐步露出 Finished Invitation 新图；5.5s）→
    横向左推 → 圆形揭幕 → 垂直上移 → 斜向擦除（各 0.4s）；
    超过 6 张时后续转场在 push/iris/slide_up/wipe 中循环；
  * 仅出画面，不含配乐；配乐由调用方（server）混，与其它模版一致。

接口
----
render_before_after(images, out_path, ratio="9x16", fps=30,
                    title="", use_text=True, progress=None)
  images : 图片路径列表（>=2 张；首张为设计稿，其余为实拍）
  ratio  : 画幅（card_pan.RATIOS，默认 9x16）
  fps    : 帧率（默认 30）
  title  : 首张标题（空则用默认句）
  use_text : 是否烧标题横条（bool，默认 True）
  返回 dict：{video, secs, images, ratio}
"""
import math
import os
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.card_pan import RATIOS, FFMPEG
from core.fonts import ttf

CRF = 17
PRESET = "veryfast"

FONT_SIZE = 76          # 标题字号（1080 宽基准，随画幅宽缩放）
BAND_TOP_FRAC = 0.0625  # 横条顶部 ≈ 6% 高度
BAND_H_FRAC = 0.068     # 横条高 ≈ 130/1920
PAD_X_FRAC = 0.024      # 底纹左右留白 ≈ 26/1080
BAND_RGB = (255, 255, 255)
TEXT_RGB = (18, 16, 14)

HOLD_FIRST = 0.7        # 首张停留秒数（开场不等待太久即开始拉幕）
HOLD_MID = 1.2          # 中间各张停留秒数
HOLD_LAST = 1.5         # 末张停留秒数
TRANS_SHORT = 0.4       # 常规转场时长
TRANS_REVEAL = 5.5      # 1→2 向右拉揭幕时长（匀速推进，用户定稿 5-6s）

_TRANS_SEQ = ["pull", "push", "iris", "slide_up", "wipe"]
_TRANS_CYCLE = ["push", "iris", "slide_up", "wipe"]


def _smooth(p):
    return p * p * (3 - 2 * p)


def _still(path, W, H):
    """按画幅居中裁剪铺满：无留白、无背景填充、无拉伸。"""
    im = Image.open(path).convert("RGB")
    iw, ih = im.size
    k = max(W / iw, H / ih)
    nw, nh = max(W, int(round(iw * k))), max(H, int(round(ih * k)))
    im = im.resize((nw, nh), Image.LANCZOS)
    x, y = (nw - W) // 2, (nh - H) // 2
    return np.asarray(im.crop((x, y, x + W, y + H)))


def _title_banner(fr, txt, alpha, W, H, font):
    """白底纹 + 黑色衬线大字；底纹宽度贴合文字。"""
    if alpha <= 0:
        return fr
    im = Image.fromarray(fr, "RGB").convert("RGBA")
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    bb = d.textbbox((0, 0), txt, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    band_h = int(round(H * BAND_H_FRAC))
    band_top = int(round(H * BAND_TOP_FRAC))
    pad = int(round(W * PAD_X_FRAC))
    tx = (W - tw) / 2 - bb[0]
    ty = band_top + (band_h - th) / 2 - bb[1]
    d.rectangle((tx + bb[0] - pad, band_top, tx + bb[0] + tw + pad, band_top + band_h),
                fill=BAND_RGB + (int(235 * alpha),))
    d.text((tx, ty), txt, font=font, fill=TEXT_RGB + (int(255 * alpha),))
    return np.asarray(Image.alpha_composite(im, lay).convert("RGB"))


def _trans_reveal(a, b, p, W, H):
    """向右拉揭幕（复刻参考片 NEW-Before-and-SnapYT.App.mp4 开头）：
    一条竖直边界以匀速从左向右推进，边界左侧露出新图 b（Finished Invitation），
    右侧仍是旧图 a（mockup）；硬切、无闪白、无缩放、无缓动。
    p 线性递增 → edge 匀速推进，p=0 全旧图，p=1 全新图。"""
    p = min(1.0, max(0.0, p))
    edge = int(round(W * p))
    out = a.copy()
    if edge > 0:
        out[:, :edge] = b[:, :edge]
    return out


def _trans_push(a, b, p, W, H):
    off = int(round(W * p))
    out = np.empty_like(a)
    out[:, :W - off] = a[:, off:]
    out[:, W - off:] = b[:, :off]
    return out


def _trans_dissolve(a, b, p, W, H):
    return (a * (1 - p) + b * p).astype(np.uint8)


def _trans_slide_up(a, b, p, W, H):
    off = int(round(H * p))
    out = np.empty_like(a)
    out[:H - off] = a[off:]
    out[H - off:] = b[:off]
    return out


def _trans_iris(a, b, p, W, H):
    """圆形揭幕：中心羽化扩散。"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W / 2.0, H * 0.45
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = math.hypot(cx, H - cy) * 1.05
    feather = W * 0.10
    t = p * (rmax + feather) - feather
    m = np.clip((dist - t) / feather, 0, 1)[..., None]
    return np.clip(b.astype(np.float32) * (1 - m) + a.astype(np.float32) * m,
                   0, 255).astype(np.uint8)


def _trans_wipe(a, b, p, W, H):
    yy, xx = np.mgrid[0:H, 0:W]
    mask = ((xx / W) * 0.45 + (yy / H) * 0.55) <= p
    out = a.copy()
    out[mask] = b[mask]
    return out


_TRANS_FN = dict(pull=_trans_reveal, push=_trans_push, dissolve=_trans_dissolve,
                 iris=_trans_iris, slide_up=_trans_slide_up, wipe=_trans_wipe)


def render_before_after(images, out_path, ratio="9x16", fps=30,
                        title="", label="", use_text=True, progress=None):
    if isinstance(use_text, str):
        use_text = use_text.lower() in ("1", "true", "yes", "on")
    W, H = RATIOS.get(ratio, RATIOS["9x16"])
    fps = int(fps or 30)
    n = len(images)
    raw = [_still(p, W, H) for p in images]

    font = ImageFont.truetype(ttf("serif"),
                              max(28, int(round(FONT_SIZE * W / 1080.0))))
    labels = [(title or "Mock Up")] + [(label or "Finished Invitation")] * (n - 1)

    if use_text:
        texted = [_title_banner(fr, lb, 1.0, W, H, font) for fr, lb in zip(raw, labels)]
    else:
        texted = raw
        labels = [""] * n

    holds = [int(round(HOLD_FIRST * fps))] + \
            [int(round(HOLD_MID * fps))] * (n - 2) + [int(round(HOLD_LAST * fps))]
    types = []
    for i in range(n - 1):
        if i < len(_TRANS_SEQ):
            types.append(_TRANS_SEQ[i])
        else:
            types.append(_TRANS_CYCLE[(i - len(_TRANS_SEQ)) % len(_TRANS_CYCLE)])
    durs = [TRANS_REVEAL if t == "pull" else TRANS_SHORT for t in types]

    total = sum(holds) + sum(int(round(d * fps)) for d in durs)
    done = 0

    def tick(msg=""):
        nonlocal done
        done += 1
        if progress:
            try:
                progress(done, total, msg)
            except Exception:
                pass

    def gen():
        for i in range(n):
            for _ in range(holds[i]):
                tick()
                yield texted[i]
            if i < n - 1:
                t = types[i]
                steps = max(2, int(round(durs[i] * fps)))
                fn = _TRANS_FN[t]
                for j in range(steps):
                    tick()
                    p = (j + 1) / (steps + 1)
                    if t in ("pull", "dissolve"):
                        fr = fn(raw[i], raw[i + 1], p, W, H)
                        if use_text:
                            a1 = 1.0 if p < 0.30 else max(0.0, 1.0 - (p - 0.30) / 0.20)
                            a2 = 0.0 if p < 0.50 else min(1.0, (p - 0.50) / 0.20)
                            fr = _title_banner(fr, labels[i], a1, W, H, font)
                            fr = _title_banner(fr, labels[i + 1], a2, W, H, font)
                    else:
                        fr = fn(texted[i], texted[i + 1], p, W, H)
                    yield fr

    cmd = [FFMPEG, "-y", "-hide_banner",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (W, H), "-r", str(fps), "-i", "-",
           "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", str(CRF), "-preset", PRESET, "-movflags", "+faststart",
           out_path]
    err_f = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=err_f, stderr=err_f)
    try:
        for fr in gen():
            if fr.dtype != np.uint8:
                fr = np.clip(fr, 0, 255).astype(np.uint8)
            proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        err_f.seek(0)
        msg = err_f.read(400).decode("utf-8", "replace")
        raise RuntimeError("before_after ffmpeg failed rc=%s: %s" % (proc.returncode, msg))
    return {"video": out_path, "secs": round(total / fps, 2),
            "images": n, "ratio": ratio}
