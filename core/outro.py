# -*- coding: utf-8 -*-
"""客户品牌尾板（专属视频）——视频结尾追加一段居中 Logo 收尾页。

设计（2026-10-30 与用户对齐，2026-10-30b 改为请柬纸感）：
- 仅当客户上传了 Logo 才追加；不传则主片完全不变（调用方保证 branding=None/无 logo 时不调用）。
- 底色：**米色调纸面**（暖白 #FDFAF4 → 米 #F0E7D8 竖向渐变 + 极轻暗角 + 低频纸纹），
  贴合请柬/婚礼品类的高级纸感，不再是通用黑底。
- 颜色：文字一律墨色 #221F1B；logo 若本身是「浅色单色」（常见白底反白版），
  自动重着色为同款墨色，保证与背景对比；彩色 logo 原样保留。
  若上传的是白底 JPG（无透明通道），自动把**与边缘连通的白**抠成透明，避免出现白方块。
- 尺寸一致：logo 统一裁到「可见内容」外接框后再等比缩放，因此 JPG（白底抠除后）
  与 PNG（自带透明）在尾板里渲染尺寸一致，不会因四周留白而偏小。
- 出场特效（约 1.1s 内全部完成，余下时间静置定版）：
    1. 背景受光渐亮（柔光从 0.35 抬到 1.0）；
    2. Logo 淡入 + 由 0.95 微放大到 1.0 + 上浮 12px（ease-out）；
    3. 标语淡入 + 上浮（logo 下方不再保留下划线 / 细墨线）。
- 尾板的分辨率/帧率必须与主片严格一致，拼接才干净：从主片 ffmpeg -i 解析 WxH/fps/音轨，
  尾板强制对齐；最后用 filter_complex concat 重新编码（自带音轨匹配，最稳）。
- 失败安全：任何异常都返回 False，由 server 端决定是否保留原片（一律保留）。

依赖：numpy / Pillow / opencv-python-headless / imageio-ffmpeg（与现有栈一致）。
"""

import os
import re
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from core.ffproc import run_ffmpeg, _default_ff, probe_info, duration_of
from core import safedel


# 配色：与前端 --ink / --paper 同一套
INK = (34, 31, 27)          # #221F1B 墨色（文字/细线/浅色 logo 重着色）
PAPER_TOP = np.array([253, 250, 244], np.float32)   # #FDFAF4
PAPER_BOT = np.array([240, 231, 216], np.float32)   # #F0E7D8

OUTRO_SECS = 2.8            # 尾板默认时长（含约 1.1s 出场动效 + 约 1.7s 静置）

# 出场节奏（秒，可按尾板时长整体压缩）
T_LOGO0, T_LOGO_D = 0.06, 0.72
T_TAG0,  T_TAG_D = 0.58, 0.62
T_LIGHT_D = 0.95


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _clamp01(v):
    return 0.0 if v < 0 else (1.0 if v > 1 else v)


def _ease_out(p):
    """ease-out-cubic：起步快、收尾稳，最像镜头落定。"""
    p = _clamp01(p)
    return 1.0 - (1.0 - p) ** 3


def _ease_in_out(p):
    p = _clamp01(p)
    return p * p * (3.0 - 2.0 * p)


def _with_alpha(img, a):
    """按比例缩放 RGBA 图像的 alpha（a=1 原样返回）。"""
    if a >= 1.0:
        return img
    arr = np.array(img)
    arr[:, :, 3] = (arr[:, :, 3].astype(np.float32) * max(0.0, a)).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def _has_cjk(text):
    for ch in text or "":
        o = ord(ch)
        if 0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0x3000 <= o <= 0x30FF:
            return True
    return False


def _cjk_candidates():
    """中文标签的字体候选（随包优先，其次常见系统路径）。

    注意：随包目前只有拉丁字体，线上 Linux 若没有中文字体，中文标语会回落成方框
    —— 需要中文标语时请把 Noto Sans SC（OFL）放进 static/fonts/ 即可自动生效。
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fd = os.path.join(base, "static", "fonts")
    cands = []
    try:
        for f in sorted(os.listdir(fd)):
            low = f.lower()
            if low.endswith((".ttf", ".otf", ".ttc")) and (
                "cjk" in low or "noto" in low or "han" in low or "source" in low
            ):
                cands.append(os.path.join(fd, f))
    except Exception:
        pass
    cands += [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    return cands


def _font_for(text, size):
    from core import fonts
    path = fonts.ttf("serif")
    if _has_cjk(text):
        for cand in _cjk_candidates():
            if cand and os.path.isfile(cand):
                path = cand
                break
    try:
        return ImageFont.truetype(path, size) if path else ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# ffmpeg -i 解析：主片 WxH / fps / 是否含音轨
# ---------------------------------------------------------------------------
def _parse_probe(err, W=None, H=None, fps=None):
    m = re.search(r"Stream #0:\d.*?Video:.*?(\d{2,5})x(\d{2,5})", err)
    if m:
        W, H = int(m.group(1)), int(m.group(2))
    m2 = re.search(r"(\d+(?:\.\d+)?)\s*fps", err)
    if m2:
        fps = float(m2.group(1))
    has_audio = "Audio:" in err
    return (W or 1080), (H or 1920), (fps or 30.0), has_audio


# ---------------------------------------------------------------------------
# 背景：米色纸面（竖向渐变 + 极轻暗角 + 低频纸纹）
# ---------------------------------------------------------------------------
def _paper_bg(W, H):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = np.clip(0.78 * (yy / max(1, H - 1)) + 0.22 * (xx / max(1, W - 1)), 0, 1)
    col = PAPER_TOP[None, None, :] * (1 - t[..., None]) + PAPER_BOT[None, None, :] * t[..., None]

    # 极轻暗角：只在四角压 ~4%，让中间像被灯照着的纸
    rv = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vig = 1.0 - 0.042 * np.clip(rv - 0.55, 0, None)
    col *= vig[..., None]

    # 低频纸纹：避免大面积纯色渐变出现色带，同时有纸张肌理
    rng = np.random.default_rng(7)
    nh, nw = max(2, H // 14), max(2, W // 14)
    noise = rng.normal(0.0, 1.0, (nh, nw)).astype(np.float32)
    noise = cv2.resize(noise, (W, H), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), max(1.0, H * 0.0055))
    noise /= (float(np.abs(noise).max()) + 1e-6)
    col += noise[..., None] * 2.4

    return np.clip(col, 0, 255).astype(np.uint8)


def _glow_field(W, H):
    """中心偏上的柔光场（0..1），用于出场时「灯渐亮」。"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W * 0.5) / (W * 0.68)) ** 2 + ((yy - H * 0.40) / (H * 0.62)) ** 2)
    g = np.clip(1.0 - r, 0, 1) ** 1.7
    return (g * 9.0)[..., None]          # 最多提亮 ~9 级，非常克制


# ---------------------------------------------------------------------------
# Logo：载入 → 白底抠除 → 浅色单色重着色
# 形状保真红线（2026-10-30 客户要求：logo 长什么样就渲染什么样，圆形保持圆形）：
#   1. 载入用 PIL + exif_transpose（cv2 不认 EXIF，手机/设计软件导出的 logo 会转歪）；
#   2. 缩放一律 contain（thumbnail / 等比 resize），绝不拉伸；
#   3. 文件本就带透明通道 → 完全不做抠图（避免侵蚀圆边/羽边）；
#   4. 重着色只允许「真透明文件的反白 logo」或「白上加白（不抠就隐形）」两种情形，
#      白底 JPG 上客户自带的浅色块/圆底一律保持原色。
# ---------------------------------------------------------------------------
def _load_logo_rgba(logo_path):
    try:
        from PIL import ImageOps
        im = Image.open(logo_path)
        im = ImageOps.exif_transpose(im)          # EXIF 旋转归正
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        return np.asarray(im, dtype=np.uint8)     # 已是 RGBA
    except Exception:
        pass
    raw = np.fromfile(logo_path, dtype=np.uint8)  # 非 ASCII 路径安全
    dec = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if dec is None:
        return None
    if dec.ndim == 2:
        dec = cv2.cvtColor(dec, cv2.COLOR_GRAY2BGRA)
    if dec.shape[2] == 4:
        return cv2.cvtColor(dec, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGBA)


def _key_white_border(rgba, tol=16):
    """把「与画布边缘连通的近白」抠成透明（白底 JPG → 干净 logo）。

    只抠与边缘连通的部分，logo 内部的白色（徽章内衬、白色字）不受影响。
    返回 (out, leftover_cov)：leftover_cov = 抠后剩余不透明占比；
    若抠后几乎全空（logo 本身近白、与底同色）返回原稿，绝不交付一个「隐形 logo」。
    """
    try:
        rgb = rgba[:, :, :3]
        near = (rgb.min(axis=2) >= 255 - tol).astype(np.uint8)
        if float(near.mean()) < 0.12:
            return rgba, float((rgba[:, :, 3] > 40).mean())
        h, w = near.shape
        im = near.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        step = max(1, min(h, w) // 48)
        for x in range(0, w, step):
            for y in (0, h - 1):
                if im[y, x] == 1:
                    cv2.floodFill(im, mask, (x, y), 2)
        for y in range(0, h, step):
            for x in (0, w - 1):
                if im[y, x] == 1:
                    cv2.floodFill(im, mask, (x, y), 2)
        filled = (im == 2)
        if float(filled.mean()) < 0.10:
            return rgba, float((rgba[:, :, 3] > 40).mean())
        out = rgba.copy()
        a = np.where(filled, 0, 255).astype(np.uint8)
        a = cv2.erode(a, np.ones((3, 3), np.uint8), iterations=1)   # 去白边残留
        a = cv2.GaussianBlur(a, (0, 0), 0.8)
        out[:, :, 3] = a
        cov = float((out[:, :, 3] > 40).mean())
        if cov < 0.02:
            # 抠图把整个 logo 都吞了（白上加白）→ 保留客户上传的原样
            return rgba, float((rgba[:, :, 3] > 40).mean())
        return out, cov
    except Exception:
        return rgba, float((rgba[:, :, 3] > 40).mean())


def _recolor_light_mono(rgba, allow=True):
    """浅色单色 logo（反白版）→ 重着色为墨色，保证在米底上看得清。

    allow=False（客户上传的是白底 JPG 且抠完后仍保留大块形状）时直接原样返回：
    此时 logo 自带底色/浅色块（如浅色圆底徽章），重着色会破坏客户原稿，宁可不动。
    返回 (rgba, was_recolored)。彩色 logo / 深色 logo 原样返回。
    """
    if not allow:
        return rgba, False
    try:
        a = rgba[:, :, 3].astype(np.float32) / 255.0
        op = a > 0.12
        if int(op.sum()) < 100:
            return rgba, False
        px = rgba[:, :, :3].astype(np.float32)[op]
        luma = 0.299 * px[:, 0] + 0.587 * px[:, 1] + 0.114 * px[:, 2]
        sat = px.max(axis=1) - px.min(axis=1)
        if float(luma.mean()) > 150.0 and float(sat.mean()) < 34.0:
            out = rgba.copy()
            out[:, :, 0] = INK[0]
            out[:, :, 1] = INK[1]
            out[:, :, 2] = INK[2]
            return out, True
    except Exception:
        pass
    return rgba, False


def _crop_to_content(img):
    """裁到 alpha>阈值的可见内容外接框，剔除透明留白 / 抠图后残留的透明边。

    作用：白底 JPG 抠图后、或自带透明留白的 PNG，都按「真实 logo 内容」缩放，
    保证 JPG 与 PNG 在尾板里渲染尺寸一致（不再因四周留白而偏小）。
    含完全透明（理论上不会发生）或裁剪残差过小则原样返回。
    """
    try:
        a = np.array(img.split()[3], np.uint8)
        ys, xs = np.where(a > 16)
        if ys.size == 0:
            return img
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        if (y1 - y0) < 2 or (x1 - x0) < 2:
            return img
        return img.crop((x0, y0, x1, y1))
    except Exception:
        return img


def _fit_contain(img, max_w, max_h):
    """等比缩放到「恰好放进」max_w×max_h——**允许放大**。

    thumbnail() 只缩不放，裁边后内容较小的 logo（如高清画布上的小徽章）
    不会被放大到最大框，导致显得小。这里统一改为始终填满最大框，
    保证 JPG / PNG、大图 / 小图渲染尺寸一致且足够大。
    """
    w, h = img.size
    s = min(max_w / max(1, w), max_h / max(1, h))
    if s <= 0:
        return img
    return img.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.LANCZOS)


# ---------------------------------------------------------------------------
# 标语：自动换行 + 字距微调
# ---------------------------------------------------------------------------
def _measure(fnt, text):
    try:
        return float(fnt.getlength(text))
    except Exception:
        return float(len(text)) * float(getattr(fnt, "size", 20))


def _render_tagline(text, W, maxw_frac=0.74):
    """渲染标语为透明层（墨色、自动换行、拉丁字加字距）。返回 RGBA Image。"""
    size = max(14, int(W * 0.040))
    fnt = _font_for(text, size)
    tracking = size * (0.02 if _has_cjk(text) else 0.10)
    maxw = int(W * maxw_frac)

    words = text.split()
    if not words:
        words = [text]
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if not cur or _measure(fnt, test) + tracking * len(test) <= maxw:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    ls = int(size * 1.55)
    th = ls * len(lines) + int(size * 0.35)
    tw = 0
    for ln in lines:
        tw = max(tw, int(_measure(fnt, ln) + tracking * max(0, len(ln) - 1)))
    tw = max(1, tw)

    img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        x = 0.0
        for ch in ln:
            d.text((x, i * ls), ch, font=fnt, fill=INK + (238,))
            x += _measure(fnt, ch) + tracking
    return img


# ---------------------------------------------------------------------------
# 渲染尾板
# ---------------------------------------------------------------------------
def make_outro(logo_path, tagline, W, H, fps, secs=OUTRO_SECS, out_path=None):
    """生成一段与主片同 WxHxfps 的尾板 mp4（带静音音轨，便于与主片拼接）。

    返回 out_path；失败返回 None。
    """
    ff = _default_ff()
    rgba = _load_logo_rgba(logo_path)
    if rgba is None:
        return None
    had_alpha = bool(rgba[:, :, 3].min() < 250)
    if had_alpha:
        # 客户的文件本就带透明通道 → 一个像素都不动（圆形/羽边/半影全部原样保留）
        pass
    else:
        rgba, _cov_after = _key_white_border(rgba)
    # 抠完后剩余不透明占比：很低说明是「白上加白」白底文件里带深浅细节的 logo，允许转墨色；
    # 否则说明 logo 自带浅色底块（如圆形徽章），保持客户原色。
    after_cov = float((rgba[:, :, 3] > 40).mean())
    rgba, _recolored = _recolor_light_mono(rgba, allow=had_alpha or after_cov < 0.10)

    pil_logo = Image.fromarray(rgba)
    # 统一裁到可见内容外接框：白底 JPG 抠图后、或自带透明留白的 PNG，
    # 都按真实 logo 内容缩放，保证 JPG 与 PNG 渲染尺寸一致（不会因四周留白而偏小）。
    pil_logo = _crop_to_content(pil_logo)
    max_w, max_h = int(W * 0.47), int(H * 0.40)
    pil_logo = _fit_contain(pil_logo, max_w, max_h)
    lw0, lh0 = pil_logo.size
    cx = W // 2

    gap = int(H * 0.045)                  # logo 底 → 标语

    # 标语先渲出来拿高度，然后「logo + 标语」整块上下左右居中
    tag_img = None
    if tagline and tagline.strip():
        tag_img = _render_tagline(tagline.strip(), W)

    block_h = lh0 + (gap + tag_img.size[1] if tag_img is not None else 0)
    top = max(0, (H - block_h) // 2)
    ly = top                              # logo 顶部
    tag_pos = None
    if tag_img is not None:
        tag_pos = (cx - tag_img.size[0] // 2, ly + lh0 + gap)

    base = _paper_bg(W, H)
    glow = _glow_field(W, H)

    # 时长偏短时按比例压缩出场节奏，避免还没演完就结束
    k = min(1.0, float(secs) / 2.40)
    t_logo0, t_logo_d = T_LOGO0 * k, T_LOGO_D * k
    t_tag0, t_tag_d = T_TAG0 * k, T_TAG_D * k
    t_light_d = T_LIGHT_D * k

    total = max(1, int(round(secs * fps)))
    tmp_v = out_path + ".novid.mp4"
    vw = cv2.VideoWriter(tmp_v, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    if not vw.isOpened():
        try:
            safedel.unlink(tmp_v)
        except Exception:
            pass
        return None

    for i in range(total):
        t = i / float(fps)

        # 1) 背景受光渐亮
        lp = _ease_in_out(_clamp01(t / max(1e-3, t_light_d)))
        amp = 0.30 + 0.70 * lp
        frame = np.clip(base.astype(np.float32) + glow * amp, 0, 255).astype(np.uint8)
        img = Image.fromarray(frame)                       # RGB

        # 2) Logo：淡入 + 0.95→1.0 微放大 + 上浮（缩放保持中心不动）
        p = _ease_out(_clamp01((t - t_logo0) / max(1e-3, t_logo_d)))
        if p > 0:
            s = 0.95 + 0.05 * p
            w2, h2 = max(1, int(round(lw0 * s))), max(1, int(round(lh0 * s)))
            lg = pil_logo.resize((w2, h2), Image.LANCZOS)
            lg = _with_alpha(lg, p)
            dx = cx - w2 // 2
            dy = (ly + lh0 // 2) - h2 // 2 + int(round((1.0 - p) * H * 0.014))
            img.paste(lg, (dx, dy), lg.split()[3])

        # 3) 标语：淡入 + 上浮（logo 下方不再保留下划线）
        if tag_img is not None:
            q = _ease_out(_clamp01((t - t_tag0) / max(1e-3, t_tag_d)))
            if q > 0:
                tg = _with_alpha(tag_img, q)
                img.paste(tg, (tag_pos[0], tag_pos[1] + int(round((1.0 - q) * H * 0.010))),
                          tg.split()[3])

        vw.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    vw.release()

    # 加一段静音音轨（立体声 44100），让尾板与主片流数量一致，concat 最稳
    rc, _ = run_ffmpeg([ff, "-y", "-i", tmp_v,
                        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                        "-c:v", "copy", "-c:a", "aac", "-shortest",
                        "-map", "0:v", "-map", "1:a", out_path])
    try:
        safedel.unlink(tmp_v)
    except Exception:
        pass
    if rc != 0 or not os.path.isfile(out_path):
        return None
    return out_path


def append_outro(main_path, logo_path, tagline, fps=30.0, final_path=None,
                 has_audio=None, W=None, H=None, secs=OUTRO_SECS):
    """把品牌尾板拼接到主片之后，结果写回 final_path（默认覆盖主片同名 .brand.mp4）。

    返回 True 表示拼接成功；任何失败返回 False（调用方保留原片）。
    """
    if not os.path.isfile(main_path) or not os.path.isfile(logo_path):
        return False
    if final_path is None:
        final_path = main_path + ".brand.mp4"
    err = probe_info(main_path)
    W2, H2, fps2, has_audio2 = _parse_probe(err, W, H, fps)
    if has_audio is None:
        has_audio = has_audio2
    W, H, fps = W2, H2, fps2

    outro = final_path + ".outro.mp4"
    made = make_outro(logo_path, tagline, W, H, fps, secs=secs, out_path=outro)
    if not made:
        return False
    ff = _default_ff()
    if has_audio:
        flt = "[0:v][1:v]concat=n=2:v=1[v];[0:a][1:a]concat=n=2:v=0:a=1[a]"
        cmd = [ff, "-y", "-i", main_path, "-i", outro, "-filter_complex", flt,
               "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-movflags", "+faststart", final_path]
    else:
        flt = "[0:v][1:v]concat=n=2:v=1[v]"
        cmd = [ff, "-y", "-i", main_path, "-i", outro, "-filter_complex", flt,
               "-map", "[v]", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", final_path]
    rc, _ = run_ffmpeg(cmd, timeout=300)
    try:
        safedel.unlink(outro)
    except Exception:
        pass
    return rc == 0 and os.path.isfile(final_path)


def duration_of_safe(path):
    try:
        return duration_of(path)
    except Exception:
        return 0.0
