# -*- coding: utf-8 -*-
"""New Collection template (侧小中大轮播 / carousel)
====================================================
N product photos (2-8) -> elegant card carousel:
every photo shows as a large centred card while the neighbours peek in
smaller from the left/right edges; on each beat the whole chain slides one
slot and every card scales continuously (small at the edges -> full size at
centre). Cards stay fully opaque during the slide (reference-matched).

Replicates the reference ad (document_810w-La8R5_QeYCQ.mp4, measured
2026-10-28) on a 1080x1920 canvas:
  * main card 655x1055 (aspect 0.621), centre x=540, vertical centre y=1043
  * CENTRE card is full size; the immediate LEFT/RIGHT neighbours peek in
    SMALLER (scale 0.795) from the edges (user brief: sides show other
    styles, smaller than the centre) - the whole chain slides right-to-left
  * bg RGB(241,234,230); reference-matched type in Poppins:
    bold gold headline (title2, "NEW COLLECTION", top), light sans subtitle
    (title1, "Our Wedding Invitations", under it), thin gold divider line, then footer
    block "Explore Now" + dash + circular arrow badge with the website below
    - all three texts user-editable
  * each product holds centre ~1.5 s; then a quick 0.5 s right-to-left swap
    (smootherstep eased) - the centre card slides LEFT while the next card
    slides in from the RIGHT and grows to full size. Neighbours are drawn at
    rel -2..+2 so the chain reads as one unbroken conveyor (no edge pop),
    and the 2026-10-28 smoother() easing bug (returning -1 at p=1) is fixed.

Zero AI dependency; fonts come from static/fonts (Poppins, OFL).
Completely independent from the other templates' pipelines.
"""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from core.ffproc import run_ffmpeg, duration_of
import imageio_ffmpeg

SLOT = 2.0          # seconds per photo (hold 1.5 s + slide 0.5 s)
TRANS = 0.5         # quick right-to-left swap duration
S_SIDE = 0.795      # side-card scale (smaller than the centre card)
S_AMP = 1.0 - S_SIDE
BG = (241, 234, 230)          # cream canvas
GOLD = (166, 124, 82)         # A67C52 wedding antique gold (headline/CTA)
INK2 = (88, 81, 73)           # warm taupe (subtitle / website)

_FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "static", "fonts")
_FONT_MAP = {
    "head": ("Poppins-Bold.ttf", "arialbd.ttf"),
    "sub": ("Poppins-Light.ttf", "arial.ttf"),
    "reg": ("Poppins-Regular.ttf", "arial.ttf"),
}


def pick_font(size, kind="head"):
    p = os.path.join(_FONT_DIR, _FONT_MAP[kind][0])
    if not os.path.exists(p):
        p = os.path.join(_FONT_DIR, _FONT_MAP[kind][1])
    return ImageFont.truetype(p, size)


def _dims(ratio):
    r = (ratio or "9x16").lower()
    if "1x1" in r or "square" in r:
        return 1080, 1080
    return 1080, 1920


def smoother(p):
    # Standard smootherstep (Perlin): 6p^5 - 15p^4 + 10p^3.
    # 1st AND 2nd derivative = 0 at 0 and 1. 2026-10-28 修正：上一版误写成
    # 七次式，p=1 时等于 -1（e 从 0 走到 -1），导致每次切换实际退回上一张
    # 再跳下一张——用户看到的"没停到切换的这张图、跳到另外一张"即此根因。
    p = min(1.0, max(0.0, p))
    return p * p * p * (10.0 + p * (-15.0 + p * 6.0))


def _load_card(fp, tw, th):
    """Read photo (non-ASCII path safe), cover-crop to card size."""
    img = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("decode fail: " + fp)
    h, w = img.shape[:2]
    s = max(tw / float(w), th / float(h))
    nw, nh = int(round(w * s + 0.5)), int(round(h * s + 0.5))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = (nw - tw) // 2
    y0 = (nh - th) // 2
    img = img[y0:y0 + th, x0:x0 + tw]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # -> RGB (pipeline is rgb24)


def _rounded_mask(tw, th, rad):
    m = np.zeros((th, tw), np.float32)
    cv2.rectangle(m, (rad, 0), (tw - rad - 1, th - 1), 1.0, -1)
    cv2.rectangle(m, (0, rad), (tw - 1, th - rad - 1), 1.0, -1)
    for cx, cy in ((rad, rad), (tw - rad - 1, rad),
                   (rad, th - rad - 1), (tw - rad - 1, th - rad - 1)):
        cv2.circle(m, (cx, cy), rad, 1.0, -1)
    return m


def _draw_text(dr, W, cy, txt, font, fill, kind="reg", target_w=None):
    """Centred text; target_w (px) distributes letter-spacing (caps headline)."""
    if target_w:
        nat = dr.textlength(txt, font=font)
        if len(txt) > 1 and nat < target_w:
            tr = (target_w - nat) / (len(txt) - 1)
            x = (W - (nat + tr * (len(txt) - 1))) / 2.0
            for chx in txt:
                dr.text((x, cy), chx, font=font, fill=fill, anchor="ls")
                x += dr.textlength(chx, font=font) + tr
            return
    tw = dr.textlength(txt, font=font)
    if tw > W * 0.88:                     # auto-shrink long custom text
        font = pick_font(max(18, int(font.size * W * 0.88 / tw)), kind)
        tw = dr.textlength(txt, font=font)
    dr.text(((W - tw) / 2.0, cy), txt, font=font, fill=fill, anchor="ls")


def _bg_canvas(W, H, SS, t1, t2, footer):
    """Static background+text, reference-matched type, wedding gold on cream."""
    W2, H2 = W * SS, H * SS
    img = Image.new("RGB", (W2, H2), BG)
    d = ImageDraw.Draw(img)
    f_head = pick_font(int(0.050 * H) * SS, "head")
    f_sub = pick_font(int(0.0305 * H) * SS, "sub")
    f_exp = pick_font(int(0.0370 * H) * SS, "head")
    f_site = pick_font(int(0.0225 * H) * SS, "reg")
    # top block: bold gold headline (title2), light subtitle (title1) under it
    if t2:
        _draw_text(d, W2, int(0.098 * H2) + int(f_head.size * 0.35),
                   t2, f_head, GOLD, "head", target_w=W2 * 0.66)
    if t1:
        _draw_text(d, W2, int(0.147 * H2) + int(f_sub.size * 0.35),
                   t1, f_sub, INK2, "sub")
    if t1 or t2:                          # thin gold divider (reference-matched)
        dw = int(0.175 * W2)
        dh = max(SS, int(0.0028 * H2))
        dy = int(0.190 * H2)
        d.rectangle([(W2 - dw) // 2, dy, (W2 + dw) // 2, dy + dh], fill=GOLD)
    # footer block: "Explore Now" + dash + circular arrow badge, website below
    ey = int(0.880 * H2)                  # optical centre of the CTA row
    lw = d.textlength("Explore Now", font=f_exp)
    dash_w = int(0.040 * W2)
    gap = int(0.022 * W2)
    r = int(0.031 * W2)
    total = lw + gap + dash_w + gap + 2 * r
    x = (W2 - total) / 2.0
    d.text((x, ey + int(f_exp.size * 0.35)), "Explore Now",
           font=f_exp, fill=GOLD, anchor="ls")
    xd = x + lw + gap
    yd = ey - int(0.002 * H2)
    d.rectangle([xd, yd, xd + dash_w, yd + max(SS, int(0.0028 * H2))], fill=GOLD)
    bx = xd + dash_w + gap + r
    by = ey + int(0.004 * H2)
    d.ellipse([bx - r, by - r, bx + r, by + r], fill=GOLD)
    aw = max(2 * SS, int(r * 0.16))       # cream arrow, pointing right
    d.line([bx - 0.45 * r, by, bx + 0.32 * r, by], fill=BG, width=aw)
    d.line([bx + 0.02 * r, by - 0.30 * r, bx + 0.35 * r, by], fill=BG, width=aw)
    d.line([bx + 0.02 * r, by + 0.30 * r, bx + 0.35 * r, by], fill=BG, width=aw)
    if footer:
        _draw_text(d, W2, int(0.930 * H2) + int(f_site.size * 0.35),
                   footer, f_site, INK2, "reg")
    return np.asarray(img).astype(np.float32)


def render_carousel(srcs, out, bgm=None, ratio="9x16", fps=30, progress=None,
                    title1="Our Wedding Invitations", title2="NEW COLLECTION",
                    footer="www.janicepaper.com", silent=False):
    if progress is None:
        progress = lambda d, t, m="": None
    W, H = _dims(ratio)
    SS = 2
    n = len(srcs)
    progress(2, 100, "Reading %d photos" % n)

    # card geometry (fractions measured on the 1080x1920 reference rebuild)
    card_h = int(round(0.5493 * H)) * SS
    card_w = int(round(card_h * 0.6207))
    cyc = 0.5431 * H * SS
    rad = max(4, int(round(0.01615 * H)) * SS)
    slot_d = card_w * 0.96                      # centre spacing between slots

    cards = [_load_card(fp, card_w, card_h) for fp in srcs]
    mask = _rounded_mask(card_w, card_h, rad)
    W2, H2 = W * SS, H * SS
    BG2 = _bg_canvas(W, H, SS, title1, title2, footer)
    N = int(fps * SLOT * n)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = out + ".raw.mp4"
    vcmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "%dx%d" % (W, H), "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    err_f = open(os.devnull, "wb")
    proc = __import__("subprocess").Popen(vcmd, stdin=__import__("subprocess").PIPE,
                                          stdout=err_f, stderr=err_f)

    def put(frame):
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def paste(fr, card, msk, cx, cy):
        """Alpha-composite card (scaled size already applied) centred at (cx, cy)."""
        mh, mw = msk.shape[:2]
        x0 = int(round(cx - mw / 2.0))
        y0 = int(round(cy - mh / 2.0))
        xa, xb = max(0, x0), min(W2, x0 + mw)
        ya, yb = max(0, y0), min(H2, y0 + mh)
        if xb <= xa or yb <= ya:
            return
        a = msk[ya - y0:yb - y0, xa - x0:xb - x0][:, :, None]
        fr[ya:yb, xa:xb] = (fr[ya:yb, xa:xb] * (1 - a) +
                            card[ya - y0:yb - y0, xa - x0:xb - x0] * a)

    for i in range(N):
        t = i / float(fps)
        k = int(t // SLOT)
        u = (t - k * SLOT) / SLOT
        e = 0.0
        if u >= 1 - TRANS / SLOT and k < n - 1:
            # 2026-10-28 定稿：右->左传送带。缓动用已修复的 smootherstep
            # （6p^5-15p^4+10p^3，0->1 单调）。每帧画 rel -2..+2 五张卡：
            # 中间卡(rel=0)满尺寸居中；左右邻卡(rel=±1)缩小 0.795 从边缘
            # peek 露边（用户口径：两侧也要有其它款式、比中间小）；rel=±2
            # 在屏外待命、滑行时连续补入，左右缘永不空、无弹出跳变。
            e = smoother((u - (1 - TRANS / SLOT)) / (TRANS / SLOT))
        frame = BG2.copy()
        # continuous right-to-left chain: card coord c = rel - e; scale shrinks
        # with distance from centre (centre full size, neighbours smaller).
        for rel in (-2, -1, 0, 1, 2):
            c = rel - e
            s = 1.0 - S_AMP * min(1.0, abs(c))
            sw2, sh2 = int(round(card_w * s)), int(round(card_h * s))
            idx = (k + rel) % n
            if s >= 0.999:
                card, msk = cards[idx], mask
            else:
                card = cv2.resize(cards[idx], (sw2, sh2), interpolation=cv2.INTER_AREA)
                msk = cv2.resize(mask, (sw2, sh2), interpolation=cv2.INTER_AREA)
            paste(frame, card, msk, (W2 / 2.0 + c * slot_d), cyc)
        out_small = cv2.resize(frame.astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA)
        put(out_small)
        if i % (fps * 2) == 0:
            progress(6 + int(84 * i / float(N)), 100, "Frame %d/%d" % (i, N))
    proc.stdin.close()
    rc = proc.wait(timeout=900)
    err_f.close()
    if rc != 0:
        raise RuntimeError("ffmpeg encode failed rc=%d" % rc)

    use_bgm = (not silent) and bgm and os.path.isfile(bgm)
    if use_bgm:
        progress(92, 100, "Adding music")
        dur = N / float(fps)
        bgm_dur = duration_of(bgm)
        trim = min(dur, bgm_dur)
        fade_out = max(0.5, trim - 1.5)
        acmd = [ff, "-y", "-i", tmp, "-i", bgm, "-filter_complex",
                "[1:a]atrim=0:%.2f,asetpts=PTS-STARTPTS,volume=0.85,"
                "afade=t=in:st=0:d=0.6,afade=t=out:st=%.2f:d=1.5[a]" % (trim, fade_out),
                "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac",
                "-b:a", "192k", "-movflags", "+faststart", out]
        rc2, err2 = run_ffmpeg(acmd, timeout=600)
        if rc2 == 0 and os.path.isfile(out):
            try:
                os.remove(tmp)
            except BaseException:
                pass
        elif os.path.isfile(tmp):
            os.replace(tmp, out)
    else:
        if os.path.isfile(tmp):
            os.replace(tmp, out)

    secs = duration_of(out)
    return {"secs": secs, "images": n, "ratio": ratio}
