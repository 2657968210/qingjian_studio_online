# -*- coding: utf-8 -*-
"""TOP 5 Countdown template (倒数榜)
==================================
N product photos -> title card + per-photo camera-move segments + bottom-left
maroon number block + style SKU beside it.

Online (2026-10-27) parameterised build:
  * 3-8 photos (caller-truncated; server enforces 3-8)
  * per-photo SKU (style) + rank number, both user-fillable
  * rank blank -> auto countdown 01..N (first photo = N, last = 01)
  * BGM optional (default user_paulyudin)
  * 5 camera moves (push/panLR/panDiag/pull/rot) greedy by image traits,
    adjacent never repeats: 3-5 photos fully unique, 6-8 cycle
  * layout (final): bottom-left maroon number block (centred) + SKU white text
    to its right, vertically centred, no plate / no filename
  * reuses core.image_motion pure functions; does not touch other templates.
"""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from core.image_motion import (analyze, focal_window, _render_win, _rot_render,
                               _upscale_bgr, FS_PUSH, FS_PULL, FS_ROT, FS_BASE_ZOOM)
from core.ffproc import run_ffmpeg, duration_of
import imageio_ffmpeg

FPS = 30
TITLE_F = 45            # 1.5s title card
SEG_F = 99              # 3.3s per photo segment

MAROON = (158, 42, 74, 255)
TITLE_BG = (233, 231, 228)

PAN_ZOOM = 1.18
PAN_FRAC = 0.30

POOL = ["panLR", "panDiag", "rot", "push", "pull"]

# Sans-serif (Arial) shipped with the app so the online sandbox matches the
# local render; fall back to the bundled Cormorant Garamond if Arial is absent.
_FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "static", "fonts")
FONT_BOLD = os.path.join(_FONT_DIR, "arialbd.ttf")
FONT_REG = os.path.join(_FONT_DIR, "arial.ttf")
_FONT_FALLBACK = os.path.join(_FONT_DIR, "CormorantGaramond-SemiBold.ttf")


def pick_font(size, bold=False):
    cand = FONT_BOLD if bold else FONT_REG
    if os.path.exists(cand):
        return ImageFont.truetype(cand, size)
    return ImageFont.truetype(_FONT_FALLBACK, size)


def ease_flow(p):
    """Near-constant-speed easing (three_screen round-6): 55% linear + 45% smoothstep."""
    s = p * p * (3.0 - 2.0 * p)
    return 0.55 * p + 0.45 * s


def _dims(ratio):
    r = (ratio or "9x16").lower()
    if "1x1" in r or "square" in r:
        return 1080, 1080
    return 1080, 1920


def title_frame(W, H, main="TOP 5", sub=None):
    img = Image.new("RGB", (W, H), TITLE_BG)
    d = ImageDraw.Draw(img)
    f1 = pick_font(168)
    tw = d.textlength(main, font=f1)
    if tw > W - 120:                       # auto-shrink long custom headlines
        f1 = pick_font(max(64, int(168 * (W - 120) / tw)))
        tw = d.textlength(main, font=f1)
    fs = f1.size
    top = H * 0.43
    d.text(((W - tw) / 2, top), main, font=f1, fill=(38, 38, 38))
    # 2026-10-28：下划线加长（用户要求"长一点"）——原固定 128px（±64）太短，
    # 改为随标题宽度联动（约 0.4×标题宽，下限 200px），长标题也不会短于一个字。
    _uw = min(max(200.0, tw * 0.40), W - 160.0)
    d.rectangle([(W - _uw) / 2, top + fs + 14, (W + _uw) / 2, top + fs + 20], fill=MAROON)
    if sub:
        f2 = pick_font(44)
        tw2 = d.textlength(sub, font=f2)
        if tw2 > W - 120:
            f2 = pick_font(max(24, int(44 * (W - 120) / tw2)))
        d.text(((W - d.textlength(sub, font=f2)) / 2, top + fs + 56), sub,
               font=f2, fill=(100, 100, 100))
    return np.asarray(img)


def overlay(frame_rgb, num_str, style, W, H):
    """Bottom-left maroon number block (centred) + SKU white text to its right,
    vertically centred, no plate / no filename."""
    img = Image.fromarray(frame_rgb)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    bw = 210
    bx0 = 30
    by1 = H - 170                      # block bottom edge, lifted to leave room
    d.rectangle([bx0, by1 - bw, bx0 + bw, by1], fill=MAROON)
    # number (centred in block)
    nf = pick_font(100, bold=True)
    nw = d.textlength(num_str, font=nf)
    d.text((bx0 + (bw - nw) / 2, by1 - bw + (bw - 100) / 2 - 4), num_str,
           font=nf, fill=(255, 255, 255))
    # SKU beside the block, lower part of the block (user-tuned 2026-10-28:
    # moved down from block-centre so it sits in the bottom quarter of the block).
    # No scrim / plate behind the text (user 2026-10-28: plain white + stroke only) —
    # a busy photo in this region is handled by swapping the photo, not a backdrop.
    if style:
        sf = pick_font(56, bold=True)
        tb = d.textbbox((0, 0), style, font=sf)
        tx = bx0 + bw + 24
        ty = (by1 - 50.0) - (tb[1] + tb[3]) / 2.0
        d.text((tx, ty), style, font=sf, fill=(255, 255, 255),
               stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB"))


def render_move(img, move, focal_n, p, W, H, push=FS_PUSH, pull=FS_PULL, rot=FS_ROT):
    ih, iw = img.shape[:2]
    ASP_F = W / float(H)
    fx, fy = focal_n
    fx_px, fy_px = fx * iw, fy * ih
    e = min(1.0, max(0.0, p))

    if move == "pull":
        z = pull + (1.0 - pull) * e
        _, _, w, h = focal_window(ih, iw, ASP_F, z, focal_n)
        cx = min(max(fx_px, w / 2), iw - w / 2)
        cy = min(max(fy_px, h / 2), ih - h / 2)
        bgr = _render_win(img, cx - w / 2, cy - h / 2, w, h, W, H)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if move in ("panLR", "panDiag"):
        z = PAN_ZOOM
        _, _, w, h = focal_window(ih, iw, ASP_F, z, focal_n)
        slack_x = (iw - w) / 2.0
        slack_y = (ih - h) / 2.0
        trav_x = min(PAN_FRAC * w, max(0.0, slack_x - 8.0))
        trav_y = min(PAN_FRAC * h, max(0.0, slack_y - 8.0))
        # Centre the pan swing on the focal point but guarantee the FULL travel
        # stays inside the valid window region. Otherwise, when the focal point
        # is near an edge, the start of the pan clamps against the crop boundary
        # and the shot looks frozen for the first ~1-2s (only the tail moves).
        cx_c = min(max(fx_px, w / 2.0 + trav_x / 2.0), iw - w / 2.0 - trav_x / 2.0)
        cy_c = min(max(fy_px, h / 2.0 + trav_y / 2.0), ih - h / 2.0 - trav_y / 2.0)
        if move == "panLR":                 # left -> right
            dx = -trav_x / 2 + trav_x * e
            dy = 0.0
        else:                               # panDiag: bottom-right -> top-left
            dx = trav_x / 2 - trav_x * e
            dy = trav_y / 2 - trav_y * e
        cx = cx_c + dx
        cy = cy_c + dy
        bgr = _render_win(img, cx - w / 2, cy - h / 2, w, h, W, H)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if move == "rot":
        z = FS_BASE_ZOOM
        _, _, w, h = focal_window(ih, iw, ASP_F, z, focal_n)
        cx = min(max(fx_px, w / 2), iw - w / 2)
        cy = min(max(fy_px, h / 2), ih - h / 2)
        ang = (e - 0.5) * 2.0 * rot
        bgr = _rot_render(img, cx - w / 2, cy - h / 2, w, h, W, H, ang)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # default push (zoom-in)
    _, _, w0, h0 = focal_window(ih, iw, ASP_F, 1.0, focal_n)
    z = 1.0 + push * e
    w = w0 / z
    h = h0 / z
    cx = min(max(fx_px, w / 2), iw - w / 2)
    cy = min(max(fy_px, h / 2), ih - h / 2)
    bgr = _render_win(img, cx - w / 2, cy - h / 2, w, h, W, H)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def pref_rank(an, aspect):
    paper, peak = an["paper"], an["peak_ratio"]
    s = {m: 1.0 for m in POOL}
    if paper:
        s["push"] = 3.0; s["pull"] = 1.6; s["panDiag"] = 1.1; s["panLR"] = 0.9; s["rot"] = 1.0
    elif aspect > 1.15:
        s["panLR"] = 3.0; s["panDiag"] = 2.4; s["pull"] = 1.8; s["push"] = 1.0; s["rot"] = 0.8
    elif peak < 1.6:
        s["rot"] = 3.0; s["panDiag"] = 2.2; s["pull"] = 1.5; s["push"] = 1.0; s["panLR"] = 1.0
    else:
        s["push"] = 2.6; s["panDiag"] = 2.0; s["panLR"] = 1.6; s["pull"] = 1.5; s["rot"] = 1.0
    return s


def assign_moves(anlist, aspects, fixed=None):
    """Greedy by per-image preference; adjacent never repeat.
    3-5 photos -> all unique; 6-8 -> cycle (still no adjacent repeat).
    `fixed` is an optional {index: move} map that pins specific shots to a move
    (e.g. {0: "panLR", 3: "panDiag"}). Pinned shots keep their move even across
    the greedy pass, and the remaining shots avoid every already-used move
    (including pinned ones) so no duplicate occurs when N == len(POOL)."""
    if fixed is None:
        fixed = {}
    chosen_list = []
    for idx, (an, asp) in enumerate(zip(anlist, aspects)):
        if idx in fixed:
            chosen_list.append(fixed[idx])
            continue
        s = pref_rank(an, asp)
        prev = chosen_list[-1] if chosen_list else None
        used = set(chosen_list) | set(fixed.values())
        cand = sorted(POOL, key=lambda m: -s[m])
        pick = None
        for m in cand:                      # prefer unused AND != prev
            if m != prev and m not in used:
                pick = m
                break
        if pick is None:                    # all used (>=6 photos) -> just != prev
            for m in cand:
                if m != prev:
                    pick = m
                    break
        if pick is None:
            pick = cand[0]
        chosen_list.append(pick)
    return chosen_list


def render_top5(srcs, out, skus=None, nums=None, bgm=None, ratio="9x16", fps=30,
                progress=None, title_main="TOP 5", title_sub=None, silent=False,
                force_moves=None):
    if progress is None:
        progress = lambda d, t, m="": None
    # Template default: shot 1 pans left->right, shot 4 pans bottom-right->top-left.
    if force_moves is None:
        force_moves = {0: "panLR", 3: "panDiag"}
    W, H = _dims(ratio)
    if skus is None:
        skus = ["" for _ in srcs]
    if nums is None:
        nums = ["" for _ in srcs]
    n = len(srcs)
    progress(2, 100, "Reading %d photos" % n)

    analyzed = []
    for fp in srcs:
        img = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("decode fail: " + fp)
        img = _upscale_bgr(img)
        an = analyze(img, name=os.path.basename(fp), aspect=W / float(H))
        an["_ih"], an["_iw"] = img.shape[:2]
        aspect = an["_iw"] / float(an["_ih"])
        analyzed.append((img, an, aspect))

    anlist = [a[1] for a in analyzed]
    aspects = [a[2] for a in analyzed]
    moves = assign_moves(anlist, aspects, fixed=force_moves)

    def num_str(i):
        if i < len(nums) and str(nums[i]).strip():
            return str(nums[i]).strip()
        return "%02d" % (n - i)

    progress(8, 100, "Rendering %d segments" % n)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = out + ".raw.mp4"
    vcmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
            "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    err_f = open(os.devnull, "wb")
    proc = __import__("subprocess").Popen(vcmd, stdin=__import__("subprocess").PIPE,
                                          stdout=err_f, stderr=err_f)

    def put(frame):
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    put(title_frame(W, H, title_main, title_sub))
    for _ in range(1, TITLE_F):
        put(title_frame(W, H, title_main, title_sub))
    for si, (img, an, aspect) in enumerate(analyzed):
        move = moves[si]
        style = skus[si] if si < len(skus) and skus[si].strip() else ""
        ns = num_str(si)
        for k in range(SEG_F):
            p = ease_flow(k / float(SEG_F - 1))
            fr = render_move(img, move, an["focal_n"], p, W, H)
            put(overlay(fr, ns, style, W, H))
        progress(8 + int(82 * (si + 1) / n), 100, "Shot %d/%d (%s)" % (si + 1, n, move))
    proc.stdin.close()
    rc = proc.wait(timeout=600)
    if rc != 0:
        raise RuntimeError("ffmpeg encode failed rc=%d" % rc)

    # BGM
    use_bgm = (not silent) and bgm and os.path.isfile(bgm)
    if use_bgm:
        progress(92, 100, "Adding music")
        dur = TITLE_F / fps + SEG_F * n / fps
        bgm_dur = duration_of(bgm)
        trim = min(dur, bgm_dur)
        fade_out = max(0.5, trim - 1.2)
        acmd = [ff, "-y", "-i", tmp, "-i", bgm, "-filter_complex",
                "[1:a]atrim=0:%.2f,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.2,"
                "afade=t=out:st=%.2f:d=1.2[a]" % (trim, fade_out),
                "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac",
                "-b:a", "192k", "-movflags", "+faststart", out]
        rc2, err2 = run_ffmpeg(acmd, timeout=600)
        if rc2 == 0 and os.path.isfile(out):
            try:
                os.remove(tmp)
            except BaseException:
                pass
        elif os.path.isfile(tmp):          # BGM mux failed -> keep silent master
            os.replace(tmp, out)
    else:
        if os.path.isfile(tmp):
            os.replace(tmp, out)

    secs = duration_of(out)
    return {"secs": secs, "images": n, "ratio": ratio, "moves": moves}
