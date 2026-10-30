# -*- coding: utf-8 -*-
"""three_screen.py —— 三屏开场错峰揭示 + 单屏图像感知运镜 模版（Three-Screen Story）。

用户 2026-10-23 定稿口径（沿用 trend5 验证版 + 原台既有模版契约）：

  * 开场（段一）：三条横带（上/中/下）分屏，从上到下**错峰揭示**
    （参考 Trust_The_Process 视频：上带先淡入+轻微下滑，随后中带、再下带，
    各隔约 1.3s），三带全亮后一起做分层推近运镜并停留。
  * 单屏段（段二）：**先切重点、再运镜**（2026-10-28 用户定稿）：每镜起帧把
    取景窗口钉在图片重点（纸品块/主文字卡/能量最优窗口）中心——重点产品居中，
    然后从重点出发做一种纯运镜（不叠加二次分量）；镜型按图片特征匹配
    （横幅场景->pan 单向扫、低对比平铺->微旋、竖幅主体/纸品->推近，
    纸品带 1.18 基础放大看清文字），不按固定顺序轮换；
    2026-10-29：单屏每镜时长/幅度参考 Collection Showcase（4 拍 ≈2.37s/镜，
    横扫 32% 窗宽、推近 15%）。
  * 转场：全片交叉溶解（cross-dissolve）；字幕可选（step4「Add Captions」勾选后
    烧一行奶油白衬线斜体在画面底部居中），默认无字幕。
  * 低清源先一次性 LANCZOS 超分到长边>=2400，消除推近/平移的像素爬行抖动。
  * 仅出画面，不含配乐；配乐由调用方（server）混，与本工作台其它模版一致。

与 four_grid / multi_suite / scroll_gallery / fast_cut 完全独立：本模块只做
「三屏错峰揭示 + 单屏图像感知运镜」。

接口
----
render_three_screen(images, out_path, ratio="9x16", fps=30, progress=None)
  images : 图片路径列表（5~12 张；首/次/末 3 张做三屏条带，全部 N 张做单屏段）
  ratio  : 画幅（默认 9x16）
  fps    : 帧率（默认 30）
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

# 字幕字体（与 multi_suite 同款 Cormorant Garamond；不依赖系统字体，发布/本地一致）
_FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "static", "fonts")
FONT_SUB = os.path.join(_FONT_DIR, "CormorantGaramond-SemiBoldItalic.ttf")

# ---------------------------------------------------------------------------
# 节奏 / 风格常量（与用户 2026-10-23 定稿一致）
# ---------------------------------------------------------------------------
BEAT = 0.5923            # 音乐拍长（与既有模版一致）
TRI_BEATS = 8            # 开场三屏段拍数（错峰揭示 + 停留）
SGL_BEATS_BASE = 4       # 单屏每镜拍数（2026-10-29 用户定稿：运镜速度/方式参考
                         # Collection Showcase——该模版每段 2.5s，4 拍 ≈2.37s 最接近）
SGL_BEATS_LAST = 5       # 末镜拍数（收尾更长，对齐 CS「结尾整图平移」的从容感）
KH = 10                  # 交叉溶解半窗宽（帧，骑在切点上；4 帧 0.13s 太生硬，10 帧 0.33s 柔和过渡）

ROT_TOTAL = 3.5          # 旋转总角（度）—— 用户要求小幅度
PAPER_ZS = 1.18          # 纸品 push 的基础放大倍率（看清纸品文字）
REF_PAN_FRAC = 0.32      # 单屏横扫行程 = 裁窗宽的 32%（2026-10-29 参考 Collection
                         # Showcase 的 V_SWEEP=0.40 放大；原 0.13 偏小、观感接近静止。
                         # 行程仍受该侧余量钳制，起帧重点必居中不变）
REF_PUSH = 0.15          # 推近幅度（2026-10-29 参考 CS 的 zoom close 0.70-0.84 即
                         # 16%~30% 推近，原 0.09 偏弱；取 0.15 保守贴近）
PULL_Z0 = 1.18           # 拉远起幅（CS 的 zoom|out 家族；起幅足以框住重点）
UPSCALE_LONG = 2400      # 预超分长边：低清源推近会像素爬行抖动，预放大消除

# 一图一手法（2026-10-23 用户定稿）：每张图只用一种纯运镜，不再叠加二次分量
# （旧复合：push+呼吸浮动 / pan+轻微推进，已按用户要求移除）
PAN_TRAV_MIN = 28.0      # 横扫最小行程（px，输出宽 1080 的约 2.6%，保证"动起来了"）
UPSCALE_LONG = 2400      # 预超分长边：低清源推近会像素爬行抖动，预放大消除

# 三屏条带分层推近（zoom, dx, dy, rot）A->B；幅度/方向分层保持层次感
TOP_A = (1.08, -40.0, 0.0, 0.0)
TOP_B = (1.26,  40.0, 0.0, 0.0)
MID_A = (1.05,  50.0, 0.0, 0.0)
MID_B = (1.22, -50.0, 0.0, 0.0)
BOT_A = (1.10,   0.0, 0.0, -1.5)   # 底部条旋转幅度调小（±1.5 度）
BOT_B = (1.32,   0.0, 0.0,  1.5)
# 错峰揭示起点（u=[0,1]）：上 0 / 中 0.24 / 下 0.48，各自 0.20 内 smoothstep
OFF_TOP, OFF_MID, OFF_BOT = 0.00, 0.06, 0.12

CRF = 19
PRESET = "medium"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _imread(p):
    """cv2 中文路径安全读取（红线：np.fromfile + imdecode）。"""
    arr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("cannot read %s" % p)
    from core.imgproc import normalize_luma   # 2026-10-27 亮度归一（用户需求）
    return normalize_luma(arr)


def _upscale(im):
    ih, iw = im.shape[:2]
    s = max(1.0, UPSCALE_LONG / float(max(ih, iw)))
    if s > 1.0:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)
    return im


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def smoother(t):
    """六次 smoothstep：一/二阶导数在两端都为 0 —— 起停更柔、中段更匀，
    比三次 smoothstep 少「冲一下再停」的机械感（单屏运镜专用）。"""
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (t * (t * 6 - 15) + 10)


def ease_flow(t):
    """近匀速缓动（单屏运镜专用，2026-10-23 用户反馈「运镜硬、不顺滑」后定稿）：
    线性 55% + 三次 smoothstep 45%。六次 smoothstep 的速度曲线是
    「两端几乎静止、中段冲刺」，短镜里观感就是顿住-加速-急刹；
    本曲线端点速度 0.55x 平均、峰值 1.22x 平均，全程接近匀速——
    运镜持续可感知、无急刹，参考片实测也是近匀速钟形 [0.7,1.5,0.8]。"""
    t = min(max(t, 0.0), 1.0)
    return 0.55 * t + 0.45 * smooth(t)


def lerp(a, b, e):
    return a + (b - a) * e


# ---------------- 字幕层（step4「Add Captions」可填一行） ----------------
def _sub_text_layer(text, ow, size_frac=0.052, color=(251, 251, 235), font_path=None):
    """把一行字幕渲染成 RGBA 图层（带柔影，奶油白，婚礼衬线斜体）。

    text 空 → 返回 None（调用方据此跳过）。
    font_path: 客户自选字体（2026-10-26 core.fonts）；None = 斜体 CG 默认。
    """
    font = None
    fpath = font_path or FONT_SUB
    try:
        font = ImageFont.truetype(fpath, int(round(ow * size_frac)))
    except Exception:
        try:
            font = ImageFont.truetype(FONT_SUB, int(round(ow * size_frac)))
        except Exception:
            font = ImageFont.load_default()
    text = str(text or "").strip()
    if not text:
        return None
    pad = int(round(ow * size_frac * 0.7))
    bb = font.getbbox(text)
    w = bb[2] - bb[0]; h = bb[3] - bb[1]
    img = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((pad + 2, pad + 3), text, font=font, fill=(8, 8, 6, 170))   # 柔影
    d.text((pad, pad), text, font=font, fill=tuple(color) + (255,))      # 本体
    return np.array(img)


def _blend_sub(frame, layer, cx, cy, alpha):
    """把 RGBA 字幕层按 alpha 叠到 frame(BGR) 的 (cx, cy) 中心。"""
    if layer is None or alpha <= 0.01:
        return frame
    H, W = frame.shape[:2]
    lh, lw = layer.shape[:2]
    x0 = int(round(cx - lw / 2.0)); y0 = int(round(cy - lh / 2.0))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x0 + (lw - sx0)), min(H, y0 + (lh - sy0))
    if x1 <= x0 or y1 <= y0:
        return frame
    sub = layer[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)].astype(np.float32)
    a = (sub[:, :, 3:4] / 255.0) * float(alpha)
    bgr = sub[:, :, 2::-1]      # PIL 是 RGB → cv2 BGR
    dst = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (dst * (1.0 - a) + bgr * a).astype(np.uint8)
    return frame


def _sub_scrim(frame, cy, half):
    """字幕底部的柔边压暗带（保证浅色照片上字幕可读；与画面融合不突兀）。"""
    H, W = frame.shape[:2]
    y0 = max(0, int(H * (cy - half))); y1 = min(H, int(H * (cy + half)))
    if y1 <= y0:
        return frame
    t = np.abs(np.linspace(-1.0, 1.0, y1 - y0, dtype=np.float32))
    m = 1.0 - 0.28 * np.clip(1.0 - t, 0.0, 1.0) ** 1.3
    f = frame[y0:y1].astype(np.float32) * m[:, None, None]
    frame[y0:y1] = f.astype(np.uint8)
    return frame


def rev_alpha(u, s, d):
    """错峰揭示：在 [s, s+d] 内 smoothstep 0->1（上->中->下 依次出现）。"""
    return smooth((u - s) / max(1e-6, d))


def place_band(content, alpha, STRIPE, W):
    """把一条带淡入 + 从上方轻微下滑进位；alpha=0 时整条为黑。"""
    band = np.zeros((STRIPE, W, 3), np.uint8)
    dy = int(round((1.0 - alpha) * STRIPE * 0.12))
    if dy > 0:
        band[dy:STRIPE] = content[0:STRIPE - dy]
    else:
        band[:] = content
    return (band.astype(np.float32) * alpha).astype(np.uint8)


def render_win(img, x, y, w, h, ow, oh):
    ih, iw = img.shape[:2]
    x = max(0, min(x, iw - w))
    y = max(0, min(y, ih - h))
    xi, yi = int(round(x)), int(round(y))
    wi = min(int(round(w)), iw - xi)
    hi = min(int(round(h)), ih - yi)
    crop = img[yi:yi + hi, xi:xi + wi]
    interp = cv2.INTER_AREA if (wi > ow or hi > oh) else cv2.INTER_LINEAR
    return cv2.resize(crop, (ow, oh), interpolation=interp)


def rot_render(img, x, y, w, h, ow, oh, ang):
    """1.18 画布 overscan 再旋转裁回，防旋转穿帮露边。"""
    S = 1.18
    bw, bh = int(round(ow * S)), int(round(oh * S))
    big = render_win(img, x, y, w, h, bw, bh)
    M = cv2.getRotationMatrix2D((bw / 2.0, bh / 2.0), ang, 1.0)
    rot = cv2.warpAffine(big, M, (bw, bh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    sx = (bw - ow) // 2
    sy = (bh - oh) // 2
    return rot[sy:sy + oh, sx:sx + ow].copy()


def strip_frame(img, zoom, dx_out, dy_out, rot, focal_n, ASP_S, W, STRIPE):
    """一条横带：窗口按 focal_n 居中，zoom/dx/dy/rot 做分层运镜。"""
    ih, iw = img.shape[:2]
    x, y, w, h = imn.focal_window(ih, iw, ASP_S, zoom, focal_n)
    kx = w / W
    ky = h / STRIPE
    cx = x + w / 2 + dx_out * kx
    cy = y + h / 2 + dy_out * ky
    cx = min(max(cx, w / 2), iw - w / 2)
    cy = min(max(cy, h / 2), ih - h / 2)
    if rot == 0.0:
        return render_win(img, cx - w / 2, cy - h / 2, w, h, W, STRIPE)
    return rot_render(img, cx - w / 2, cy - h / 2, w, h, W, STRIPE, rot)


def _decide(an, asp_f, k, prev_kind=None):
    """单屏运镜 —— 2026-10-28 用户定稿：**先切重点、再运镜，重点产品居中，
    镜型按图片特征匹配**（不再按 REF_MIX 固定顺序轮换）：
      - analyze 已把焦点钉在重点（纸品块 / 主文字卡 / 能量最优窗口）中心，
        render_shot 所有镜型的取景窗口都钉在 focal 上 => 每镜起帧重点必居中；
      - 镜型按 analyze 的按图判定：横幅场景 -> pan（向该侧余量足的方向单向扫）、
        低对比平铺 -> rot、竖幅主体 / 纸品 -> push；
      - 纸品 push 带 PAPER_ZS=1.18 基础放大（看清文字）；
      - pan 该侧余量不足或近满幅图 -> 慢推近（避免死帧）。
    ★ 2026-10-29 运镜方式参考 Collection Showcase（multi_suite）：CS 的既定
      口径是「同一条片里相邻两段不撞运镜家族」。按图判别会把一整组图全映射到
      push（实测本素材 8 张全 push），观感就是 8 连推。故在按图选型为底之上，
      相邻两镜撞家族时依次轮换 pan（余量足的一侧）/ pull / rot 备选——
      与 CS 的「家族 + 方向」防重逻辑同构。
    返回 (kind, param, zs)。
    """
    fx, fy = an["focal_n"]
    ih, iw = an["_ih"], an["_iw"]
    motion = an["motion"]
    zs = PAPER_ZS if an["paper"] else 1.0

    def pan_move():
        x0, _, w0, _ = imn.focal_window(ih, iw, asp_f, zs, an["focal_n"])
        lm, rm = x0, iw - w0 - x0
        if rm >= w0 * 0.04 and rm >= lm:
            return ("panR", REF_PAN_FRAC, zs)
        if lm >= w0 * 0.04:
            return ("panL", REF_PAN_FRAC, zs)
        return None

    def push_move():
        return ("push", REF_PUSH, zs)

    def pull_move():
        return ("pull", PULL_Z0, 1.0)

    def rot_move():
        return ("rot", ROT_TOTAL, 1.0)

    if motion in ("panL", "panR"):
        first = pan_move() or push_move()
        alts = (push_move, pull_move, rot_move)
    elif motion == "rot":
        first = rot_move()
        alts = (push_move, pan_move, pull_move)
    else:
        first = push_move()
        alts = (pan_move, pull_move, rot_move)

    kind, param, zsv = first
    if prev_kind is not None and kind == prev_kind:
        for mk in alts:
            r = mk()
            if r is not None and r[0] != prev_kind:
                return r
    return (kind, param, zsv)


# ---------------------------------------------------------------------------
# 主渲染
# ---------------------------------------------------------------------------
def render_three_screen(images, out_path, ratio="9x16", fps=30, progress=None,
                        captions=None, font=None):
    """N(5~12) 张图 -> 三屏错峰揭示开场 + 单屏图像感知运镜成片（仅画面，无音轨）。

    captions : 可选字幕文本（str）；非空时烧一行奶油白衬线斜体在画面底部居中；
               留空/None 则不烧字幕。
    font     : 客户自选字体键（core.fonts；None = 斜体 CG 默认）。
    """
    from core import fonts as _fonts
    paths = [p for p in (images or []) if p and os.path.isfile(p)]
    if len(paths) < 5:
        raise ValueError("Three-Screen Story needs at least 5 photos")
    if len(paths) > 12:
        paths = paths[:12]

    OW, OH = cp.ratio_size(ratio)
    W, H = int(OW), int(OH)
    STRIPE = H // 3
    ASP_S = W / STRIPE
    ASP_F = W / H

    if progress:
        progress(6, 100, "Preparing photos")
    imgs = [_upscale(_imread(p)) for p in paths]
    N = len(imgs)

    ANS = []
    for k, im in enumerate(imgs):
        a = imn.analyze(im, "T%d" % k, aspect=ASP_F)
        a["_ih"], a["_iw"] = im.shape[:2]
        ANS.append(a)

    # 三屏条带：首 / 次 / 末 三张做上 / 中 / 下
    TRI = [0, 1, N - 1]
    # 单屏段（2026-10-28 定稿）：全部 N 张，镜型按图匹配（横幅->pan、平铺->rot、
    # 竖幅主体/纸品->push），pan 方向也按图（向余量足的一侧单向扫）——
    # 不再按 REF_MIX 固定顺序轮换。
    # 2026-10-29：相邻两镜不撞家族（CS 式防重），撞了就轮换 pan/pull/rot 备选。
    PLAN = []
    _prev_kind = None
    for k in range(N):
        _d = _decide(ANS[k], ASP_F, k, prev_kind=_prev_kind)
        _prev_kind = _d[0]
        PLAN.append((k,) + _d)
    SINGLES = [(pi, knd, prm, zs, ANS[pi]["focal_n"]) for pi, knd, prm, zs in PLAN]

    def b2f(nb):
        return int(round(nb * BEAT * fps))

    TRI_FRAMES = b2f(TRI_BEATS)
    sgl_beats = [SGL_BEATS_BASE] * N
    sgl_beats[-1] = SGL_BEATS_LAST
    SGL_FRAMES = [b2f(b) for b in sgl_beats]
    LENS = [TRI_FRAMES] + SGL_FRAMES
    N_SHOTS = 1 + len(SGL_FRAMES)
    STARTS = [0]
    for n in LENS:
        STARTS.append(STARTS[-1] + n)
    TOTAL_F = STARTS[-1]
    total_secs = TOTAL_F / float(fps)

    # 字幕（step4「Add Captions」）：一行奶油白衬线斜体，居中底部 + 柔边压暗带
    sub_layer = _sub_text_layer(captions, W, font_path=_fonts.ttf(font)) if captions else None
    SUB_CY = 0.90
    SUB_HALF = 0.10

    def tri_progress(u, off):
        return smooth((u - off) / max(1e-6, (1.0 - off)))

    def render_shot(idx, i):
        n = LENS[idx]
        i = min(max(i, 0), n - 1)
        if idx == 0:
            u = i / (TRI_FRAMES - 1)
            et = tri_progress(u, OFF_TOP)
            em = tri_progress(u, OFF_MID)
            eb = tri_progress(u, OFF_BOT)
            top = strip_frame(imgs[TRI[0]], lerp(TOP_A[0], TOP_B[0], et),
                              lerp(TOP_A[1], TOP_B[1], et), lerp(TOP_A[2], TOP_B[2], et),
                              lerp(TOP_A[3], TOP_B[3], et), ANS[TRI[0]]["focal_n"], ASP_S, W, STRIPE)
            mid = strip_frame(imgs[TRI[1]], lerp(MID_A[0], MID_B[0], em),
                              lerp(MID_A[1], MID_B[1], em), lerp(MID_A[2], MID_B[2], em),
                              lerp(MID_A[3], MID_B[3], em), ANS[TRI[1]]["focal_n"], ASP_S, W, STRIPE)
            bot = strip_frame(imgs[TRI[2]], lerp(BOT_A[0], BOT_B[0], eb),
                              lerp(BOT_A[1], BOT_B[1], eb), lerp(BOT_A[2], BOT_B[2], eb),
                              lerp(BOT_A[3], BOT_B[3], eb), ANS[TRI[2]]["focal_n"], ASP_S, W, STRIPE)
            at = rev_alpha(u, 0.00, 0.20)
            am = rev_alpha(u, 0.24, 0.20)
            ab = rev_alpha(u, 0.48, 0.20)
            return np.vstack([place_band(top, at, STRIPE, W),
                              place_band(mid, am, STRIPE, W),
                              place_band(bot, ab, STRIPE, W)])
        pi, kind, param, zs, focal = SINGLES[idx - 1]
        img = imgs[pi]
        ih, iw = img.shape[:2]
        fx_px, fy_px = focal[0] * iw, focal[1] * ih
        e = ease_flow(i / (n - 1)) if n > 1 else 1.0
        if kind == "push":
            # 纯推近（一图一手法）：起幅 = max(zs, min_zoom_centre)——focal
            # 偏边的图起帧也把重点框进画面中心（「先切重点」），然后缓慢推近，
            # 窗口中心全程钉在重点上。
            z0 = max(zs, imn.min_zoom_centre(ih, iw, ASP_F, focal))
            z = z0 * (1.0 + param * e)
            w, h = imn.focal_window(ih, iw, ASP_F, z, focal)[2:4]
            cx = min(max(fx_px, w / 2), iw - w / 2)
            cy = min(max(fy_px, h / 2), ih - h / 2)
            return render_win(img, cx - w / 2, cy - h / 2, w, h, W, H)
        if kind == "pull":
            # 纯拉远（CS zoom|out 家族，2026-10-29 引入）：起幅取
            # max(PULL_Z0, min_zoom_centre)——与 push 同口径，focal 偏边的图
            # 起帧也把重点框进画面中心，然后拉远到全幅揭示环境。
            z0 = max(param, imn.min_zoom_centre(ih, iw, ASP_F, focal))
            z = z0 + (1.0 - z0) * e
            w, h = imn.focal_window(ih, iw, ASP_F, z, focal)[2:4]
            cx = min(max(fx_px, w / 2), iw - w / 2)
            cy = min(max(fy_px, h / 2), ih - h / 2)
            return render_win(img, cx - w / 2, cy - h / 2, w, h, W, H)
        if kind == "rot":
            # 微旋（低对比平铺图）：起幅足以让重点居中，绕重点 ±param 度缓动
            z = max(1.0, imn.min_zoom_centre(ih, iw, ASP_F, focal))
            x0, y0, w0, h0 = imn.focal_window(ih, iw, ASP_F, z, focal)
            ang = (e - 0.5) * 2.0 * param
            return rot_render(img, x0, y0, w0, h0, W, H, ang)
        if kind in ("panL", "panR"):
            # 纯横扫（2026-10-28 改）：起帧窗口以重点为中心（先切重点），
            # 然后向该侧单向扫（行程 = 裁窗宽 × param，受该侧余量限制）——
            # 旧版是跨焦点往返式，起帧重点不在画面中心。
            # 2026-10-29：起幅 zoom 补 min_zoom_centre（与 push 同口径）——
            # focal 偏边的图 zoom=1/1.18 时窗口钳在画幅内、重点无法居中；
            # 放大起幅让窗口装得下「以重点为中心」，产品起帧即居中。
            z_pan = max(zs, imn.min_zoom_centre(ih, iw, ASP_F, focal))
            _, y0, w0, h0 = imn.focal_window(ih, iw, ASP_F, z_pan, focal)
            x0 = min(max(fx_px - w0 / 2, 0.0), iw - w0)
            if kind == "panR":
                trav = max(8.0, min(param * w0, (iw - w0 - x0) - 6))
                x = x0 + trav * e
            else:
                trav = max(8.0, min(param * w0, x0 - 6))
                x = x0 - trav * e
            return render_win(img, x, y0, w0, h0, W, H)
        raise ValueError(kind)

    def fx_dissolve(a, b, t):
        e = smooth(t)
        return cv2.addWeighted(a, 1 - e, b, e, 0)

    if progress:
        progress(12, 100, "Rendering")
    cmd = [cp.FFMPEG, "-y", "-hide_banner",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (W, H), "-r", str(fps), "-i", "-",
           "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", str(CRF), "-preset", PRESET, "-movflags", "+faststart",
           out_path]
    err_f = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=err_f, stderr=err_f)
    try:
        for f in range(TOTAL_F):
            j = 0
            while j < N_SHOTS - 1 and f >= STARTS[j + 1]:
                j += 1
            if j >= 1 and STARTS[j] - KH <= f < STARTS[j] + KH:
                c = STARTS[j]
                t = (f - (c - KH) + 0.5) / (2 * KH)
                fa = render_shot(j - 1, f - STARTS[j - 1])
                fb = render_shot(j, f - STARTS[j])
                frame = fx_dissolve(fa, fb, t)
            else:
                frame = render_shot(j, f - STARTS[j])
            if sub_layer is not None:
                frame = _sub_scrim(frame, SUB_CY, SUB_HALF)
                frame = _blend_sub(frame, sub_layer, W * 0.5, H * SUB_CY, 1.0)
            proc.stdin.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
            if progress and (f % 60 == 0):
                progress(int(12 + 86 * f / max(TOTAL_F, 1)), 100, "Rendering")
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed rc=%s" % proc.returncode)

    if progress:
        progress(100, 100, "Done")
    return {"video": out_path, "secs": round(total_secs, 2),
            "images": N, "ratio": ratio}
