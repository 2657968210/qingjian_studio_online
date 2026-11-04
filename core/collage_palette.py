# -*- coding: utf-8 -*-
"""婚礼 Look 拼贴 + 主色色块模版（Wedding Look Collage）。

客户上传照片后：
  * 色块按上传照片识别，规则与参考片 `_ref_trend/wedding_palettes_final`
    完全一致：**同一色系的浅→深 4 档**（蓝调照片 → 近白→浅蓝→蓝→藏青，
    即 "Dusty Blue & White"；暖调 → "Terracotta & Rust"；绿调 →
    "Olive & Sage"），不会出现跨色系杂色（2026-11-03 15:23 用户定稿）。
  * 开场：双列全出血拼贴（masonry，两列等宽、各自居中，柔和模糊底补白），
    中央文字（kicker 衬线正体 + script 衬线斜体大标题，副标题留空则按
    识别出的色系自动起名，如 "Weddings 2027 / Dusty Blue & White"），
    下方一排同色系色块（浅→深）。
  * 中段：全部 N 张照片依次单图满幅展示（纯硬切、无运镜）。
  * 仅出画面，不含配乐；配乐由 server 端混（默认 color trend）。

与 four_grid / scroll_gallery 完全独立：本模块只做「拼贴 + 色块 + 单图」。

接口
----
render_collage_palette(images, out_path, ratio="9x16", fps=30,
                       title="", subtitle="", show_text=True, progress=None,
                       font=None, palette_override=None)
  images   : 图片路径列表（≥3 张；开场拼贴取全部、单图段也展示全部）
  title    : 中央 kicker 衬线小字（如 Weddings 2027）；空串则不叠
  subtitle : 中央 script 衬线斜体大标题；留空则自动按色系起名
             （Dusty Blue & White / Terracotta & Rust …）
  show_text: False 则完全不叠文字（色块仍保留——它是本模版的核心产物）
  palette_override: 显式指定色板（封面样片定稿用）；不传则按图识别
  返回 dict：{video, secs, images, ratio, palette:[(r,g,b),...], palette_name}
（只出画面，不含配乐）
"""
import os
import re
import json
import subprocess
import tempfile
import traceback

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from core.card_pan import FFMPEG, ratio_size, norm_ratio

# ---------------------------------------------------------------------------
# 字体（随包 static/fonts/，Linux 部署不依赖系统字体）
# ---------------------------------------------------------------------------
_FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "fonts")
POPPINS = os.path.join(_FONT_DIR, "Poppins-Light.ttf")
# 2026-11-03 用户定稿（参考图 "Weddings 2027 / Terracotta & Rust"）：
# 两行都用 Cormorant 衬线——第一行正体、第二行斜体；弃用 GreatVibes 手写体。
CORMORANT = os.path.join(_FONT_DIR, "CormorantGaramond-SemiBold.ttf")
CORMORANT_ITALIC = os.path.join(_FONT_DIR, "CormorantGaramond-MediumItalic.ttf")

# ---------------------------------------------------------------------------
# 节奏 / 风格常量
# ---------------------------------------------------------------------------
INTRO = 3.0        # 拼贴 + 文字/色块淡入停留
HOLD = 1.55        # 每张单图展示时长（s，纯硬切、无运镜）
GAP = 0            # 拼贴列内/列间间隙（0 = 全出血无缝）

# 橘色系（Terracotta & Rust）——仅用于封面定稿 / palette_override：
# 2026-11-03 15:06 起，线上客户上传不再强制套用此色板，改走 detect_palette
# 按图片识别；此处保留供封面样片与显式 override 使用。
# 顺序：象牙白 → 陶土橘 → 锈棕 → 深可可棕（浅→深）。
ORANGE_PALETTE = [
    (243, 233, 217),   # Ivory      象牙白
    (206, 126, 78),    # Terracotta 陶土橘
    (164, 96, 46),     # Rust       锈棕
    (74, 53, 35),      # Cocoa      深可可棕
]

CRF = 18
PRESET = "medium"

# 色块条（2026-11-03 用户参考片定稿：窄横条胶囊，浅→深，从上到下逐条淡入）
# 几何全部按参考片实测（1080×1920 帧）：条宽 415px、条高 ~95px、净间距 ~77px
N_SWATCH = 4        # 2026-11-03 用户定：只保留 4 个色块
SW_WF = 0.20        # 圆形色块直径 / 画面宽（2026-11-03 17:29 用户：圆形缩小一点）
SW_HF = 0.20        # （=直径，圆形无高宽差；保留常量位）
SW_GAPF = 0.034     # 圆间距 / 画面宽
GAP_LINES_F = 0.012   # 两行文字之间的间距 / 画面宽（行距小）
GAP_TEXT_SW_F = 0.050 # 文字块与色块堆之间的间距 / 画面宽（参考片实测 ~55/1080）
# 参考片动画时序：标题先现，4 条色块从上到下错峰淡入（实测 t=0.5s 时
# 第 1 条已实、第 2 条半透、第 3 条微现、第 4 条未出，1s 内全部到位）
TXT_DELAY = 0.15    # 标题淡入起始（s）
TXT_DUR = 0.55      # 标题淡入时长
SW_DELAY0 = 0.30    # 第一条色块起始（s）
SW_STAG = 0.12      # 相邻条错峰间隔
SW_DUR = 0.42       # 单条淡入时长


# ---------------------------------------------------------------------------
# 图片读取
# ---------------------------------------------------------------------------
def _imread_bgr(p):
    """cv2 中文路径安全读取（红线：np.fromfile + imdecode），不调色。"""
    arr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("cannot read %s" % p)
    return arr


def _load(paths):
    """返回 {path: bgr(ndarray)} 与 rgb 列表（供主色识别）。"""
    out = {}
    for p in paths:
        bgr = _imread_bgr(p)
        out[p] = bgr
    return out


# ---------------------------------------------------------------------------
# 主色识别：参考片同款「同一色系浅→深 4 档」（见 detect_palette）
# ---------------------------------------------------------------------------


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def _far(c, lst, thr=45):
    return all(_dist2(c, p) >= thr * thr for p in lst)


def _pad_variants(selected, n):
    """照片真实色不足 n 个时，用**照片自身颜色**的深浅变体补齐。"""
    out = list(selected)
    if not out:
        return out
    guard = 0
    while len(out) < n and guard < 32:
        cands = []
        for c in list(out):
            for t in (0.18, 0.38):
                cands.append(tuple(int(round(v + (255 - v) * t)) for v in c))
                cands.append(tuple(int(round(v * (1.0 - t))) for v in c))
        best, bd = None, -1
        for c in cands:
            d = min(_dist2(c, p) for p in out)
            if d > bd:
                bd, best = d, c
        if best is None:
            break
        out.append(best)
        guard += 1
    return out[:n]


def _lum(c):
    """感知亮度（Rec.601 加权）。"""
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _sat_of(c):
    mx, mn = float(max(c)), float(min(c))
    return 0.0 if mx <= 0 else (mx - mn) / mx


def _hue_of(c):
    """色相 0-360；无彩色返回 -1。"""
    r, g, b = c
    mx, mn = max(c), min(c)
    d = mx - mn
    if d <= 0:
        return -1.0
    if mx == r:
        h = (g - b) / d
    elif mx == g:
        h = 2.0 + (b - r) / d
    else:
        h = 4.0 + (r - g) / d
    return (h * 60.0) % 360.0


def _structure_palette(real, cand, n=N_SWATCH):
    """参考片四层结构：浅中性基底 → 主色 → 深色 → 最深锚色（浅→深排列）。"""
    cs = sorted(real, key=_lum)          # 暗→亮
    if not cs:
        return _pad_variants([(206, 198, 188)], n)
    light, anchor = cs[-1], cs[0]
    uniq = list(cs)
    for _, c in cand:
        if all(_dist2(c, u) > 60 * 60 for u in uniq):
            uniq.append(c)
    ll = _lum(light)
    acc = None
    for c in sorted(uniq, key=_sat_of, reverse=True):
        if ll - 18 > _lum(c) > _lum(anchor) + 18:
            acc = c
            break
    if acc is None:
        mids = cs[1:-1] or (cs[1:] if len(cs) > 1 else cs)
        acc = max(mids, key=_sat_of) if mids else light
    ha = _hue_of(acc)
    la = _lum(acc)
    deep, bd = None, None
    for c in uniq:
        if c == acc:
            continue
        h = _hue_of(c)
        if ha >= 0 and h >= 0:
            hd = abs(h - ha)
            hd = min(hd, 360.0 - hd)
        elif ha < 0 and h < 0:
            hd = 0.0
        else:
            hd = 999.0
        l = _lum(c)
        if hd < 26 and 0.45 * la <= l <= 0.88 * la:
            d = _dist2(c, acc)
            if bd is None or d < bd:
                bd, deep = d, c
    if deep is None:
        deep = tuple(int(round(v * 0.70)) for v in acc)
    if anchor == light or _lum(anchor) > la - 30 or _dist2(anchor, deep) < 40 * 40:
        anchor = tuple(int(round(v * 0.42)) for v in acc)
    out = [light, acc, deep, anchor]
    return sorted(out, key=_lum)[:n]


def _is_orange(rgb):
    """判定「明亮暖橘」——橘色系照片的标志性色。"""
    r, g, b = rgb
    if not (r >= g >= b):
        return False
    if r - g < 40:
        return False
    if r - b < 55:
        return False
    if r < 150:
        return False
    if (r + g + b) / 3.0 < 95:
        return False
    return True


_HUE_BANDS = (('red', 335, 375), ('warm', 15, 45), ('gold', 45, 70),
              ('green', 70, 150), ('teal', 150, 190), ('blue', 190, 250),
              ('purple', 250, 310), ('mauve', 310, 345))


# ---------------------------------------------------------------------------
# 婚礼配色族表（2026-11-04 用户选 A：锚色最近邻，15 彩色族 + 中性）
# ---------------------------------------------------------------------------
# 旧实现按 8 条色相区间粗判，实测 36 个婚礼标准色只有 6 个能叫对：
# 金色→陶土、黄油/芥末→橄榄、粉/紫/正红全归酒红、莫兰迪淡色全归中性。
# 现改为「每族给若干锚色（浅/中/深），像素取最近锚色定族」：
#   · 匹配以 LAB 色度(a/b)为主、明度只占低权重 —— 同族的浅档与深档
#     不会被拆到两族；
#   · 中性（白/象牙/香槟/沙/灰/炭黑）是独立锚组，不再靠饱和度门槛
#     一刀切，淡粉/淡蓝/淡紫/浅鼠尾草因此不再被吞成「香槟」。
# 每个元组 = (族 key, 主色名, 辅色名, 锚色 RGB 列表)。
_FAMILIES = (
    ("blush",      "Blush",      "Champagne",  ((233, 205, 199), (240, 213, 216), (245, 226, 218))),
    ("rose",       "Dusty Rose", "Blush",      ((196, 142, 140), (172, 130, 140), (194, 24, 91), (216, 163, 146))),
    ("burgundy",   "Burgundy",   "Ivory",      ((122, 60, 72), (142, 74, 78), (163, 36, 48), (110, 42, 68))),
    ("terracotta", "Terracotta", "Ivory",      ((206, 126, 78), (194, 90, 42), (165, 84, 52), (242, 201, 168), (239, 177, 131), (184, 115, 72))),
    ("gold",       "Gold",       "Ivory",      ((201, 162, 92), (201, 160, 60), (193, 154, 107), (140, 98, 60))),
    ("butter",     "Butter",     "Cream",      ((248, 230, 168), (242, 214, 140))),
    ("olive",      "Olive",      "Sage",       ((122, 131, 85), (110, 118, 72), (138, 143, 96))),
    ("sage",       "Sage",       "Eucalyptus", ((158, 174, 148), (140, 160, 150), (214, 232, 220))),
    ("forest",     "Forest",     "Sage",       ((47, 90, 60), (60, 100, 70), (32, 72, 50))),
    ("emerald",    "Emerald",    "Ivory",      ((15, 110, 86), (29, 158, 117), (20, 130, 120))),
    ("dustyblue",  "Dusty Blue", "White",      ((152, 172, 190), (190, 205, 220), (168, 203, 224), (110, 130, 155))),
    ("navy",       "Navy",       "Ivory",      ((70, 90, 120), (46, 109, 180), (30, 60, 100))),
    ("lilac",      "Lilac",      "White",      ((211, 201, 224), (185, 168, 206))),
    ("plum",       "Plum",       "Blush",      ((107, 74, 102), (123, 94, 167), (90, 60, 90))),
    ("cocoa",      "Cocoa",      "Sand",       ((125, 100, 70), (92, 64, 44), (72, 52, 38))),
)
_NEUTRAL_ANCHORS = ((250, 250, 248), (244, 241, 230), (236, 231, 219),
                    (231, 220, 201), (203, 184, 158), (181, 165, 144),
                    (200, 196, 188), (70, 70, 72))

# 细族 → 大组（跨图投票按大组汇总，避免同组细族互相分票）
_FAMILY_GROUP = {
    "blush": "pink", "rose": "pink",
    "burgundy": "pink",
    "terracotta": "orange",
    "gold": "yellow", "butter": "yellow",
    "olive": "green", "sage": "green", "forest": "green", "emerald": "green",
    "dustyblue": "blue", "navy": "blue",
    "lilac": "purple", "plum": "purple",
    "cocoa": "brown",
}
FAMILY_KEYS = tuple(f[0] for f in _FAMILIES)
# 旧版 8 条色相带的族名 → 新细族（老会话已落库的 color_profile 用旧名，
# 不映射会被 pick_family 当成独立组投出去，再喂给 detect_palette 时匹配
# 不到任何族 → 整片退化成中性「香槟」）
_LEGACY_FAMILY = {
    "warm": "terracotta", "red": "burgundy", "gold": "gold",
    "green": "sage", "teal": "emerald", "blue": "dustyblue",
    "purple": "lilac", "mauve": "rose",
}
_W_L_PIXEL = 0.30     # 像素分族：明度权重低（浅档/深档必须同族）
_W_L_PAL = 0.55       # 色板定族：明度权重高（Blush / Rose / Burgundy 靠明暗分开）
_ANCHOR_LAB = None
_ANCHOR_KEYS = None


def _lab_from_bgr(px):
    """BGR 像素 (N,3) → LAB 浮点 (N,3)：L 0-100，a/b 以 0 为中心。"""
    arr = np.ascontiguousarray(np.asarray(px, dtype=np.uint8).reshape(-1, 1, 3))
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return np.stack([lab[:, 0] * (100.0 / 255.0),
                     lab[:, 1] - 128.0, lab[:, 2] - 128.0], axis=1)


def _anchor_table():
    """懒构建锚色 LAB 表（族锚 + 中性锚）。"""
    global _ANCHOR_LAB, _ANCHOR_KEYS
    if _ANCHOR_LAB is None:
        rgbs, keys = [], []
        for key, _n, _l, anchors in _FAMILIES:
            for c in anchors:
                rgbs.append(c)
                keys.append(key)
        for c in _NEUTRAL_ANCHORS:
            rgbs.append(c)
            keys.append("neutral")
        _ANCHOR_LAB = _lab_from_bgr(np.asarray(rgbs, dtype=np.uint8)[:, ::-1])
        _ANCHOR_KEYS = np.array(keys, dtype=object)
    return _ANCHOR_LAB, _ANCHOR_KEYS


def _pixel_keys(bgr_flat):
    """每个像素 → 最近锚色的族名（中性为 "neutral"）。"""
    lab, keys = _anchor_table()
    px = _lab_from_bgr(bgr_flat)
    d = ((_W_L_PIXEL * (px[:, 0:1] - lab[None, :, 0])) ** 2 +
         (px[:, 1:2] - lab[None, :, 1]) ** 2 +
         (px[:, 2:3] - lab[None, :, 2]) ** 2)
    return keys[d.argmin(axis=1)]


def _key_of_rgb(rgb):
    """单个 RGB → 族名（像素口径，明度权重低）。"""
    return str(_pixel_keys(np.asarray([rgb], dtype=np.uint8)[:, ::-1])[0])


_GROUP_LAB = None
_GROUP_LAB_KEYS = None


def _group_table():
    """大组级锚表：组内所有细族锚色的并集。"""
    global _GROUP_LAB, _GROUP_LAB_KEYS
    if _GROUP_LAB is None:
        rgbs, keys = [], []
        for key, _n, _l, anchors in _FAMILIES:
            g = _FAMILY_GROUP.get(key, key)
            for c in anchors:
                rgbs.append(c)
                keys.append(g)
        _GROUP_LAB = _lab_from_bgr(np.asarray(rgbs, dtype=np.uint8)[:, ::-1])
        _GROUP_LAB_KEYS = np.array(keys, dtype=object)
    return _GROUP_LAB, _GROUP_LAB_KEYS


def _key_of_palette(palette):
    """整块色板（浅→深）→ 细族：两级判定。

    一级「定大组」用饱和度加权的 Lab 平均色（明度权重 0.55）——只看
    色度，所以浅档/深档不会把大组带偏；
    二级「组内定细族」看平均明度（粉红组 blush/rose/burgundy、蓝组
    dustyblue/navy、紫组 lilac/plum、黄组 gold/butter 全靠明暗分开），
    绿组再加「最深档是否够深」「色度是否偏青」区分 forest/emerald。

    为什么不直接取单档：同色系 4 档里最饱和档往往是最深档（Dusty Blue
    的深档就是 Navy、Blush 的深档就是 Dusty Rose），取它必判错族。
    """
    rgbs = np.asarray(list(palette), dtype=np.uint8)
    lab = _lab_from_bgr(rgbs[:, ::-1])
    ws = []
    for c in palette:
        mx, mn = float(max(c)), float(min(c))
        ws.append(0.15 + (0.0 if mx <= 0 else (mx - mn) / mx))
    w = np.asarray(ws, dtype=np.float32)
    avg = (lab * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-6)

    tab, gkeys = _group_table()
    d = ((_W_L_PAL * (avg[0] - tab[:, 0])) ** 2 +
         (avg[1] - tab[:, 1]) ** 2 + (avg[2] - tab[:, 2]) ** 2)
    g = str(gkeys[int(d.argmin())])
    dmin = float(d.min())

    # 中性竞争：平均色离中性锚不比离彩色组锚远太多 → 判中性
    ntab, nkeys = _anchor_table()
    nm = (nkeys == "neutral")
    dn = ((_W_L_PAL * (avg[0] - ntab[nm][:, 0])) ** 2 +
          (avg[1] - ntab[nm][:, 1]) ** 2 + (avg[2] - ntab[nm][:, 2]) ** 2)
    if float(dn.min()) <= dmin * 1.15:
        return "neutral"

    L, a, b = float(avg[0]), float(avg[1]), float(avg[2])
    if g == "pink":
        return "blush" if L >= 78 else ("rose" if L >= 62 else "burgundy")
    if g == "orange":
        return "terracotta"
    if g == "yellow":
        return "butter" if L >= 84 else "gold"
    if g == "green":
        # 绿组四族靠「黄绿 vs 纯绿」+「明度」分开。不能用最深档判
        # forest —— detect_palette 会把最深档统一压暗到 L≈30，几乎必然
        # 触发（实测橄榄色板被判成 Forest & Sage）。olive 的黄绿在 Lab
        # 的 b 值上很明显（b≥20），forest 的 b 只有 10 上下。
        if a <= -25.0:            # 青绿（祖母绿/孔雀蓝绿）：a 极负、b 不高
            return "emerald"
        if float(lab[:, 2].max()) >= 20.0:   # 有一档明显黄绿 → 橄榄系
            return "olive"
        if L >= 72:
            return "sage"
        return "forest"
    if g == "blue":
        return "navy" if L <= 58 else "dustyblue"
    if g == "purple":
        return "lilac" if L >= 68 else "plum"
    if g == "brown":
        return "cocoa"
    return "neutral"


def _family_bands(bgr_flat, S, min_px=40, keys=None):
    """按细族统计 (share, 平均饱和01, 掩码, 族名)；中性族不计入。"""
    if keys is None:
        keys = _pixel_keys(bgr_flat)
    n = float(len(keys))
    bands = []
    for k in FAMILY_KEYS:
        bm = (keys == k)
        cnt = int(bm.sum())
        if cnt < min_px:
            continue
        bands.append((cnt / n, float(S[bm].mean()) / 255.0, bm, k))
    return bands


def _tag_mask(Hh, chrom, tag):
    """按色相带标签返回布尔掩码（Hh 为 0-360 展开色相）。"""
    for t, lo, hi in _HUE_BANDS:
        if t != tag:
            continue
        if hi > 360:
            return ((Hh >= lo) | (Hh < hi - 360)) & chrom
        return (Hh >= lo) & (Hh < hi) & chrom
    return None


def _bands_from_hsv(Hh, S, chrom, min_band=40):
    """按 8 条粗色相带统计 (share, 平均饱和01, 掩码, tag)。"""
    bands = []
    for tag, lo, hi in _HUE_BANDS:
        bm = ((Hh >= lo) | (Hh < hi - 360)) & chrom if hi > 360 \
            else (Hh >= lo) & (Hh < hi) & chrom
        if int(bm.sum()) >= min_band:
            bands.append((float(bm.mean()), float(S[bm].mean()) / 255.0, bm, tag))
    return bands


def _vote_family(bands, lenient=False):
    """粗带两档族决策（2026-11-03 16:35 用户酒红案例定稿）。

    单一「面积×彩度」公式无法同时服务淡雾蓝（大面积低饱和）与深酒红
    （小面积高饱和）：酒红案例 warm/gold 大色块总赢。改为：
      A = 面积×(0.10+S) 最大者（主体族）；
      B = 面积×S² 最大者（焦点族，即最抢眼的高饱和族）；
      若 B≠A 且 B 面积≥6%、平均饱和≥0.40、A 平均饱和<0.35 → 选 B；
      A 平均饱和<0.14 → 近中性，走中性暖族（Champagne & Ivory 式）。
    lenient=True（单图画像用）：近中性时不直接放弃，取一条「有彩度但面积
    小」的点缀族（meanS≥0.20、share≥1.5%），以便白底点缀色照片也能投票。
    """
    if not bands:
        return None
    A = max(bands, key=lambda b: b[0] * (0.10 + b[1]))
    # 焦点族竞争排除绿色带（70-150°）：婚礼摄影里绿=叶材是默认背景，
    # 绿只在它是主体族时通过 A 胜出（鼠尾草主题）；否则抢跑的
    # chartreuse 会把酒红/粉主题带偏（用户酒红案例实锤）。
    non_green = [b for b in bands if _FAMILY_GROUP.get(b[3]) != 'green']
    B = max(non_green or bands, key=lambda b: b[0] * b[1] * b[1])
    if A[1] < 0.14:
        if lenient:
            acc = [b for b in bands
                   if b[1] >= 0.20 and b[0] >= 0.015
                   and _FAMILY_GROUP.get(b[3]) != 'green']
            if acc:
                return max(acc, key=lambda b: b[0] * b[1])[3]
        return None
    chosen = A
    # 主题族规则：在面积≥10% 的主要族里选平均饱和最高者，
    # 若其饱和≥0.45 即为主题族（酒红/宝蓝这类重彩花材是婚礼
    # 主题色最强信号；小面积点缀 share<0.10 或淡族 S 低不触发）。
    # 注意红兰花常横跨 red/mauve 两带，按带面积门槛可避免被
    # 小 mauve 带抢走判定）
    major = [b for b in bands if b[0] >= 0.10]
    if major:
        best_s = max(major, key=lambda b: b[1])
        if best_s[1] >= 0.45:
            chosen = best_s
    if chosen is A:
        if B is not A and B[0] >= 0.06 and B[1] >= 0.40 and A[1] < 0.35:
            chosen = B             # 焦点族兜底
    return chosen[3]


def _small_data(bgr):
    """单图 → 80×80 展平 BGR 采样（与 detect_palette 同一采样口径）。"""
    h, w = bgr.shape[:2]
    s = min(80, min(h, w))
    small = cv2.resize(bgr, (s, s), interpolation=cv2.INTER_AREA)
    return small.reshape(-1, 3).astype(np.float32)


def image_color_profile(bgr):
    """单张照片的色彩画像（上传时预计算，落库到 entry["color_profile"]）。

    返回 {family, main, peak_sat, chroma}：
      family   色相带标签（red/warm/gold/green/teal/blue/purple/mauve），
               近中性照片为空串；
      main     该族内最饱和 30% 像素的均值 RGB（主色候选）；
      peak_sat 该族高光饱和（0-1，用于跨图投票加权）；
      chroma   有彩像素占比（0-1，画面里颜色多不多）。
    这样「识别」变成上传时一次性的确定性结果，出片阶段只需对每张图
    的画像做投票，不再把所有像素混在一起重新聚类——多一张/少一张图
    或一张偏色图都不会轻易推翻整体主题色。
    """
    data = _small_data(bgr)
    try:
        hsv = cv2.cvtColor(data.astype(np.uint8).reshape(-1, 1, 3),
                           cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
        Hh, S, V = hsv[:, 0] * 2.0, hsv[:, 1], hsv[:, 2]
    except Exception:
        return {"family": "", "main": None, "peak_sat": 0.0, "chroma": 0.0}
    chrom = (S >= 12) & (V >= 30) & (V <= 250)
    chroma = float(chrom.mean()) if chrom.size else 0.0
    _keys = _pixel_keys(data)
    bands = _family_bands(data, S, min_px=12, keys=_keys)
    tag = _vote_family(bands, lenient=True)
    main, peak = None, 0.0
    if tag:
        m = (_keys == tag)
        if int(m.sum()) >= 8:
            sub, sm = data[m], S[m]
            k = max(4, int(len(sub) * 0.3))
            idx = np.argsort(sm)[-k:]
            mean = sub[idx].mean(axis=0)
            main = [int(round(mean[2])), int(round(mean[1])), int(round(mean[0]))]
            peak = float(np.quantile(sm, 0.90) / 255.0)
    return {"family": tag or "", "main": main,
            "peak_sat": round(peak, 3), "chroma": round(chroma, 3)}


def pick_family(profiles):
    """跨图投票：选出整体主题族（2026-11-04 二次修正）。

    旧权重「1 + 2·chroma·peak_sat」基数 1.0，导致多张轻微偏色的
    场景图（chroma 0.5 也拿 ~1.7 票）能压过一张明确主题色的请柬。
    改为**乘性权重**：w = chroma × (0.30 + 2·peak_sat) ——
      · 近中性照片（chroma 低）几乎不投票；
      · 重彩、高饱和的照片权重显著变大（主题意图最强）。
    平手裁决：前两名权重差 <20% 时，取平均 peak_sat 更高的族
    （更饱和 = 更有主题意图）。绿带保护：green 胜出但优势 <1.5 倍
    且非绿族更饱和时让位（叶材是婚礼摄影默认背景；鼠尾草主题的
    绿是大面积低饱和，正常权重下仍会明显领先，不受影响）。
    全部近中性则返回空串（走 Champagne & Ivory 中性暖族）。
    """
    score = {}
    pks = {}
    for p in (profiles or []):
        if not isinstance(p, dict):
            continue
        f = (p.get("family") or "").strip()
        f = _LEGACY_FAMILY.get(f, f)
        if not f:
            continue
        chroma = float(p.get("chroma") or 0.0)
        peak = float(p.get("peak_sat") or 0.0)
        w = chroma * (0.30 + 2.0 * peak)
        score[f] = score.get(f, 0.0) + w
        pks.setdefault(f, []).append(peak)
    if not score:
        return ""

    def _mean_pk(keys):
        vals = []
        for k in keys:
            vals.extend(pks.get(k) or [0.0])
        return sum(vals) / len(vals) if vals else 0.0

    # 大组归并（2026-11-04）：细族先并到 pink/red/orange/yellow/green/
    # blue/purple/brown 八个大组再比票，否则同组细族互相分票 —— 例如
    # Blush 主题的图分到 blush+rose 各半票，会被 3 张绿色场景图的 sage
    # 各个击破。大组胜出后再取组内票最高的细族。
    grp, members = {}, {}
    for f, w in score.items():
        g = _FAMILY_GROUP.get(f, f)
        grp[g] = grp.get(g, 0.0) + w
        members.setdefault(g, []).append(f)

    ranked = sorted(grp.items(), key=lambda kv: kv[1], reverse=True)
    top_g, top_w = ranked[0]
    if len(ranked) > 1:
        sec_g, sec_w = ranked[1]
        # 平手裁决：差距 <20% → 更饱和者胜
        if sec_w >= top_w * 0.8 and _mean_pk(members[sec_g]) > _mean_pk(members[top_g]) + 0.05:
            top_g = sec_g
        # 绿组保护：叶材是婚礼摄影默认背景（鼠尾草主题是大面积低饱和绿，
        # 正常权重下仍明显领先，不受影响）
        elif top_g == "green" and sec_w >= top_w / 1.5 \
                and _mean_pk(members[sec_g]) > _mean_pk(members[top_g]):
            top_g = sec_g
    pool = members.get(top_g) or []
    return max(pool, key=lambda f: score[f]) if pool else ""


def detect_palette(bgr_images, n=N_SWATCH, forced_family=None):
    """参考片同款「同色系浅→深」识别（2026-11-03 15:23 用户定稿）。

    参考片 wedding_palettes_final 的逻辑——每个配色都是**同一个色系**
    的 4 级浅→深渐变（如 Dusty Blue & White = 近白→浅蓝→蓝→藏青）。
    实现：先定照片的主色相族（饱和像素色相直方图峰值），再在族内按
    明度分位取 4 档：最浅(近白/象牙)→浅中→主色(饱和核心)→最深锚。
    全片低饱和时退化为中性暖族（Champagne & Ivory 式）。

    forced_family（2026-11-04 稳定性改造）：传色相带标签（warm/blue/…）
    则跳过整体直方图投票、直接用该族取阶梯；传空串则强制走中性暖族。
    该值来自上传时逐图画像的跨图投票（pick_family），比把所有像素混在
    一起重新聚类稳定得多。
    """
    samples = []
    for im in bgr_images:
        h, w = im.shape[:2]
        s = min(80, min(h, w))
        small = cv2.resize(im, (s, s), interpolation=cv2.INTER_AREA)
        samples.append(small.reshape(-1, 3).astype(np.float32))
    if not samples:
        return _pad_variants([(202, 198, 192)], n)
    data = np.vstack(samples)

    try:
        hsv = cv2.cvtColor(data.astype(np.uint8).reshape(-1, 1, 3),
                           cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
        Hh, S, V = hsv[:, 0] * 2.0, hsv[:, 1], hsv[:, 2]
    except Exception:
        mean = data.mean(axis=0)
        return _pad_variants([tuple(int(round(c)) for c in mean[::-1])], n)

    def _avg(mask):
        if mask is None or int(np.asarray(mask).sum()) < 8:
            return None
        m = data[mask].mean(axis=0)
        return (int(round(m[2])), int(round(m[1])), int(round(m[0])))  # BGR→RGB

    # -- 1) 主色相族：优先用调用方给的投票结果，否则现场投票 --
    chrom = (S >= 12) & (V >= 30) & (V <= 250)
    keys_all = _pixel_keys(data)
    fam = None
    if forced_family is not None:
        fam = (keys_all == forced_family) if forced_family else None
        if fam is not None and int(fam.sum()) < 12:
            fam = None                     # 该族在本批图里几乎没有像素 → 退化中性
    elif int(chrom.sum()) >= max(64, len(data) // 25):
        tag = _vote_family(_family_bands(data, S, 40, keys=keys_all))
        if tag:
            fam = (keys_all == tag)

    pal = []
    # -- 2) 最浅档：全图最亮像素（白纱/桌布等中性高光），不够白则向白靠 --
    light = _avg(V >= np.quantile(V, 0.92)) or (243, 233, 217)
    lv = sum(light) / 3.0
    if lv < 225:
        light = tuple(int(round(c + (255 - c) * 0.55)) for c in light)
    pal.append(light)

    if fam is not None:
        fv = V[fam]
        # -- 3) 浅中档：族内 65-82 明度分位（明显浅于主色）--
        m1 = _avg(fam & (V >= np.quantile(fv, 0.65))
                  & (V <= np.quantile(fv, 0.82))) or light
        pal.append(m1)
        # -- 4) 主色档：族内最饱和 30% 的均值（核心色） --
        core = _avg(fam & (S >= np.quantile(S[fam], 0.70))) or m1
        pal.append(core)
        # -- 5) 最深锚：族内最暗 12%，不够深则压暗到锚位 --
        deep = _avg(fam & (V <= np.quantile(fv, 0.12))) or core
        dlv = sum(deep) / 3.0
        if dlv > 95:
            k = 88.0 / dlv
            deep = tuple(max(0, int(round(c * k))) for c in deep)
        pal.append(deep)
    else:
        # -- 中性暖族：ivory → champagne → tan → 深可可（全图明度分位） --
        for q0, q1 in ((0.62, 0.78), (0.38, 0.55), (0.0, 0.14)):
            c = _avg((V >= np.quantile(V, q0)) & (V <= np.quantile(V, q1))) or pal[-1]
            pal.append(c)
        dlv = sum(pal[-1]) / 3.0
        if dlv > 120:
            k = 110.0 / dlv
            pal[-1] = tuple(max(0, int(round(c * k))) for c in pal[-1])

    pal = sorted(pal, key=_lum, reverse=True)   # 浅→深
    return _pad_variants(pal[:n], n)


# ---------------------------------------------------------------------------
# 自动配色命名（2026-11-03 用户定稿：色板自动起婚礼配色名）
# ---------------------------------------------------------------------------
_PALETTE_NAMES = [
    ((250, 250, 248), "White"),
    ((244, 241, 230), "Ivory"),
    ((236, 231, 219), "Pearl"),
    ((240, 228, 208), "Cream"),
    ((231, 220, 201), "Champagne"),
    ((233, 205, 199), "Blush"),
    ((214, 190, 170), "Linen"),
    ((203, 184, 158), "Sand"),
    ((181, 165, 144), "Greige"),
    ((196, 142, 140), "Dusty Rose"),
    ((172, 130, 140), "Mauve"),
    ((193, 154, 107), "Camel"),
    ((201, 162, 92),  "Gold"),
    ((197, 138, 84),  "Caramel"),
    ((184, 115, 72),  "Copper"),
    ((201, 110, 78),  "Terracotta"),
    ((224, 122, 57),  "Paprika"),
    ((165, 84, 52),   "Rust"),
    ((158, 140, 120), "Taupe"),
    ((125, 100, 70),  "Cocoa"),
    ((122, 60, 72),   "Burgundy"),
    ((92, 64, 44),    "Chocolate"),
    ((72, 52, 38),    "Espresso"),
    ((158, 174, 148), "Sage"),
    ((140, 160, 150), "Eucalyptus"),
    ((120, 150, 130), "Fir"),
    ((190, 205, 220), "Powder Blue"),
    ((152, 172, 190), "Dusty Blue"),
    ((110, 130, 155), "Steel Blue"),
    ((70, 90, 120),   "Navy"),
    ((70, 70, 72),    "Charcoal"),
]


def _nearest_color_name(rgb):
    """RGB → 最近的婚礼色名（最近邻锚色）。"""
    best, bd = None, None
    for anchor, name in _PALETTE_NAMES:
        d = _dist2(rgb, anchor)
        if bd is None or d < bd:
            best, bd = name, d
    return best or "Neutral"


# 色系族名（参考片命名法：族名 + 浅档描述词）
# Dusty Blue & White / Olive & Sage / Terracotta & Rust / Champagne & Ivory
# 按色相区间映射到婚礼常用族名：(lo, hi, 主名, 浅档名)
_HUE_FAMILIES = [
    (0,   15,  "Burgundy",    "Ivory"),
    (15,  42,  "Terracotta",  "Ivory"),
    (42,  95,  "Olive",       "Sage"),
    (95,  145, "Sage",        "Eucalyptus"),
    (145, 175, "Eucalyptus",  "Ivory"),
    (175, 200, "Beryl",       "Champagne"),
    (200, 232, "Dusty Blue",  "White"),
    (232, 262, "Navy",        "Ivory"),
    (262, 300, "Lavender",    "White"),
    (300, 335, "Mauve",       "Blush"),
    (335, 361, "Burgundy",    "Ivory"),
]


def _family_names(palette):
    """色板 → (族名, 辅色名)，婚礼行业「主色 & 辅色」命名法。

    2026-11-04 用户选 A 起改为锚色最近邻（16 族）：不再用色相区间 +
    饱和度门槛，淡粉/淡蓝/淡紫不再被吞成「香槟」，金色不再被判成陶土，
    粉/正红/酒红各自成族。
    """
    if not palette:
        return "Champagne", "Ivory"
    key = _key_of_palette(palette)
    if key == "neutral":
        return "Champagne", "Ivory"
    # 封面定稿名 "Terracotta & Rust"（2026-11-03）：深档够深够彩时用 Rust
    if key == "terracotta":
        deep = min(palette, key=_lum)
        if _lum(deep) < 130 and (max(deep) - min(deep)) > 60:
            return "Terracotta", "Rust"
    for k, name, light, _anchors in _FAMILIES:
        if k == key:
            return name, light
    return "Champagne", "Ivory"


def name_palette(palette):
    """把识别出的色板起成婚礼配色名（参考视频 "Terracotta & Rust" 式）。

    2026-11-03 15:23 起改为**同色系命名**：先由主色相定色系族
    （Dusty Blue / Olive / Terracotta / Champagne …），再配浅档描述词
    （White / Ivory / Sage …），与参考片命名法一致。
    """
    if not palette:
        return ""
    fam, light = _family_names(palette)
    return "%s & %s" % (fam, light)


# ---------------------------------------------------------------------------
# AI 辅助配色（2026-11-04 用户：接入火山方舟视觉模型做语义识别）
# 混合口径：精确 4 档 RGB 仍由 OpenCV 像素法给出（零偏差），AI 只做语义层——
# 确认主色族、生成自然品牌化命名。不可用时返回 None，调用方回退 OpenCV 命名。
# 复用 core.showcase.chat_vision（同一把 Ark Key / 模型，零新增依赖）。
# ---------------------------------------------------------------------------

def _montage_thumb(bgr_list, max_n=6, side=320, gap=14):
    """最多取 max_n 张，等比缩放到高 side，拼成 2 列暖白底缩略图（AI 输入用）。"""
    pick = list(bgr_list)[:max_n]
    if not pick:
        return None
    thumbs = []
    for im in pick:
        h, w = im.shape[:2]
        s = min(1.0, float(side) / max(h, 1))
        t = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))),
                       interpolation=cv2.INTER_AREA)
        thumbs.append(t)
    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    cell_w = max(t.shape[1] for t in thumbs)
    cell_h = max(t.shape[0] for t in thumbs)
    canvas = np.full((rows * (cell_h + gap) - gap, cols * (cell_w + gap) - gap, 3),
                     (245, 243, 238), dtype=np.uint8)   # 暖白底
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = c * (cell_w + gap) + (cell_w - t.shape[1]) // 2
        y = r * (cell_h + gap) + (cell_h - t.shape[0]) // 2
        canvas[y:y + t.shape[0], x:x + t.shape[1]] = t
    return canvas


def ai_palette_hint(bgr_list, cfg, timeout=120):
    """火山方舟视觉模型对图集做语义识别，返回 {palette_name, family, note} 或 None。

    仅在 cfg 含 api_key 时尝试；任何异常（无 Key / 模型未开通 / 网络或解析失败）
    一律返回 None，由 render 端回退 OpenCV 命名——保证模版零 AI 时仍可用。
    """
    try:
        key = ((cfg or {}).get("api_key") or "").strip()
        if not key:
            return None
        mont = _montage_thumb(bgr_list)
        if mont is None:
            return None
        fd, tmp = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            ok, buf = cv2.imencode(".jpg", mont, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                return None
            with open(tmp, "wb") as f:
                f.write(buf.tobytes())
            from core import showcase
            system_prompt = (
                "You are a professional wedding color palette analyst. "
                "Given a set of wedding photos (florals, decor, invitations, table settings), "
                "identify the dominant color palette. Respond with STRICT JSON only, no prose, "
                'in this exact shape: {"palette_name": "<short English Title Case title, '
                '2-4 words, ampersand-separated, e.g. \\"Terracotta & Rust\\" or '
                '\\"Dusty Blue & White\\">", "family": "<one of: red, terracotta, burgundy, '
                'pink, blush, copper, gold, champagne, green, sage, olive, blue, dusty blue, '
                'navy, lavender, mauve, neutral, ivory, beige, cocoa, rust>", "note": "<one '
                'short sentence describing the palette>"}')
            user_text = ("Analyze this wedding photo collage. What is the dominant color "
                         "palette of the florals and decor? Return only the JSON.")
            txt, _usage, _el = showcase.chat_vision(
                cfg, system_prompt, user_text, tmp,
                model=None, max_tokens=400, temperature=0.15, timeout=timeout)
        finally:
            try:
                os.remove(tmp)
            except BaseException:
                pass
        m = re.search(r"\{.*\}", txt or "", re.S)
        if not m:
            return None
        obj = json.loads(m.group(0))
        name = str(obj.get("palette_name") or "").strip().strip(" \"'.,;:")
        fam = str(obj.get("family") or "").strip().lower()
        note = str(obj.get("note") or "").strip()
        if not (2 <= len(name) <= 40):
            return None
        return {"palette_name": name, "family": fam, "note": note}
    except BaseException:
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# 文字 + 色块 overlay（一次生成 RGBA）
# ---------------------------------------------------------------------------
def _fit_font(path, text, target_w, tracking=0):
    lo, hi = 12, 400
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(path, mid)
        tw = sum(probe.textlength(c, font=f) for c in text) + tracking * (len(text) - 1)
        if tw < target_w:
            lo = mid
        else:
            hi = mid
    return ImageFont.truetype(path, lo)


def _draw_tracked(d, cx, cy, text, font, tracking, fill):
    ws = [d.textlength(c, font=font) for c in text]
    total = sum(ws) + tracking * (len(text) - 1)
    x = cx - total / 2
    asc, desc = font.getmetrics()
    y = cy - (asc + desc) / 2
    for c, wd in zip(text, ws):
        d.text((x, y), c, font=font, fill=fill)
        x += wd + tracking


def _build_overlay(kicker, script, palette, W, H):
    """返回 (text_layer, pill_layers)，供渲染端按参考片时序分层淡入。

    text_layer : RGBA，两行标题（含阴影）——整体一次淡入；
    pill_layers: [RGBA, ...] 每条色块独立一层（含各自投影），从上到下
                 逐条淡入（参考片动画）。

    2026-11-03 用户参考片定稿（wedding_palettes_final 实测）：
      · 标题两行宽约 0.40W / 0.45W（"Weddings 2027 / Terracotta & Rust"）；
      · 色块窄条 0.384W，组块整体垂直居中（实测组中心 ≈ 0.497H），
        标题在色块组上方。
    palette: [(r,g,b), ...] 已识别主色（RGB），内部按明度浅→深排列。
    色块条始终渲染（本模版核心产物）；kicker / script 为空则跳过对应文字。
    """
    # 2026-11-04 用户：文字部分稍微放大一点（fit 宽度放宽 → 字号变大）
    fk = _fit_font(CORMORANT, kicker, int(0.66 * W), tracking=2) if kicker else None
    fs = _fit_font(CORMORANT_ITALIC, script, int(0.72 * W)) if script else None
    kh = sum(fk.getmetrics()) if fk else 0
    sh_ = sum(fs.getmetrics()) if fs else 0

    cols = sorted(palette, key=lambda c: -(c[0] + c[1] + c[2]))[:N_SWATCH] if palette else []
    # 2026-11-03 17:15 用户：长条胶囊 → 圆形色块（直径 = SW_WF*W），竖排、整组居中
    bw = int(W * SW_WF)
    bh = bw                          # 圆形：高 = 直径
    gap = int(W * SW_GAPF)
    stack_h = len(cols) * bh + (len(cols) - 1) * gap if cols else 0

    gap_lines = int(W * GAP_LINES_F)
    gap_tsw = int(W * GAP_TEXT_SW_F)
    text_h = 0
    if kicker:
        text_h += kh
    if kicker and script:
        text_h += gap_lines
    if script:
        text_h += sh_
    # 2026-11-03 14:42 用户：文字 + 色块堆作为「一个整体」上下居中，
    # 而不是只居中色块、文字堆在上方。
    if text_h > 0:
        total_h = text_h + gap_tsw + stack_h
    else:
        total_h = stack_h
    group_top = (H - total_h) // 2
    text_top = group_top
    y0 = group_top + (text_h + gap_tsw if text_h > 0 else 0)

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    txt = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(txt)
    white = (255, 255, 255, 255)
    if kicker:
        ky = text_top + kh / 2
        _draw_tracked(sd, W // 2, ky + 3, kicker, fk, 3, (0, 0, 0, 170))
        _draw_tracked(d, W // 2, ky, kicker, fk, 3, white)
    if script:
        sy = text_top + (kh + gap_lines if kicker else 0) + sh_ / 2
        sd.text((W // 2, sy + 4), script, font=fs, fill=(0, 0, 0, 185), anchor="mm")
        d.text((W // 2, sy), script, font=fs, fill=white, anchor="mm")
    shadow = shadow.filter(ImageFilter.GaussianBlur(5))
    text_layer = Image.alpha_composite(shadow, txt)

    pill_layers = []
    if cols:
        x0 = (W - bw) // 2
        yy = y0
        for (rr, gg, bb) in cols:
            lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            sd2 = ImageDraw.Draw(sh)
            sd2.ellipse([x0 + 2, yy + 8, x0 + bw + 2, yy + bh + 8],
                        fill=(0, 0, 0, 90))
            sh = sh.filter(ImageFilter.GaussianBlur(9))
            lay = Image.alpha_composite(sh, lay)
            dl = ImageDraw.Draw(lay)
            dl.ellipse([x0, yy, x0 + bw, yy + bh],
                       fill=(rr, gg, bb, 255))
            pill_layers.append(lay)
            yy += bh + gap
    return text_layer, pill_layers


# ---------------------------------------------------------------------------
# 拼贴板（双列全出血 masonry）
# ---------------------------------------------------------------------------
def _build_collage(bgr_imgs, W, H):
    """两列等宽(W/2) masonry：按累计高度贪心分配，各列居中，模糊底补白。

    返回 H 高的 RGB ndarray（已中心裁切到 9:16）。
    """
    paths = list(bgr_imgs.keys())
    cw = W // 2
    native = {}
    for p in paths:
        ih, iw = bgr_imgs[p].shape[:2]
        native[p] = max(2, int(round(cw * ih / max(1, iw))))
    cols = [[], []]
    ch = [0, 0]
    for p in paths:
        ci = 0 if ch[0] <= ch[1] else 1
        cols[ci].append(p)
        ch[ci] += native[p]
    MAXF = 2.0
    scale = [min(MAXF, max(1.0, H / float(max(1, ch[ci])))) for ci in range(2)]
    heights = {}
    for ci in range(2):
        for p in cols[ci]:
            heights[p] = max(2, int(round(native[p] * scale[ci])))
    ch = [sum(heights[p] for p in cols[ci]) for ci in range(2)]
    board_h = max(max(ch), 1)

    acc = np.zeros((96, 54, 3), np.float32)
    for im in bgr_imgs.values():
        acc += cv2.resize(im, (54, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
    acc /= max(1, len(bgr_imgs))
    backdrop = cv2.resize(acc, (W, board_h), interpolation=cv2.INTER_LANCZOS4)
    backdrop = cv2.GaussianBlur(backdrop, (0, 0), 40)
    board = Image.fromarray(backdrop.astype(np.uint8)).convert("RGB")

    x = 0
    for ci, col in enumerate(cols):
        y = (board_h - ch[ci]) // 2
        for p in col:
            tile = _cover(bgr_imgs[p], cw, heights[p])
            board.paste(Image.fromarray(tile), (x, y))
            y += heights[p]
        x += cw

    arr = np.asarray(board).astype(np.uint8)
    if arr.shape[0] < H:
        canvas = cv2.GaussianBlur(cv2.resize(arr, (W, H), interpolation=cv2.INTER_AREA),
                                  (0, 0), 60)
        y0 = (H - arr.shape[0]) // 2
        canvas[y0:y0 + arr.shape[0], :] = arr
        arr = canvas
    arr = arr.astype(np.uint8)
    top = max(0, (arr.shape[0] - H) // 2)
    return arr[top:top + H]


# ---------------------------------------------------------------------------
# 单图满幅（2026-11-03 用户终版：纯硬切、零运镜）
# ---------------------------------------------------------------------------
def _cover(bgr, tw, th):
    ih, iw = bgr.shape[:2]
    s = max(tw / iw, th / ih)
    nw, nh = int(round(iw * s + 0.5)), int(round(ih * s + 0.5))
    im = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    x0, y0 = (nw - tw) // 2, (nh - th) // 2
    return im[y0:y0 + th, x0:x0 + tw]


def _ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


# 全局尺寸（在 render 里赋值，供渲染用）
W0, H0 = 1080, 1920


def render_collage_palette(images, out_path, ratio="9x16", fps=30,
                           title="", subtitle="", show_text=True, progress=None,
                           font=None, palette_override=None, forced_family=None,
                           cfg=None):
    """N(≥3) 张照片 → 双列拼贴开场(橘色色块) + 逐张单图（仅画面，无音轨）。

    palette_override: 传入 [(r,g,b),...] 则跳过默认橘色系、改用该色板
    （封面/样片定稿用；不传则默认 Terracotta & Rust 橘色系）。
    """
    paths = [p for p in (images or []) if p and os.path.isfile(p)]
    if len(paths) < 3:
        raise ValueError("Wedding Look Collage needs at least 3 photos")

    global W0, H0
    OW, OH = ratio_size(ratio)
    W0, H0 = int(OW), int(OH)
    W, H = W0, H0

    if progress:
        progress(6, 100, "Reading photos")

    bgr = _load(paths)

    if progress:
        progress(10, 100, "Recognizing colors")
    if palette_override:
        # 显式定稿色板（如封面 Terracotta & Rust 四色）
        palette = [tuple(int(x) for x in c) for c in palette_override][:N_SWATCH]
        palette_name = name_palette(palette)
    else:
        # 2026-11-03 15:06 用户反转：色块按上传图片自动识别匹配，
        # 不再套用模版固定橘色系；蓝/绿等冷色如实出现。
        # 2026-11-04 稳定性改造：优先用上传时逐图画像的跨图投票结果
        # （forced_family），没有画像才现场重新聚类。
        palette = detect_palette(list(bgr.values()), N_SWATCH,
                                 forced_family=forced_family)
        palette_name = name_palette(palette)
        # 2026-11-04 用户：接入火山方舟视觉模型做语义识别——精确 4 档 RGB 仍由
        # OpenCV 给出（零偏差），AI 只提供自然品牌化命名与语义族；无 Key / 模型
        # 未开通 / 解析失败一律回退 OpenCV 命名（palette_name 不变）。
        ai_hint = ai_palette_hint(list(bgr.values()), cfg) if cfg is not None else None
        if ai_hint:
            palette_name = ai_hint["palette_name"]

    # 2026-11-02 修 R/B 反转（红线 13：PIL/ffmpeg 走 RGB，cv2 出 BGR）：
    # 旧版把 BGR 数组直接塞进 Image.fromarray / rgb24 管线，成片红蓝整体
    # 对调（陶土橙被渲成蓝）。这里显示路径统一先转 RGB；取色仍吃 BGR。
    rgb = {p: v[:, :, ::-1] for p, v in bgr.items()}

    # 文字（受 show_text 控制；色块始终渲染）
    # 副标题留空 → 用自动识别出的色板名（如 "Terracotta & Sand"）；客户手填则用客户的；
    # 显式 override 时则为 override 色板名。
    sub_in = (subtitle or "").strip()
    kicker = (title or "").strip() if show_text else ""
    script = (sub_in or palette_name) if show_text else ""
    text_layer, pill_layers = _build_overlay(kicker, script, palette, W, H)
    text_np = np.asarray(text_layer)
    pill_nps = [np.asarray(l) for l in pill_layers]

    if progress:
        progress(16, 100, "Building collage")
    collage = _build_collage(rgb, W, H)

    singles = [_cover(b, W, H) for b in rgb.values()]

    n_intro = int(round(INTRO * fps))
    n_single = int(round(HOLD * fps))
    total = n_intro + n_single * len(singles)
    total_secs = total / fps

    if progress:
        progress(20, 100, "Rendering")

    cmd = [FFMPEG, "-y", "-hide_banner",
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
            if i < n_intro:
                t = i / fps
                frame = collage.astype(np.float32)
                # 参考片时序：标题先淡入，4 条色块从上到下错峰淡入
                comps = [(text_np, _ease((t - TXT_DELAY) / TXT_DUR))]
                for j, pn in enumerate(pill_nps):
                    comps.append((pn, _ease((t - (SW_DELAY0 + j * SW_STAG)) / SW_DUR)))
                if any(a > 0 for _, a in comps):
                    base = Image.fromarray(collage).convert("RGBA")
                    for pn, a in comps:
                        if a <= 0:
                            continue
                        lay = pn.astype(np.float32).copy()
                        lay[:, :, 3] = lay[:, :, 3] * a
                        base = Image.alpha_composite(
                            base, Image.fromarray(lay.astype(np.uint8)))
                    frame = np.asarray(base.convert("RGB")).astype(np.float32)
            else:
                # 纯硬切单图：每张静止展示 HOLD 秒，切换不使用任何运镜
                k = (i - n_intro) // n_single
                frame = singles[k].astype(np.float32)
            proc.stdin.write(frame.astype(np.uint8).tobytes())
            if progress and (i % 60 == 0):
                progress(int(20 + 75 * i / max(total, 1)), 100, "Rendering")
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed rc=%s" % proc.returncode)

    if progress:
        progress(100, 100, "Done")
    return {"video": out_path, "secs": round(total_secs, 2),
            "images": len(paths), "ratio": ratio, "palette": palette,
            "palette_name": palette_name, "family": forced_family,
            "ai_used": bool(ai_hint),
            "palette_ai_name": (ai_hint or {}).get("palette_name"),
            "palette_ai_family": (ai_hint or {}).get("family"),
            "palette_ai_note": (ai_hint or {}).get("note")}
