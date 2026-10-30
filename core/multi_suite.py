# -*- coding: utf-8 -*-
"""多图合集运镜（N 张合集图 → 一条 ≤15s 的成片）—— 独立于单图裁卡模版。

为什么单独一个模块
------------------
线上现有的 `Product Showcase` 模版是「1 张图 → 识别图内的信封/内页/工艺 →
围绕这一张图做运镜」。新模版是**多图入**：客户上传 N 张合集图（每张一个款式），
要的是「开头多款同屏 → 逐款一段 + 款式名牌」的成片结构。

两者的编排逻辑、时长预算、文字层都不一样，**混在一起改会碰坏线上那条**，
所以本模块完全独立：只 import card_pan 的底层原语（mk / gen_shot / 缓动 /
窗口夹取 / 速度红线），不 import 也不修改它的 build_plan_from_regions。

成片结构（用户 2026-10-14 二次确认：开场不要四宫格合集板，照参考片
—— 第一张图直接开场、大标题叠在它上面，之后逐款一段 + 款式名牌）
--------------------------------------------------------------------------
    0.0 – op+seg0  第一款：**旋转环绕**开场（时长 = 开场预算 + 首段预算）
                   标题淡入 → 在 op 时刻淡出 → 该款名牌接着淡入
    之后            逐款分段：每张图一段（硬切），运镜取自 8 种池且互不重复
                   （用户 2026-10-14：每张图运镜不要重复、速度稍快）
（参考片实测：标题压在第一个款式上约 2.5s，不另拼合集板；名牌都在画面上方）

文字样式 = 参考片（E:\\推广素材\\案例\\多款 的 DIY 片）实测复刻：
主标题近白奶油两行（正体+斜体）、名牌淡奶油黄衬线斜体在画面上方、
无固定底板，仅亮白背景时自适应加一条软边压暗带保证可读。

参考视频来源：`E:\\推广素材\\案例\\多款`（5 条实拍 9:16 案例，2026-10-14 学习）。

时长口径（硬红线：成片 ≤15s；每段目标 3s，用户 2026-10-14）
------------------------------------------------------------
    seg = min(3.0, (TOTAL - OPENER) / N)，下限 MIN_SEG
    N=2 → 3.0s×2（合计 8.2s）      N=3 → 3.0s×3（11.2s）
    N=4 → 3.0s×4（14.2s）          N=5 → 2.56s×5（15.0s）
    N=6 → 2.13s×6（15.0s）         N=8 → 1.6s×8（15.0s）
    图少时每段 3s 左右、长不足 15s（不硬凑时长拖节奏），图多时自动变快剪。

文字与字体
----------
名牌/标题用 Cormorant Garamond（OFL 开源，随包放在 static/fonts/）。
**不能依赖系统字体**：本地是 Windows、发布环境是 Linux，只有内置才两边一致。
字号随输出宽度等比缩放，所以换画幅（9:16 / 4:5 / 1:1）排版不乱。

性能
----
* 源图先降到长边 ≤2600（输出才 1080x1920，再大只是白烧 CPU）；
* 首镜是旋转镜头 → 走 gen_shot 的超采样慢路径（约 0.1s/帧，仅此一镜）；
  其余段落全部非旋转 → 直接按输出分辨率 warp（快速路径，约 0.01s/帧）；
* 全部帧写进同一个 ffmpeg 管道（硬切天然成立），只有一次编码。

依赖：cv2 / numpy / Pillow / imageio_ffmpeg。
"""
import math
import os
import subprocess
import tempfile
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import card_pan as cp

# ---------------- 常量 ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(BASE_DIR, "static", "fonts")
FONT_LABEL = os.path.join(FONT_DIR, "CormorantGaramond-SemiBoldItalic.ttf")
FONT_TITLE = os.path.join(FONT_DIR, "CormorantGaramond-SemiBold.ttf")

TOTAL_MAX = 15.0      # 成片总时长红线（用户 2026-10-14：不要超过 15s）
OPENER = 2.2          # 标题占用预算（叠在第一个款式上，参考片 ≈2.5s）
OPENER_SHORT = 1.6    # 图少于 4 张时标题窗口更短
MIN_SEG = 1.6         # 单段最短（图多时自动快剪的下限）
TARGET_SEG = 2.5      # 单段目标时长（用户 2026-10-15：镜头速度偏慢→缩短单段、更利落；图少不硬凑 15s）
MAX_IMAGES = 8        # 最多支持的合集图数（再多就超出 15s 预算）
MIN_IMAGES = 2

FPS = 30
CRF = 18
PRESET = "veryfast"   # 与单图管线一致：CRF 不变、清晰度几乎无差别、编码快一倍

INK = (250, 246, 238)         # 象牙白文字（#FAF6EE，旧版兜底色）

# 文字颜色取自参考片实测（E:\推广素材\案例\多款 的 DIY 片逐帧取色）：
# 主标题近白奶油 #FBFBEB，款式名牌淡奶油黄 #FEFFDA。RGB 顺序（_blend_layer 会转 BGR）。
CREAM_TITLE = (251, 251, 235)
CREAM_LABEL = (254, 252, 214)

# 参考片的排版：名牌在画面上方 ~14%、无底板、大号衬线斜体、只有柔影；
# 主标题居中偏下，两行时第一行正体 + 第二行斜体。白底图上奶油字会糊，
# 所以压暗带改成「自适应」：背景越亮带越深（亮白平铺图才出带，深色图完全无带）。
LABEL_Y = 0.145       # 名牌纵向位置（画面高度比例）
TITLE_Y = 0.53        # 标题纵向位置
BAND_HALF = 0.125     # 自适应压暗带的半高（画面高度比例）
BAND_MAX = 0.40       # 压暗带最深的暗度
BAND_LUM_LO = 0.70    # 背景平均亮度低于此 → 完全不出带
BAND_LUM_HI = 0.88    # 高于此 → 出满带

LABEL_SIZE = 0.094    # 名牌字号 = 输出宽 × 该比例（用户 2026-10-15：再大一点；比标题 0.100 略小、保留层级）
TITLE_SIZE = 0.100    # 标题字号
TRACK_LABEL = 0.10    # 字距（相对字号）——参考片名牌字距不大
TRACK_TITLE = 0.06    # 主标题字距同样收小，贴参考片
TITLE_MAXW = 0.88     # 标题最大宽度（超出自动缩字号）

WORK_MAX = 2600       # 源图工作边长上限

# ============================================================================
#  文字层
# ============================================================================
def _load_font(path, size):
    try:
        return ImageFont.truetype(path, int(size))
    except Exception:
        # 字体缺失时退回 Pillow 默认字体（不抛异常，宁可排版难看也不让任务挂掉）
        return ImageFont.load_default()

def _text_layer(text, path, size, tracking, color=INK, shadow=True):
    """把一行字渲染成 RGBA 图层（返回 np.uint8 的 (h, w, 4)）。

    逐字符绘制以做出「字距」（Pillow 没有 letter-spacing）。
    """
    font = _load_font(path, size)
    text = str(text or "")
    if not text.strip():
        return None
    pad = int(size * 0.5)
    # 逐字符量宽度
    widths = []
    for ch in text:
        try:
            bb = font.getbbox(ch)
            widths.append(bb[2] - bb[0])
        except Exception:
            widths.append(int(size * 0.5))
    gap = size * tracking
    total_w = int(sum(widths) + gap * max(0, len(text) - 1))
    asc, desc = font.getmetrics()
    total_h = asc + desc
    img = Image.new("RGBA", (total_w + pad * 2, total_h + pad * 2), (0, 0, 0, 0))
    if shadow:
        # 柔化黑晕（参考片同款 soft shadow）：整行字画黑 → 高斯模糊 → 垫在字下面。
        # 比逐字偏移描影立得住——亮白纸面上奶油字也能读清，且不破坏「无底板」观感。
        from PIL import ImageFilter
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ds = ImageDraw.Draw(sh)
        x = float(pad + size * 0.04)
        for ch, w in zip(text, widths):
            ds.text((x, pad + size * 0.05), ch, font=font, fill=(10, 10, 8, 215))
            x += w + gap
        sh = sh.filter(ImageFilter.GaussianBlur(radius=max(2.0, size * 0.05)))
        img = Image.alpha_composite(sh, img)
    d = ImageDraw.Draw(img)
    x = float(pad)
    for ch, w in zip(text, widths):
        d.text((x, pad), ch, font=font, fill=tuple(color) + (255,))
        x += w + gap
    return np.array(img)

def _blend_layer(frame, layer, cx, cy, alpha):
    """把 RGBA 图层按 alpha 叠到 frame 的 (cx, cy) 中心处（越界自动裁）。"""
    if layer is None or alpha <= 0.002:
        return frame
    H, W = frame.shape[:2]
    lh, lw = layer.shape[:2]
    x0 = int(round(cx - lw / 2.0))
    y0 = int(round(cy - lh / 2.0))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x0 + (lw - sx0)), min(H, y0 + (lh - sy0))
    if x1 <= x0 or y1 <= y0:
        return frame
    sub = layer[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)].astype(np.float32)
    a = (sub[:, :, 3:4] / 255.0) * float(alpha)
    bgr = sub[:, :, 2::-1]      # PIL 是 RGB、cv2 帧是 BGR，必须换序（奶油黄会因此变蓝）
    dst = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (dst * (1.0 - a) + bgr * a).astype(np.uint8)
    return frame

def _strip_lum(img, cy, half):
    """文字条带区域的平均亮度（0~1）。用来源图近似 —— 运镜位移小，够用。"""
    H = img.shape[0]
    y0 = max(0, int(H * (cy - half)))
    y1 = min(H, int(H * (cy + half)))
    if y1 <= y0:
        return 0.5
    g = cv2.cvtColor(img[y0:y1], cv2.COLOR_BGR2GRAY)
    return float(g.mean()) / 255.0

def _band_dark(lum):
    """自适应压暗：背景越亮带越深；深色/中性背景完全不出带（贴参考片的观感）。"""
    t = (lum - BAND_LUM_LO) / float(BAND_LUM_HI - BAND_LUM_LO)
    return BAND_MAX * max(0.0, min(1.0, t))

def _band_scrim(OH, cy=TITLE_Y, half=BAND_HALF, dark=BAND_MAX):
    """标题压暗带：以 cy 为中心的水平软边暗带，返回 (keep, y0, y1)。

    参考片的大标题都压在画面中部的平铺图上 —— 白底不压暗，白色衬线字必然发白。
    keep 是亮度保持系数（1=不变），只对 [y0, y1) 行生效。
    """
    m = np.ones((OH,), np.float32)
    c = int(OH * cy)
    h = int(OH * half)
    y0, y1 = max(0, c - h), min(OH, c + h)
    if y1 > y0:
        t = np.abs(np.linspace(-1.0, 1.0, y1 - y0, dtype=np.float32))
        m[y0:y1] = 1.0 - dark * np.clip(1.0 - t, 0.0, 1.0) ** 1.2
    return m, y0, y1

def _fit_width(mk_layers, size, max_w, min_size=26):
    """从 size 起渲染文字层，超宽就按 0.92 缩字号重画，直到装得下。"""
    layers = mk_layers(size)
    while max(l.shape[1] for l in layers) > max_w and size > min_size:
        size = int(size * 0.92)
        layers = mk_layers(size)
    return layers

def _title_layers(text, OW, font_path=None):
    """主标题层（参考片样式）：支持两行 —— 第一行正体、第二行斜体。

    用户输入里带竖线「|」或换行即视为两行；单行就一层正体。返回层列表或 None。
    font_path: 客户自选字体（2026-10-26）；None 用模版默认（正体 CG）。
    """
    text = str(text or "").strip()
    if not text:
        return None
    fpath = font_path or FONT_TITLE
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == 1 and "|" in text:
        lines = [s.strip() for s in text.split("|") if s.strip()]
    lines = (lines[:2] or [text])
    def mk(sz):
        return [_text_layer(ln, fpath if (i and len(lines) > 1) else fpath,
                            sz, TRACK_TITLE, color=CREAM_TITLE, shadow=True)
                for i, ln in enumerate(lines)]
    return _fit_width(mk, max(24, int(round(OW * TITLE_SIZE))), int(OW * TITLE_MAXW))

def _line_w(text, path, size, tracking):
    """估一行字的总宽（含两侧 pad，与 _text_layer 的实际层宽一致）。"""
    font = _load_font(path, size)
    text = str(text or "")
    if not text:
        return 0
    widths = []
    for ch in text:
        try:
            bb = font.getbbox(ch)
            widths.append(bb[2] - bb[0])
        except Exception:
            widths.append(int(size * 0.5))
    gap = size * tracking
    return int(sum(widths) + gap * max(0, len(text) - 1)) + int(size * 0.5) * 2

def _wrap2(text, path, size, tracking, max_w):
    """一行放不下的文字按词拆两行：优先两行都不超宽，其次两行宽度均衡。"""
    words = text.split()
    if len(words) < 2:
        return [text]
    best, best_cost = None, None
    for k in range(1, len(words)):
        l1, l2 = " ".join(words[:k]), " ".join(words[k:])
        w1 = _line_w(l1, path, size, tracking)
        w2 = _line_w(l2, path, size, tracking)
        over = max(0, w1 - max_w) + max(0, w2 - max_w)   # 超宽总量越小越好
        cost = (over, abs(w1 - w2))                      # 再取两行均衡
        if best_cost is None or cost < best_cost:
            best_cost, best = cost, (l1, l2)
    return list(best)

def _label_layers(text, OW, font_path=None):
    """名牌层：先试单行原字号；装不下按词拆两行（用户 2026-10-15：一行显示
    不全就放 2 行）；两行仍超宽才缩字号兜底。

    font_path: 客户自选字体（2026-10-26）；None 用模版默认（斜体 CG）。

    返回 None 或 (layers, gap, block)：
    layers 是 1~2 个 RGBA 图层；gap 是堆叠时加在整层高度上的步进修正
    （图层自带上下各 0.5 字号的透明 pad，不收紧的话两行视觉空隙近一行高）；
    block 是整叠总高。多行排布：top += l.shape[0] + gap。

    注意必须走本函数的显式宽度判断——直接把单层 ndarray 喂给 _fit_width 会
    静默失效（对数组迭代取到的是像素行，宽度恒为 4），名牌因此从缩过字号。
    """
    text = str(text or "").strip()
    if not text:
        return None
    fpath = font_path or FONT_LABEL
    size = max(20, int(round(OW * LABEL_SIZE)))
    max_w = int(OW * 0.92)
    if _line_w(text, fpath, size, TRACK_LABEL) <= max_w:
        layers = [_text_layer(text, fpath, size, TRACK_LABEL,
                              color=CREAM_LABEL, shadow=True)]
        return layers, 0, layers[0].shape[0]
    lines = _wrap2(text, fpath, size, TRACK_LABEL, max_w)

    def mk(sz):
        return [_text_layer(ln, fpath, sz, TRACK_LABEL,
                            color=CREAM_LABEL, shadow=True) for ln in lines]
    layers = _fit_width(mk, size, max_w, min_size=20)
    pad = int(size * 0.5)
    gap = -2 * pad + max(4, int(0.18 * size))   # 抵消两层相邻 pad，留 ~0.18 字号行距
    block = sum(l.shape[0] for l in layers) + gap * (len(layers) - 1)
    return layers, gap, block

# ============================================================================
#  画面取景
# ============================================================================
def _load_work(path):
    """读图 + 降到工作尺寸（长边 ≤ WORK_MAX）。非 ASCII 路径必须 imdecode。"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    from core.imgproc import normalize_luma   # 2026-10-27 亮度归一（用户需求）
    img = normalize_luma(img)
    h, w = img.shape[:2]
    m = max(h, w)
    if m > WORK_MAX:
        s = WORK_MAX / float(m)
        img = cv2.resize(img, (max(2, int(round(w * s))), max(2, int(round(h * s)))),
                         interpolation=cv2.INTER_AREA)
    return img

# ============================================================================
#  时长预算
# ============================================================================
def plan_timing(n, total=TOTAL_MAX):
    """返回 (开场秒, [每段秒] * n)。每段按 3s 目标，图多时自动压到 15s 红线内。"""
    n = max(MIN_IMAGES, min(MAX_IMAGES, int(n)))
    op = OPENER if n >= 4 else OPENER_SHORT
    seg = min(float(TARGET_SEG), (float(total) - op) / n)
    if seg < MIN_SEG:
        seg = MIN_SEG
        # 段数太多塞不进 15s：先把开场压到最短，再放弃开场（图多就纯快剪）
        if op + seg * n > total:
            op = OPENER_SHORT
        if op + seg * n > total:
            op = 1.0
        if op + seg * n > total:
            op = 0.0
    return round(op, 2), [math.floor(seg * 100) / 100.0] * n

# ============================================================================
#  逐段运镜
# ============================================================================
# 运镜池：8 种互不相同的手法（用户 2026-10-14：每张图的运镜不要重复）。
# 顺序刻意交替「横移 / 推拉 / 反向 / 对角 / 垂直」——相邻两段观感差异明显。
# ★ 2026-10-15 用户反馈「几段都是同一个运镜、都是从左到右」→ 光靠池的轮转不够：
#   判别（detail/full/rotate）会把多张图映射到同一个「固定镜头」，看起来就重复。
#   现在改成「家族 + 方向」签名 + 全片防重集合（见 _pick_cand），每个家族都备足变体。
_VPOOL = ("scan_lr", "push", "scan_rl", "pull", "diag_ur", "vup", "diag_dl", "vdown")
V_SPEED = 0.34        # 横移类速度上限（用户 2026-10-15：运镜明显+更快；原 0.24，9:16 横移原被压成 0.7 幅度）
V_SPEED_ZOOM = 0.32   # 推拉 / 对角 / 垂直（原 0.22）
V_SWEEP = 0.40        # 横移幅度（视窗宽比例；用户 2026-10-15 二次加大：0.34→0.40）
V_TILT = 0.16         # 垂直「摇」幅度（视窗高比例；纵轴余量通常比横轴小，故低于 V_SWEEP）
V_LIFT = 0.12         # 垂直位移（视窗高比例；0.10→0.12）

# 位移「可用」阈值 & 两轴都没余量时的缩放兜底（2026-10-15 修「1:1 有整段静止」）：
# 位移小于视窗的 5% 时观众只会读成静止帧，这种方向的平移一律视为不可用，
# 改走另一个轴；两个轴都不够（方形源图 → 方形出片）时靠缩放 + 轻 roll 兜底。
MIN_MOVE = 0.05
ZOOM_SOFT = 0.80      # 1:1 兜底「微推旋转」的推近幅度（0.86→0.80，推得更明显）

# 两轴都顶满时（方形源图 + 1:1 出片，客户实拍请柬就是 1:1）没有免费余量可用，
# 只能把窗口等比收紧一点、把画面边缘裁掉几个百分点，换出平移空间：
# NEED_HALF 是「希望换到的半幅位移」（占视窗比例，全程 = 2 倍），
# SHRINK_MIN 是收紧下限（窗口最小收到基准的 82%，即每边最多裁 ~9%；
# 收得越小余量越大、平移越明显，同时窗口越小离原图边越远、条纹风险越低）。
NEED_HALF = 0.10      # 1:1 方图换平移余量：提到 0.10（配合 SHRINK_MIN=0.82 可换到 ex≈0.11ww，横移幅度翻倍）
SHRINK_MIN = 0.82

def _pick_move(i):
    """第 i 段的运镜名：0 = 旋转环绕开场；其后按池轮转（不重复）。"""
    if i <= 0:
        return "orbit"
    return _VPOOL[(i - 1) % len(_VPOOL)]

def _room_for(cx, cy, ww, wh, W, H, need_x, need_y):
    """给「顶满原图」的轴换出一点平移余量：等比收紧窗口（画面裁掉边缘几个百分点）。

    整图取景的窗口在某轴顶满原图时，该轴平移余量恰为 0（位移会被 clamp_win
    夹回原地 → 整段静止）。免费余量只有另一个轴才有；当两个轴都顶满
    （源图长宽比 == 出片比例，例如客户的 1:1 方图出 1:1），就只能让窗口略小于
    原图来换位移 —— 代价是裁掉边缘 SHRINK_MIN 以内的几个百分点。
    窗口宽高比锁死为出片比例，所以只能等比收；收到够用即停，最多到 SHRINK_MIN。
    """
    def rooms(w, h):
        return (max(0.0, min(cx - w / 2.0, W - cx - w / 2.0)),
                max(0.0, min(cy - h / 2.0, H - cy - h / 2.0)))

    rx, ry = rooms(ww, wh)
    if rx >= need_x * ww and ry >= need_y * wh:
        return ww, wh
    s = 1.0
    while s > SHRINK_MIN + 1e-9:
        s = max(SHRINK_MIN, round(s - 0.01, 4))
        w, h = ww * s, wh * s
        rx, ry = rooms(w, h)
        if rx >= need_x * w and ry >= need_y * h:
            return w, h
    return ww * SHRINK_MIN, wh * SHRINK_MIN

# ---------------- 单镜「原子」构造 ----------------
# 每个原子返回一个 mk() 运动；几何上做不出来（该轴没有平移余量）就返回 None，
# 由候选选择器跳去下一个变体 —— 保证「被选中的那一镜一定看得见运动」。
def _rooms(cx, cy, w, h, W, H):
    """窗口中心在两个方向上的可用半幅位移（顶满原图的那一轴恰为 0）。"""
    return (max(0.0, min(cx - w / 2.0, W - cx - w / 2.0)),
            max(0.0, min(cy - h / 2.0, H - cy - h / 2.0)))

def _fit(m, limit, dur):
    cp.fit_motion2(m, limit=limit, dmax=dur)
    m["dur"] = min(m["dur"], dur)
    return m

def _spin(cx, cy, ww, dur, direction, close, roll_deg, tag):
    """平面旋转环绕：**只 roll，绝不 yaw/pitch**（红线见模块顶部）。

    direction='in' 一边旋一边推近（close = 推近后的窗口宽度比例），'out' 一边旋一边拉远。
    roll_deg 取正负可让旋转方向相反 → 同片内两次旋转也不重样。
    """
    if direction == "in":
        p0, p1 = [cx, cy, ww], [cx, cy, ww * close]
    else:
        p0, p1 = [cx, cy, ww * close], [cx, cy, ww]
    sgn = 1.0 if direction == "in" else -1.0
    m = cp.mk("orbit", p0, p1, dur, "smooth", yaw=(0.0, 0.0),
              roll=(sgn * roll_deg, -sgn * roll_deg), margin=1.8, tag=tag)
    return _fit(m, V_SPEED, dur)

def _pan(cx, cy, ww, wh, W, H, dur, axis, sign, tag, frac=None):
    """沿 axis 平移（sign 定方向，+1 = 右/下）。该轴没余量时先收窗换余量，仍不够返回 None。

    顶满原图的轴平移会被 clamp_win 原样夹回 → 整段静止帧，所以这里一律先量余量；
    不足就先缩窗（方形源图 + 1:1 只能这么换），再不足就放弃这个变体。
    """
    w, h = ww, wh
    rx, ry = _rooms(cx, cy, w, h, W, H)
    ok = (rx >= w * MIN_MOVE) if axis == "x" else (ry >= h * MIN_MOVE)
    if not ok:
        w2, h2 = _room_for(cx, cy, ww, wh, W, H,
                           NEED_HALF if axis == "x" else 0.0,
                           NEED_HALF if axis == "y" else 0.0)
        rx, ry = _rooms(cx, cy, w2, h2, W, H)
        ok = (rx >= w2 * MIN_MOVE) if axis == "x" else (ry >= h2 * MIN_MOVE)
        if not ok:
            return None
        w, h = w2, h2
    if axis == "x":
        e = min(w * (V_SWEEP if frac is None else frac), rx)
        if e < w * MIN_MOVE:
            return None
        m = cp.mk("scan", [cx - sign * e, cy, w], [cx + sign * e, cy, w],
                  dur, "smooth", tag=tag)
        return _fit(m, V_SPEED, dur)
    e = min(h * V_TILT, ry)
    if e < h * MIN_MOVE:
        return None
    m = cp.mk("vpan", [cx, cy - sign * e, w], [cx, cy + sign * e, w],
              dur, "smooth", tag=tag)
    return _fit(m, V_SPEED, dur)

def _zoom(cx, cy, ww, dur, direction, close, tag):
    """纯推近 / 拉远（只缩放、无位移）—— 任何比例都安全，是兜底的主力。"""
    if direction == "in":
        p0, p1 = [cx, cy, ww], [cx, cy, ww * close]
    else:
        p0, p1 = [cx, cy, ww * close], [cx, cy, ww]
    m = cp.mk("push" if direction == "in" else "pull", p0, p1, dur, "smooth", tag=tag)
    return _fit(m, V_SPEED_ZOOM, dur)

def _diag(cx, cy, ww, wh, W, H, dur, sx, sy, tag):
    """对角位移 + 推近（位移吃满可用余量；两轴都没余量则 None）。"""
    rx, ry = _rooms(cx, cy, ww, wh, W, H)
    ex = min(ww * V_SWEEP * 0.45, rx)
    ey = min(wh * V_LIFT, ry)
    if ex < ww * MIN_MOVE and ey < wh * MIN_MOVE:
        return None
    m = cp.mk("diag", [cx - sx * ex, cy + sy * ey, ww],
              [cx + sx * ex, cy - sy * ey, ww * 0.84], dur, "smooth", tag=tag)
    return _fit(m, V_SPEED_ZOOM, dur)

def _refocus_box(img, fbox):
    """把 detail 的 focus 框（xyxy 归一化）中心精确到框内细节最密点（2026-10-26）。

    用户反馈「切细节没切到重点」：识别 focus 框常比真正的重点大一圈，微距
    推近以框心为目标就会偏。借 card_pan.refine_focus_pt 找框内细节最密点，
    保持框尺寸不变、把框心平移过去（平移后仍完整落在原框内，绝不放大/缩小，
    失败原样返回 —— 宁可不精确也不切飞）。
    """
    try:
        x0, y0, x1, y1 = [float(v) for v in fbox]
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:
            return fbox
        pt = cp.refine_focus_pt(img, [x0, y0, bw, bh])
        if not pt:
            return fbox
        fx, fy = pt
        nx0 = min(max(fx - bw / 2.0, x0), x1 - bw)
        ny0 = min(max(fy - bh / 2.0, y0), y1 - bh)
        return [nx0, ny0, nx0 + bw, ny0 + bh]
    except BaseException:
        return fbox


def _detail_win(cx, cy, ww, wh, W, H, R, focus, pad):
    """detail 的目标小窗 (fx, fy, wwf)：focus 归一化框 / 无 focus 时按整图的 55% 推近。

    ★ 防御性几何安全网（用户 2026-10-16）：即使焦点绕过识别直达渲染，也先用
    recog_label.safe_detail_focus 拒掉「配饰型」焦点（极小框 / 框心贴边），
    命中则退回以卡片为中心的居中推近。
    """
    fx, fy = cx, cy
    wwf = None
    if focus and len(focus) == 4:
        try:
            from core import recog_label
            sf = recog_label.safe_detail_focus(focus)
            if sf:
                x0, y0, x1, y1 = sf
                fx = (x0 + x1) / 2.0 * W
                fy = (y0 + y1) / 2.0 * H
                wwf = max((x1 - x0) * W / pad, (y1 - y0) * H * R / pad)
        except Exception:
            pass
    if wwf is None:
        # 没有焦点框：pad 直接当「目标窗口占整图的比例」用（不同 pad → 不同推近深度，
        # 这样无 focus 时各变体仍然是肉眼可分的，不会签名不同、画面一样）。
        wwf = ww * pad
    wwf = min(wwf, ww * 0.92)           # 必然比整图小 → 推近成立
    wwf = max(wwf, ww * 0.25)           # 微距下限（原 0.45，为「微距拉近」放开）
    wwf = min(wwf, W, H * R)
    fx, fy, wwf, _ = cp.clamp_win(fx, fy, wwf, wwf / R, W, H)
    return fx, fy, wwf

def _macro(cx, cy, ww, wh, W, H, R, dur, focus, direction, pad, tag, drift=None):
    """微距推近 / 拉远：整图 ↔ focus 小窗之间的缩放（可带轻微侧移）。

    drift = (fx, fy) 侧移比例（相对可移动量）：0.25 = 边推近边横向带 1/4 行程。
    带侧移的变体与纯推近**观感不同**，所以各有签名，同片里可以并存。
    """
    fx, fy, wwf = _detail_win(cx, cy, ww, wh, W, H, R, focus, pad)
    if wwf >= ww * 0.92:                # 推近深度不足 8% → 观感上不像「微距」，弃用
        return None
    dx = dy = 0.0
    if drift:
        dx = drift[0] * (ww - wwf)
        dy = drift[1] * (wh - wwf / R)
    if direction == "in":
        w0, c0, w1, c1 = ww, (cx, cy), wwf, (fx + dx, fy + dy)
    else:
        w0, c0, w1, c1 = wwf, (fx + dx, fy + dy), ww, (cx, cy)
    a = cp.clamp_win(c0[0], c0[1], w0, w0 / R, W, H)
    b = cp.clamp_win(c1[0], c1[1], w1, w1 / R, W, H)
    m = cp.mk("push" if direction == "in" else "pull",
              [a[0], a[1], a[2]], [b[0], b[1], b[2]], dur, "smooth", tag=tag)
    return _fit(m, V_SPEED_ZOOM, dur)

# ---------------- 候选：按「家族」备足变体（同片内尽量互不重复） ----------------
# ★ 签名必须只描述**画面观感**（方向 / 推拉 / 旋转方向 / 推近深度），
#   绝不能带判别来源（full / auto / detail …）—— 否则 `full|pan-lr` 与
#   `auto|pan-lr` 会被当成两种，实际渲染出来是一模一样的「左 → 右」，用户看到的就是重复。
#   同一套签名全片共享，跨家族也不会撞。
def _cands_rotate(cx, cy, ww, wh, W, H, R, dur):
    # ★ 用户 2026-10-16：**相邻两段旋转角度必须不同** → 整个池 8 个变体给
    #   8 个互不相同的 roll 角度（正负/大小都错开），按列表顺序被 _pick_cand 挑中时，
    #   相邻两段必然拿到不同角度。
    # ★ 纯 roll（平面旋转），不带任何「上下颠」（bob）。
    return [
        ("spin|in",         lambda: _spin(cx, cy, ww, dur, "in", 0.80, 11.0, "旋转推近")),
        ("spin|out",        lambda: _spin(cx, cy, ww, dur, "out", 0.86, -8.0, "反向旋转拉远")),
        ("spin|in-deep",    lambda: _spin(cx, cy, ww, dur, "in", 0.72, 14.0, "大幅旋转推近")),
        ("spin|out-deep",   lambda: _spin(cx, cy, ww, dur, "out", 0.84, -12.0, "大幅反向旋转拉远")),
        ("spin|flat",       lambda: _spin(cx, cy, ww, dur, "in", 0.90, 6.0, "轻旋")),
        ("spin|in-rev2",    lambda: _spin(cx, cy, ww, dur, "in", 0.80, 16.0, "反向大幅旋转推近")),
        ("spin|in-rev",     lambda: _spin(cx, cy, ww, dur, "in", 0.86, -9.0, "反向旋转")),
        ("spin|flat-rev",   lambda: _spin(cx, cy, ww, dur, "in", 0.92, 5.0, "轻旋反向")),
    ]

def _cands_detail(cx, cy, ww, wh, W, H, R, dur, focus):
    return [
        ("macro|in",        lambda: _macro(cx, cy, ww, wh, W, H, R, dur, focus, "in", 0.55, "微距推近")),
        ("macro|out",       lambda: _macro(cx, cy, ww, wh, W, H, R, dur, focus, "out", 0.55, "微距拉远")),
        ("macro|in-close",  lambda: _macro(cx, cy, ww, wh, W, H, R, dur, focus, "in", 0.36, "微距特写推近")),
        ("macro|out-close", lambda: _macro(cx, cy, ww, wh, W, H, R, dur, focus, "out", 0.40, "微距特写拉远")),
        ("macro|in-loose",  lambda: _macro(cx, cy, ww, wh, W, H, R, dur, focus, "in", 0.78, "微距中景推近")),
        ("macro|out-loose", lambda: _macro(cx, cy, ww, wh, W, H, R, dur, focus, "out", 0.80, "微距中景拉远")),
        ("macro|in-drift",  lambda: _macro(cx, cy, ww, wh, W, H, R, dur, focus, "in", 0.55, "微距推近侧移", drift=(0.45, 0.0))),
        ("macro|out-drift", lambda: _macro(cx, cy, ww, wh, W, H, R, dur, focus, "out", 0.55, "微距拉远侧移", drift=(-0.45, 0.0))),
        ("macro|in-lift",   lambda: _macro(cx, cy, ww, wh, W, H, R, dur, focus, "in", 0.50, "微距推近上移", drift=(0.0, 0.45))),
        ("macro|out-drop",  lambda: _macro(cx, cy, ww, wh, W, H, R, dur, focus, "out", 0.50, "微距拉远下移", drift=(0.0, -0.45))),
    ]

def _cands_full(cx, cy, ww, wh, W, H, R, dur):
    return [
        ("pan-x+",  lambda: _pan(cx, cy, ww, wh, W, H, dur, "x", +1, "整图横移 →")),
        ("pan-x-",  lambda: _pan(cx, cy, ww, wh, W, H, dur, "x", -1, "整图横移 ←")),
        ("pan-y-",  lambda: _pan(cx, cy, ww, wh, W, H, dur, "y", -1, "整图上摇 ↑")),
        ("pan-y+",  lambda: _pan(cx, cy, ww, wh, W, H, dur, "y", +1, "整图下摇 ↓")),
        ("diag|ur", lambda: _diag(cx, cy, ww, wh, W, H, dur, +1, +1, "整图对角 ↗")),
        ("diag|dl", lambda: _diag(cx, cy, ww, wh, W, H, dur, -1, -1, "整图对角 ↙")),
        ("diag|ul", lambda: _diag(cx, cy, ww, wh, W, H, dur, -1, +1, "整图对角 ↖")),
        ("diag|dr", lambda: _diag(cx, cy, ww, wh, W, H, dur, +1, -1, "整图对角 ↘")),
        ("zoom|in", lambda: _zoom(cx, cy, ww, dur, "in", 0.78, "整图缓推")),
        ("zoom|out", lambda: _zoom(cx, cy, ww, dur, "out", 0.78, "整图缓拉")),
    ]

def _cands_auto(i, cx, cy, ww, wh, W, H, R, dur):
    """未判别（auto）时的候选顺序：刻意交替家族，且结尾留旋转 / 微距，
    这样即使前面几段把横移/推拉用掉了，后面也不会重样。
    签名与 full / detail 家族共用同一套（见上），跨判别也不会撞。"""
    return [
        ("pan-x+",    lambda: _pan(cx, cy, ww, wh, W, H, dur, "x", +1, "横移 →")),
        ("zoom|in",   lambda: _zoom(cx, cy, ww, dur, "in", 0.70, "推近")),
        ("pan-x-",    lambda: _pan(cx, cy, ww, wh, W, H, dur, "x", -1, "横移 ←")),
        ("zoom|out",  lambda: _zoom(cx, cy, ww, dur, "out", 0.74, "拉远")),
        ("diag|ur",   lambda: _diag(cx, cy, ww, wh, W, H, dur, +1, +1, "对角 ↗")),
        ("pan-y-",    lambda: _pan(cx, cy, ww, wh, W, H, dur, "y", -1, "上摇 ↑")),
        ("diag|dl",   lambda: _diag(cx, cy, ww, wh, W, H, dur, -1, -1, "对角 ↙")),
        ("pan-y+",    lambda: _pan(cx, cy, ww, wh, W, H, dur, "y", +1, "下摇 ↓")),
        ("spin|in",   lambda: _spin(cx, cy, ww, dur, "in", 0.84, 10.0, "旋转推近")),
        ("macro|in",  lambda: _macro(cx, cy, ww, wh, W, H, R, dur, None, "in", 0.45, "微距推近")),
        ("spin|out",  lambda: _spin(cx, cy, ww, dur, "out", 0.84, -10.0, "旋转拉远")),
        ("diag|dr",   lambda: _diag(cx, cy, ww, wh, W, H, dur, +1, -1, "对角 ↘")),
    ]

# 兼容旧的池名（若有调用方直接传 "scan_lr" 之类）；签名映射到同一套「观感」签名
_POOL_AXIS = {"scan_lr": ("x", +1, "pan-x+"), "scan_rl": ("x", -1, "pan-x-"),
              "vup": ("y", -1, "pan-y-"), "vdown": ("y", +1, "pan-y+")}

def _pick_cand(cands, used):
    """从候选里挑第一个「签名未被本片用过且几何上做得出」的变体。

    全都用过 / 都做不出时退回第一个可做的（允许重复，但至少有运动）。
    """
    first_ok = None
    for sig, build in cands:
        m = build()
        if m is None:
            continue
        if first_ok is None:
            first_ok = (sig, m)
        if sig not in used:
            used.add(sig)
            m["_sig"] = sig
            return m
    if first_ok is not None:
        sig, m = first_ok
        m["_sig"] = sig
        return m
    return None

def _variant(i, W, H, OW, OH, dur, shot=None, focus=None, used=None):
    """第 i 段的运动参数（确定性：同图同序 → 每次跑出来一模一样）。

    运镜怎么定（2026-10-15 重构，用户：「几段都是同一个运镜、都是从左到右」）
    -----------------------------------------------------------------------
    每段先按**判别**选家族：
        rotate → 旋转环绕（首镜默认也是它，用户 2026-10-14）
        detail → 微距推近 / 拉远（focus 框指向要放大的细节）
        full   → 整图平移 / 摇 / 对角 / 缓推拉
        auto   → 家族交替的候选池（含旋转与微距）
    家族内**备足互不相同的变体**（方向、推/拉、旋转方向、推近深度都不同），
    再用全片共享的 `used` 签名集合挑第一个没用过的 → **同一条片里不会出现两段
    一样的运镜**；变体用完才允许重复（图 ≤8，变体 5~10 个，实际够用）。

    ★ 两条几何红线（改动前务必读）
    -----------------------------
    1) **旋转只做 roll，绝不 yaw/pitch**：本模版整图取景、窗口 ww 顶满原图宽度，
       roll 是绕光轴旋转，旋转后只是矩形外扩，`clamp_win` 的轴对齐包围盒公式恰好
       精确覆盖它 → 任何角度、任何比例都不飞出原图；yaw/pitch 是真正三维转动，
       透视会让取样飞出源图被 BORDER_REPLICATE 补成条纹（实测扫到 k=0.4 仍炸）。
    2) **平移必须按原图余量夹取**：整图取景下窗口必有一轴（方形图则两轴）顶满原图，
       该轴平移余量为 0，会被 clamp_win 原样夹回 → **整段静止帧**（1:1 方图 + 1:1
       出片时两轴同时归零）。位移不足 5% 视窗（MIN_MOVE）的方向一律判为不可用，
       先缩窗换余量（`_room_for`，方形图专属），仍不够就换变体 —— 见 `_pan`。
    """
    R = OW / float(OH)
    cx, cy, ww = cp._win_for_box([0.0, 0.0, 1.0, 1.0], 0.92, R, W, H)
    wh = ww / R
    if used is None:
        used = set()

    kind = str(shot or "").strip().lower()
    if kind == "rotate":
        cands = _cands_rotate(cx, cy, ww, wh, W, H, R, dur)
    elif kind == "detail":
        cands = _cands_detail(cx, cy, ww, wh, W, H, R, dur, focus)
    elif kind == "full":
        cands = _cands_full(cx, cy, ww, wh, W, H, R, dur)
    elif kind in _POOL_AXIS:
        axis, sign, sig = _POOL_AXIS[kind]
        cands = [(sig, (lambda a=axis, s=sign, t=kind:
                        _pan(cx, cy, ww, wh, W, H, dur, a, s, t)))]
    elif kind == "push":
        cands = [("zoom|in", lambda: _zoom(cx, cy, ww, dur, "in", 0.70, "推近"))]
    elif kind == "pull":
        cands = [("zoom|out", lambda: _zoom(cx, cy, ww, dur, "out", 0.74, "拉远"))]
    elif kind in ("diag_ur", "diag_dl"):
        sx, sy = (1, 1) if kind == "diag_ur" else (-1, -1)
        sig = "diag|ur" if kind == "diag_ur" else "diag|dl"
        cands = [(sig, (lambda a=sx, b=sy, t=kind:
                        _diag(cx, cy, ww, wh, W, H, dur, a, b, t)))]
    elif i is not None and i <= 0:
        # 首镜（未判别）默认旋转开场
        cands = _cands_rotate(cx, cy, ww, wh, W, H, R, dur)
    else:
        cands = _cands_auto(i, cx, cy, ww, wh, W, H, R, dur)

    m = _pick_cand(cands, used)
    if m is not None:
        return m
    # 兜底：所有候选几何上都做不出（理论上不会）→ 纯缓推，缩放一定有运动
    m = cp.mk("push", [cx, cy, ww], [cx, cy, ww * ZOOM_SOFT], dur, "smooth", tag="缓推")
    m["_sig"] = "fallback|zoom"
    m["dur"] = min(m["dur"], dur)
    return m

# ============================================================================
#  主渲染
# ============================================================================
def render_multi(images, out_path, labels=None, title=None, ratio="9x16",
                 fps=FPS, total=TOTAL_MAX, progress=None, shots=None, focuses=None,
                 font=None):
    """N 张合集图 → 一条成片（只出画面，不含配乐；配乐由调用方混）。

    images : 图片路径列表（2~8 张；超出按前 8 张处理）
    labels : 每张图对应的款式名牌（可与 images 等长；空串/None = 该段不出名牌）
    title  : 开场大标题（None = 默认 "Our Collection"；"" = 不叠标题，
             供 step4「Add Captions」关闭时使用）
    ratio  : 画幅 id（card_pan.RATIOS 真源，默认 9x16）
    font   : 客户自选字体键（core.fonts 注册表；None = 模版默认）
    返回 dict：{video, secs, segs, labels, opener}
    """
    from core import fonts as _fonts
    fp = _fonts.ttf(font) or FONT_LABEL
    imgs_paths = [p for p in (images or []) if p]
    if len(imgs_paths) < MIN_IMAGES:
        raise ValueError("Need at least %d collection photos" % MIN_IMAGES)
    imgs_paths = imgs_paths[:MAX_IMAGES]
    labels = list(labels or [])
    # 每张图的「运镜判别」（detail / full / rotate）；缺省 None → 回落池轮转（向后兼容）
    shots = list(shots or [])
    focuses = list(focuses or [])

    def _shot_at(k):
        s_ = shots[k] if k < len(shots) and shots[k] else None
        f_ = focuses[k] if k < len(focuses) and focuses[k] else None
        return s_, f_

    # 全片共享的「已用运镜签名」——同一条片里尽量不出现两段一样的运镜
    used = set()
    seg_sigs = []

    OW, OH = cp.ratio_size(ratio)
    n = len(imgs_paths)
    op_dur, seg_durs = plan_timing(n, total)

    work = [_load_work(p) for p in imgs_paths]
    if progress:
        progress(8, 100, "Reading photos")

    # ---- 文字层（名牌 / 标题）预渲染一次，逐帧只做混合（参考片样式）----
    # title=None → 默认 "Our Collection"；title="" → captions 关闭，不叠标题
    if title is None:
        title = "Our Collection"
    title_layers = _title_layers(title, OW, fp) if title else []
    lab_layers = []
    for i in range(n):
        txt = (labels[i] if i < len(labels) else "") or ""
        lab_layers.append(_label_layers(txt, OW, fp))

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ---- 一次编码：开场帧 + 各段帧 全部写进同一个管道（段间硬切） ----
    cmd = [cp.FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", "%dx%d" % (OW, OH), "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(CRF),
           "-preset", PRESET, "-movflags", "+faststart", out_path]
    err_f = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=err_f, stderr=err_f)
    wrote = 0
    t0 = time.time()
    try:
        # ① 第一张图直接开场（参考片做法）：**旋转环绕**（用户 2026-10-14），
        #    大标题叠在第一个款式上淡入淡出，不再另拼 2x2 合集板。
        #    镜头时长 = 开场预算 + 首段预算，一镜到底。
        first_dur = op_dur + seg_durs[0]
        # ★ 开头固定为「旋转环绕」：与 Collection Showcase 模版一致（模版预览片
        #   开头即平面旋转）。原识别结果对第 0 段不再生效 —— 否则识别成
        #   detail/full 时开头就不是旋转，和模版对不上（用户 2026-10-16）。
        m0 = _variant(0, work[0].shape[1], work[0].shape[0], OW, OH, first_dur,
                      shot="rotate", focus=None, used=used)
        seg_sigs.append(m0.get("_sig") or _pick_move(0))
        H, W = work[0].shape[:2]
        nf0 = max(2, int(round(m0["dur"] * fps)))
        stack0 = lab_layers[0]
        # 该款名牌压暗带（与后续段落同口径：亮白背景才出带）。两行名牌更高，
        # 带半高随名牌总高走；整体中心下移钳制，保证不顶出画面上沿。
        band_lab = None
        lab_cy0 = OH * LABEL_Y
        if stack0 is not None:
            layers0, gap0, block0 = stack0
            lab_cy0 = max(lab_cy0, block0 / 2.0 + 0.03 * OH)
            cy_f = lab_cy0 / float(OH)
            half0 = max(BAND_HALF * 1.15, (block0 * 0.5 + 0.05 * OH) / float(OH))
            dark = _band_dark(_strip_lum(work[0], cy_f, half0 * 0.9))
            if dark > 0.01:
                bk, by0, by1 = _band_scrim(OH, cy=cy_f, half=half0, dark=dark)
                band_lab = (bk, by0, by1)
        # 标题压暗带：只随标题一起淡入淡出（带的出现/消失才不突兀）
        band_title = None
        if title_layers:
            dark = _band_dark(_strip_lum(work[0], TITLE_Y, BAND_HALF))
            if dark > 0.01:
                bk, by0, by1 = _band_scrim(OH, cy=TITLE_Y, half=BAND_HALF, dark=dark)
                band_title = (bk, by0, by1)
        k_title_out = int(round(min(max(op_dur, 1.5), first_dur * 0.7) * fps))
        fin_t = max(1, int(0.50 * fps))          # 标题淡入 0.5s
        fout_t = max(1, int(0.60 * fps))         # 标题淡出 0.6s
        fin_l = max(1, int(0.30 * fps))          # 名牌淡入 0.3s
        fout_l = max(1, int(0.25 * fps))         # 名牌淡出 0.25s
        for k, frame in enumerate(cp.gen_shot(work[0], OW, OH, fps, m0, W, H)):
            # ---- 标题：约 0.1s 后淡入，k_title_out 前淡出 ----
            if title_layers:
                a = 1.0
                if k < fin_t:
                    a = k / float(fin_t)
                if k > k_title_out - fout_t:
                    a = min(a, max(0.0, (k_title_out - k) / float(fout_t)))
                if a > 0.002:
                    if band_title is not None:
                        bk, by0, by1 = band_title
                        f = frame[by0:by1].astype(np.float32)
                        f = f * (1.0 - (1.0 - bk[by0:by1, None, None]) * a)
                        frame[by0:by1] = f.astype(np.uint8)
                    gap = max(6, int(0.22 * max(l.shape[0] for l in title_layers)))
                    block = sum(l.shape[0] for l in title_layers) + gap * (len(title_layers) - 1)
                    top = OH * TITLE_Y - block / 2.0
                    for l in title_layers:
                        frame = _blend_layer(frame, l, OW * 0.5, top + l.shape[0] / 2.0, a)
                        top += l.shape[0] + gap
            # ---- 名牌：标题退场后淡入，本镜末尾淡出 ----
            if stack0 is not None:
                a = min(1.0, max(0.0, (k - k_title_out) / float(fin_l)))
                if k >= nf0 - fout_l:
                    a *= max(0.0, (nf0 - k) / float(fout_l))
                if a > 0.002:
                    if band_lab is not None:
                        bk, by0, by1 = band_lab
                        f = frame[by0:by1].astype(np.float32) * bk[by0:by1, None, None]
                        frame[by0:by1] = f.astype(np.uint8)
                    top = lab_cy0 - block0 / 2.0
                    for l in layers0:
                        frame = _blend_layer(frame, l, OW * 0.5,
                                             top + l.shape[0] / 2.0, a)
                        top += l.shape[0] + gap0
            proc.stdin.write(frame.tobytes()); wrote += 1
        if progress:
            progress(35, 100, "Opening with your first design")

        # ② 逐款分段（从第二张开始，与首镜之间硬切）
        for i in range(1, n):
            dur = seg_durs[i]
            if i == n - 1:
                # ★ 结尾固定为「整图平移/推拉」家族（full）：模版收尾是平移（实测
                #   roll 不收敛、scale=1，即平移而非旋转/缩放）。full 候选优先平移，
                #   方形源无平移余量时由 _room_for 缩窗换余量、仍不够则降级缩放——
                #   保证结尾永远有动感、且和模版一致（非旋转、非微距）。
                #   原识别结果对最后一段不再生效（用户 2026-10-16）。
                _sh_i, _fo_i = "full", None
            else:
                _sh_i, _fo_i = _shot_at(i)
            # detail 镜焦点精确化（2026-10-26）：focus 框心→框内细节最密点
            if _sh_i == "detail" and _fo_i:
                _fo_i = _refocus_box(work[i], _fo_i)
            m = _variant(i, work[i].shape[1], work[i].shape[0], OW, OH, dur,
                         shot=_sh_i, focus=_fo_i, used=used)
            seg_sigs.append(m.get("_sig") or _pick_move(i))
            H, W = work[i].shape[:2]
            nf = max(2, int(round(m["dur"] * fps)))
            stack = lab_layers[i]
            band = None
            if stack is not None:
                layers_i, gap_i, block_i = stack
                # 自适应压暗带：背景亮（白纸平铺）才出带；深色图贴参考片原样、无带。
                # 两行名牌：带随总高走，中心下移钳制不顶出上沿。
                lab_cy = max(OH * LABEL_Y, block_i / 2.0 + 0.03 * OH)
                cy_f = lab_cy / float(OH)
                half = max(BAND_HALF * 1.15, (block_i * 0.5 + 0.05 * OH) / float(OH))
                dark = _band_dark(_strip_lum(work[i], cy_f, half * 0.9))
                if dark > 0.01:
                    bk, by0, by1 = _band_scrim(OH, cy=cy_f, half=half, dark=dark)
                    band = (bk, by0, by1)
            for k, frame in enumerate(cp.gen_shot(work[i], OW, OH, fps, m, W, H)):
                if band is not None:
                    bk, by0, by1 = band
                    f = frame[by0:by1].astype(np.float32) * bk[by0:by1, None, None]
                    frame[by0:by1] = f.astype(np.uint8)
                if stack is not None:
                    a = min(1.0, k / max(1.0, 0.30 * fps))
                    if k >= nf - int(0.25 * fps):
                        a *= max(0.0, (nf - k) / max(1.0, 0.25 * fps))
                    if a > 0.002:
                        top = lab_cy - block_i / 2.0
                        for l in layers_i:
                            frame = _blend_layer(frame, l, OW * 0.5,
                                                 top + l.shape[0] / 2.0, a)
                            top += l.shape[0] + gap_i
                proc.stdin.write(frame.tobytes()); wrote += 1
            if progress:
                progress(35 + int(60.0 * (i + 1) / n), 100, "Rendering look %d/%d" % (i + 1, n))
        proc.stdin.close()
        proc.wait(timeout=900)
    except BaseException:
        try:
            proc.kill()
        except BaseException:
            pass
        try:
            proc.wait(timeout=10)
        except BaseException:
            pass
        raise
    finally:
        err_f.seek(0)
        tail = err_f.read(1600).decode("utf-8", "replace")
        err_f.close()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg rc=%s\n%s" % (proc.returncode, tail))

    secs = round(wrote / float(fps), 2)
    if progress:
        progress(97, 100, "Finishing")
    return {"video": out_path, "secs": secs, "frames": wrote,
            "images": n, "opener": op_dur, "seg": seg_durs[0] if seg_durs else 0,
            "moves": seg_sigs[:n],
            "labels": labels[:n], "ratio": cp.norm_ratio(ratio)}

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) >= 2:
        render_multi(args[:-1], args[-1], progress=lambda d, t, m: print(d, m))
