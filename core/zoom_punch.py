# -*- coding: utf-8 -*-
"""Dynamic Collection Showcase（zoom_punch）—— N 张产品图 → 剪映风格规整转场 + 白条深色衬线标题。

编排取自参考视频 韩梦瑶分享的视频.mp4（2026-11-10 用户提供，16.7s 竖屏）；
转场按用户 2026-11-11 定稿口径改为「剪映里的常见转场」：干净、规整、边缘锐利，
不再使用模糊 / 甩镜 / 拖影类效果。

  * 每张图出场 1 次、按顺序播放、不循环；每镜 ~1.8s **画面完全静止**
    （2026-11-11 用户定稿：转场之后图片不再加运镜。居中 cover 取景一次生成，
     整镜复用同一帧，产品恒在画面中央不被裁切；全片只有转场本身在动）；
    可选 ken_burns=True 才启用旧的 1.0→1.10 慢推/慢拉（默认 False，零回归）；
  * 镜与镜之间 = **逐个不同的剪映风格转场**（0.4s），13 种手法按序列轮换、
    相邻永不重复，分三组：
      推入 / 滑入 / 擦除：push_lr 左右推入(左右切入) / push_tb 上下推入 /
                           wipe_lr 左右擦除 / wipe_tb 上下擦除
      基础类            ：dissolve 叠化 / flash 闪白 / black 黑场 /
                           zoom 缩放推近(或缩小拉远) / rotate 旋转
      特效类            ：glitch 故障抖动 / mosaic 马赛克 /
                           mirror 镜像分屏 / flip_page 翻页
    序列前 9 个即覆盖三组（10 张图成片可见 2 次左右切入）。
    variant 决定推入方向 / 分屏方向 / 故障幅度，重复出现也不重样。
    可用 transitions= 显式指定序列覆盖默认轮换。
  * 标题文字 = 参考片开头 "2027 New Collection · Laser cut invites" 的样式：
    **深色衬线字压白色标签条**（每行一条白 bar）+ 首行两侧四角星装饰；
    2026-11-11 定稿：每行白条宽度上限 70% 屏宽，**放不下自动折行**（最多 4 行），
    不再靠无限缩小字号塞进一行（长标题仍清晰可读）；显式换行符 / "·" / " - " 强制分段；
    开场 0.3s 淡入，约 5.5s（跨前两镜）淡出；字体随包 Cormorant SemiBold；
  * 原图直出，不加滤镜调色；仅出画面不含配乐（BGM 由 server 混 canon_in_d）。

与 fastcut / decor_show / four_grid 完全独立。

接口
----
render_zoom_punch(images, out_path, ratio="9x16", fps=30,
                  title=DEFAULT_TITLE, show_title=True, progress=None,
                  font=None, transitions=None, ken_burns=False)
  images     : 图片路径列表（3~99 张）
  title      : 标题文字（"·" 或换行分行；空则回退默认句）
  show_title : False 则完全不叠标题
  font       : 字体键（core.fonts；None = 衬线正体 CG，即参考片观感）
  transitions: 可选，转场序列（长度 >= n-1）；None = 自动轮换
  ken_burns  : False（默认）= 镜头内画面静止，只有转场动；
               True = 恢复旧的逐帧 1.0→1.10 慢推/慢拉
  返回 dict：{video, secs, images, ratio, transitions, still}
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
TRANS = 0.4              # 转场时长（s）
ZB_NEAR, ZB_FAR = 1.00, 1.10     # 可选 Ken Burns 收紧范围（默认静止只用 ZB_NEAR=1.0 全景）

# 剪映风格转场序列：三组交错、左右切入优先、相邻永不重复。
# 10 张图（9 个接点）即覆盖 推入/擦除 4 次、基础 3 次、特效 3 次。
# 超过 13 个接点后循环，variant 改方向 / 分屏方向 / 故障幅度避免重样。
TRANSITION_SEQUENCE = (
    "push_lr", "dissolve", "glitch", "wipe_lr", "flash", "push_tb",
    "mosaic", "zoom", "mirror", "rotate", "wipe_tb", "flip_page", "black",
)

# 标题（参考片原文）
DEFAULT_TITLE = "2027 New Collection \u00b7 Laser cut invites"
TITLE_W_RATIO = 0.70     # 每行白条宽度上限（占屏宽）；超出自动折行，不再缩成小字
TITLE_MIN_SIZE_RATIO = 0.040  # 行数仍超上限时允许的最小字号（占屏高）
TITLE_MAX_LINES = 4      # 标题最多行数（长句不盖满画面）
TITLE_MAX_SEGMENTS = 4   # 显式分隔符（换行/·/ - ）最多段数
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
CRF = 15                    # 2026-11-11 用户：画质要高清（17→15，近无损）
PRESET = "slow"             # 同上：medium→slow，压缩效率更高、动态转场更干净


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
# 剪映风格转场库（干净、规整、硬边或整幅位移；无模糊甩镜）
# 统一签名：(fout, fin, p, W, H, variant=0) -> BGR 帧 (H,W,3) uint8
#   fout / fin 是「已 cover 到 W×H 的镜头边界帧」，由 render 预生成，
#   因此转场内部不再触碰 4000px 源图 —— 可全分辨率合成、边缘锐利且极快。
# 三组：推入/滑入/擦除、基础、特效；variant 决定方向 / 幅度，重复不重样。
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


def _solid(W, H, color):
    """生成 H×W 纯色帧（W/H 可互换用于窄条）。"""
    return np.full((H, W, 3), color, np.uint8)


def _scale_center(frame, z, W, H):
    """中心放大 z 倍后裁回 W×H。"""
    if z <= 1.001:
        return frame
    nw, nh = int(round(W * z)), int(round(H * z))
    big = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0, y0 = (nw - W) // 2, (nh - H) // 2
    return big[y0:y0 + H, x0:x0 + W]


# --- 组 1：推入 / 滑入 / 擦除 -------------------------------------------------
def _push_lr(fout, fin, p, W, H, variant=0):
    """左右推入（左右切入）：整幅胶片平移，旧图被新图整幅推出画面。"""
    off = max(0, min(W, int(round(W * cp.smoothstep(p)))))
    out = np.empty((H, W, 3), np.uint8)
    if variant % 2 == 0:                    # 新图从右向左推入
        if off < W:
            out[:, :W - off] = fout[:, off:]
        if off > 0:
            out[:, W - off:] = fin[:, :off]
    else:                                   # 新图从左向右推入
        if off < W:
            out[:, off:] = fout[:, :W - off]
        if off > 0:
            out[:, :off] = fin[:, W - off:]
    return out


def _push_tb(fout, fin, p, W, H, variant=0):
    """上下推入：垂直方向整幅平移。"""
    off = max(0, min(H, int(round(H * cp.smoothstep(p)))))
    out = np.empty((H, W, 3), np.uint8)
    if variant % 2 == 0:                    # 新图从下向上推入
        if off < H:
            out[:H - off] = fout[off:]
        if off > 0:
            out[H - off:] = fin[:off]
    else:                                   # 新图从上向下推入
        if off < H:
            out[off:] = fout[:H - off]
        if off > 0:
            out[:off] = fin[H - off:]
    return out


def _wipe_lr(fout, fin, p, W, H, variant=0):
    """左右擦除：旧图静止，新图以硬边从左 / 右逐渐覆盖。"""
    off = max(0, min(W, int(round(W * cp.smoothstep(p)))))
    out = fout.copy()
    if off:
        if variant % 2 == 0:
            out[:, :off] = fin[:, :off]
        else:
            out[:, W - off:] = fin[:, W - off:]
    return out


def _wipe_tb(fout, fin, p, W, H, variant=0):
    """上下擦除：旧图静止，新图以硬边从上 / 下逐渐覆盖。"""
    off = max(0, min(H, int(round(H * cp.smoothstep(p)))))
    out = fout.copy()
    if off:
        if variant % 2 == 0:
            out[:off] = fin[:off]
        else:
            out[H - off:] = fin[H - off:]
    return out


# --- 组 2：基础类 -----------------------------------------------------------
def _dissolve(fout, fin, p, W, H, variant=0):
    """叠化溶解：平滑交叉淡化，中点轻微提亮。"""
    fr = _blend(fout, fin, cp.smoothstep(p))
    lift = 0.10 * math.sin(math.pi * p)
    if lift > 0.002:
        fr = _blend(fr, _solid(W, H, 255), lift)
    return fr


def _flash(fout, fin, p, W, H, variant=0):
    """闪白：中点全白瞬间换镜。"""
    tri = math.sin(math.pi * p)
    base = fout if p < 0.5 else fin
    return _blend(base, _solid(W, H, 255), 0.96 * tri)


def _black(fout, fin, p, W, H, variant=0):
    """黑场：淡出到黑再淡入。"""
    tri = math.sin(math.pi * p)
    base = fout if p < 0.5 else fin
    return _blend(base, _solid(W, H, 0), 0.97 * tri)


def _zoom(fout, fin, p, W, H, variant=0):
    """缩放：偶数 variant 推近冲出，奇数 variant 缩小拉远。"""
    if variant % 2 == 0:
        zf = 1.38
        if p <= 0.5:
            q = p / 0.5
            z, src = 1.0 + (zf - 1.0) * (q * q), fout
        else:
            q = (p - 0.5) / 0.5
            z = zf - (zf - 1.0) * (1.0 - (1.0 - q) ** 2)
            src = fin
        return _scale_center(src, z, W, H)
    out = fin.copy()
    s = 1.0 - cp.smoothstep(p)
    if s > 0.02:
        sw, sh = max(2, int(W * s)), max(2, int(H * s))
        small = cv2.resize(fout, (sw, sh), interpolation=cv2.INTER_LINEAR)
        x0, y0 = (W - sw) // 2, (H - sh) // 2
        out[y0:y0 + sh, x0:x0 + sw] = small
    return out


def _rotate(fout, fin, p, W, H, variant=0):
    """旋转：整幅边旋转边放大，中点放大最多时换镜。"""
    spin = 1.0 if variant % 2 == 0 else -1.0
    ang = spin * 24.0 * p
    s = 1.0 + 0.26 * math.sin(math.pi * p)
    src = fout if p < 0.5 else fin
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, s)
    return cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


# --- 组 3：特效类 -----------------------------------------------------------
def _glitch(fout, fin, p, W, H, variant=0):
    """故障抖动：RGB 通道错位 + 横向条带位移 + 少量色块，中点最强。"""
    tri = math.sin(math.pi * p)
    base = _blend(fout, fin, cp.smoothstep(p))
    g = base.copy()
    sh = int(W * 0.035 * tri) * (1 if variant % 2 == 0 else -1)
    if sh:
        g[:, :, 0] = np.roll(base[:, :, 0], sh, axis=1)
        g[:, :, 2] = np.roll(base[:, :, 2], -sh, axis=1)
    nb = 6 + 2 * (variant % 3)
    bh = max(1, H // nb)
    for b in range(nb):
        if (b + variant) % 3 == 0:
            d = int(W * 0.06 * tri * (1 if b % 2 == 0 else -1))
            y0, y1 = b * bh, min(H, (b + 1) * bh)
            if d:
                g[y0:y1] = np.roll(g[y0:y1], d, axis=1)
    if tri > 0.55 and W > 40:
        bw = int(W * 0.22 * tri)
        y0 = int(H * (0.28 + 0.12 * (variant % 3)))
        y1 = min(H, y0 + max(2, int(H * 0.02)))
        if y1 > y0 and bw > 0:
            g[y0:y1, :bw] = (40, 40, 235) if variant % 2 == 0 else (235, 90, 40)
    return g


def _mosaic(fout, fin, p, W, H, variant=0):
    """马赛克：中点最粗、两端清晰。"""
    tri = math.sin(math.pi * p)
    base = _blend(fout, fin, cp.smoothstep(p))
    block = max(2, int(2 + 24 * tri))
    sw, sh = max(2, W // block), max(2, H // block)
    small = cv2.resize(base, (sw, sh), interpolation=cv2.INTER_LINEAR)
    big = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
    return _blend(base, big, tri)


def _mirror(fout, fin, p, W, H, variant=0):
    """镜像 / 分屏切换：左右两半分别自上下反向滑入，中点合拢。"""
    e = cp.smoothstep(p)
    half = W // 2
    out = fout.copy()
    shift = int(H * 0.42 * (1.0 - e))
    s = 1 if variant % 2 == 0 else -1
    left = np.roll(fin[:, :half], s * shift, axis=0)
    right = np.roll(fin[:, half:], -s * shift, axis=0)
    out[:, :half] = _blend(fout[:, :half], left, e)
    out[:, half:] = _blend(fout[:, half:], right, e)
    return out


def _flip_page(fout, fin, p, W, H, variant=0):
    """翻页：旧画面沿竖轴压缩翻走（背面渐暗），逐渐露出新画面。"""
    e = cp.smoothstep(p)
    nw = max(1, int(round(W * (1.0 - e))))
    strip = cv2.resize(fout, (nw, H), interpolation=cv2.INTER_LINEAR)
    strip = _blend(strip, _solid(nw, H, 0), e * 0.40)
    out = fin.copy()
    if variant % 2 == 0:
        out[:, :nw] = strip
    else:
        out[:, W - nw:] = strip
    if 1 < nw < W:
        x = max(0, min(W - 2, nw if variant % 2 == 0 else W - nw))
        out[:, x:x + 2] = _blend(out[:, x:x + 2], _solid(2, H, 255), 0.5)
    return out


_TRANSITIONS = {
    "push_lr": _push_lr,
    "push_tb": _push_tb,
    "wipe_lr": _wipe_lr,
    "wipe_tb": _wipe_tb,
    "dissolve": _dissolve,
    "flash": _flash,
    "black": _black,
    "zoom": _zoom,
    "rotate": _rotate,
    "glitch": _glitch,
    "mosaic": _mosaic,
    "mirror": _mirror,
    "flip_page": _flip_page,
}
VALID_TRANSITIONS = tuple(_TRANSITIONS.keys())


def resolve_transitions(n, seq=None):
    """返回长度 n-1 的转场名列表；None = 按剪映风格序列轮换，相邻永不重复。"""
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
    """标题分段：手动换行 / "·" / " - " 强制分段；每段再由 _wrap_text 按宽度折行。
    2026-11-11：不再为了塞进一行而无限缩小字号 —— 宽度固定，放不下就换行。"""
    t = (text or "").strip()
    if not t:
        t = DEFAULT_TITLE
    if "\n" in t:
        parts = [s.strip() for s in t.split("\n") if s.strip()]
    elif "\u00b7" in t:
        parts = [s.strip() for s in t.split("\u00b7") if s.strip()]
    elif " - " in t:
        parts = [s.strip() for s in t.split(" - ") if s.strip()]
    else:
        parts = [t]
    return parts[:TITLE_MAX_SEGMENTS]


def _wrap_text(text, font, max_w, draw):
    """按最大宽度折行：英文按空格成词，中文 / 超长词按字符切，行宽绝不超过 max_w。"""
    def width(s):
        bb = draw.textbbox((0, 0), s, font=font)
        return bb[2] - bb[0]

    # 纯中文（无空格）：逐字折行，词间不加空格
    if any("\u4e00" <= ch <= "\u9fff" for ch in text) and " " not in text:
        lines, cur = [], ""
        for ch in text:
            if cur and width(cur + ch) > max_w:
                lines.append(cur)
                cur = ch
            else:
                cur += ch
        if cur:
            lines.append(cur)
        return lines or [""]

    tokens = []
    for wd in text.split():
        if width(wd) <= max_w:
            tokens.append(wd)
        else:
            cur = ""
            for ch in wd:
                if cur and width(cur + ch) > max_w:
                    tokens.append(cur)
                    cur = ch
                else:
                    cur += ch
            if cur:
                tokens.append(cur)
    lines, cur = [], ""
    for tk in tokens:
        trial = (cur + " " + tk) if cur else tk
        if width(trial) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = tk
    if cur:
        lines.append(cur)
    return lines or [""]


def _build_title_overlay(text, W, H, font_path=None):
    """返回 (rgb, base_alpha) —— BGR 空间 overlay 与基础 alpha 图（含白条+字+星）。"""
    fpath = font_path or SERIF
    # CJK 标题：随包衬线字体无中文字形（会渲染成方块）→ 回退系统自带中文字体。
    if any("\u4e00" <= ch <= "\u9fff" for ch in (text or "")):
        for cand in (r"C:\Windows\Fonts\msyh.ttc",
                     r"C:\Windows\Fonts\simhei.ttf",
                     "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                     "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
            if os.path.isfile(cand):
                fpath = cand
                break
    segments = _split_title(text)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # 宽度口径（2026-11-11）：白条宽度 ≤ W*TITLE_W_RATIO，字区再扣左右内边距；
    # 按起始字号折行，行数超上限才适度缩字号（有下限），绝不缩成看不清的小字。
    pad_x = int(W * TITLE_BAR_PADX)
    max_w = max(80, int(W * TITLE_W_RATIO) - 2 * pad_x)
    size0 = int(H * 0.052)
    size_min = max(12, int(H * TITLE_MIN_SIZE_RATIO))

    def _seg_w(seg, font):
        bb = d.textbbox((0, 0), seg, font=font)
        return bb[2] - bb[0]

    # ① 先在不低于 size_min 的范围内适度缩字号，让每个显式分段各自单行
    #    （保住参考片两行标题的观感，字仍够大）；
    # ② 到下限仍放不下 → 才按宽度自动折行，行数最多 TITLE_MAX_LINES。
    size, lines = size0, None
    s = size0
    while s >= size_min:
        f = ImageFont.truetype(fpath, s)
        if all(_seg_w(seg, f) <= max_w for seg in segments):
            size, lines = s, list(segments)
            break
        s -= 2
    if lines is None:
        size = size_min
        while True:
            f = ImageFont.truetype(fpath, size)
            lines = []
            for seg in segments:
                lines.extend(_wrap_text(seg, f, max_w, d))
            if len(lines) <= TITLE_MAX_LINES or size <= max(12, int(H * 0.030)):
                break
            size -= 2
        lines = lines[:TITLE_MAX_LINES]
    f = ImageFont.truetype(fpath, size)
    fonts = [(ln, f, d.textbbox((0, 0), ln, font=f)) for ln in lines]

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
    # 行数变多时白条块整体下移，但不越过画面顶部（折行后标题可能占 3~4 行）
    y = max(int(H * 0.015), int(H * TITLE_BLOCK_CY) - total_h // 2)
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
                      progress=None, font=None, transitions=None,
                      ken_burns=False):
    """N(3~99) 张产品图 → 逐个不同转场的成片（仅画面，无音轨）。

    默认 ken_burns=False：镜头内图片静止，只有 0.4s 转场本身有动效。
    """
    from core import fonts as _fonts
    paths = [p for p in (images or []) if p and os.path.isfile(p)]
    if len(paths) < MIN_PHOTOS:
        raise ValueError("Dynamic Collection Showcase needs at least %d photos" % MIN_PHOTOS)
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

    # 静止口径（2026-11-11 定稿，默认）：每张图只做一次居中 cover 取景，
    # 整镜复用同一帧；转场首尾也用同一帧 → 转场结束直接落在静止画面，无跳变。
    # ken_burns=True 才恢复旧的 1.0→1.10 奇偶推拉（逐帧重算取景）。
    if ken_burns:
        dirs = [(ZB_NEAR, ZB_FAR) if i % 2 == 0 else (ZB_FAR, ZB_NEAR)
                for i in range(n)]
    else:
        dirs = [(ZB_NEAR, ZB_NEAR) for _ in range(n)]

    # 预生成每镜「结束态 / 起始态」全尺寸帧：转场只做几何合成，
    # 不再触碰 4000px 源图 → 可全分辨率输出，推入/擦除边缘保持锐利。
    if ken_burns:
        end_frames = [_cover_crop(photos[i], dirs[i][1], W, H) for i in range(n)]
        start_frames = [_cover_crop(photos[i], dirs[i][0], W, H) for i in range(n)]
        still_frames = None
    else:
        still_frames = [_cover_crop(photos[i], ZB_NEAR, W, H) for i in range(n)]
        end_frames = start_frames = still_frames

    # 每个接点的转场手法：按剪映风格序列轮换，相邻永不重复
    trans_kinds = resolve_transitions(n, transitions)

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
            z0, z1 = dirs[si]
            # 静止模式：底图恒定，只在标题 alpha 变化时重算叠加并缓存，
            # 整镜不再逐帧做 4000px LANCZOS 取景 → 快且画面绝对静止。
            scene_base = None if ken_burns else still_frames[si]
            title_cache = {}
            for k in range(scene_f):
                if ken_burns:
                    frame = _kb_frame(photos[si], z0, z1,
                                      k / float(scene_f - 1), W, H)
                else:
                    frame = scene_base
                # 标题（淡入淡出窗口内叠加）
                if ttl_a is not None:
                    t = frame_no / float(fps)
                    a = _title_alpha(t, total_secs, n)
                    if a > 0.003:
                        if ken_burns:
                            frame = (frame.astype(np.float32) * (1 - ttl_a * a)
                                     + ttl_rgb * (ttl_a * a)).astype(np.uint8)
                        else:
                            ck = int(round(a * 500))
                            if ck not in title_cache:
                                aq = ck / 500.0
                                title_cache[ck] = (
                                    frame.astype(np.float32) * (1 - ttl_a * aq)
                                    + ttl_rgb * (ttl_a * aq)).astype(np.uint8)
                            frame = title_cache[ck]
                proc.stdin.write(frame.tobytes())
                frame_no += 1
            if si < n - 1:
                fn = _TRANSITIONS[trans_kinds[si]]
                variant = si
                fo, fi = end_frames[si], start_frames[si + 1]
                for k in range(trans_f):
                    p = (k + 0.5) / trans_f
                    frame = fn(fo, fi, p, W, H, variant)
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
            "transitions": trans_kinds,
            "still": not bool(ken_burns)}
