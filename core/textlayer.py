# -*- coding: utf-8 -*-
"""画面文字层 —— 所有模板 / 所有字体共用同一套字号与折行规则。

2026-11-12 用户定稿：无文字模板勾选「Add Captions」后，文字字号必须与
Beat-Synced Showcase (fast_cut) 一致，无论选哪种字体都一样，长文案自动换行。

为什么 captions 不再用 ASS/libass 烧录：
    libass 按字体自己的 OS/2 ascent/descent 缩放，同一 Fontsize 下
    Great Vibes 的可见字只有 Cormorant 的约 1/8（1080x1920 实测：
    italic 785x45px / 9937 白像素，script 374x40px / 1294）。
    用补偿系数只能治标；改为与 Beat-Synced 完全相同的 PIL 直绘后，
    两条链路共用 layout() / render()，字号数值天然一致，也不再有字体度量歧义。

消费方：
    core/fastcut._build_text_overlay  → numpy RGBA 叠帧（Beat-Synced 基准）
    core/voiceover.apply_caption      → PNG 文字层 + ffmpeg overlay（后处理）

2026-11-13 用户定稿（5 个模版：Wedding Signage & Decor / Wedding Color
Inspiration / Cinematic Showcase / Product Spotlight / Dynamic Collection
Showcase）：wide=True 时文字区放宽到红框实测的 0.78 画宽，字号在旧基准上
加大 2 个字号（+4px），按整词折行、超长单词也不拆字符（独占一行）。
其它模版仍走默认 0.62 画宽，保持与 Beat-Synced 一致。
"""
import os

from PIL import Image, ImageDraw, ImageFont

W_RATIO = 0.62          # 每行最大宽度（占屏宽），同 fastcut.TEXT_W_RATIO
CY = 0.44               # 文字块垂直中心（占屏高），同 fastcut.TEXT_CY
SIZE_START = 100        # 起始字号，同 fastcut 旧算法
SIZE_MIN = 14
SIZE_STEP = 2           # 1 个字号 = 2px（历史算法每次 -2）
SINGLE_MIN = 40         # 单行拟合的最低字号，再小就改用折行
MAX_LINES = 4
WIDE_RATIO = 0.78       # wide 预设行宽 = 用户红框实测值（395/505 画宽）
WIDE_SIZE_BOOST = 2 * SIZE_STEP   # 「加大 2 个字号」= +4px
HARD_RATIO = 0.94       # 整词独占一行时的绝对上限（留边距、不出画）
LINE_GAP = 1.18

_FONT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "fonts")
REF_FONT = os.path.join(_FONT_DIR, "CormorantGaramond-SemiBoldItalic.ttf")


def _draw():
    return ImageDraw.Draw(Image.new("RGBA", (8, 8)))


def _width(d, text, font):
    b = d.textbbox((0, 0), text, font=font)
    return b[2] - b[0]


def _wrap(d, text, font, max_w, keep_words=False):
    """按词贪心折行；**不缩小整句字号**。

    用户口径 2026-11-12：「确保每个单词字号大小能正常显示」。
    用户口径 2026-11-13：keep_words=True 时超长单词也保持完整、独占一行，
    绝不按字符拆开（后续由 layout 统一缩字号保证该行不出画）。
    """
    # 中文（无空格）没有「单词」概念：按字符折行，避免整句被当成
    # 一个超长单词而缩到看不清；英文仍严格按整词处理。
    if keep_words and " " not in text.strip() and any(
            "\u4e00" <= ch <= "\u9fff" for ch in text):
        out, cur = [], ""
        for ch in text.strip():
            if cur and _width(d, cur + ch, font) > max_w:
                out.append(cur)
                cur = ch
            else:
                cur += ch
        if cur:
            out.append(cur)
        return out or [text]

    lines, cur = [], ""
    for word in text.split():
        if _width(d, word, font) > max_w:
            if cur:
                lines.append(cur)
                cur = ""
            if keep_words:
                lines.append(word)
                continue
            chunk = ""
            for ch in word:
                if chunk and _width(d, chunk + ch, font) > max_w:
                    lines.append(chunk)
                    chunk = ch
                else:
                    chunk += ch
            cur = chunk
            continue
        trial = (cur + " " + word) if cur else word
        if cur and _width(d, trial, font) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines or [text]


def _legacy_size(d, text, W, H):
    """旧规则（0.62 画宽）拟合出的字号 —— wide 预设的字号基准。"""
    max_w = max(1, int(W * W_RATIO))
    max_h = int(H * 0.30)
    size = SIZE_START
    while size >= SINGLE_MIN:
        if _width(d, text, ImageFont.truetype(REF_FONT, size)) <= max_w:
            return size
        size -= SIZE_STEP
    size = SIZE_START
    while size > SIZE_MIN:
        f = ImageFont.truetype(REF_FONT, size)
        cand = _wrap(d, text, f, max_w)
        lh = int(round(size * LINE_GAP))
        if len(cand) <= MAX_LINES and lh * len(cand) <= max_h:
            return size
        size -= SIZE_STEP
    return max(SIZE_MIN, size)


def layout(text, W, H, font_path=None, wide=False):
    """→ (lines, font, size, line_h)；空文案返回 None。

    字号**只用参考字体（Beat-Synced 默认斜体）计算**，所有客户字体沿用
    同一个 size 数值 —— 这是「无论什么字体字号都一样」的保证；目标字体
    仅按自己的宽度决定折行，不单独缩放字号。

    wide=True（5 个指定模版）：行宽 0.78W、字号 = 旧基准 + 2 个字号，
    按整词折行；若行数/高度/整词宽度超限则**整句统一**缩小字号，
    任何情况下都不在单词中间断开。
    """
    text = (text or "").strip()
    if not text:
        return None
    d = _draw()
    max_h = int(H * 0.30)
    fpath = font_path or REF_FONT

    if wide:
        size = _legacy_size(d, text, W, H) + WIDE_SIZE_BOOST
        max_w = max(1, int(W * WIDE_RATIO))
        hard_w = int(W * HARD_RATIO)
        while True:
            target = ImageFont.truetype(fpath, size)
            lines = _wrap(d, text, target, max_w, keep_words=True)
            lh = int(round(size * LINE_GAP))
            ok = (len(lines) <= MAX_LINES and lh * len(lines) <= max_h and
                  all(_width(d, ln, target) <= hard_w for ln in lines))
            if ok or size <= SIZE_MIN:
                break
            size -= SIZE_STEP
        return lines, target, size, lh

    max_w = max(1, int(W * W_RATIO))

    # ① 单行优先：完全复刻旧版 Beat-Synced 的「100 起按 62% 屏宽收缩」
    size = SIZE_START
    while size >= SINGLE_MIN:
        if _width(d, text, ImageFont.truetype(REF_FONT, size)) <= max_w:
            break
        size -= SIZE_STEP
    if size < SINGLE_MIN:
        # ② 长文案：允许按词折行，重新从大字号拟合（每个单词字号保持正常）
        size = SIZE_START
        while size > SIZE_MIN:
            f = ImageFont.truetype(REF_FONT, size)
            cand = _wrap(d, text, f, max_w)
            lh = int(round(size * LINE_GAP))
            if len(cand) <= MAX_LINES and lh * len(cand) <= max_h:
                break
            size -= SIZE_STEP
        size = max(SIZE_MIN, size)

    # ③ 目标字体沿用同一字号；按自身宽度折行。
    #    **绝不因字体宽而单独缩小字号** —— 否则「无论什么字体都一样」就破了。
    target = ImageFont.truetype(fpath, size)
    lines = _wrap(d, text, target, max_w)
    # ④ 极宽字体（如 serif）：不降字号，改为适度放宽行宽，仍保证不出画
    if len(lines) > MAX_LINES:
        lines = _wrap(d, text, target, int(W * WIDE_RATIO))
    lh = int(round(size * LINE_GAP))
    return lines, target, size, lh


def render(text, W, H, font_path=None, wide=False):
    """→ 透明底 RGBA 图（纯白文字，块中心 44% 屏高）；空文案返回 None。"""
    got = layout(text, W, H, font_path, wide=wide)
    if not got:
        return None
    lines, font, size, lh = got
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if len(lines) == 1:
        # 与旧版 Beat-Synced 单行定位逐像素一致
        b = d.textbbox((0, 0), lines[0], font=font)
        th, tw = b[3] - b[1], b[2] - b[0]
        x = (W - tw) // 2 - b[0]
        y = int(H * CY) - th // 2 - b[1]
        d.text((x, y), lines[0], font=font, fill=(255, 255, 255, 255))
    else:
        top = int(round(H * CY - lh * len(lines) / 2.0))
        for i, ln in enumerate(lines):
            b = d.textbbox((0, 0), ln, font=font)
            tw = b[2] - b[0]
            x = (W - tw) // 2 - b[0]
            y = top + i * lh - b[1]
            d.text((x, y), ln, font=font, fill=(255, 255, 255, 255))
    return img
