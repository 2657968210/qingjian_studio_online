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
"""
import os

from PIL import Image, ImageDraw, ImageFont

W_RATIO = 0.62          # 每行最大宽度（占屏宽），同 fastcut.TEXT_W_RATIO
CY = 0.44               # 文字块垂直中心（占屏高），同 fastcut.TEXT_CY
SIZE_START = 100        # 起始字号，同 fastcut 旧算法
SIZE_MIN = 14
SINGLE_MIN = 40         # 单行拟合的最低字号，再小就改用折行
MAX_LINES = 4
WIDE_RATIO = 0.78       # 极宽字体的兜底行宽（仍不出画、字号不变）
LINE_GAP = 1.18

_FONT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "fonts")
REF_FONT = os.path.join(_FONT_DIR, "CormorantGaramond-SemiBoldItalic.ttf")


def _draw():
    return ImageDraw.Draw(Image.new("RGBA", (8, 8)))


def _width(d, text, font):
    b = d.textbbox((0, 0), text, font=font)
    return b[2] - b[0]


def _wrap(d, text, font, max_w):
    """按词贪心折行；超长单词按字符拆行，**不缩小整句字号**。

    用户口径：「确保每个单词字号大小能正常显示」—— 不能为了把一个超长词
    塞进 62% 屏宽就把整句字号压小；宁可拆行，字号始终一致。
    """
    lines, cur = [], ""
    for word in text.split():
        if _width(d, word, font) > max_w:
            if cur:
                lines.append(cur)
                cur = ""
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


def layout(text, W, H, font_path=None):
    """→ (lines, font, size, line_h)；空文案返回 None。

    字号**只用参考字体（Beat-Synced 默认斜体）计算**，所有客户字体沿用
    同一个 size 数值 —— 这是「无论什么字体字号都一样」的保证；目标字体
    仅按自己的宽度决定折行，不单独缩放字号。
    """
    text = (text or "").strip()
    if not text:
        return None
    d = _draw()
    max_w = max(1, int(W * W_RATIO))
    max_h = int(H * 0.30)
    fpath = font_path or REF_FONT

    # ① 单行优先：完全复刻旧版 Beat-Synced 的「100 起按 62% 屏宽收缩」
    size = SIZE_START
    while size >= SINGLE_MIN:
        if _width(d, text, ImageFont.truetype(REF_FONT, size)) <= max_w:
            break
        size -= 2
    if size < SINGLE_MIN:
        # ② 长文案：允许按词折行，重新从大字号拟合（每个单词字号保持正常）
        size = SIZE_START
        while size > SIZE_MIN:
            f = ImageFont.truetype(REF_FONT, size)
            cand = _wrap(d, text, f, max_w)
            lh = int(round(size * LINE_GAP))
            if len(cand) <= MAX_LINES and lh * len(cand) <= max_h:
                break
            size -= 2
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


def render(text, W, H, font_path=None):
    """→ 透明底 RGBA 图（纯白文字，块中心 44% 屏高）；空文案返回 None。"""
    got = layout(text, W, H, font_path)
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
