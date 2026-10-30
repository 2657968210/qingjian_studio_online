# -*- coding: utf-8 -*-
"""随包字体注册表（2026-10-26：客户自选字体，全模版统一）。

字体随包放在 static/fonts/（SIL OFL 开源授权，可商用），本地 Windows 与
线上 Linux 只有内置字体才能表现一致——绝不能依赖系统字体。

四个可选键（前端 #capFont 下拉与服务端 body.font 都用这些 id）：
    serif   CormorantGaramond-SemiBold        衬线正体（庄重，标题感）
    italic  CormorantGaramond-SemiBoldItalic  衬线斜体（各模版字幕的现行默认）
    light   CormorantGaramond-MediumItalic    衬线细斜体（轻盈）
    script  GreatVibes-Regular                手写花体（签名感）

两个消费面：
    PIL 直绘（fastcut/cinematic/three_screen/multi_suite/four_grid）→ ttf(key)
      给精确的字体文件，零匹配歧义；
    libass 烧录（voiceover.apply_caption 的 captions 行）→ ass_style(key)
      ASS 只认「字体族名 + Bold/Italic 旗标」，靠 subtitles 滤镜的 fontsdir
      指到随包目录完成匹配（族名是字体的内部名，不是文件名）。
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(BASE_DIR, "static", "fonts")

# key -> (文件名, ASS 字体族名, ASS Bold, ASS Italic, 前端展示名)
_FONTS = {
    "serif":  ("CormorantGaramond-SemiBold.ttf",       "Cormorant Garamond", 1, 0, "Serif"),
    "italic": ("CormorantGaramond-SemiBoldItalic.ttf", "Cormorant Garamond", 1, 1, "Serif Italic"),
    "light":  ("CormorantGaramond-MediumItalic.ttf",   "Cormorant Garamond", 0, 1, "Light Italic"),
    "script": ("GreatVibes-Regular.ttf",               "Great Vibes",        0, 0, "Script"),
}
DEFAULT_KEY = "italic"


def norm(key):
    """归一字体键：未知/缺省一律回落 italic（各模版现行默认，向后兼容）。"""
    k = str(key or "").strip().lower().replace("-", "_").replace(" ", "_")
    return k if k in _FONTS else DEFAULT_KEY


def ttf(key):
    """字体键 → ttf 绝对路径（PIL 用）。文件缺失返回 None（调用方回落自身默认）。"""
    k = norm(key)
    p = os.path.join(FONT_DIR, _FONTS[k][0])
    return p if os.path.isfile(p) else None


def ass_family(key):
    """字体键 → libass 匹配用的字体族名。"""
    return _FONTS[norm(key)][1]


def ass_flags(key):
    """字体键 → (Bold, Italic) 整型旗标（写进 ASS Style 行）。"""
    k = norm(key)
    return _FONTS[k][2], _FONTS[k][3]


def label(key):
    """字体键 → 前端展示名（后端日志/回显用）。"""
    return _FONTS[norm(key)][4]
