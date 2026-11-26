# -*- coding: utf-8 -*-
"""多图合集模版：上传后自动识别每张合集图的款式名牌 + 运镜类型。

策略（用户 2026-10-15 选定「两者结合」做名牌；2026-10-15 扩展「先判别运镜类型」）
---------------------------------------------------------------------------
  1) 名牌：优先提取图片上**印有的文字**（款式名 / 品牌名），识别不到再让视觉模型
     按画面语义生成一个简短英文标签，保证每段都至少有一个可读名牌。
  2) 运镜类型：让同一个视觉模型顺手判一张图该用哪种镜头 ——
        detail  细节图（印字 / 徽标 / 角花等小元素，推近放大看更清楚）
        full    全图（宽场景 / 完整平铺排版，整图缓移最合适）
        rotate  旋转展示（居中、近对称的主体，绕中心转最好看）
     若是 detail，模型一并返回该细节的归一化包围盒 focus=[x0,y0,x1,y1]（0~1，
     左上原点），渲染时据此推近到该区域。

复用 `core.showcase.chat_vision`（方舟图文理解，与卡片框检测同一套 key / 模型，
零新增依赖）。每张图一次调用，N≤8，与现有卡片检测同量级，成本可控。
"""
import json
import re

from core.showcase import chat_vision

SYSTEM = (
    "You help auto-caption and shot-plan invitation / stationery product photos "
    "for a short video template. For each photo do four things and respond with "
    "ONLY a JSON object (no code fences, no extra text): "
    '{"printed": <exact printed text on THE PRODUCT CARD, or "" if none>, '
    '"caption": <short English design tag, max 4 words, e.g. "Gold Foil Wreath">, '
    '"shot": <one of "detail" | "full" | "rotate">, '
    '"focus": <if shot=="detail", the normalized bounding box [x0,y0,x1,y1] of the '
    "single most important detail that BELONGS TO THE PRODUCT CARD (printed name / "
    "monogram / emblem / corner motif) to zoom into, each value 0-1, top-left origin, "
    "x=left->right, y=top->bottom; otherwise null>}. "
    "DEFINITIONS: "
    "- The PRODUCT is the printed stationery item itself: invitation card, menu, place "
    "card, table number, tag, RSVP card, envelope. It is normally the large, central, "
    "rectangular object in the photo. "
    "- ACCESSORIES / PROPS are decorative items placed around the product: flowers, "
    "greenery, rings, wax seals, pens, ribbons, candles, plates, tableware, fabric. "
    "These are NOT the product. "
    "RULES (strict): "
    "1. The detail focus MUST be a feature of the product CARD (printed name, "
    "monogram, emblem, corner motif). NEVER select an accessory or prop as the detail "
    "target. "
    "2. If the most eye-catching element in the photo is an accessory rather than the "
    "card, do NOT choose 'detail' — choose 'full' or 'rotate' instead and set focus "
    "to null. "
    "3. If the card has no small detail worth a separate zoom, prefer 'full' or "
    "'rotate' and leave focus null. "
    "shot guide: "
    '"detail" = a close-up of a small element ON THE CARD (printed name, monogram, '
    'emblem, corner motif) that benefits from a zoom-in; '
    '"full" = a wide scene or complete flat layout best appreciated as a whole (the '
    'whole card plus its setting); '
    '"rotate" = a centered, roughly symmetric design (emblem / medallion / wreath on '
    "the card) that looks good spinning around its center. "
    "The first photo is the hero opening — prefer 'rotate' for it unless it is clearly "
    "a zoomed-in detail shot. "
    "Prefer English. If printed text is non-Latin, still put the exact original in "
    "'printed' and provide an English 'caption'."
)


def _parse(txt):
    """从模型输出里宽松地抠出 JSON 对象（容忍 code fence / 多余文本）。"""
    t = (txt or "").strip()
    t = re.sub(r"^```(?:json)?", "", t, flags=re.I).strip()
    t = re.sub(r"```$", "", t).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def pick_label(j):
    """两者结合：优先图上印的英文文字；否则用语义标签；都为空返回空串。"""
    if not isinstance(j, dict):
        return ""
    printed = str(j.get("printed") or "").strip()
    caption = str(j.get("caption") or "").strip()
    # 优先用图上印的、且为纯 ASCII（拉丁字母/数字/标点/空格）、长度合理的文字
    # —— 非 ASCII（如中文品牌名）交给 caption，避免衬线字体缺字渲染成方块。
    if printed and all(ord(c) < 128 for c in printed) and 0 < len(printed) <= 40:
        return printed
    return caption  # 可能为空 → 该段不出名牌（沿用现有逻辑），但模型通常会给英文标签


_VALID_SHOTS = ("detail", "full", "rotate")


def pick_shot(j):
    """从模型输出里取运镜类型；非法 / 缺失回落 'auto'（交给渲染端按池轮转）。"""
    if not isinstance(j, dict):
        return "auto"
    s = str(j.get("shot") or "").strip().lower()
    return s if s in _VALID_SHOTS else "auto"


def safe_detail_focus(focus):
    """几何安全网（用户 2026-10-16）：归一化焦点框 [x0,y0,x1,y1] 是否「像配饰」。

    命中任一条就返回 None（detail 退回以卡片为中心的居中推近）：
      - 不是 4 元组 / 越界 / 退化框；
      - 面积 < 3%（几乎一定是小配饰）；
      - 框心跑到图像最外 12% 边缘（配饰常在四周，卡片一般在中央）。
    合法则夹进 [0.02, 0.98] 后返回。供识别与渲染两端复用。
    """
    if not isinstance(focus, (list, tuple)) or len(focus) != 4:
        return None
    try:
        vals = [max(0.02, min(0.98, float(v))) for v in focus]
    except Exception:
        return None
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        return None
    x0, y0, x1, y1 = vals
    area = (x1 - x0) * (y1 - y0)
    cx_box = (x0 + x1) / 2.0
    cy_box = (y0 + y1) / 2.0
    if area < 0.03:
        return None
    if cx_box < 0.12 or cx_box > 0.88 or cy_box < 0.12 or cy_box > 0.88:
        return None
    return vals


def pick_focus(j):
    """从模型输出里取 detail 的焦点包围盒（归一化 [x0,y0,x1,y1]，0~1，左上原点）。

    只在 shot=='detail' 时有效；其余情况返回 None。值会经 safe_detail_focus
    几何净化，配饰型焦点一律弃用 → detail 退回「以卡片为中心的居中推近」。
    """
    if not isinstance(j, dict):
        return None
    if str(j.get("shot") or "").strip().lower() != "detail":
        return None
    return safe_detail_focus(j.get("focus"))


def analyze_one(cfg, image_path, model=None):
    """对单张图调用视觉模型，返回 {label, shot, focus} 三元组（不抛异常）。

    失败（网络 / key / 超时）一律返回空 label + 'auto' + None，
    绝不阻断用户的手动输入与成片生成。
    """
    try:
        txt, _u, _e = chat_vision(
            cfg, SYSTEM, "Return the JSON now.", image_path,
            model=model, max_tokens=220, temperature=0.2, timeout=120)
    except Exception:
        return {"label": "", "shot": "auto", "focus": None}
    j = _parse(txt)
    return {
        "label": pick_label(j),
        "shot": pick_shot(j),
        "focus": pick_focus(j),
    }


def recognize_one(cfg, image_path, model=None):
    """向后兼容：只返回名牌字符串（detail / full / rotate 判别请改用 analyze_one）。"""
    return analyze_one(cfg, image_path, model=model)["label"]
