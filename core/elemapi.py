# -*- coding: utf-8 -*-
"""
AI 处理元素素材（步骤2）
=========================
把步骤1 上传的产品实拍图交给火山方舟 Doubao-Seedream：
先「识别拆解」画面中的独立纸品元素，再按所选模式输出修复后的**参考素材**。

两种模式（互斥，指令全文见 core/prompts.py）
-------------------------------------------
  fidelity  保真优化模式      —— 只做清理与保真修复（去手/去杂物/去水印/清晰化/正形/光影统一），
                                产品设计 100% 不变。默认口径。
  restage   AI 重搭配重场景   —— 元素自身设计 100% 不变，重新搭配元素组合 + 重建拍摄场景。

两种输出粒度（附加在模式正文之后）
---------------------------------
  item  逐个元素：一元素一图（贴合「先拆解、再逐个修复输出」的口径）
  set   整套场景：全部元素合成一张整图参考（适合作为「整套逐张拿起展示」视频的首/尾帧）

产物地位：写入项目 elements/，并成为下游（基础运镜源素材 / AI 视频首帧·尾帧）的**默认来源**，
上传原图降为备选（见 server.py 的 pick_first_frame / pick_last_frame）。

成本口径：与 Seedream 组图同价（4K 约 0.22 元/张；2K/3K 更低）。
约束：参考图数 + 生成图数 ≤ 15（方舟侧硬限制，imgapi.build_payload 已自动收敛）。
"""

from . import imgapi, prompts

MODES = [
    {"id": "fidelity", "name": "Fidelity restoration",
     "desc": "Cleanup and faithful restoration only: remove hands / clutter / watermarks, sharpen, correct geometry, unify lighting. Product design stays 100% unchanged.",
     "prompt_key": "ELEMENT_FIDELITY", "default_gran": "item"},
    {"id": "restage", "name": "AI restage & rescene",
     "desc": "Element designs stay 100% unchanged; recompose the arrangement and rebuild the shooting scene (new tabletop / florals / lighting / framing).",
     "prompt_key": "ELEMENT_RESTAGE", "default_gran": "set"},
]

GRANULARITIES = [
    {"id": "item", "name": "Per element (one image per element)",
     "desc": "One restored image per independently identified paper element, up to 15 (reference images included)"},
    {"id": "set", "name": "Full set (all elements in one image)",
     "desc": "All elements composed into a single scene reference — ideal as the first / last frame of a whole-set, card-by-card reveal video"},
]

DEFAULT_MODE = "fidelity"
DEFAULT_GRAN = "item"

_MODE_IDS = [m["id"] for m in MODES]
_GRAN_IDS = [g["id"] for g in GRANULARITIES]


def mode_meta(mid):
    mid = (mid or "").strip() or DEFAULT_MODE
    for m in MODES:
        if m["id"] == mid:
            return m
    return MODES[0]


def normalize(mode, granularity):
    """收敛入参 → (mode_id, gran_id)。非法值回落到默认，不抛错。"""
    mode = (mode or "").strip()
    if mode not in _MODE_IDS:
        mode = DEFAULT_MODE
    gran = (granularity or "").strip()
    if gran not in _GRAN_IDS:
        gran = mode_meta(mode)["default_gran"]
    return mode, gran


def default_prompt(mode):
    m = mode_meta(mode)
    txt = getattr(prompts, m["prompt_key"], None) or prompts.DEFAULTS.get(m["prompt_key"].lower())
    if not txt:
        raise RuntimeError("Missing prompt template for mode %s (%s)" % (m["id"], m["prompt_key"]))
    return txt


def build_prompt(mode, granularity, extra=""):
    """模式正文 + 输出粒度段落 +（可选）补充要求。"""
    mode, gran = normalize(mode, granularity)
    body = default_prompt(mode)
    body += prompts.ELEMENT_GRAN_SET if gran == "set" else prompts.ELEMENT_GRAN_ITEM
    extra = (extra or "").strip()
    if extra:
        body += "\n\n【补充要求】\n" + extra
    return body


def suggest_max_images(granularity, ref_count=0):
    """生成张数建议：方舟硬约束「参考图数 + 生成图数 ≤ 15」。"""
    cap = max(1, 15 - int(ref_count or 0))
    return 1 if granularity == "set" else min(15, cap)


def is_ready(cfg):
    return imgapi.is_ready(cfg)


def generate(cfg, out_dir, refs, mode="fidelity", granularity="item",
             prompt=None, size=None, max_images=None, output_format=None,
             watermark=False, optimize_prompt=None, progress=None, tag=None):
    """一次组图调用：产品实拍图 → 修复后的元素素材。

    refs: 参考图路径列表
    返回 imgapi.generate 的结果字典，并补充 mode/granularity/prompt 便于落库。
    """
    mode, gran = normalize(mode, granularity)
    refs = [p for p in (refs or []) if p]
    if not refs:
        raise RuntimeError("Missing reference image — upload product photos in step 1 first")
    text = (prompt or "").strip() or build_prompt(mode, gran)
    if max_images in (None, "", 0):
        max_images = suggest_max_images(gran, len(refs))
    res = imgapi.generate(
        cfg, out_dir, text, ref_images=refs, size=size, max_images=int(max_images),
        sequential="auto", output_format=output_format or "jpeg",
        watermark=bool(watermark), optimize_prompt=optimize_prompt or "standard",
        progress=progress, tag=tag or "Element assets", download=True)
    res["mode"] = mode
    res["mode_name"] = mode_meta(mode)["name"]
    res["granularity"] = gran
    res["prompt"] = text
    res["refs"] = refs
    return res
