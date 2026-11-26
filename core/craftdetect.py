# -*- coding: utf-8 -*-
"""
工艺细节识别模块（请柬卡片产品图）
=================================

任务：在请柬卡片产品图中找到 **烫金 / 浮雕(凸版) / 蕾丝(镂空)** 区域，
      输出适合微距展示的 Top-5 候选框。

严格遵守的纪律（来自需求）：
  1. 多尺度扫描：整图 → 3x3 网格 → 候选区域放大 2–3 倍。
  2. 先检测后分类：先找疑似区域，再判定工艺类型。
  3. 每个候选必须输出：bbox / label / confidence / evidence / micro_score /
     is_text / is_logo / is_pattern。
  4. 置信度 < 0.7 → 裁剪该区域放大 3 倍重新识别；仍不确定 → 标 uncertain。
  5. 不允许只给一个结果 → 输出 Top-5 候选，按 micro_score 降序。
  6. 文字 / logo / 花纹区域单独标记；不允许建议 AI 重绘。
  7. 只输出 JSON（craft_detect / craft_detect_json 的返回即 JSON）。

工艺定义（防混）：
  · 烫金 gold_foil：金属高光、金/银色、边缘亮、反光随角度变化，常在文字/边框/logo。
    —— 金色印刷 ≠ 烫金。
  · 浮雕 emboss：侧光阴影、边缘明暗过渡、凸起或压痕、letterpress 凹陷。
    —— 平面阴影 ≠ 浮雕。
  · 蕾丝 lace：高频镂空、网状、透明、边缘复杂、背景透光。
    —— 白色花纹 ≠ 蕾丝。

实现说明：
  · 视觉判定复用 core.showcase.chat_vision（同一把 ARK Key，零额外依赖）。
  · 图像读取走 cv2.imdecode(np.fromfile)（非 ASCII 路径红线），裁剪写盘走 imencode+write。
  · 网格检测用「3x3 拼接图 + 编号」一次视觉调用搞定（不放大 9 次调用）。
  · 候选区放大：整图检测给出 bbox → 裁出该区（已是 2x 级放大）→ 若 conf<0.7 再裁更紧（3x 级）。
"""

import os
import json
import uuid
import tempfile
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

from core.showcase import chat_vision, PLANNER_MODELS, _parse_json_loose

# 视觉模型输入图最长边上限（2026-10-14 性能修复）：检测只需定位/分类框，
# 不需要原图 4K 分辨率；把视觉输入压到 2048 既能保住识别质量，又能把单张上传体积
# 从 ~5MB 降到 ~1MB、并显著缩短模型推理耗时（每个调用都重复上传整图）。
_VISION_MAX_SIDE = 2048

CONF_THRESHOLD = 0.7          # 置信度阈值（低于则放大重识别）
LABELS = ("gold_foil", "emboss", "lace")
GRID_N = 3                    # 3x3 网格
TILE = 384                    # 网格拼接每格边长(px)
SEP = 8                       # 网格分隔线宽(px)
SCAN_LEVELS = ["whole", "grid3x3", "zoom2x", "zoom3x"]

# ---------------- 提示词（严格 JSON-only，含工艺定义与易混点）----------------

DETECT_WHOLE_SYSTEM = (
    "You are a craft-detail detector for wedding-invitation product photos. "
    "Scan the WHOLE image and locate every region that shows a special surface craft "
    "ON the paper goods themselves (invitation card, RSVP/insert cards, envelope). "
    "Target crafts (ONLY these three): "
    "gold_foil = metallic foil (gold/silver highlights, bright glinting edges, reflections that shift with viewing angle; "
    "usually on text, borders, monograms, logos — NOT a flat gold ink print); "
    "emboss = raised or pressed relief (side-lit shadows, light-to-dark edge gradients, visible raised/压痕, "
    "letterpress depression, engraved/engraving-style lettering, deboss — NOT a flat printed shadow); "
    "lace = high-frequency cutout / lace (mesh, see-through holes, intricate edges, background shows through — "
    "NOT a plain white printed pattern). "
    "A raised 3D paper applique on a card (paper flower / butterfly / bow glued onto the card) counts as emboss. "
    "HARD EXCLUSIONS — never box these as craft: decorative props lying AROUND the cards "
    "(bead garlands, pearl strings, gold bead chains, ribbons, jewelry, rings, candles, plates, "
    "cutlery, flowers not attached to a card, fabric/table linen texture); "
    "flat printed body text paragraphs (plain ink text is not a craft); "
    "the table/background scene itself. "
    "Output ONLY JSON, no explanation, no code fences: "
    '{"regions":[{"bbox":[x,y,w,h],"guess":"gold_foil|emboss|lace|other","reason":"one short phrase"}]}. '
    "Coordinates are normalized 0-1 relative to the whole image (x,y = top-left of box; w,h = box size). "
    "List up to 8 regions by salience, most significant first. "
    "Box each DISTINCT craft location separately (a foil word, a foil corner flourish, a lace border strip, "
    "an embossed seal — each its own box). Do NOT box the entire card; box the local craft area. "
    "Each craft box must be SMALL and LOCAL: under ~25% of the image area, hugging the craft itself. "
    "If the image shows none of the three crafts, regions:[].")

DETECT_GRID_SYSTEM = (
    "You are a craft-detail detector for wedding-invitation product photos. "
    "The attached image is a 3x3 grid of equal tiles, indices 0..8, row-major "
    "(row0 = 0,1,2; row1 = 3,4,5; row2 = 6,7,8). "
    "For each tile that contains a special surface craft ON THE PAPER GOODS (card / envelope), "
    "report its index and the craft location INSIDE that tile. "
    "Target crafts (ONLY these three): "
    "gold_foil = metallic foil highlight, glinting edges, angle-dependent reflection — NOT flat gold print; "
    "emboss = raised/pressed relief, side-lit shadow, edge gradient, letterpress, engraved lettering, "
    "deboss, 3D paper applique glued on the card — NOT flat shadow; "
    "lace = high-frequency cutout/lace, see-through, intricate edge — NOT white print. "
    "NEVER report decorative props (bead garlands, pearl strings, ribbons, jewelry, candles, "
    "plates, cutlery, loose flowers) or plain printed body text as craft. "
    "Output ONLY JSON: "
    '{"tiles":[{"idx":0,"guess":"gold_foil|emboss|lace|other","sub":[x,y,w,h],"reason":"short"}]}. '
    "sub is normalized 0-1 within that tile and should hug the local craft area. "
    "Report up to 6 tiles by salience. If none, tiles:[].")

CLASSIFY_SYSTEM = (
    "You are a precision craft classifier for a zoomed crop from a wedding-invitation product photo. "
    "Determine EXACTLY what surface craft the crop shows, ON THE PAPER GOODS. "
    "Labels: "
    "gold_foil = metallic foil (gold/silver), bright glinting edges, reflection changes with viewing angle, "
    "typically on text/border/logo; must NOT be flat gold ink print. "
    "emboss = raised or pressed relief, side-lit shadow, light-to-dark edge transition, visible raised area or 压痕, "
    "letterpress depression, engraved lettering, deboss, 3D paper applique (paper flower/butterfly on the card); "
    "must NOT be a flat printed shadow. "
    "lace = high-frequency cutout/lace, mesh, transparent holes, intricate edge, background shows through; "
    "must NOT be a plain white printed pattern. "
    "If the crop shows a decorative prop instead (bead garland, pearl string, ribbon, jewelry, candle, "
    "plate, cutlery, loose flowers, table linen), label it 'uncertain'. "
    "If the crop is ambiguous or does not clearly show one of these three, label it 'uncertain'. "
    "Output ONLY JSON: "
    '{"label":"gold_foil|emboss|lace|uncertain","confidence":0.0,"evidence":"one short English sentence '
    'stating the visual cue you relied on","micro_score":0.0,"is_text":false,"is_logo":false,"is_pattern":false}. '
    "micro_score = how suitable this region is as a close-up macro shot (high when the craft detail is fine, "
    "centered, and fills the frame; low for blurry, off-centre, or coarse regions). "
    "Do NOT suggest any AI redraw, regeneration, or inpainting. Only JSON.")


# ---------------- 图像 IO（遵守非 ASCII 路径红线）----------------

# ---------------- 卡片分区检测（2026-10-12 学习案例：运镜主体会切到单独卡片）----------------
# 案例热帖的运镜并非一直聚焦整套，而是会从「整套 → 单卡 → 工艺细节」逐层切主体。
# 为让单图 set/group（平铺套装图）也能复刻这层「切到单独卡片」的层次，需要先知道
# 图里哪些区域是「一张独立的卡」（主卡 / 信封 / RSVP / Menu / 插卡…），再让 cm16 定格镜
# 框住某张单卡。本检测只找「独立卡片/组件」的 bbox，不做工艺分类（工艺由 craft_detect 负责）。

CARD_SYSTEM = (
    "You are a layout analyzer for wedding-invitation product photos. "
    "The image is usually a flat-lay of a full invitation SUITE — several separate pieces "
    "laid out together. Find the bounding boxes of EVERY distinct individual piece, including: "
    "the main invitation card, the envelope, the RSVP / reply card, the menu card, "
    "the details / insert card, the belly band / wrap, the wax seal, "
    "and any dark outer envelope, sleeve, shell, pocket folder or box "
    "(a black / dark enclosure IS a separate piece — box it too, even if it sits at the bottom edge). "
    "Box EACH separate item on its own (do NOT box the whole arrangement as one box). "
    "NESTED items: when a card lies inside / on top of a container (a white card inside an open "
    "black folder, a small card on a sleeve, an insert inside an envelope), box the INNER item "
    "itself by its own visible edges, and box the container as its own separate piece — one box "
    "must NEVER cover both the container and what sits inside it. "
    "Do NOT include neighboring props (rings, ribbons, flowers, perfume) in any box. "
    "Typical suites have 4-6 pieces; expect and report multiple boxes. "
    "If the photo shows only ONE card with no separate components, return that single box. "
    "If there are no distinct cards (e.g. an abstract texture or a pure detail crop), boxes:[]. "
    "Output ONLY JSON, no explanation, no code fences: "
    '{"boxes":[{"bbox":[x,y,w,h],"label":"main_card|envelope|rsvp|menu|insert|band|seal|sleeve|shell|other",'
    '"confidence":0.0,"reason":"one short phrase"}]}. '
    "Coordinates normalized 0-1 relative to the whole image (x,y = top-left of box; w,h = box size). "
    "List up to 6 boxes by prominence, largest/most-central first. "
    "Every box must HUG the item's visible edges: padding beyond the item stays under ~3% of the "
    "image on each side. A box much larger than the item itself is wrong."
)


def _detect_cards_whole(cfg, img, tmp, model):
    p = os.path.join(tmp, "cards.jpg")
    _write_jpg(img, p)
    try:
        j = _vision(cfg, CARD_SYSTEM, "Find each distinct card/component box. Output JSON only.",
                    p, model, max_tokens=500, timeout=90)
    except Exception as e:
        return [], "card detect failed: %s" % e
    out = []
    for r in (j.get("boxes") or []):
        b = _norm_box(r.get("bbox"))
        if not b:
            continue
        try:
            conf = float(r.get("confidence") or 0.0)
        except Exception:
            conf = 0.0
        out.append({"bbox": b, "label": str(r.get("label") or "other"),
                    "confidence": round(min(1.0, max(0.0, conf)), 3),
                    "reason": str(r.get("reason") or "")})
    return out, None


def _merge_card_boxes(dets, iou_thr=0.5):
    """卡片框去重：置信度高的优先，IoU 过高的合并为一条。"""
    dets = sorted(dets, key=lambda d: d.get("confidence", 0.0), reverse=True)
    out = []
    for d in dets:
        if any(_iou(d["bbox"], m["bbox"]) > iou_thr for m in out):
            continue
        out.append(d)
    return out


def detect_cards(path, cfg=None, depth="fast", max_cards=3):
    """在请柬卡片产品图中找到「独立的单卡/组件」bbox，输出归一化框列表。

    返回结构：
    {
      "source": <basename>,
      "cards": [ {"bbox":[x,y,w,h], "label": str, "confidence": 0.0-1.0, "reason": str}, ... ],
      "total_found": int,
      "notes": [...]
    }
    无 ARK Key 时优雅返回空 cards（不影响生成流程）。
    """
    notes = []
    cfg = cfg or {}
    model = PLANNER_MODELS.get(depth) or PLANNER_MODELS["fast"]
    result = {"source": os.path.basename(path), "cards": [], "total_found": 0,
              "notes": notes}
    if not (cfg.get("api_key") or "").strip():
        notes.append("ARK API key not configured — cannot run card detection (empty).")
        return result
    try:
        img = _read(path)
    except Exception as e:
        notes.append("image read failed: %s" % e)
        return result
    tmp = tempfile.mkdtemp(prefix="cards_")
    try:
        dets, err = _detect_cards_whole(cfg, img, tmp, model)
        if err:
            notes.append(err)
        merged = _merge_card_boxes(dets)
        result["cards"] = merged[:max_cards]
        result["total_found"] = len(merged)
        if not merged:
            notes.append("no distinct cards/components detected (single flat product photo).")
    except Exception as e:
        notes.append("detect_cards error: %s" % e)
    finally:
        try:
            import shutil as _sh
            _sh.rmtree(tmp, ignore_errors=True)
        except BaseException:   # 本机安全删除钩子可能抛 SystemExit
            pass
    return result


def detect_cards_json(path, cfg=None, depth="fast", max_cards=3):
    """只输出 JSON 字符串。"""
    return json.dumps(detect_cards(path, cfg, depth, max_cards),
                      ensure_ascii=False, indent=2)


def _read(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Could not read image: %s" % path)
    from core.imgproc import normalize_luma   # 2026-10-27 亮度归一：暗图检出率更高
    return normalize_luma(img)


def _write_jpg(img, path, max_side=_VISION_MAX_SIDE):
    """写 jpg 给视觉模型；图最长边超过 max_side 先等比缩小（INTER_AREA），
    避免把 4K 原图整张上传（每个调用都重复上传，是检测慢的主因之一）。"""
    h, w = img.shape[:2]
    long = max(w, h)
    if long > max_side:
        sc = max_side / float(long)
        img = cv2.resize(img, (max(1, int(round(w * sc))), max(1, int(round(h * sc)))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("Image encode failed: %s" % path)
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def _crop(img, bbox, pad=0.12):
    """按归一化 bbox 从原图裁出区域（带 padding），返回裁剪图。"""
    x, y, w, h = bbox
    H, W = img.shape[:2]
    x0 = max(0.0, x - pad * w)
    y0 = max(0.0, y - pad * h)
    x1 = min(1.0, x + w + pad * w)
    y1 = min(1.0, y + h + pad * h)
    px0, py0, px1, py1 = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
    return img[py0:py1, px0:px1].copy()


def _shrink(bbox, k=0.7):
    """把 bbox 围绕中心收缩到 k 倍（更紧的放大，用于 3x 重识别）。"""
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = w * k, h * k
    return [cx - nw / 2.0, cy - nh / 2.0, nw, nh]


def _montage_grid(img, n=GRID_N, tile=TILE, sep=SEP):
    """把原图切成 n×n 网格拼成一张图（带分隔线与编号），返回 (montage, rects)。
    rects[k] = 该格在原图中的像素矩形 (x0,y0,x1,y1)，用于把 sub-bbox 映射回原图。

    2026-10-29 修复：旧版每格 resize 到固定 tile(384) 见方、但 stride 用 tw=W//n
    算 montage 尺寸 —— 图宽不能整除/不接近 3*384 时，最后一列/行的目标区域比
    384 窄，赋值直接 broadcast 越界抛异常；该异常又会把整条 craft_detect（含
    whole 路径的候选）炸成空并被调用方静默吞掉（诊断实锤：1064×1600 图
    "could not broadcast ... into shape (384,302,3)"，工艺候选常年为 0）。
    现在每格 resize 到本格实际尺寸 (tw, th)，stride 与 montage 尺寸用同一组
    tw/th，任何输入尺寸都精确对齐。
    """
    H, W = img.shape[:2]
    tw = max(1, W // n)
    th = max(1, H // n)
    montage = np.full(((th + sep) * n + sep, (tw + sep) * n + sep, 3), 255, dtype=np.uint8)
    rects = []
    for r in range(n):
        for c in range(n):
            k = r * n + c
            gx0, gy0 = c * tw, r * th
            gx1, gy1 = min(W, gx0 + tw), min(H, gy0 + th)
            rects.append((gx0, gy0, gx1, gy1))
            tile_img = img[gy0:gy1, gx0:gx1]
            if tile_img.size == 0:
                continue
            tile_img = cv2.resize(tile_img, (tw, th), interpolation=cv2.INTER_AREA)
            mx0 = sep + c * (tw + sep)
            my0 = sep + r * (th + sep)
            montage[my0:my0 + th, mx0:mx0 + tw] = tile_img
            # 编号
            cv2.rectangle(montage, (mx0, my0), (mx0 + 34, my0 + 30), (20, 20, 20), -1)
            cv2.putText(montage, str(k), (mx0 + 6, my0 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return montage, rects


# ---------------- 视觉调用封装 ----------------

def _vision(cfg, system, user_text, image_path, model, max_tokens=700, timeout=120):
    txt, _u, _e = chat_vision(cfg, system, user_text, image_path,
                              model=model, max_tokens=max_tokens, timeout=timeout)
    return _parse_json_loose(txt)


def _norm_box(b):
    try:
        b = [float(v) for v in b]
    except Exception:
        return None
    if len(b) != 4:
        return None
    x, y, w, h = b
    if w <= 0 or h <= 0:
        return None
    if x < -0.05 or y < -0.05 or x + w > 1.06 or y + h > 1.06:
        return None
    x = max(0.0, x); y = max(0.0, y)
    w = min(w, 1.0 - x); h = min(h, 1.0 - y)
    if w < 0.01 or h < 0.01:
        return None
    return [x, y, w, h]


def _detect_whole(cfg, img, tmp, model):
    p = os.path.join(tmp, "whole.jpg")
    _write_jpg(img, p)
    try:
        j = _vision(cfg, DETECT_WHOLE_SYSTEM,
                    "Detect every gold_foil / emboss / lace region in this invitation card. Output JSON.",
                    p, model)
    except Exception as e:
        return [], "whole detect failed: %s" % e
    out = []
    for r in (j.get("regions") or []):
        b = _norm_box(r.get("bbox"))
        if not b:
            continue
        out.append({"bbox": b, "guess": str(r.get("guess") or "other"),
                    "reason": str(r.get("reason") or ""), "stage": "whole"})
    return out, None


def _detect_grid(cfg, img, tmp, model):
    montage, rects = _montage_grid(img)
    p = os.path.join(tmp, "grid.jpg")
    _write_jpg(montage, p)
    try:
        j = _vision(cfg, DETECT_GRID_SYSTEM,
                    "Which 3x3 tiles contain gold_foil / emboss / lace? Report idx + sub-bbox. Output JSON.",
                    p, model)
    except Exception as e:
        return [], "grid detect failed: %s" % e
    out = []
    for t in (j.get("tiles") or []):
        try:
            idx = int(t.get("idx"))
        except Exception:
            continue
        if idx < 0 or idx >= len(rects):
            continue
        sub = _norm_box(t.get("sub"))
        if not sub:
            continue
        gx0, gy0, gx1, gy1 = rects[idx]
        gw, gh = gx1 - gx0, gy1 - gy0
        W = img.shape[1]; H = img.shape[0]
        gx0n, gy0n = gx0 / W, gy0 / H
        gwn, ghn = gw / W, gh / H
        b = _norm_box([gx0n + sub[0] * gwn, gy0n + sub[1] * ghn,
                       sub[2] * gwn, sub[3] * ghn])
        if not b:
            continue
        out.append({"bbox": b, "guess": str(t.get("guess") or "other"),
                    "reason": str(t.get("reason") or ""), "stage": "grid3x3"})
    return out, None


def _classify(cfg, img, bbox, model, tmp, tight=False):
    crop = _crop(img, bbox, pad=0.04 if tight else 0.12)
    if crop.size == 0:
        return None
    # 文件名带 uuid：并行分类时多个 _classify 共享同一 tmp，必须互不覆盖
    p = os.path.join(tmp, "cls_%s_%s.jpg" % ("tight" if tight else "wide", uuid.uuid4().hex[:8]))
    _write_jpg(crop, p)
    try:
        j = _vision(cfg, CLASSIFY_SYSTEM,
                    "Classify this zoomed craft crop. Output JSON only.",
                    p, model, max_tokens=400, timeout=90)
    except Exception as e:
        return {"label": "uncertain", "confidence": 0.0,
                "evidence": "classify call failed: %s" % e, "micro_score": 0.0,
                "is_text": False, "is_logo": False, "is_pattern": False,
                "rerecognized": tight, "zoom_stage": "zoom3x" if tight else "zoom2x"}
    label = str(j.get("label") or "uncertain").strip().lower()
    if label not in LABELS and label != "uncertain":
        label = "uncertain"
    try:
        conf = float(j.get("confidence") or 0.0)
    except Exception:
        conf = 0.0
    try:
        ms = float(j.get("micro_score") or 0.0)
    except Exception:
        ms = 0.0
    conf = min(1.0, max(0.0, conf))
    ms = min(1.0, max(0.0, ms))
    return {
        "label": label,
        "confidence": round(conf, 3),
        "evidence": str(j.get("evidence") or ""),
        "micro_score": round(ms, 3),
        "is_text": bool(j.get("is_text")),
        "is_logo": bool(j.get("is_logo")),
        "is_pattern": bool(j.get("is_pattern")),
        "rerecognized": tight,
        "zoom_stage": "zoom3x" if tight else "zoom2x",
    }


# ---------------- 候选合并 ----------------

def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0


def _merge_detections(dets, iou_thr=0.4):
    """合并整图与网格检测到的重叠框，保留显著度更高的 guess。"""
    merged = []
    for d in dets:
        b = d["bbox"]
        hit = None
        for m in merged:
            if _iou(b, m["bbox"]) > iou_thr:
                hit = m
                break
        if hit is None:
            merged.append(dict(bbox=b, guesses=[d["guess"]], reasons=[d.get("reason", "")],
                               stages=[d["stage"]]))
        else:
            hit["guesses"].append(d["guess"])
            if d.get("reason"):
                hit["reasons"].append(d["reason"])
            if d["stage"] not in hit["stages"]:
                hit["stages"].append(d["stage"])
    return merged


# ---------------- 主入口 ----------------

def craft_detect(path, cfg=None, depth="fast", use_grid=True, max_candidates=5):
    """在请柬卡片产品图中识别工艺细节，返回 Top-N 候选（JSON 字典）。

    返回结构（严格遵守纪律 3/4/5/6/7）：
    {
      "source": <basename>,
      "scan_levels": [...],
      "candidates": [            # 按 micro_score 降序，最多 max_candidates 条
        {
          "bbox": [x,y,w,h],     # 归一化 0-1
          "label": "gold_foil|emboss|lace|uncertain",
          "confidence": 0.0-1.0,
          "evidence": "英文短句",
          "micro_score": 0.0-1.0,
          "is_text": bool, "is_logo": bool, "is_pattern": bool,
          "zoom_stage": "zoom2x|zoom3x",
          "rerecognized": bool
        }, ...
      ],
      "total_found": int,        # 实际检测到的候选数（去重后）
      "returned": int,           # 实际返回条数
      "notes": [...]
    }
    """
    notes = []
    cfg = cfg or {}
    model = PLANNER_MODELS.get(depth) or PLANNER_MODELS["fast"]
    result = {
        "source": os.path.basename(path),
        "scan_levels": list(SCAN_LEVELS),
        "candidates": [],
        "total_found": 0,
        "returned": 0,
        "notes": notes,
    }
    if not (cfg.get("api_key") or "").strip():
        notes.append("ARK API key not configured — cannot run vision detection (returning empty candidates).")
        return result
    try:
        img = _read(path)
    except Exception as e:
        notes.append("image read failed: %s" % e)
        return result

    tmp = tempfile.mkdtemp(prefix="craft_")
    try:
        # ---- 尺度 1/2：整图 + 3x3 网格 并行检测 ----
        # 两个独立视觉调用互不依赖，串行要 ~2 倍单调用时；并行把这段墙钟时间砍半。
        with ThreadPoolExecutor(max_workers=2) as _ex:
            _f_whole = _ex.submit(_detect_whole, cfg, img, tmp, model)
            _f_grid = _ex.submit(_detect_grid, cfg, img, tmp, model) if use_grid else None
            try:
                dets, err = _f_whole.result()
            except Exception as e:
                dets, err = [], "whole detect crashed: %s" % e
            if err:
                notes.append(err)
            if _f_grid is not None:
                try:
                    gdets, gerr = _f_grid.result()
                except Exception as e:
                    # 单路失败不拖垮整条检测（2026-10-29：此前 grid 抛异常会把
                    # whole 路径的候选一起带走，工艺检测整条变空还被静默吞掉）
                    gdets, gerr = [], "grid detect crashed: %s" % e
                if gerr:
                    notes.append(gerr)
                dets += gdets
        # ---- 合并（去重重叠）----
        merged = _merge_detections(dets)
        result["total_found"] = len(merged)
        if not merged:
            notes.append("no craft region detected across whole + grid scan.")
            return result

        # ---- 尺度 3/4：逐候选放大分类（先 2x，conf<阈值再 3x）----
        # 各候选分类互不依赖，用受限线程池并行（max_workers=4）把 ~N×2 次串行调用
        # 压成 ~ceil(N/4)×2 波；调用失败（限流/超时）由 _classify 兜底成 uncertain，
        # 最终按 micro_score 取 Top，失败候选自然落到末尾，不影响主流程。
        def _cls_one(m):
            bbox = m["bbox"]
            c1 = _classify(cfg, img, bbox, model, tmp, tight=False)
            if c1 is None:
                return None
            # 置信度不足 → 放大 3x 重识别（纪律 4）
            if c1["confidence"] < CONF_THRESHOLD:
                c2 = _classify(cfg, img, bbox, model, tmp, tight=True)
                # 取置信度更高者；若两者都 < 阈值，标 uncertain 但保留最接近的判断作 guess
                if c2 is not None and c2["confidence"] >= c1["confidence"]:
                    c1 = c2
                if c1["confidence"] < CONF_THRESHOLD:
                    c1["label"] = "uncertain"
                    c1["guess"] = (m["guesses"][0] if m["guesses"] else "other")
            c1["bbox"] = [round(v, 4) for v in bbox]
            return c1

        cands = []
        with ThreadPoolExecutor(max_workers=4) as _ex:
            for c in _ex.map(_cls_one, merged):
                if c is not None:
                    cands.append(c)
        # 兜底：并行全失败（极端限流）时退回串行，避免整批漏检
        if not cands and merged:
            for m in merged:
                c = _cls_one(m)
                if c is not None:
                    cands.append(c)

        # ---- 纪律 5：按 micro_score 降序，Top-N，不允许只给一个 ----
        cands.sort(key=lambda c: c.get("micro_score", 0.0), reverse=True)
        result["candidates"] = cands[:max_candidates]
        result["returned"] = len(result["candidates"])
        if result["returned"] < result["total_found"]:
            notes.append("found %d candidates, returning top %d by micro_score."
                         % (result["total_found"], result["returned"]))
        if result["returned"] < max_candidates:
            notes.append("fewer than %d genuine craft regions on this card; "
                         "output is the full ranked shortlist (no fabrication)."
                         % max_candidates)
    except Exception as e:
        notes.append("craft_detect error: %s" % e)
    finally:
        try:
            import shutil as _sh
            _sh.rmtree(tmp, ignore_errors=True)
        except BaseException:   # 本机安全删除钩子可能抛 SystemExit
            pass
    return result


def craft_detect_json(path, cfg=None, depth="fast", use_grid=True, max_candidates=5):
    """只输出 JSON 字符串（纪律 7）。"""
    return json.dumps(craft_detect(path, cfg, depth, use_grid, max_candidates),
                      ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import sys
    _cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "config.json"), encoding="utf-8")).get("storyboard_api", {})
    if len(sys.argv) > 1:
        print(craft_detect_json(sys.argv[1], _cfg))
