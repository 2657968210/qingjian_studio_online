# -*- coding: utf-8 -*-
"""
分段手持展示流水线（脚本规划层 + 逐段生成编排）
=================================================
2026-09-24 用户口径：不再「一张整套图 + 一条长指令」让模型一次完成所有卡片展示。

链路：整套图 → 视觉大模型识别组件并排序（脚本规划层）
      → 每个组件单独一段（Touch→PickUp→CenterDisplay→MicroTilt→Hold，4s 生成、本地裁剪）
      → 一致性检查 → 按序拼接 → Hero 收尾（几何运镜）→ 配乐导出（复用 compose）

发给 Seedance 的指令只有 SHOWCASE_SEGMENT 渲染出的单镜形态；
v3 长指令（prompts.VIDEO_HANDHELD）退居规划层的口径依据，不直接发送。
"""
import base64
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

from core import prompts
from core.compose import FF, _duration

# 规划层可用视觉模型（须在方舟控制台开通；账号实测 2026-09-24）
PLANNER_MODELS = {
    "fast": "doubao-seed-2-0-mini-260428",   # ~15s / ~3k tokens（260215 版本已被方舟下线，2026-10-27 升版）
    "deep": "doubao-seed-2-0-lite-260428",   # ~100s / ~6k tokens，含更细画质评估
}
DEFAULT_PLANNER = "fast"

# 类别归一：label 关键词 → (排序档位, 建议成片时长, 中文名)
# 档位即标准展示顺序：主请柬 → RSVP → Details/Information/Reception → Menu → 其他 → 信封/外壳
CATEGORY_RULES = [
    ("main invitation", 1, 3.0, "主请柬"),
    ("invitation",      1, 3.0, "主请柬"),
    ("rsvp",            2, 2.5, "RSVP 卡"),
    ("reception",       3, 2.5, "Reception 卡"),
    ("details",         3, 2.5, "Details 卡"),
    ("information",     3, 2.5, "Information 卡"),
    ("menu",            4, 2.5, "Menu 卡"),
    ("wishing",         5, 2.5, "祝福卡"),
    ("envelope",        9, 2.5, "信封"),
    ("outer shell",     9, 2.5, "外壳"),
    ("pocket",          9, 2.5, "封套"),
    ("sleeve",          9, 2.5, "纸套"),
]

NOT_PRODUCT = ("not a product", "非产品", "装饰", "背景")


def _category(label):
    lab = (label or "").lower()
    for kw, rank, dur, zh in CATEGORY_RULES:
        if kw in lab:
            return rank, dur, zh
    return 5, 2.5, "其他小卡"


# ---------------- 视觉大模型调用（chat/completions） ----------------

def _imencode_u(path, max_side=2048, quality=92):
    """读图 → 缩放 → JPEG base64 data URI（OpenCV 对非 ASCII 路径不可靠，统一走 imencode）。"""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Could not read the image: %s" % path)
    h, w = img.shape[:2]
    s = min(1.0, float(max_side) / max(h, w))
    if s < 0.999:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Image encoding failed: %s" % path)
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def chat_vision(cfg, system_prompt, user_text, image_path,
                model=None, max_tokens=2500, temperature=0.2, timeout=240):
    """方舟 chat/completions（image→text）。返回 (content_text, usage, elapsed)。"""
    api_key = ((cfg or {}).get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError("Volcengine Ark API key is not configured (Settings → Image generation API; shared with video generation)")
    base = ((cfg or {}).get("endpoint") or "").strip().rstrip("/") or \
        "https://ark.cn-beijing.volces.com/api/v3"
    for suffix in ("/chat/completions", "/images/generations", "/contents/generations/tasks"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    uri = _imencode_u(image_path)
    payload = {
        "model": model or PLANNER_MODELS[DEFAULT_PLANNER],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": uri}},
            ]},
        ],
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer %s" % api_key,
                 "Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            j = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "ignore")
        try:
            err = json.loads(raw).get("error") or {}
            msg = err.get("message") or raw
        except Exception:
            msg = raw
        hint = ""
        if e.code == 404 and "ModelNotOpen" in raw:
            hint = "(this vision model is not enabled on this account — enable it in the Ark console, or switch to another planning model in settings)"
        elif e.code == 401:
            hint = "(invalid API key)"
        raise RuntimeError("Planning model call failed with HTTP %s: %s%s" % (e.code, str(msg)[:300], hint))
    except urllib.error.URLError as e:
        raise RuntimeError("Cannot reach Volcengine Ark: %s" % e.reason)
    txt = (j.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return txt, (j.get("usage") or {}), round(time.time() - t0, 1)


# ---------------- 规划层 ----------------

def _parse_json_loose(txt):
    t = (txt or "").strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        raise RuntimeError("The planning model did not return JSON: %s" % t[:200])
    return json.loads(m.group(0))


def _pos_desc(bbox):
    """bbox(0-1) → 中文方位描述（九宫格），用于单镜指令定位组件。"""
    x, y = (bbox or [0, 0, 1, 1])[0], (bbox or [0, 0, 1, 1])[1]
    col = "左侧" if x < 0.36 else ("右侧" if x > 0.62 else "中部")
    row = "上方" if y < 0.36 else ("下方" if y > 0.62 else "中部")
    if col == "中部" and row == "中部":
        return "画面中央"
    if col == "中部":
        return "画面" + row
    if row == "中部":
        return "画面" + col
    return "画面" + row + col


def _pos_desc_en(bbox):
    """Same nine-grid mapping as _pos_desc, but in English — display only."""
    x, y = (bbox or [0, 0, 1, 1])[0], (bbox or [0, 0, 1, 1])[1]
    col = "left" if x < 0.36 else ("right" if x > 0.62 else "middle")
    row = "top" if y < 0.36 else ("bottom" if y > 0.62 else "middle")
    if col == "middle" and row == "middle":
        return "centre of frame"
    if col == "middle":
        return "upper centre" if row == "top" else "lower centre"
    if row == "middle":
        return "centre " + col
    return row + " " + col


def plan_components(image_path, cfg, depth="fast"):
    """脚本规划层：识别组件 → 类别强制排序 → 生成可编辑的分段计划。

    返回 {src, scene, image_quality, issues, components, model, elapsed, usage}
    components: [{id,label,label_zh,category_rank,dur,bbox,pos,confidence,note,include}]
    """
    model = PLANNER_MODELS.get(depth) or PLANNER_MODELS[DEFAULT_PLANNER]
    txt, usage, el = chat_vision(
        cfg, prompts.SHOWCASE_PLANNER,
        "请识别这张整套请柬产品图中的独立纸品组件，按系统约定输出 JSON。",
        image_path, model=model)
    plan = _parse_json_loose(txt)
    comps = []
    for c in plan.get("components") or []:
        label = (c.get("label") or "").strip()
        if not label or any(k in label.lower() for k in NOT_PRODUCT):
            continue
        rank, dur, zh_default = _category(label)
        zh = (c.get("label_zh") or "").strip() or zh_default
        bbox = c.get("bbox") or [0, 0, 1, 1]
        try:
            bbox = [float(v) for v in bbox][:4]
        except Exception:
            bbox = [0, 0, 1, 1]
        comps.append({
            "id": "c%02d" % (len(comps) + 1),
            "label": label, "label_zh": zh,
            "category_rank": rank, "dur": dur,
            "bbox": [round(v, 4) for v in bbox],
            "pos": _pos_desc(bbox),
            "pos_en": _pos_desc_en(bbox),
            "confidence": float(c.get("confidence") or 0),
            "note": (c.get("note") or "").strip(),
            "include": True,
        })
    if not comps:
        raise RuntimeError("Planning finished but no standalone paper component was identified (the model returned %d entries)"
                           % len(plan.get("components") or []))
    # 类别强制排序：档位优先，同档按置信度（模型自报的 order 不可靠，不采信）
    comps.sort(key=lambda c: (c["category_rank"], -c["confidence"]))
    for i, c in enumerate(comps):
        c["order"] = i + 1
    return {
        "src": image_path,
        "scene": (plan.get("scene") or "").strip(),
        "image_quality": plan.get("image_quality") or "unknown",
        "issues": [str(x) for x in (plan.get("issues") or [])],
        "components": comps,
        "model": model, "elapsed": el, "usage": usage,
        "time": time.time(),
    }


def segment_prompt(comp, scene):
    """渲染单镜指令（真正发给 Seedance 的唯一形态）。"""
    scene_lock = prompts.SHOWCASE_SCENE_LOCK
    if scene:
        scene_lock += "当前场景：%s。" % scene
    return prompts.SHOWCASE_SEGMENT.format(
        label_zh=comp.get("label_zh") or comp.get("label") or "产品",
        pos=comp.get("pos") or "画面中",
        scene_lock=scene_lock)


# ---------------- 逐段生成编排（服务端任务体内调用） ----------------

def trim_keep_tail(src, out, keep, progress=None):
    """把 4s 成片裁剪为末尾 keep 秒（保留拿起→中央展示→Hold 收尾窗口）。

    重编码裁剪（关键帧精确），输出统一 1080x1920/30fps 与拼接口径一致。
    """
    dur = _duration(src)
    if dur <= 0:
        raise RuntimeError("Could not read the clip duration: %s" % src)
    keep = max(0.5, min(float(keep), dur))
    ss = max(0.0, dur - keep)
    cmd = [FF, "-y", "-ss", "%.3f" % ss, "-i", src,
           "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,"
                  "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1",
           "-an", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    from core import ffproc
    rc, err = ffproc.run_ffmpeg(cmd, timeout=600, err_tail=400)
    if rc != 0:
        raise RuntimeError("Trim failed: %s" % err)
    return out


def check_segments(files):
    """一致性检查：逐段 ffprobe 时长/宽高/帧率，报告口径是否统一。"""
    report = []
    for f in files:
        if not os.path.isfile(f):
            report.append({"file": os.path.basename(f), "ok": False, "reason": "File not found"})
            continue
        from core import ffproc
        err = ffproc.probe_info(f)
        m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
        ms = re.search(r"Stream #.*Video:.* (\d{2,5})x(\d{2,5})", err)
        mfps = re.search(r"([\d.]+) fps", err)
        dur = (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))) if m else 0
        report.append({
            "file": os.path.basename(f), "ok": bool(m and ms),
            "duration": round(dur, 2),
            "size": ("%sx%s" % (ms.group(1), ms.group(2))) if ms else "?",
            "fps": mfps.group(1) if mfps else "?",
            "reason": "" if (m and ms) else "Could not parse the video stream",
        })
    sizes = {x.get("size") for x in report if x.get("ok")}
    if len(sizes) > 1:
        for x in report:
            if x.get("ok"):
                x["warn"] = "resolution differs from the other clips"
    return report
