# -*- coding: utf-8 -*-
"""
图生视频 API 直连（火山方舟 / Volcengine Ark · Doubao-Seedance 系列）
=====================================================================
官方接口（截至 2026-09）:
  创建任务  POST {base}/contents/generations/tasks
  查询任务  GET  {base}/contents/generations/tasks/{id}
  鉴权      Authorization: Bearer <ARK_API_KEY>

base 默认 https://ark.cn-beijing.volces.com/api/v3

默认走「图生视频 · 首帧」模式：把分镜图作为 first_frame 传入，配合指令词生成
与静态分镜图连续的视频（卡片/文字/背景由首帧锚定，最大程度保真）。

注意
----
* 图片可用公网 URL 或 Base64 内联（格式 data:image/png;base64,xxx，图片类型必须小写）
* 单张图片 < 30MB，请求体 < 64MB；宽高比 [0.4, 2.5]，边长 [300, 6000] px
* 生成结果 video_url 有效期 24 小时，须及时下载转存
* watermark=false 可关闭平台水印（默认 false）
"""
import base64
import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL = "doubao-seedance-2-0-mini-260615"   # 项目既定口径：Seedance 2.0 mini · 720P

# 常用模型与其能力区间（dur 秒 / res 分辨率档位；用于前端提示与参数收敛）
MODELS = [
    {"id": "doubao-seedance-2-0-mini-260615", "name": "Seedance 2.0 Mini (project default · 480p/720p · lowest cost)",
     "dur": [4, 15], "res": ["480p", "720p"]},
    {"id": "doubao-seedance-2-0-260128", "name": "Seedance 2.0 (highest quality · 480p–4K)",
     "dur": [4, 15], "res": ["480p", "720p", "1080p"]},
    {"id": "doubao-seedance-2-0-fast-260128", "name": "Seedance 2.0 Fast (balanced speed/cost · 480p/720p)",
     "dur": [4, 15], "res": ["480p", "720p"]},
    {"id": "doubao-seedance-2-5-260628", "name": "Seedance 2.5 (30s long-form · multimodal reference)",
     "dur": [4, 30], "res": ["480p", "720p", "1080p"]},
    {"id": "doubao-seedance-1-5-pro-251215", "name": "Seedance 1.5 Pro (⚠ officially being retired)",
     "dur": [4, 12], "res": ["480p", "720p", "1080p"]},
    {"id": "doubao-seedance-1-0-pro-250528", "name": "Seedance 1.0 Pro (stable)",
     "dur": [2, 12], "res": ["480p", "720p", "1080p"]},
    {"id": "doubao-seedance-1-0-pro-fast-251015", "name": "Seedance 1.0 Pro Fast (fast · low cost)",
     "dur": [2, 12], "res": ["480p", "720p", "1080p"]},
]

RATIOS = ["adaptive", "9:16", "16:9", "1:1", "3:4", "4:3", "21:9"]

_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".webp": "image/webp", ".bmp": "image/bmp", ".tif": "image/tiff",
         ".tiff": "image/tiff", ".gif": "image/gif", ".heic": "image/heic",
         ".heif": "image/heif"}

MAX_IMAGE_BYTES = 30 * 1024 * 1024


# ---------------- HTTP 基础 ----------------

def _request(url, api_key, method="GET", payload=None, timeout=60):
    data = None
    headers = {"Authorization": "Bearer %s" % api_key}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "ignore")
        try:
            j = json.loads(raw)
            msg = j.get("error", {}).get("message") or j.get("message") or raw
            code = j.get("error", {}).get("code") or j.get("code") or ""
        except Exception:
            msg, code = raw, ""
        hint = ""
        if e.code == 401:
            hint = "(invalid API key, or the model is not enabled — check \"API key management\" in the Ark console)"
        elif e.code == 404:
            hint = "(incorrect model ID or endpoint path)"
        elif e.code == 429:
            hint = "(rate limited, or insufficient balance)"
        raise RuntimeError("Volcengine Ark HTTP %s: %s %s%s" % (e.code, code, msg[:400], hint))
    except urllib.error.URLError as e:
        raise RuntimeError("Cannot reach Volcengine Ark: %s (check your network or proxy)" % e.reason)


def api_base(cfg):
    ep = (cfg or {}).get("endpoint") or ""
    ep = ep.strip()
    if not ep:
        return DEFAULT_BASE
    # 允许用户填完整任务路径，自动回退到 base
    for suffix in ("/contents/generations/tasks", "/contents/generations/tasks/"):
        if ep.endswith(suffix):
            ep = ep[: -len(suffix)]
            break
    return ep.rstrip("/")


def model_id(cfg):
    return ((cfg or {}).get("model") or "").strip() or DEFAULT_MODEL


def is_ready(cfg):
    return bool((cfg or {}).get("api_key", "").strip())


def meta(mid=None):
    mid = mid or DEFAULT_MODEL
    for m in MODELS:
        if m["id"] == mid:
            return m
    # 未知/自定义模型 id：按最宽能力处理，交给服务端报错
    return {"id": mid, "name": mid, "dur": [2, 30],
            "res": ["480p", "720p", "1080p"]}


def resolve_params(cfg, resolution=None, duration=None):
    """把请求参数收敛到当前模型的能力区间，返回 (resolution, duration, notes)。

    例如 Seedance 2.0 mini 只支持 480p/720p：若前端选了 1080p，这里会回落到 720p
    并给出提示，避免直接抛 400 浪费一次调用。
    """
    m = meta(model_id(cfg))
    notes = []
    res = (resolution or "").strip() or (cfg or {}).get("resolution") or "720p"
    if res not in m["res"]:
        fallback = "720p" if "720p" in m["res"] else m["res"][0]
        notes.append("%s does not support %s — fell back to %s" % (m["id"], res, fallback))
        res = fallback
    lo, hi = m["dur"]
    try:
        dur = int(duration)
    except Exception:
        dur = lo
    if dur < lo or dur > hi:
        dur = max(lo, min(hi, dur))
        notes.append("Duration clamped into the %s-supported %d–%ds range (using %ds)" % (m["id"], lo, hi, dur))
    return res, dur, notes


# ---------------- 图片内联 ----------------

def encode_image(path, max_side=4096):
    """读图并返回 (data_uri, note)。超大图自动压缩以满足 30MB 限制。"""
    ext = os.path.splitext(path)[1].lower()
    mime = _MIME.get(ext, "image/jpeg")
    size = os.path.getsize(path)
    note = ""
    raw = open(path, "rb").read()

    need_shrink = size > MAX_IMAGE_BYTES or max_side > 0
    if need_shrink and mime in ("image/jpeg", "image/png", "image/webp", "image/bmp"):
        try:
            import cv2
            import numpy as np
            arr = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                long_side = max(h, w)
                scale = min(1.0, float(max_side) / long_side) if max_side else 1.0
                if size > MAX_IMAGE_BYTES:
                    scale = min(scale, (MAX_IMAGE_BYTES * 0.85 / size) ** 0.5)
                if scale < 0.999:
                    img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_AREA)
                    note = "Compressed %d×%d→%d×%d" % (w, h, img.shape[1], img.shape[0])
                ok, buf = cv2.imencode(".jpg", img,
                                       [cv2.IMWRITE_JPEG_QUALITY, 94])
                if ok:
                    raw = buf.tobytes()
                    mime = "image/jpeg"
        except Exception:
            pass

    if len(raw) > MAX_IMAGE_BYTES:
        raise RuntimeError("Image exceeds the 30MB limit (%0.1fMB) — compress it first, or use the upscale/crop operators in step 3"
                           % (len(raw) / 1048576.0))
    b64 = base64.b64encode(raw).decode("ascii")
    return "data:%s;base64,%s" % (mime.lower(), b64), note


# ---------------- 任务创建 / 查询 ----------------

def create_task(cfg, image_path, prompt, duration=5, resolution="720p",
                ratio="adaptive", generate_audio=False, watermark=False,
                camera_fixed=False, seed=-1, image_url=None, last_frame_path=None):
    """创建图生视频任务，返回 (task_id, 提交信息 dict)。"""
    api_key = (cfg or {}).get("api_key", "").strip()
    if not api_key:
        raise RuntimeError("Volcengine Ark API key is not configured (Settings → Video generation API)")

    # 参数按模型能力收敛（如 Seedance 2.0 mini 只有 480p/720p）
    resolution, duration, param_notes = resolve_params(cfg, resolution, duration)

    content = [{"type": "text", "text": prompt}]
    note = ""
    if image_url:
        content.append({"type": "image_url",
                        "image_url": {"url": image_url}, "role": "first_frame"})
    elif image_path:
        uri, note = encode_image(image_path)
        content.append({"type": "image_url",
                        "image_url": {"url": uri}, "role": "first_frame"})
        if last_frame_path:
            lf, _ = encode_image(last_frame_path)
            content.append({"type": "image_url",
                            "image_url": {"url": lf}, "role": "last_frame"})

    payload = {
        "model": model_id(cfg),
        "content": content,
        "resolution": resolution,
        "ratio": ratio,
        "duration": int(duration),
        "watermark": bool(watermark),
        "camera_fixed": bool(camera_fixed),
    }
    if generate_audio:
        payload["generate_audio"] = True
    if seed is not None and int(seed) >= 0:
        payload["seed"] = int(seed)

    url = api_base(cfg) + "/contents/generations/tasks"
    j = _request(url, api_key, method="POST", payload=payload, timeout=120)
    tid = j.get("id")
    if not tid:
        raise RuntimeError("Task creation failed — no task ID returned: %s" % json.dumps(j, ensure_ascii=False)[:300])
    return tid, {"payload_note": note, "model": payload["model"],
                 "duration": payload["duration"], "resolution": resolution,
                 "ratio": ratio, "param_notes": param_notes,
                 "frames": ("first+last" if (last_frame_path and image_path and not image_url)
                            else "first")}


def query_task(cfg, task_id):
    api_key = (cfg or {}).get("api_key", "").strip()
    url = api_base(cfg) + "/contents/generations/tasks/%s" % task_id
    return _request(url, api_key, method="GET", timeout=60)


def cancel_task(cfg, task_id):
    api_key = (cfg or {}).get("api_key", "").strip()
    url = api_base(cfg) + "/contents/generations/tasks/%s" % task_id
    return _request(url, api_key, method="DELETE", timeout=60)


def list_tasks(cfg, page_size=3):
    api_key = (cfg or {}).get("api_key", "").strip()
    url = api_base(cfg) + "/contents/generations/tasks?page_num=1&page_size=%d" % page_size
    return _request(url, api_key, method="GET", timeout=30)


def video_url_of(task):
    c = (task or {}).get("content") or {}
    if isinstance(c, dict):
        return c.get("video_url") or c.get("url") or ""
    if isinstance(c, list) and c:
        return (c[0] or {}).get("video_url", "")
    return ""


def wait_task(cfg, task_id, progress=None, interval=6, timeout=1800):
    """轮询直到 succeeded/failed/expired。progress(pct, msg) 回调。"""
    t0 = time.time()
    last = ""
    while True:
        if time.time() - t0 > timeout:
            raise RuntimeError("Task timed out (unfinished after %d min), task ID: %s" % (timeout // 60, task_id))
        task = query_task(cfg, task_id)
        status = task.get("status", "")
        el = time.time() - t0
        # 排队/运行阶段的进度估算（视频生成一般 1-4 分钟）
        pct = min(88, int(12 + el * 100 / 180.0))
        if progress:
            progress(pct, "Volcengine Ark task %s (waited %ds)" % (status or "Processing", int(el)))
        if status == "succeeded":
            return task
        if status in ("failed", "expired", "cancelled"):
            err = (task.get("error") or {}).get("message") or task.get("message") or status
            raise RuntimeError("Volcengine Ark task%s: %s" % (status, str(err)[:400]))
        if status and status != last:
            last = status
        time.sleep(interval)


def download(url, out_path, timeout=600):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = out_path + ".part"
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, out_path)
    return os.path.getsize(out_path)


def generate(cfg, image_path, out_dir, prompt, duration=5, resolution="720p",
             ratio="adaptive", generate_audio=False, watermark=False,
             camera_fixed=False, seed=-1, progress=None, task_id_hint=None,
             last_frame_path=None):
    """完整链路：创建任务 → 轮询 → 下载成片。返回结果 dict。

    last_frame_path 传入时启用「首尾帧」模式（Seedance 2.0 系列均支持）：
    content 中带两个 image_url，role 分别为 first_frame / last_frame，
    视频从首帧自然过渡到尾帧（如 合上外壳 → 翻开内页）。
    官方说明：首尾帧宽高比不一致时以首帧为主，尾帧自动裁剪适配。
    """
    t0 = time.time()
    tid, info = create_task(cfg, image_path, prompt, duration, resolution, ratio,
                            generate_audio, watermark, camera_fixed, seed,
                            last_frame_path=last_frame_path)
    if progress:
        progress(10, "Task created: %s — waiting for generation…" % tid)
    task = wait_task(cfg, tid, progress=progress)
    vurl = video_url_of(task)
    if not vurl:
        raise RuntimeError("Task succeeded but returned no video_url: %s"
                           % json.dumps(task, ensure_ascii=False)[:300])
    if progress:
        progress(92, "Generation complete — downloading the final cut…")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "ai_generated_%s_%s.mp4"
                       % (task_id_hint or time.strftime("%m%d%H%M%S"), tid[-6:]))
    size = download(vurl, out)
    if progress:
        progress(100, "Final cut downloaded")
    return {"task_id": tid, "file": out, "size": size,
            "video_url": vurl, "model": info["model"],
            "duration": info["duration"], "resolution": info["resolution"],
            "elapsed": round(time.time() - t0, 1),
            "param_notes": info.get("param_notes") or [],
            "frames": info.get("frames", "first"),
            "image_note": info.get("payload_note", "")}
