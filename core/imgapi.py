# -*- coding: utf-8 -*-
"""
图片生成 API 直连（火山方舟 / Volcengine Ark · Doubao-Seedream 系列）
=====================================================================
用途：步骤5「手持展示分镜图」——把产品实拍图喂给 Seedream，一次产出多张
      「真人双手手持单卡」的竖版分镜图（组图模式），供步骤6 图生视频当首帧。

官方接口（截至 2026-09）:
  生成      POST {base}/images/generations      （同步返回，非任务制）
  鉴权      Authorization: Bearer <ARK_API_KEY>
base 默认 https://ark.cn-beijing.volces.com/api/v3

请求体关键字段
--------------
model                           模型 ID，见 IMAGE_MODELS
prompt                          指令词
image                           参考图：URL 或 data:image/xxx;base64,...（**必须小写 mime**）
                                单值或数组，5.0-lite 最多 14 张
size                            预设档位 2K / 3K / 4K，或显式 "宽x高"
sequential_image_generation     auto = 组图（模型自行判断张数）/ disabled = 单图
sequential_image_generation_options.max_images   1~15
                                **约束：输入参考图数 + 生成图片数 ≤ 15**
response_format                 url（24h 有效，需及时转存）/ b64_json
output_format                   仅 5.0 系列可自定义，jpeg / png
watermark                       默认 true（带「AI生成」水印），本项目默认 false
optimize_prompt_options         {"mode": "standard"|"fast"}，5.0-lite 仅 standard

响应：{model, created, data:[{url|b64_json, size}], usage:{generated_images,...}}
"""

import base64
import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_IMAGE_MODEL = "doubao-seedream-5-0-260128"      # Seedream 5.0 Lite 主 ID

# 常用模型能力（size 档位 / 是否支持组图 / 参考图上限）
IMAGE_MODELS = [
    {"id": "doubao-seedream-5-0-260128", "name": "Seedream 5.0 Lite (recommended · image sets + up to 4K)",
     "alias": "doubao-seedream-5-0-lite-260128",
     "sizes": ["2K", "3K", "4K"], "sequential": True, "ref_max": 14, "out_fmt": ["jpeg", "png"]},
    {"id": "doubao-seedream-5-0-flash-260915", "name": "Seedream 5.0 Flash (faster, cheaper)",
     "sizes": ["2K", "3K", "4K"], "sequential": True, "ref_max": 14, "out_fmt": ["jpeg", "png"]},
    {"id": "doubao-seedream-5-0-pro-260628", "name": "Seedream 5.0 Pro (fine single image · 1K/2K)",
     "sizes": ["1K", "2K"], "sequential": False, "ref_max": 10, "out_fmt": ["jpeg", "png"]},
    {"id": "doubao-seedream-4-5-251128", "name": "Seedream 4.5 (2K/4K · stable)",
     "sizes": ["2K", "4K"], "sequential": True, "ref_max": 14, "out_fmt": ["jpeg"]},
    {"id": "doubao-seedream-4-0-250828", "name": "Seedream 4.0 (1K/2K/4K · classic)",
     "sizes": ["1K", "2K", "4K"], "sequential": True, "ref_max": 14, "out_fmt": ["jpeg"]},
]

MAX_REF_BYTES = 9 * 1024 * 1024      # 单张参考图 ≤10MB，留余量
MAX_REF_SIDE = 4096                  # 参考图总像素 ≤ 6000x6000

_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".webp": "image/webp", ".bmp": "image/bmp", ".tif": "image/tiff",
         ".tiff": "image/tiff", ".gif": "image/gif", ".heic": "image/heic",
         ".heif": "image/heif"}


# ---------------- HTTP 基础（与 videoapi 同构，独立以便单独演进）----------------

def _request(url, api_key, method="GET", payload=None, timeout=600):
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
            err = j.get("error") or {}
            msg = err.get("message") or j.get("message") or raw
            code = err.get("code") or j.get("code") or ""
        except Exception:
            msg, code = raw, ""
        hint = ""
        if e.code == 401:
            hint = "(invalid API key, or the image generation model is not enabled)"
        elif e.code == 404:
            hint = "(model or endpoint missing, or not enabled — check the model ID and your account's activation status)"
        elif e.code == 429:
            hint = "(rate limited, or insufficient balance)"
        elif e.code == 400:
            hint = "(invalid parameters — usually the size tier, image count, or too many reference images)"
        raise RuntimeError("Volcengine Ark image generation HTTP %s: %s %s%s"
                           % (e.code, code, str(msg)[:400], hint))
    except urllib.error.URLError as e:
        raise RuntimeError("Cannot reach Volcengine Ark: %s (check your network or proxy)" % e.reason)


def api_base(cfg):
    ep = ((cfg or {}).get("endpoint") or "").strip()
    if not ep:
        return DEFAULT_BASE
    for suffix in ("/images/generations", "/images/generations/"):
        if ep.endswith(suffix):
            ep = ep[: -len(suffix)]
            break
    return ep.rstrip("/")


def model_id(cfg):
    return ((cfg or {}).get("model") or "").strip() or DEFAULT_IMAGE_MODEL


def meta(mid=None):
    mid = mid or DEFAULT_IMAGE_MODEL
    for m in IMAGE_MODELS:
        if m["id"] == mid or m.get("alias") == mid:
            return m
    return IMAGE_MODELS[0]


def is_ready(cfg):
    return bool((cfg or {}).get("api_key", "").strip())


# ---------------- 账号模型可用性（零成本）----------------
# 方舟提供 GET {base}/models，返回该账号可见的全部模型及其 status。
# 这是核对「模型 ID 是否正确 + 是否已开通」最可靠且完全免费的方式。

def list_models(cfg, timeout=30):
    api_key = ((cfg or {}).get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError("Enter ARK_API_KEY first")
    j = _request(api_base(cfg) + "/models", api_key, method="GET", timeout=timeout)
    return j.get("data") or []


def check_model(cfg, mid):
    """在账号模型列表中核对指定模型。返回 found/status/warning/task_type。"""
    mid = (mid or "").strip()
    for m in list_models(cfg):
        if m.get("id") != mid:
            continue
        st = (m.get("status") or "").lower()
        warn = ""
        if st == "shutdown":
            warn = "This model has been shut down and can no longer be called"
        elif st == "retiring":
            warn = "The vendor has announced this model is retiring — migrate soon"
        return {"found": True, "id": mid, "status": m.get("status") or "Available",
                "name": m.get("name"), "task_type": m.get("task_type"),
                "modalities": m.get("modalities"), "warning": warn}
    return {"found": False, "id": mid, "status": "NotAccessible",
            "warning": "Model not found in the account's model list: the ID may be wrong, or access is not enabled yet"}


def normalize_size(cfg, size):
    """把用户/前端的 size 收敛到当前模型支持的档位。"""
    m = meta(model_id(cfg))
    size = (size or "").strip() or (cfg or {}).get("size") or "4K"
    if size in m["sizes"]:
        return size
    # 显式像素值直接放行（模型端会校验范围）
    if "x" in size.lower():
        return size
    return m["sizes"][-1]      # 取该模型最高档


# ---------------- 参考图内联 ----------------

def encode_image(path, max_side=MAX_REF_SIDE):
    """读参考图 → data URI。超限自动等比压缩，返回 (uri, note)。"""
    ext = os.path.splitext(path)[1].lower()
    mime = _MIME.get(ext, "image/jpeg")
    size = os.path.getsize(path)
    raw = open(path, "rb").read()
    note = ""

    if (size > MAX_REF_BYTES or max_side) and mime in (
            "image/jpeg", "image/png", "image/webp", "image/bmp"):
        try:
            import cv2
            import numpy as np
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                scale = min(1.0, float(max_side) / max(h, w)) if max_side else 1.0
                if size > MAX_REF_BYTES:
                    scale = min(scale, (MAX_REF_BYTES * 0.85 / size) ** 0.5)
                if scale < 0.999:
                    img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_AREA)
                    note = "Reference image compressed %d×%d→%d×%d" % (w, h, img.shape[1], img.shape[0])
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok:
                    raw, mime = buf.tobytes(), "image/jpeg"
        except Exception:
            pass

    if len(raw) > MAX_REF_BYTES:
        raise RuntimeError("Reference image exceeds the 10MB limit (%0.1fMB) — compress it first or process it in step 3"
                           % (len(raw) / 1048576.0))
    return "data:%s;base64,%s" % (mime.lower(), base64.b64encode(raw).decode("ascii")), note


# ---------------- 生成 ----------------

def build_payload(cfg, prompt, image_uris, size=None, max_images=3,
                  sequential="auto", output_format="jpeg", watermark=False,
                  optimize_prompt=None):
    m = meta(model_id(cfg))
    payload = {
        "model": model_id(cfg),
        "prompt": prompt,
        "size": normalize_size(cfg, size),
        "response_format": "url",
        "watermark": bool(watermark),
    }
    if image_uris:
        payload["image"] = image_uris[0] if len(image_uris) == 1 else list(image_uris)
    # 组图（仅 5.0-lite/4.5/4.0 支持）
    if m["sequential"] and max_images and int(max_images) > 1:
        payload["sequential_image_generation"] = "auto" if sequential != "disabled" else "disabled"
        if payload["sequential_image_generation"] == "auto":
            refs = len(image_uris or [])
            payload["sequential_image_generation_options"] = {
                "max_images": max(1, min(15 - refs, int(max_images)))}
    # 输出格式仅 5.0 系列可自定义
    if "png" in m["out_fmt"]:
        payload["output_format"] = output_format if output_format in m["out_fmt"] else "jpeg"
    if optimize_prompt:
        payload["optimize_prompt_options"] = {"mode": optimize_prompt}
    return payload


def generate(cfg, out_dir, prompt, ref_images=None, size="4K", max_images=3,
             sequential="auto", output_format="jpeg", watermark=False,
             optimize_prompt=None, progress=None, tag=None, download=True):
    """一次组图调用：参考图 + 指令词 → 多张分镜图（已下载到 out_dir）。

    ref_images: 参考图路径列表（≤14 张）
    返回 {files:[...], urls:[...], note:str, elapsed:float, size:str, model:str, count:int}
    """
    api_key = ((cfg or {}).get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError("ARK_API_KEY is not configured (Settings → Storyboard API)")
    if not (prompt or "").strip():
        raise RuntimeError("The prompt cannot be empty")

    m = meta(model_id(cfg))
    refs = list(ref_images or [])
    if len(refs) > m["ref_max"]:
        refs = refs[: m["ref_max"]]

    t0 = time.time()
    uris, notes = [], []
    for i, p in enumerate(refs):
        uri, nt = encode_image(p)
        uris.append(uri)
        if nt:
            notes.append(nt)
        if progress:
            progress(5 + int(15 * (i + 1) / max(len(refs), 1)),
                     "Encoding reference %d/%d" % (i + 1, len(refs)))

    payload = build_payload(cfg, prompt, uris, size, max_images, sequential,
                            output_format, watermark, optimize_prompt)
    note = "；".join(notes)
    if progress:
        progress(22, "Submitted to Seedream %s (%s)…" % (payload["model"], payload["size"]))
    j = _request(api_base(cfg) + "/images/generations", api_key,
                 method="POST", payload=payload, timeout=900)
    items = j.get("data") or []
    if not items:
        raise RuntimeError("No image returned: %s" % json.dumps(j, ensure_ascii=False)[:300])
    if progress:
        progress(70, "%d image(s) generated — downloading…" % len(items))

    out_dir = out_dir or "."
    os.makedirs(out_dir, exist_ok=True)
    ext = ".png" if payload.get("output_format") == "png" else ".jpg"
    prefix = tag or time.strftime("storyboard_%m%d%H%M%S")
    urls, files = [], []
    for i, it in enumerate(items):
        url = it.get("url") or ""
        if url:
            urls.append(url)
            if download:
                fp = os.path.join(out_dir, "%s_%02d%s" % (prefix, i + 1, ext))
                download_file(url, fp)
                files.append(fp)
        elif it.get("b64_json"):
            fp = os.path.join(out_dir, "%s_%02d%s" % (prefix, i + 1, ext))
            with open(fp, "wb") as f:
                f.write(base64.b64decode(it["b64_json"]))
            files.append(fp)
        if progress:
            progress(70 + int(28 * (i + 1) / len(items)),
                     "Downloaded %d/%d" % (i + 1, len(items)))

    return {"files": files, "urls": urls, "count": len(items), "note": note,
            "size": payload["size"], "model": payload["model"],
            "elapsed": round(time.time() - t0, 1),
            "usage": j.get("usage") or {}}


def download_file(url, out_path, timeout=600):
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


# ---------------- 免费连通性探测 ----------------

def probe(cfg):
    """零成本探测：不生成任何图片。

    做法：用合法凭据发一个「模型名非法」的请求。
    方舟的校验顺序是 **先鉴权、后模型**，因此：
      · 401          → API Key 无效 / 未开通
      · 404 / 400    → 鉴权已通过、端点可达（服务端拒绝的是探测用的非法模型名，属预期）
    不产生计费，也不留下任务。
    """
    api_key = ((cfg or {}).get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError("Enter ARK_API_KEY first")
    try:
        _request(api_base(cfg) + "/images/generations", api_key, method="POST",
                 payload={"model": "__auth_probe__", "prompt": "x"}, timeout=30)
        return {"ok": True, "detail": "Endpoint responding normally"}
    except RuntimeError as e:
        s = str(e)
        if "HTTP 401" in s:
            raise RuntimeError("Invalid API key, or the image generation model is not enabled: %s" % s)
        if "HTTP 404" in s or "HTTP 400" in s:
            # 鉴权已通过（坏 Key 会先返回 401），404 是探测用的非法模型名所致
            return {"ok": True,
                    "detail": "Authentication passed · the server confirms the probe model does not exist (expected), so "
                              "both key and endpoint work"}
        raise
