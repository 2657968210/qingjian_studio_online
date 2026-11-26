# -*- coding: utf-8 -*-
"""
图片处理（不合格素材补救）
===========================
本地算子（零 AI 算力）:
  upscale  超分放大 2x（Lanczos + 振铃抑制锐化）
  denoise  去噪（非局部均值）
  sharpen  锐化（USM）
  wb       白平衡（灰世界）
  tone     自动对比度/亮度（CLAHE 限制对比度）
  crop     中心/自定义裁切（percent: [x0,y0,x1,y1] 0-100）
云端算子（需在 config.json 配置 cloud_token, 消耗平台算力）:
  ai_enhance  保真增强      ai_restore  老片修复      ai_erase  去水印/杂物
"""
import os, sys, json, subprocess
import numpy as np
import cv2

_LOCAL_OPS = ["upscale", "denoise", "sharpen", "wb", "tone", "crop"]
_AI_OPS = ["ai_enhance", "ai_restore", "ai_erase"]
LOCAL_OPS = list(_LOCAL_OPS)   # 公开别名：步骤2 的「本地降级通道」以此为准


def _imread(path):
    """cv2.imread 不支持 Windows 中文路径, 用 fromfile+imdecode 替代。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _imwrite(path, img):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        buf.tofile(path)
    return ok


def normalize_luma(img, lo=106.0, hi=170.0, target=124.0, target_hi=150.0):
    """自适应亮度归一（2026-10-27 用户需求：确保每张图的亮度一致）。

    低光手机照（如室内闪粉请柬）直接渲染会比其它图明显偏暗；这里对整图做
    自适应 gamma，把平均亮度拉进一致的观感档：
      平均亮度 < lo → gamma 提亮到 target 附近（封顶 0.35，防噪声放大）
      平均亮度 > hi → gamma 压暗到 target_hi 附近（封顶 1.5，防发灰）
      中间带原样返回（避免无意义重处理与画质损失）
    gamma 是逐像素 LUT，通道序无关——BGR/RGB/带 alpha 的图都能用。
    处理失败一律原样返回，绝不影响渲染主链路。
    """
    try:
        if img is None or img.ndim < 2 or img.dtype != np.uint8:
            return img
        g = img if img.ndim == 2 else cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        mean = float(g.mean())
        if mean <= 1.0 or lo <= mean <= hi:
            return img
        tgt = target if mean < lo else target_hi
        gamma = float(np.log(tgt / 255.0) / np.log(mean / 255.0))
        gamma = max(0.35, min(1.5, gamma))
        lut = np.array([min(255, int(round(255.0 * (i / 255.0) ** gamma)))
                        for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, lut)
    except BaseException:
        return img


def _usm(img, amount=0.6, radius=1.6):
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def apply_local(path, ops, out_path, crop_percent=None):
    img = _imread(path)
    if img is None:
        raise ValueError("Could not read the image: %s" % path)
    log = []
    for op in ops:
        if op == "upscale":
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
            img = _usm(img, 0.35, 1.2)
            log.append("Upscale 2×")
        elif op == "denoise":
            img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
            log.append("Denoise")
        elif op == "sharpen":
            img = _usm(img, 0.7, 1.8)
            log.append("Sharpen")
        elif op == "wb":
            result = img.astype(np.float32)
            means = result.reshape(-1, 3).mean(axis=0)
            gray = means.mean()
            gains = gray / np.maximum(means, 1e-3)
            gains = np.clip(gains, 0.85, 1.18)
            img = np.clip(result * gains[None, None, :], 0, 255).astype(np.uint8)
            log.append("White balance")
        elif op == "tone":
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
            img = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
            log.append("Auto contrast")
        elif op == "crop":
            h, w = img.shape[:2]
            cp = crop_percent or [5, 5, 95, 95]
            x0, y0, x1, y1 = [int(float(v) / 100 * (w if i % 2 == 0 else h))
                              for i, v in enumerate(cp)]
            x0, x1 = sorted((max(0, x0), min(w, x1)))
            y0, y1 = sorted((max(0, y0), min(h, y1)))
            if x1 - x0 > 10 and y1 - y0 > 10:
                img = img[y0:y1, x0:x1]
                log.append("Crop %s%%" % cp)
    _imwrite(out_path, img)
    return {"out": out_path, "ops": log, "size": [img.shape[1], img.shape[0]]}


def _run_ai(operation, path, out_path, token, prompt=""):
    """调用 buddy-image-processing 云端算子。

    注意：脚本不提供 --output-dir（传了会直接参数报错），结果文件落在
    当前工作目录（processed_image_*），这里从 stdout 的 JSON 里拿真实路径再搬走。
    """
    script = (r"E:\workbuddy\resources\app.asar.unpacked\resources\plugins"
              r"\workbuddy-builtin\skills\buddy-image-processing\scripts"
              r"\buddy-image-processing.py")
    prompts = {
        "ai_enhance": "Enhance image fidelity and sharpness, keep all text and patterns unchanged",
        "ai_restore": "Restore and repair image, remove compression artifacts, keep text unchanged",
        "ai_erase": "Remove watermarks and unwanted objects, fill background naturally",
    }
    cmd = [sys.executable, script, "image-edit", "--operation",
           {"ai_enhance": "enhance", "ai_restore": "restore", "ai_erase": "erase"}[operation],
           "--image-file", path,
           "--prompt", prompt or prompts[operation],
           "--token", token]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="ignore", timeout=900, cwd=os.path.dirname(out_path) or None)
    if r.returncode != 0:
        raise RuntimeError("Cloud processing failed: %s" % (r.stdout or r.stderr)[-400:])
    # stdout 末尾是一段 JSON，里面有 result_files[].path
    src = None
    txt = r.stdout or ""
    i = txt.find("{")
    if i >= 0:
        try:
            j = json.loads(txt[i:])
            files = (j.get("result_files") or [])
            if files and files[0].get("path"):
                src = files[0]["path"]
        except Exception:
            src = None
    if not src or not os.path.isfile(src):
        # 兜底：找 cwd 里最新的 processed_image_*
        import glob, time as _t
        cand = sorted(glob.glob(os.path.join(os.path.dirname(out_path) or ".",
                                             "processed_image_*")),
                      key=os.path.getmtime)
        if not cand:
            raise RuntimeError("Cloud processing returned no file")
        src = cand[-1]
    import shutil
    shutil.move(src, out_path)
    return out_path


def process(path, ops, out_dir, token="", crop_percent=None):
    """主入口。返回 {out, ops, ai_used}"""
    base = os.path.splitext(os.path.basename(path))[0]
    stamp = str(int(__import__("time").time()))
    local_ops = [o for o in ops if o in _LOCAL_OPS]
    ai_ops = [o for o in ops if o in _AI_OPS]
    cur = path
    results = {"ops": []}
    if local_ops:
        out1 = os.path.join(out_dir, "%s_%s_local.jpg" % (base, stamp))
        r = apply_local(cur, local_ops, out1, crop_percent)
        results["ops"] += r["ops"]
        cur = out1
    if ai_ops:
        if not token:
            raise RuntimeError("Cloud AI processing requires cloud_token in config.json (currently missing or expired)."
                               "You can fall back to the local operators.")
        out2 = os.path.join(out_dir, "%s_%s_ai.jpg" % (base, stamp))
        for op in ai_ops:
            _run_ai(op, cur, out2, token)
            results["ops"].append(op)
            cur = out2
    results["out"] = cur
    return results
