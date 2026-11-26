# -*- coding: utf-8 -*-
"""
质检不过 → 自动修复 → 复检（autofix）
=====================================
把 core.smart 的本地体检结论映射成修复算子，修完自动复检，
如实返回「翻盘 / 好转 / 没救回来」。

映射依据（2026-09-29 真图实测）：
  分辨率不足  → 超分 2x + 轻锐化        实测翻盘（信息只是被压小，没丢）
  偏暗        → 自适应 gamma 提亮 + CLAHE + 轻锐化
                （现有 tone 力度不够：br 24 只能救到 37.9；gamma 后 68.6 翻盘）
  过曝        → tone（力度有限，如实标注）
  轻微软糊    → USM 锐化
  真失焦/糊   → 本地锐化基本无效（实测 17.8→24.1）；
                有 AI 通道时走 enhance（实测 9.9→61.8 翻盘），没有则如实报修不了
装饰类图 / 请柬类图通用：本地算子逐像素运算，不改内容不改文字。
"""
import os
import numpy as np

from core import imgproc, smart

# 本地体检 issues 里的关键词 → 修复动作
_KEYWORDS = ("too low", "a little low", "very dark", "overexposed",
             "blurry or out of focus", "soft", "could not be read")


def _checks(path):
    loc = smart._local_checks(path)
    hard = any(k in i for i in (loc.get("issues") or [])
               for k in ("too low", "blurry", "very dark", "overexposed", "could not be read"))
    return loc, ("fail" if hard else "pass")


def _gamma_lift(img, target=120.0):
    """自适应 gamma：把平均亮度拉到 target 附近（实测暗图翻盘的关键算子）。"""
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    if mean >= target - 5 or mean <= 1:
        return img
    g = float(np.log(target / 255.0) / np.log(mean / 255.0))
    g = max(0.35, min(1.0, g))          # 只提亮不压暗，且封顶防噪声放大
    lut = np.array([min(255, int(round(255.0 * (i / 255.0) ** g))) for i in range(256)],
                   dtype=np.uint8)
    return cv2.LUT(img, lut)


def plan(issues, ai_available=False):
    """按体检结论出修复方案。返回 {ops, ai_op, notes, hopeless}。"""
    ops, ai_op, notes = [], None, []
    txt = " ".join(issues or []).lower()
    if "could not be read" in txt:
        return {"ops": [], "ai_op": None, "notes": ["File is unreadable - cannot repair."],
                "hopeless": True}
    if "too low" in txt or "a little low" in txt:
        ops += ["upscale"]
    if "very dark" in txt:
        notes.append("brighten")
    if "overexposed" in txt:
        ops += ["tone"]
        notes.append("Blown-out highlights can only be eased, not restored.")
    if "blurry or out of focus" in txt:
        if ai_available:
            ai_op = "ai_enhance"       # 实测：AI 才救得回真糊片
        else:
            ops += ["sharpen"]
            notes.append("True blur cannot be fully repaired locally - "
                         "a sharper original is best.")
    elif "soft" in txt:
        ops += ["sharpen"]
    if "too low" in txt or "a little low" in txt:
        if "sharpen" not in ops:
            ops += ["sharpen"]
    return {"ops": ops, "ai_op": ai_op, "notes": notes, "hopeless": False}


def fix(path, out_path, ai_available=False, token="", progress=None):
    """修复单张图并复检。返回：
    {out, before:{...}, after:{...}, flipped, improved, ops, ai_used, notes}
    修不好时 flipped=False 且 after 仍 fail —— 调用方据此决定提示重传。
    """
    import cv2

    def say(*a):
        if progress:
            progress(*a)

    loc_b, v_b = _checks(path)
    p = plan(loc_b.get("issues"), ai_available)
    notes = list(p["notes"])
    say(5, 100, "Planning repair")

    # 红线：任何算子都不许写回源图 —— 一律先复制到临时文件再加工
    import shutil
    work = out_path + ".work.jpg"
    shutil.copy2(path, work)
    cur = work
    if p["ops"]:
        imgproc.apply_local(cur, p["ops"], cur)
        say(35, 100, "Applying local repair")
    if "brighten" in notes:
        data = np.fromfile(cur, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if img is not None:
            img = _gamma_lift(img)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                buf.tofile(cur)
        say(55, 100, "Brightening")

    ai_used = None
    if p["ai_op"] and token:
        try:
            r = imgproc.process(cur, [p["ai_op"]], os.path.dirname(out_path), token=token)
            cur = r["out"]
            ai_used = p["ai_op"]
            say(80, 100, "AI enhancement")
        except Exception as e:
            notes.append("AI repair unavailable (%s)" % str(e)[:120])
    elif p["ai_op"] and not token:
        notes.append("AI repair is not configured on this studio.")

    # 统一落盘为 jpg
    if cur != out_path:
        data = np.fromfile(cur, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if img is None:
            from . import safedel
            safedel.unlink(work, quiet=False)
            return {"out": None, "before": loc_b, "after": loc_b, "verdict_before": v_b,
                    "verdict_after": v_b, "flipped": False, "improved": False,
                    "ops": p["ops"], "ai_used": None, "notes": notes + ["Repair failed."]}
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            buf.tofile(out_path)
    cleanup = [work]
    if cur != work and cur != out_path:
        cleanup.append(cur)          # AI 中间产物
    from . import safedel
    safedel.unlink_all(cleanup, tries=2)

    loc_a, v_a = _checks(out_path)
    n_b = len(loc_b.get("issues") or [])
    n_a = len(loc_a.get("issues") or [])
    return {"out": out_path, "before": loc_b, "after": loc_a,
            "verdict_before": v_b, "verdict_after": v_a,
            "flipped": v_b == "fail" and v_a == "pass",
            "improved": (v_a == "pass" and v_b != "pass") or n_a < n_b,
            "ops": p["ops"], "ai_used": ai_used, "notes": notes}
