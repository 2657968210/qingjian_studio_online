# -*- coding: utf-8 -*-
"""主体分割抠图（u2netp + OpenCV DNN）。

2026-11-25 第 8 版新增。客户口径「抠图不行」的根因是 GrabCut 在
「浅色产品 + 同色调背景」上无法收敛（实测蛋糕图覆盖率 0.69~0.80，
等于把背景整块留下）。改用 U^2-Net portable（4.5MB，Apache-2.0）做
显著主体分割，并且**先按视觉模型给的产品框裁剪再推理**——主体在裁剪图里
占比大，模型输出就是干净蛋糕/婚纱/请柬纸品。

依赖只有 opencv（cv2.dnn），不需要 onnxruntime / torch。
模型缺失、加载失败、推理异常一律返回 None，调用方回退 GrabCut。
"""
import os
import shutil
import tempfile
import threading

import numpy as np
import cv2

_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255.0
_STD = np.array([0.229, 0.224, 0.225], np.float32) * 255.0
_SIZE = 320

_NET = {"net": None, "state": "init"}      # init / ok / fail
_LOCK = threading.Lock()                   # cv2.dnn 的 setInput/forward 非线程安全
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(os.path.dirname(_HERE), "models", "u2netp.onnx")


def _load_net(path):
    # cv2.dnn 在 Windows 中文路径下会「Can't read ONNX file」，
    # 先用原路径，失败则复制到 ASCII 临时目录再读（部署环境是 Linux，不受影响）。
    try:
        return cv2.dnn.readNet(path)
    except Exception:
        try:
            tmp = os.path.join(tempfile.gettempdir(), "u2netp.onnx")
            if not os.path.isfile(tmp) or os.path.getsize(tmp) != os.path.getsize(path):
                shutil.copyfile(path, tmp)
            return cv2.dnn.readNet(tmp)
        except Exception:
            return None


def net():
    """惰性加载模型；失败只记一次状态，不抛异常。"""
    if _NET["state"] == "ok":
        return _NET["net"]
    if _NET["state"] == "fail":
        return None
    with _LOCK:
        if _NET["state"] == "ok":
            return _NET["net"]
        if _NET["state"] == "fail":
            return None
        _NET["state"] = "fail"
        try:
            if not os.path.isfile(_MODEL_PATH):
                return None
            n = _load_net(_MODEL_PATH)
            if n is None:
                return None
            _NET["net"] = n
            _NET["state"] = "ok"
            return n
        except Exception:
            return None


def available():
    return net() is not None


def _forward(n, bgr):
    h, w = bgr.shape[:2]
    inp = cv2.resize(bgr, (_SIZE, _SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    inp = (inp - _MEAN) / _STD
    blob = cv2.dnn.blobFromImage(inp, 1.0, (_SIZE, _SIZE), (0, 0, 0),
                                 swapRB=True, crop=False)
    with _LOCK:                 # 同一 net 实例并发推理会串输入，必须串行
        n.setInput(blob)
        out = n.forward()
    m = out.reshape(out.shape[-2], out.shape[-1]).astype(np.float32)
    lo, hi = float(m.min()), float(m.max())
    m = (m - lo) / max(hi - lo, 1e-6)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC)


def _segment_crop(bgr, box=None, pad=0.12):
    """按产品框裁剪后推理，返回 (alpha, (X0,Y0,X1,Y1))。"""
    n = net()
    if n is None or bgr is None:
        return None, None
    try:
        h, w = bgr.shape[:2]
        if box is not None and len(box) == 4:
            x0, y0, x1, y1 = [float(v) for v in box]
            gx, gy = (x1 - x0) * pad, (y1 - y0) * pad
            X0 = max(0, int((x0 - gx) * w)); X1 = min(w, int(np.ceil((x1 + gx) * w)))
            Y0 = max(0, int((y0 - gy) * h)); Y1 = min(h, int(np.ceil((y1 + gy) * h)))
            if X1 - X0 < 16 or Y1 - Y0 < 16:
                X0, Y0, X1, Y1 = 0, 0, w, h
        else:
            X0, Y0, X1, Y1 = 0, 0, w, h
        crop = bgr[Y0:Y1, X0:X1]
        m = _forward(n, crop)
        full = np.zeros((h, w), np.float32)
        full[Y0:Y1, X0:X1] = m
        return full, (X0, Y0, X1, Y1)
    except Exception:
        return None, None


def segment(bgr, box=None, pad=0.12):
    """返回与原图同尺寸的 alpha（float32 0..1）；失败返回 None。

    box 是视觉模型给的归一化产品框 [x0,y0,x1,y1]。给了就裁到框（外扩 pad）
    再推理，主体占比更大、背景更少，实测蛋糕图从「只抠出花」变成「完整三层
    蛋糕 + 花艺」。
    """
    a, _rect = _segment_crop(bgr, box=box, pad=pad)
    return a


def _clean(alpha, box=None):
    """低阈值取主连通块 → 填洞 → 得到实心基础掩膜。

    注意 u2netp 在大面积浅色主体（蛋糕侧面、纸卡内页）上的输出并不是
    均匀高置信：内部常出现 0.2~0.4 的低值区，直接按 0.45 阈值二值化会把
    主体打成蜂窝。所以用低阈值（0.22）取主体轮廓，再填洞补实内部；
    真正的软边只在轮廓带附近用原始 alpha 过渡。
    """
    h, w = alpha.shape
    m = (alpha > 0.22).astype(np.uint8)
    if m.sum() == 0:
        return None
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    items = sorted([(st[i, cv2.CC_STAT_AREA], i) for i in range(1, n)], reverse=True)
    keep = {items[0][1]}
    bx0 = by0 = 0; bx1 = w; by1 = h
    if box is not None:
        bx0, by0, bx1, by1 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
    main = items[0][1]
    mx, my = st[main, cv2.CC_STAT_LEFT], st[main, cv2.CC_STAT_TOP]
    mw, mh = st[main, cv2.CC_STAT_WIDTH], st[main, cv2.CC_STAT_HEIGHT]
    for area, i in items[1:]:
        if area < items[0][0] * 0.10:
            continue
        x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
        ww, hh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        near = (x <= mx + mw * 1.15 and x + ww >= mx - mw * 0.15 and
                y <= my + mh * 1.15 and y + hh >= my - mh * 0.15)
        inside = (x >= bx0 - 4 and y >= by0 - 4 and x + ww <= bx1 + 4 and y + hh <= by1 + 4)
        if near and inside:
            keep.add(i)
    out = np.isin(lab, list(keep)).astype(np.uint8)
    # 填洞（洞面积上限放到 40%——浅色主体内部大片低置信区都要补实）
    nh, hl = cv2.connectedComponents((out == 0).astype(np.uint8), 4)
    for i in range(1, nh):
        mm = hl == i
        if (mm[:2, :].any() or mm[-2:, :].any() or mm[:, :2].any() or mm[:, -2:].any()):
            continue
        if mm.sum() < h * w * 0.40:
            out[mm] = 1
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return out


def _touches_edge(mask, rect, shape):
    """掩膜是否贴住裁剪边界（=主体可能被框裁断）。"""
    h, w = shape[:2]
    X0, Y0, X1, Y1 = rect
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return False
    return bool((xs.min() <= X0 + 2 and X0 > 0) or (ys.min() <= Y0 + 2 and Y0 > 0) or
                (xs.max() >= X1 - 3 and X1 < w) or (ys.max() >= Y1 - 3 and Y1 < h))


def _seed_mask(alpha):
    """低阈值取最大连通块 → 大核闭运算 → 填洞，得到实心主体掩膜。

    u2netp 在浅色主体内部常出 0.2~0.4 的低置信区，直接二值化会把蛋糕/纸卡
    打成蜂窝（客户截图里的「抠图不行」就是这个）。这里用 0.18 低阈值取轮廓，
    再用按图像尺寸缩放的核补实内部。
    """
    h, w = alpha.shape
    m = (alpha > 0.18).astype(np.uint8)
    if m.sum() == 0:
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = (lab == big).astype(np.uint8)
    k = max(9, int(min(h, w) * 0.025) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    nh, hl = cv2.connectedComponents((m == 0).astype(np.uint8), 4)
    for i in range(1, nh):
        mm = hl == i
        if (mm[:2, :].any() or mm[-2:, :].any() or mm[:, :2].any() or mm[:, -2:].any()):
            continue
        if mm.sum() < h * w * 0.40:
            m[mm] = 1
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))


def cutout(bgr, box=None, stroke=0):
    """分割 + 修边，返回 (rgba, score)。失败返回 (None, 0.0)。

    stroke > 0 时在返回前加白描边（与 GrabCut 路径同一套描边口径）。

    多裁剪候选（pad 0.06 / 0.12 / 0.20）+ 「取最大的不贴边主体」：
      * 紧裁剪让主体占满画面，u2netp 才不会把旁边花艺/道具当成主体；
      * 但框偏紧时主体会被裁断（贴住裁剪边界），所以同时跑松裁剪；
      * 主体是画面里最大的显著物体，用「面积最大且不贴边」选，能稳定命中
        蛋糕/婚纱整体，而不是花瓣这类高对比子物体。
    """
    cands = []
    for pad in (0.06, 0.12, 0.20):
        a, rect = _segment_crop(bgr, box=box, pad=pad)
        if a is None:
            continue
        m = _seed_mask(a)
        if m is None:
            continue
        cov = float(m.mean())
        if cov < 0.015 or cov > 0.92:
            continue
        cands.append((_touches_edge(m, rect, bgr.shape), float(m.sum()), a, m, rect, cov))
    if not cands:
        return None, 0.0
    free = [c for c in cands if not c[0]]
    a, binm, rect, cov = max(free or cands, key=lambda c: c[1])[2:]
    # 软边：内部按掩膜补实，轮廓带用原始 alpha 做 1~2px 过渡
    band = np.clip((cv2.GaussianBlur(a, (0, 0), 1.0) - 0.30) / 0.35, 0, 1)
    soft = np.maximum(band * 255.0, binm.astype(np.float32) * 250.0)
    soft = np.clip(soft, 0, 255).astype(np.uint8)
    soft[binm == 0] = 0
    ys, xs = np.where(soft > 40)
    if len(ys) == 0:
        return None, 0.0
    rgba = np.dstack([bgr, soft])[max(0, ys.min() - 4):ys.max() + 5,
                                  max(0, xs.min() - 4):xs.max() + 5]
    if stroke:
        from core.moodboard_showcase import _with_stroke
        rgba = _with_stroke(rgba, stroke)
    return rgba, round(cov, 3)
