# -*- coding: utf-8 -*-
"""
素材质检（SOP 第0步）
======================
自动化判定项（本地零 AI 算力）:
  1. 分辨率     长边 >= 1500px
  2. 清晰度     拉普拉斯方差（归一化到高 1080 后计算）>= 阈值
  3. 曝光       平均亮度 90~210, 过曝/欠曝像素占比 < 8%
  4. 色偏       RGB 通道均值差 <= 20
  5. 噪点       平坦区高频能量 < 阈值
人工复核项（AI 无法可靠判定, 界面提示人工确认）:
  6. 文字还原   拼写/换行/字体层级正确（红线: 有问题一律回设计源文件修正, 禁止 AI 改文字）
  7. 画面干净   水印/杂物/遮挡 + 套装完整性
"""
import os, json
import numpy as np
import cv2

CHECK_NAMES = {
    "resolution": "分辨率（长边≥1500px）",
    "sharpness": "清晰度",
    "exposure": "曝光",
    "colorcast": "色彩平衡",
    "noise": "噪点",
}
MANUAL_ITEMS = [
    "文字还原：拼写/换行/字体层级正确（若异常请回设计源文件修正，禁止用 AI 改文字）",
    "画面干净：无水印、杂物、遮挡、反光",
    "工艺表现：浮雕向上凸、烫金为金属反光、LP 为压入凹痕",
    "套装完整性：主卡/信封/RSVP/Detail 卡/垫纸齐全且同色系",
]


def _down_gray(img, h=1080):
    s = h / img.shape[0]
    w = max(1, int(img.shape[1] * s))
    g = cv2.cvtColor(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    return g


def _imread(path):
    """cv2.imread 不支持 Windows 中文路径, 用 fromfile+imdecode 替代。"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def qc_image(path):
    """返回 {file, checks:[{key,name,pass,value,detail}], verdict: A/C, reasons:[], manual:[...]}"""
    img = _imread(path)
    if img is None:
        return {"file": os.path.basename(path), "path": path, "verdict": "C",
                "reasons": ["文件无法读取（可能已损坏或格式不支持）"], "checks": [], "manual": MANUAL_ITEMS}
    h, w = img.shape[:2]
    checks = []
    # 1 分辨率
    long_edge = max(w, h)
    checks.append({"key": "resolution", "name": CHECK_NAMES["resolution"],
                   "pass": bool(long_edge >= 1500),
                   "value": "%dx%d" % (w, h),
                   "detail": "长边 %dpx, 门槛 1500px" % long_edge})
    # 2 清晰度
    g = _down_gray(img)
    lap = cv2.Laplacian(g, cv2.CV_64F).var()
    checks.append({"key": "sharpness", "name": CHECK_NAMES["sharpness"],
                   "pass": bool(lap >= 55),
                   "value": "%.0f" % lap,
                   "detail": "清晰度指数 %.0f（阈值 55, 越高越锐利）" % lap})
    # 3 曝光
    mean_l = float(g.mean())
    over = float((g > 250).mean()) * 100
    under = float((g < 8).mean()) * 100
    ok = 90 <= mean_l <= 215 and over < 8 and under < 8
    checks.append({"key": "exposure", "name": CHECK_NAMES["exposure"],
                   "pass": bool(ok), "value": "%.0f" % mean_l,
                   "detail": "平均亮度 %.0f, 过曝 %.1f%%, 欠曝 %.1f%%" % (mean_l, over, under)})
    # 4 色偏
    b, gr, r = [float(c.mean()) for c in cv2.split(img)]
    dev = max(abs(r - gr), abs(r - b), abs(gr - b))
    checks.append({"key": "colorcast", "name": CHECK_NAMES["colorcast"],
                   "pass": bool(dev <= 22), "value": "%.0f" % dev,
                   "detail": "通道均值最大差 %.0f（≤22 视为无整体偏色）" % dev})
    # 5 噪点（平坦区高频能量）
    blur = cv2.GaussianBlur(g, (0, 0), 1.4)
    hf = np.abs(g.astype(np.float32) - blur.astype(np.float32))
    # 取高频能量最低的 20% 区域估算本底噪声
    flat = np.sort(hf.ravel())[: max(1, int(hf.size * 0.2))]
    noise = float(flat.mean())
    checks.append({"key": "noise", "name": CHECK_NAMES["noise"],
                   "pass": bool(noise <= 6.0), "value": "%.1f" % noise,
                   "detail": "本底噪声 %.1f（≤6.0 合格；偏高建议去噪）" % noise})
    reasons = [c["detail"] for c in checks if not c["pass"]]
    verdict = "A" if not reasons else "C"
    return {"file": os.path.basename(path), "path": path, "size": [w, h],
            "checks": checks, "verdict": verdict, "reasons": reasons,
            "manual": MANUAL_ITEMS}


def qc_project(files):
    results = [qc_image(f) for f in files]
    n_pass = sum(1 for r in results if r["verdict"] == "A")
    return {"results": results, "pass": n_pass, "total": len(results)}
