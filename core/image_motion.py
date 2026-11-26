# -*- coding: utf-8 -*-
"""image_motion.py —— 图像感知运镜（零 AI，纯 OpenCV）

职责：
  1) 显著性能量图（Sobel 梯度 + 饱和度）
  2) 纸品 / 文字区域检测（桌卡 / 菜单 / sign / 请柬 等婚礼纸品）
     —— 用「高细节 + 高印刷密度 + 近似矩形 + 与背景分离」作为代理信号
  3) 按图特征自动选运镜类型（push / panL / panR / diag / rot）
  4) 主体居中取窗：把取景窗口中心钉在识别出的主体上

说明：真正的「分类这张到底是 menu 还是 invitation」需要视觉模型；
本模块用 heuristic 把「印刷密集的矩形纸状物」识别出来并优先作为运镜中心，
这恰好就是用户要的「桌卡/菜单/sign/请柬 优先居中」的可操作流程。
后续要精确分类，可在 analyze() 里把 detect_paper 换成视觉模型调用即可。
"""
import numpy as np
import cv2

# ---------------- 底层特征图 ----------------
def _energy_map(img, size=480):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(img, (sw, sh))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1]
    mag = mag / (mag.max() + 1e-6)
    sat = sat / 255.0
    energy = 0.7 * mag + 0.3 * sat
    sX = max(1.0, sw * 0.02)
    sY = max(1.0, sh * 0.02)
    energy = cv2.GaussianBlur(energy, (0, 0), sigmaX=sX, sigmaY=sY)
    return energy, (sw, sh)


def _print_density(img, size=480):
    """高印刷密度图：文字 / 线条 / 印刷元素 的代理（桌卡/菜单/sign/请柬 普遍密集）。"""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(img, (sw, sh))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    fine = np.sqrt(gx * gx + gy * gy)
    lap = lap / (lap.max() + 1e-6)
    fine = fine / (fine.max() + 1e-6)
    density = 0.65 * lap + 0.35 * fine
    sX = max(2.0, sw * 0.012)
    sY = max(2.0, sh * 0.012)
    density = cv2.GaussianBlur(density, (0, 0), sigmaX=sX, sigmaY=sY)
    return density, (sw, sh)


def _text_map(img, size=768):
    """细笔画过剩度图（升级版：治奶油/低对比调）。

    三步：
      1) 局部对比度归一化（同态）：(gray - blur(gray)) / (blur(gray) + 8)，
         把整体窄动态范围的奶油调上的细节提出来（旧版直接 Laplacian 会被
         全局 max 归一化压平，导致低对比纸品文字响应不足而漏检）；
      2) 多尺度高频：Laplacian ksize=3/5 加权，兼顾小字与粗标题；
      3) 高频过剩度：高于邻域平均的部分才是「笔画」，大面积柔纹理不算。

    返回 (textness, (sw, sh), gray_small)。gray_small 供方向/台阶判据用。
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(img, (sw, sh))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(4.0, sw * 0.035), sigmaY=max(4.0, sh * 0.035))
    norm = (gray - bg) / (bg + 8.0)
    lap3 = np.abs(cv2.Laplacian(norm, cv2.CV_32F, ksize=3))
    lap5 = np.abs(cv2.Laplacian(norm, cv2.CV_32F, ksize=5))
    fine = 0.55 * lap3 + 0.45 * lap5
    broad = cv2.GaussianBlur(fine, (0, 0), sigmaX=max(3.0, sw * 0.02), sigmaY=max(3.0, sh * 0.02))
    textness = np.clip(fine - 0.8 * broad, 0, None)
    textness = cv2.GaussianBlur(textness, (0, 0), sigmaX=2.0, sigmaY=2.0)
    return textness, (sw, sh), gray


def _edge_contrast(gray, x, y, w, h, m=8):
    """纸品与背景的亮度台阶：bbox 外环均值 vs 内部均值差（相对值）。
    卡片浮在场景上 -> 台阶明显；均质桌面/泳池/海景 -> 台阶小。
    这是排除「横竖结构丰密的自然场景」误报的核心判据。"""
    H, W = gray.shape
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(W, x + w + m), min(H, y + h + m)
    ox, oy = x - x0, y - y0
    outer = gray[y0:y1, x0:x1]
    mask = np.ones(outer.shape, bool)
    mask[oy:oy + h, ox:ox + w] = False
    in_mean = float(gray[y:y + h, x:x + w].mean())
    out_mean = float(outer[mask].mean()) if mask.any() else in_mean
    return abs(in_mean - out_mean) / (out_mean + 8.0)


def _orient_score(gray, x, y, w, h):
    """区域内强边缘的方向集中度（0~1）：印刷文字横竖笔画 -> 接近 1；
    自然纹理方向杂乱 -> 接近 0。仅作诊断输出，不作为闸门（粗体标题/数字
    如桌号"2"是曲线笔画，方向集中度天然低，会误伤真纸品）。"""
    if w < 6 or h < 6:
        return 0.0
    roi = gray[y:y + h, x:x + w].astype(np.float32)
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    sel = mag > max(np.percentile(mag, 75), 1e-3)
    if int(sel.sum()) < 30:
        return 0.0
    ang = np.arctan2(gy[sel], gx[sel])
    cardinal = np.abs(np.sin(2.0 * ang))
    return float(1.0 - cardinal.mean())


# ---------------- 主体 / 纸品检测 ----------------
def _best_window(score, sw, sh, aspects, size_fracs, ng=14):
    """在 score 图上用积分图滑窗，找均值最高的窗口（限幅在画面内）。
    位置按窗口尺寸动态采样，确保每个尺寸的窗口都能被评到。"""
    I = cv2.integral(score.astype(np.float64))  # (sh+1, sw+1)
    best = None
    for ar in aspects:
        for sf in size_fracs:
            ww = max(4, int(sw * sf))
            hh = max(4, int(ww / ar))
            if hh > sh * 0.90:
                hh = int(sh * 0.90)
                ww = max(4, int(hh * ar))
            if ww > sw or hh > sh or ww < 5 or hh < 5:
                continue
            xs_pos = np.linspace(0, max(0, sw - ww), ng, dtype=int)
            ys_pos = np.linspace(0, max(0, sh - hh), ng, dtype=int)
            for ty in ys_pos:
                for tx in xs_pos:
                    s = (I[ty + hh, tx + ww] - I[ty, tx + ww]
                         - I[ty + hh, tx] + I[ty, tx]) / (ww * hh)
                    if best is None or s > best[0]:
                        best = (s, int(tx), int(ty), int(ww), int(hh))
    return best  # (score, x, y, w, h)


def _mean_outside(I, W, H, x, y, w, h):
    """窗口外区域的均值（判断该窗口是否与背景分离）。"""
    tot = I[H, W] - I[0, W] - I[H, 0] + I[0, 0]
    in_ = (I[y + h, x + w] - I[y, x + w] - I[y + h, x] + I[y, x])
    out_area = W * H - w * h
    out_ = (tot - in_) / max(1, out_area)
    return in_ / (w * h), out_


def analyze(img, name="", verbose=False, aspect=None):
    h, w = img.shape[:2]
    energy, (sw, sh) = _energy_map(img)
    density, _ = _print_density(img)

    # 显著性加权质心（带轻微中心先验，避免飘到边缘噪点）—— 全部用归一化坐标
    ys, xs = np.mgrid[0:sh, 0:sw].astype(np.float32)
    xn = xs / sw
    yn = ys / sh
    cxn, cyn = 0.5, 0.5
    prior = 1.0 - 0.55 * (((xn - cxn) ** 2 + (yn - cyn) ** 2) * 2.2)
    prior = np.clip(prior, 0.18, 1.0)
    wmap = energy * prior
    tot = wmap.sum() + 1e-6
    fx_n = (xn * wmap).sum() / tot
    fy_n = (yn * wmap).sum() / tot

    # 2026-10-28：产品居中裁剪（用户红线「比例裁剪一定要产品上下左右居中」）。
    # 加权质心会被高光噪点/背景纹理带偏，裁窗跟着偏。改用「显著性主体外接框
    # 中心」做焦点：能量图 97 分位阈值 -> 闭运算连块 -> 最大连通域外接框，
    # 框面积占比合理（3%~85%）才采用，否则维持质心。focal_window 以该中心
    # 取景，按比例裁剪时产品即上下左右居中。纸品检测命中后仍会覆盖此焦点。
    try:
        _sal = (energy > np.percentile(energy, 97)).astype(np.uint8) * 255
        _sal = cv2.morphologyEx(_sal, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        _sal = cv2.dilate(_sal, np.ones((7, 7), np.uint8))
        _cnts, _ = cv2.findContours(_sal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if _cnts:
            _bx, _by, _bw, _bh = cv2.boundingRect(max(_cnts, key=cv2.contourArea))
            _frac = (_bw * _bh) / float(sw * sh)
            if 0.03 < _frac < 0.85:
                fx_n = (_bx + _bw / 2.0) / sw
                fy_n = (_by + _bh / 2.0) / sh
    except BaseException:
        pass

    # 2026-10-28 第三轮：焦点=「按裁剪比例的能量最优窗口中心」。上一版的最大
    # 连通域外接框会误选画面里的小块高细节区（实拍：0W5A8298 的右上角花纹把
    # 焦点带到 0.676，裁切整体偏右、产品出框）。改为在能量图上按调用方的输出
    # 画幅（aspect，如 9:16=0.5625）滑窗积分取均值最高窗口，窗口中心即焦点
    # ——直接优化「裁剪窗框住的产品能量最大」，产品必然在裁剪里居中。
    # 纸品检测命中仍在后面最后覆盖。
    if aspect and aspect > 0:
        try:
            _bw = _best_window(energy, sw, sh, [float(aspect)],
                               [0.95, 0.80, 0.62])
            if _bw:
                _s, _wx, _wy, _ww, _wh = _bw
                fx_n = (_wx + _ww / 2.0) / sw
                fy_n = (_wy + _wh / 2.0) / sh
        except BaseException:
            pass

    # 2026-10-28 第四轮：主文字卡定心（治多产品平铺图）。能量最优窗口会被
    # 邻近产品/花艺带偏（实拍 0W5A8298 两组产品取到 0.555，主卡仍偏左）。
    # 放宽文字块闸门找「强文字卡」（edge_c>=0.15 且 inner_d>=1.9，面积
    # 0.03~0.45），取 edge_c 最高者的框中心定焦——TOP5 场景里这就是 SKU
    # 主卡，裁剪必须以它居中。严格纸品检测命中仍在后面最后覆盖。
    try:
        _tmap2, (_tw2, _th2), _g2 = _text_map(img)
        _tt2 = np.percentile(_tmap2, 97)
        _m2 = (_tmap2 > _tt2).astype(np.uint8) * 255
        _m2 = cv2.morphologyEx(_m2, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        _m2 = cv2.dilate(_m2, np.ones((5, 5), np.uint8))
        _cnts2, _ = cv2.findContours(_m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _tmean2 = float(_tmap2.mean()) + 1e-6
        _best2 = None
        for _c in sorted(_cnts2, key=cv2.contourArea, reverse=True)[:5]:
            _bx, _by, _bw2, _bh2 = cv2.boundingRect(_c)
            _af = (_bw2 * _bh2) / float(_tw2 * _th2)
            _arb = _bw2 / max(1, _bh2)
            if not (0.03 < _af < 0.45) or not (0.25 < _arb < 4.0):
                continue
            _ec2 = _edge_contrast(_g2, _bx, _by, _bw2, _bh2)
            _id2 = float(_tmap2[_by:_by + _bh2, _bx:_bx + _bw2].mean() / _tmean2)
            if _ec2 >= 0.15 and _id2 >= 1.9:
                if _best2 is None or _ec2 > _best2[0]:
                    _best2 = (_ec2, (_bx + _bw2 / 2.0) / _tw2,
                              (_by + _bh2 / 2.0) / _th2)
        if _best2:
            fx_n = _best2[1]
            fy_n = _best2[2]
    except BaseException:
        pass

    peak = energy.max()
    mean = energy.mean()
    peak_ratio = peak / (mean + 1e-6)

    varx = (xn * wmap).sum() / tot
    vary = (yn * wmap).sum() / tot
    covx = ((xn - varx) ** 2 * wmap).sum() / tot
    covy = ((yn - vary) ** 2 * wmap).sum() / tot
    spread = np.sqrt(covx + covy)      # 归一化散布 0..~0.5
    aniso = np.sqrt(covy + 1e-6) / (np.sqrt(covx + 1e-6) + 1e-6)

    # 纸品检测（2026-10-23 升级：治奶油/低对比调漏检与场景误报）：
    #   a) 局部归一化 textness（见 _text_map）—— 低对比纸品文字也能起来；
    #   b) 97 分位 mask -> CLOSE15 连字成块 -> dilate5 外扩，取面积 top5 候选；
    #   c) 双闸门：亮度台阶 edge_c>=0.09（卡片浮在背景上）+ 块内细节密度
    #      inner_d>=2.9（文字密集）——两关同时过才判纸品。
    #   阈值来自 _tune_paper.py 在 8 张真实婚礼图 + 合成请柬卡上的标定：
    #   桌号卡/请幡/BAR 牌/合成卡全过，长桌/泳池/室内餐桌/海边全景全不过。
    tmap, (tw, thh), gsmall = _text_map(img)
    tt = np.percentile(tmap, 97)
    tmask = (tmap > tt).astype(np.uint8) * 255
    tmask = cv2.morphologyEx(tmask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    tmask = cv2.dilate(tmask, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(tmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    is_paper = False
    pbox_n = None
    paper_score = -1.0
    paper_par = 1.0
    if contours:
        tmean = float(tmap.mean()) + 1e-6
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            bx, by, bw2, bh2 = cv2.boundingRect(c)
            area_frac = (bw2 * bh2) / float(tw * thh)
            ar_b = bw2 / max(1, bh2)
            # 面积适中 + 近矩形 => 候选纸品块（下限 0.022：排除桌上小件高细节物件）
            if not (0.022 < area_frac < 0.45) or not (0.25 < ar_b < 3.5):
                continue
            # 2026-10-29 贴边闸门：真「浮在场景上的卡片」不会顶到画面边。
            # 贴边（1% 内）的块多半是桌沿/桌布/背景横带，取它当焦点会把重点
            # 钉到画面角落（实测请柬平铺/桌号卡组：焦点被带到 y=0.07 / y=0.95 /
            # x=0.93，起帧几何上无法居中）。命中贴边弃用该候选、看下一个。
            _mx, _my = max(1, int(tw * 0.01)), max(1, int(thh * 0.01))
            if (bx <= _mx or by <= _my or
                    bx + bw2 >= tw - _mx or by + bh2 >= thh - _my):
                continue
            ec = _edge_contrast(gsmall, bx, by, bw2, bh2)
            idens = float(tmap[by:by + bh2, bx:bx + bw2].mean() / tmean)
            if ec >= 0.09 and idens >= 2.9:
                is_paper = True
                paper_score = float(area_frac)
                paper_par = float(ar_b)
                pbox_n = (bx / tw, by / thh, (bx + bw2) / tw, (by + bh2) / thh)
                # 纸品优先：把主体钉到纸品块中心
                fx_n = (bx + bw2 / 2) / tw
                fy_n = (by + bh2 / 2) / thh
                break

    # 运镜选型（婚纱调性：纸品/竖幅主体->推近，横幅场景->横移，平铺->旋转）
    ih, iw = img.shape[:2]
    aspect_src = iw / ih                      # 注意 shape 是 (H,W)：aspect = 宽/高
    if is_paper:
        motion = "push"          # 纸品 = 主视觉，直接推近聚焦
    elif aspect_src > 1.15:
        # 横向宽幅（风景 / 大场景）：横向 Ken Burns 平移最自然
        motion = "panL" if fx_n > 0.5 else "panR"
    elif peak_ratio < 1.6:
        motion = "rot"           # 低对比平铺 -> 慢旋转
    else:
        motion = "push"          # 竖幅居中主体 -> 优雅推近

    result = {
        "name": name,
        "focal_n": (float(fx_n), float(fy_n)),
        "focal": (int(fx_n * w), int(fy_n * h)),
        "peak_ratio": float(peak_ratio),
        "spread": float(spread),
        "aniso": float(aniso),
        "paper": bool(is_paper),
        "paper_score": float(paper_score),
        "paper_par": float(paper_par),
        "paper_box_n": pbox_n,
        "motion": motion,
    }
    if verbose:
        print("[%s] motion=%s paper=%s(par=%.2f) peak=%.2f spread=%.3f aniso=%.2f pscore=%.3f focal=(%.2f,%.2f)"
              % (name, motion, is_paper, paper_par, peak_ratio, spread, aniso, paper_score, fx_n, fy_n))
    return result


# ---------------- 主体居中取窗 ----------------
def focal_window(ih, iw, aspect, zoom=1.0, focal_n=(0.5, 0.5)):
    """返回以 focal_n（归一化主体中心）为中心的、符合 aspect 的取景窗口 (x,y,w,h)。"""
    fx, fy = focal_n
    win_h = ih / zoom
    win_w = win_h * aspect
    if win_w > iw / zoom:
        win_w = iw / zoom
        win_h = win_w / aspect
    # 窗口中心钉在主体上（先按比例缩到像素，再钳制使窗口完整）
    cx_px = fx * iw
    cy_px = fy * ih
    x = cx_px - win_w / 2
    y = cy_px - win_h / 2
    x = min(max(x, 0.0), iw - win_w)
    y = min(max(y, 0.0), ih - win_h)
    return x, y, win_w, win_h


def focal_strip_frame(img, zoom, dx_out, dy_out, rot, focal_n):
    """与 wed_threescreen 的 strip_frame 等价，但窗口中心钉在 focal_n 上。"""
    ih, iw = img.shape[:2]
    x, y, w, h = focal_window(ih, iw, 1080.0 / 640.0, zoom, focal_n)
    kx = w / 1080.0
    ky = h / 640.0
    cx = x + w / 2 + dx_out * kx
    cy = y + h / 2 + dy_out * ky
    cx = min(max(cx, w / 2), iw - w / 2)
    cy = min(max(cy, h / 2), ih - h / 2)
    if rot == 0.0:
        return _render_win(img, cx - w / 2, cy - h / 2, w, h, 1080, 640)
    return _rot_render(img, cx - w / 2, cy - h / 2, w, h, 1080, 640, rot)


def _render_win(img, x, y, w, h, ow, oh):
    ih, iw = img.shape[:2]
    x = max(0, min(x, iw - w))
    y = max(0, min(y, ih - h))
    xi, yi = int(round(x)), int(round(y))
    wi = min(int(round(w)), iw - xi)
    hi = min(int(round(h)), ih - yi)
    crop = img[yi:yi + hi, xi:xi + wi]
    interp = cv2.INTER_AREA if (wi > ow or hi > oh) else cv2.INTER_LINEAR
    return cv2.resize(crop, (ow, oh), interpolation=interp)


def _rot_render(img, x, y, w, h, ow, oh, ang):
    S = 1.18
    bw, bh = int(round(ow * S)), int(round(oh * S))
    big = _render_win(img, x, y, w, h, bw, bh)
    M = cv2.getRotationMatrix2D((bw / 2.0, bh / 2.0), ang, 1.0)
    rot = cv2.warpAffine(big, M, (bw, bh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    sx = (bw - ow) // 2
    sy = (bh - oh) // 2
    return rot[sy:sy + oh, sx:sx + ow].copy()


# ---------------------------------------------------------------------------
# 全屏单图运镜（供 Grid Showcase / Colour Inspiration 单图段复用）
# ---------------------------------------------------------------------------
# 默认运镜参数（2026-10-26 引自 three_screen 定稿口径，适配单图满屏 9:16）
FS_PUSH = 0.07        # 推近幅度（zoom 1.0 -> 1.07，慢而稳）
FS_PAN = 0.13         # 横扫行程系数（× 窗口宽；视觉舒适不晃）
FS_PULL = 1.11        # 拉远起点 zoom（1.11 -> 1.0 揭示环境）
FS_ROT = 4.0          # 旋转幅度（度，±FS_ROT 内缓动）
FS_BASE_ZOOM = 1.12   # 纸品 / 近满幅静态放大倍数（看清文字）


def _upscale_bgr(bgr, long_min=1800):
    """低清源先 LANCZOS 超分，避免单图满屏被拉糊（与 card_pan 同款思路）。"""
    if bgr is None:
        return None
    ih, iw = bgr.shape[:2]
    long = max(ih, iw)
    if long < long_min:
        s = long_min / float(long)
        bgr = cv2.resize(bgr, (int(round(iw * s)), int(round(ih * s))),
                         interpolation=cv2.INTER_LANCZOS4)
    return bgr


def min_zoom_centre(ih, iw, asp, focal_n, zmax=1.35, step=1.06):
    """「先切重点」所需的最小起幅 zoom（2026-10-28 用户定稿：重点产品要在
    中心位置）。zoom=1.0 全幅时若 focal 偏离中心，取景窗口被钳制在画幅内、
    重点无法落到画面中心；逐步放大（缩小视野）直到窗口能以 focal 为中心
    放下，返回该 zoom。封顶 zmax——过强的放大破坏构图，残余偏差可接受。"""
    fx, fy = focal_n
    px, py = fx * iw, fy * ih
    z = 1.0
    while z < zmax - 1e-9:
        win_h = ih / z
        win_w = win_h * asp
        if win_w > iw / z:
            win_w = iw / z
            win_h = win_w / asp
        if (win_w <= 2.0 * min(px, iw - px) + 1e-6 and
                win_h <= 2.0 * min(py, ih - py) + 1e-6):
            return z
        z *= step
    return zmax


def decide_fullscreen(an, ASP_F):
    """全屏单图运镜决策（2026-10-28 定稿：先切重点再运镜、按图选型）：
    对 pan 改为按「该方向侧的单侧余量」判定（余量不足 -> 改推近，避免越界 /
    死帧）；rot 靠 _rot_render 的 1.18 overscan，近满幅也可用。

    an: image_motion.analyze 结果（需已写入 _ih / _iw）。
    返回 (motion, param)：motion∈{push,panL,panR,rot}，param 为对应幅度。
    """
    motion = an["motion"]
    ih, iw = an["_ih"], an["_iw"]
    if an.get("paper"):
        # 纸品优先看清文字：只有该侧余量足够才横扫，否则静态推近
        if motion in ("panL", "panR"):
            x0, _, w0, _ = focal_window(ih, iw, ASP_F, 1.0, an["focal_n"])
            margin = (iw - w0 - x0) if motion == "panR" else x0
            if margin >= w0 * 0.04:
                return (motion, FS_PAN)
        return ("push", FS_PUSH)
    if motion in ("panL", "panR"):
        x0, _, w0, _ = focal_window(ih, iw, ASP_F, 1.0, an["focal_n"])
        margin = (iw - w0 - x0) if motion == "panR" else x0
        if margin < w0 * 0.04:
            return ("push", FS_PUSH)          # 该侧无余量 -> 推近
        return (motion, FS_PAN)
    if motion == "rot":
        return ("rot", FS_ROT)
    return ("push", FS_PUSH)


def render_single_shot(img, motion, focal_n, p, W, H,
                       push=FS_PUSH, pan=FS_PAN, pull=FS_PULL, rot=FS_ROT,
                       pan_zoom=None):
    """全屏单图运镜一帧（输出 RGB，供 ffmpeg rgb24）。

    img     : BGR uint8（建议先 _upscale_bgr 保证分辨率）
    motion  : 'push' | 'panL' | 'panR' | 'rot' | 'pull'
    focal_n : 归一化主体中心 (fx, fy)
    p       : 进度 0..1（缓动已在外部完成，这里线性映射到运镜轨迹）
    W, H    : 输出尺寸（9:16 -> 1080x1920）
    pan_zoom: pan 起幅 zoom（2026-10-29，Colour Inspiration 传
              max(1.0, min_zoom_centre(...)) 让起帧重点居中）；None=维持
              旧口径 zoom=1.0（Grid Showcase 不受影响）。
    """
    ih, iw = img.shape[:2]
    ASP_F = W / float(H)
    fx, fy = focal_n
    fx_px, fy_px = fx * iw, fy * ih
    e = min(1.0, max(0.0, p))

    if motion == "pull":
        # 拉远揭示环境：起点放大足以把重点框进中心，终点回到全幅
        z0 = max(pull, min_zoom_centre(ih, iw, ASP_F, focal_n))
        z = z0 + (1.0 - z0) * e
        _, _, w, h = focal_window(ih, iw, ASP_F, z, focal_n)
        cx = min(max(fx_px, w / 2), iw - w / 2)
        cy = min(max(fy_px, h / 2), ih - h / 2)
        bgr = _render_win(img, cx - w / 2, cy - h / 2, w, h, W, H)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if motion in ("panL", "panR"):
        # 2026-10-28 用户定稿：起帧窗口以重点为中心（先切重点），然后向该侧
        # 单向横扫——旧版是跨焦点往返式（起点偏在焦点一侧 trav/2 处，起帧
        # 重点不在画面中心）。
        # 2026-10-29：起幅 zoom 可由调用方传入（pan_zoom）——focal 偏边的图
        # zoom=1.0 时窗口钳在画幅内、重点无法居中，放大起幅后窗口装得下
        # 「以重点为中心」。默认 None 维持 zoom=1.0（Grid Showcase 口径不变）。
        z_pan = pan_zoom if (pan_zoom and pan_zoom > 1.0) else 1.0
        _, y0, w0, h0 = focal_window(ih, iw, ASP_F, z_pan, focal_n)
        x0 = min(max(fx_px - w0 / 2, 0.0), iw - w0)
        if motion == "panR":
            trav = max(8.0, min(pan * w0, (iw - w0 - x0) - 6))
            x = x0 + trav * e
        else:
            trav = max(8.0, min(pan * w0, x0 - 6))
            x = x0 - trav * e
        bgr = _render_win(img, x, y0, w0, h0, W, H)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if motion == "rot":
        # 起幅足以让重点居中（FS_BASE_ZOOM 与 min_zoom_centre 取大者）
        z = max(FS_BASE_ZOOM, min_zoom_centre(ih, iw, ASP_F, focal_n))
        _, _, w, h = focal_window(ih, iw, ASP_F, z, focal_n)
        cx = min(max(fx_px, w / 2), iw - w / 2)
        cy = min(max(fy_px, h / 2), ih - h / 2)
        ang = (e - 0.5) * 2.0 * rot
        bgr = _rot_render(img, cx - w / 2, cy - h / 2, w, h, W, H, ang)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 默认 push（推近）：起幅 = min_zoom_centre —— focal 偏边的图起帧也能把
    # 重点框进画面中心（「先切重点、重点居中」，2026-10-28 用户定稿），
    # 然后从该起幅缓慢推近。
    z0 = min_zoom_centre(ih, iw, ASP_F, focal_n)
    z = z0 * (1.0 + push * e)
    _, _, w, h = focal_window(ih, iw, ASP_F, z, focal_n)
    cx = min(max(fx_px, w / 2), iw - w / 2)
    cy = min(max(fy_px, h / 2), ih - h / 2)
    bgr = _render_win(img, cx - w / 2, cy - h / 2, w, h, W, H)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
