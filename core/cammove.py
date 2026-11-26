# -*- coding: utf-8 -*-
"""
2.5D 基础运镜引擎（cv2.remap 加速版）
======================================
复刻 probe/output/product_10s_25d.mp4 的运镜手法（上移/横移/特写/细节），
压缩为 6-7s 结构。全部几何位移，文字零变形，零 AI 算力。

与原 shotlib 的差异（均为修正）:
  1. build_canvas 保持宽高比（原实现 nw=max(W+40, w*s) 会把 1080 宽源图拉伸 3.7%）
  2. 像素采样用 cv2.remap（比 numpy 花式索引快 10 倍以上）
  3. pan 行程按画布边界自动钳制，适配任意尺寸上传图
"""
import os, math, re, time
import numpy as np
import cv2
from PIL import Image, ImageFilter
import imageio_ffmpeg

W, H, FPS = 1080, 1920, 30
CANVAS_H = 1920
# 2026-10-10 用户口径：Step 2 由「时长档」改为「视频比例」，仅两档 —— 9:16 竖屏 / 1:1 方图。
# 视口（= 成片）尺寸由比例决定；RATIOS 是画幅的唯一真源（前端 / server 都读它）。
RATIOS = {"9:16": (1080, 1920), "1:1": (1080, 1080)}
DEFAULT_RATIO = "9:16"
PAPER_RGB = 245          # 纸白（9:16 窄图左右补白用；1:1 已改为截取方框、不再留边）
# 「对着工艺区域取景」时允许的 pan 边界余量（比常规 70 小）：
# 工艺区贴近画面边缘时，只有放宽边距才能既把视口压在工艺区上、又留出位移行程。
ANCHOR_MARGIN = 24.0


def _int_secs(n):
    """镜头时长取整到秒（四舍五入到最近整秒，0.5 向上靠）：2.5→3, 1.5→2, 3.0→3。"""
    return int(math.floor(float(n) + 0.5))

# ---------------- 画布 ----------------

def build_canvas(src_path, ratio=DEFAULT_RATIO):
    """源图 → 画布（含视口尺寸）。ratio 决定成片画幅（见 RATIOS）。

    9:16（默认，原口径）：源图等比缩放到高 1920，宽不足视口时左右补纸白 —— 竖屏取景。
    1:1（2026-10-10 用户口径 v2「按方框截取 + 2.5D 运镜」）：从源图**截取中心最大内接
      正方形**（cover 方框、四周不留白），再等比放大到视口的 1.22 倍留出运镜余量；
      方图因此吃到与 9:16 **同一套 2.5D 运镜**（扫/推/微距），不再是一张静态图浮在白底上。
    返回 CanvasCtx（附 vw/vh 视口尺寸、src_wh/scale/padx/pady 原图→画布映射）。
    """
    vw, vh = RATIOS.get(ratio) or RATIOS[DEFAULT_RATIO]
    img = np.asarray(Image.open(src_path).convert("RGB"))
    h, w = img.shape[:2]
    if ratio == "1:1":
        # [2026-10-10 用户口径 v2] 不再「整图入方框 + 纸白留边」，改为**从原图截取正方形画面**：
        #   取原图中心的最大内接正方形（cover 方框、四周不留白），再等比放大到视口的 1.22 倍，
        #   留出与 9:16 同量级的运镜余量 —— 方图因此吃到**同一套 2.5D 运镜**（扫/推/微距）。
        side_src = min(w, h)
        x0 = max(0, (w - side_src) // 2)
        y0 = max(0, (h - side_src) // 2)
        sq = img[y0:y0 + side_src, x0:x0 + side_src]
        side = int(round(vw * 1.22))                       # 画布 > 视口 → 留运镜余量
        img = np.asarray(Image.fromarray(sq).resize((side, side), Image.LANCZOS))
        s = side / float(max(1, side_src))
        pad_l, pad_t = -x0 * s, -y0 * s                    # 原图→画布映射（截取后仍需对位）
    else:
        s = CANVAS_H / h
        nw = max(1, int(round(w * s)))
        img = np.asarray(Image.fromarray(img).resize((nw, CANVAS_H), Image.LANCZOS))
        pad_l, pad_t = 0, 0
        if nw < vw + 40:  # 画布比视口窄: 左右补白
            pad = (vw + 40) - nw
            pad_l = pad // 2
            img = np.pad(img, ((0, 0), (pad_l, pad - pad_l), (0, 0)),
                         constant_values=PAPER_RGB)
    ctx = CanvasCtx(img, vw, vh)
    # 原图 → 画布的变换（工艺区域框映射用）：x_canvas = x_src * scale + padx
    ctx.src_wh = (w, h)
    ctx.scale = s
    ctx.padx = pad_l
    ctx.pady = pad_t
    return ctx


class CanvasCtx:
    def __init__(self, img, vw=W, vh=H):
        self.img = img                      # uint8 RGB
        self.hh, self.ww = img.shape[:2]
        self.vw, self.vh = int(vw), int(vh)  # 视口（= 成片）尺寸，画幅由 build_canvas 决定
        # 原图 → 画布的变换参数（build_canvas 覆盖；直接构造时按 1:1 兜底）
        self.src_wh = (self.ww, self.hh)
        self.scale = 1.0
        self.padx = 0
        self.pady = 0
        f = img.astype(np.float32)
        gray = f @ np.array([0.299, 0.587, 0.114], np.float32)
        # --- 浅浮雕深度图 (与 shotlib.build_depth_relief 等价) ---
        hh, ww = gray.shape
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        nx, ny = (xx / ww - 0.5) * 2, (yy / hh - 0.5) * 2
        dm = 0.012 * np.exp(-(nx * nx + ny * ny) * 1.1)
        g01 = gray / 255.0
        blur = np.asarray(Image.fromarray(gray.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(6)), np.float32) / 255.0
        edge = np.clip(np.abs(g01 - blur) * 2.0, 0, 1)
        r, g, b = f[..., 0], f[..., 1], f[..., 2]
        gold = np.clip((r - b) * 1.4 + (g - b) * 0.9 - 0.18, 0, 1) * np.clip(g01 * 1.1, 0, 1)
        d = 0.5 + dm + edge * 0.045 + gold * 0.018
        d = np.asarray(Image.fromarray((np.clip(d, 0, 1) * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(1.2)), np.float32) / 255.0 - 0.5
        self.depth = d.astype(np.float32)
        # --- 景深用模糊图层 ---
        self.blur = np.asarray(Image.fromarray(img).filter(
            ImageFilter.GaussianBlur(9)))
        # --- 静态网格缓存 ---
        self.yy, self.xx = np.mgrid[0:self.vh, 0:self.vw].astype(np.float32)
        self.cx, self.cy = self.vw / 2.0, self.vh / 2.0
        self._dof_cache = {}
        # --- 采样缓冲（2026-09-29 性能优化：逐帧复用，避免每帧分配 16 MB）---
        # 注意：map 是**视口**坐标系 (H, W)，remap 的输出尺寸由 map 决定。
        self.xs = np.arange(self.vw, dtype=np.float32)   # 视口列坐标 0..vw-1
        self.ys = np.arange(self.vh, dtype=np.float32)   # 视口行坐标 0..vh-1
        self._maps = None
        self._dscale = {}
        self._vig_cache = {}          # 暗角掩膜缓存（按强度）—— 见 vignette_apply

    def maps(self):
        """复用两张 (H, W) float32 采样坐标缓冲（视口坐标系）。"""
        if self._maps is None:
            self._maps = (np.empty((self.vh, self.vw), np.float32),
                          np.empty((self.vh, self.vw), np.float32))
        return self._maps

    def depth_scaled(self, parallax):
        """深度图按 parallax 预缩放并缓存（逐帧只剩 d * pan 两次乘加）。"""
        key = round(float(parallax), 2)
        d = self._dscale.get(key)
        if d is None:
            d = (self.depth * (key / 800.0)).astype(np.float32)
            self._dscale[key] = d
        return d

    # --- 暗角（2026-10-10 性能修复）---
    def vignette_mask_u8(self, amount):
        """按强度缓存的 uint8 三通道暗角增益（255=不压暗）。
        权重只依赖「视口尺寸 + 强度」，与相机位姿无关 → 整条片子只需算一次。"""
        key = round(float(amount), 3)
        m = self._vig_cache.get(key)
        if m is None:
            nx = (self.xx / self.vw - 0.5) * 2
            ny = (self.yy / self.vh - 0.5) * 2
            v = 1.0 - key * np.clip(nx * nx + ny * ny - 0.35, 0, 1.2)
            m = np.repeat(np.clip(v * 255.0, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
            m = np.ascontiguousarray(m)
            self._vig_cache[key] = m
        return m

    def vignette_apply(self, out, amount):
        """把暗角乘到帧上（等价 out * v，v∈[0,1]）。uint8 走定点乘法（快 ~25×）。"""
        if out.dtype == np.uint8:
            return cv2.multiply(out, self.vignette_mask_u8(amount), scale=1.0 / 255.0)
        nx = (self.xx / self.vw - 0.5) * 2
        ny = (self.yy / self.vh - 0.5) * 2
        v = 1.0 - amount * np.clip(nx * nx + ny * ny - 0.35, 0, 1.2)
        return out * v[..., None]

    # --- 景深权重（按锚点缓存, 基于画布坐标）---
    # [2026-09-24] 浅景深模糊已弃用(用户否决: 整帧混合导致文字模糊), 保留机制勿在新成片启用
    def dof_layer(self, anchor, sigma, soft):
        key = (int(anchor[0]), int(anchor[1]), round(sigma, 1), round(soft, 1))
        if key not in self._dof_cache:
            yy, xx = np.mgrid[0:self.hh, 0:self.ww].astype(np.float32)
            dist = np.sqrt((xx - anchor[0]) ** 2 + (yy - anchor[1]) ** 2)
            wgt = (1.0 / (1.0 + ((dist - sigma) / soft) ** 2)).astype(np.float32)
            self._dof_cache[key] = wgt
        return self._dof_cache[key]


def ease_io(t):
    return t * t * (3 - 2 * t)


def smoother(t):
    """smootherstep：起止速度为 0 且加速度连续，比 ease_io 更"缓起缓收"。"""
    return t * t * t * (t * (t * 6 - 15) + 10)


def even(t):
    """近匀速曲线：主体线性 + 两端轻微缓起缓收。
    标准指令要求「镜头移动速度保持均匀，不突然加速或减速」，
    但完全线性会在起止处出现速度突变的生硬感，故用 30% smootherstep 混合。"""
    return 0.70 * t + 0.30 * smoother(t)


def curve_in(t):
    """缓起曲线：起手慢、收尾快（位移前段被压缩）。
    [2026-10-10 用户口径] 给扫类镜头注入"运动性格"差异，避免所有平移都是同一种匀速滑法。
    导数 0.5→1.5：前段慢、后段快，与 even 形成可感知的节奏对比。"""
    return 0.50 * t + 0.50 * t * t


def curve_out(t):
    """缓收曲线：起手快、收尾慢（位移后段被压缩），落点更"稳"。
    [2026-10-10 用户口径] 与 even / curve_in 互不相同，增加单条片子里镜头的节奏层次；
    cm07 工艺近景用它让推入"停稳"在细节上。导数 1.5→0.5：前段快、后段慢。"""
    return 1.50 * t - 0.50 * t * t


# ---------------- 相机 ----------------

class Cam:
    def __init__(self, zoom=1.05, ax=960.0, ay=960.0, pan=(0.0, 0.0),
                 roll=0.0, parallax=200.0, dof=None, vignette=0.0):
        self.zoom, self.ax, self.ay = float(zoom), float(ax), float(ay)
        self.panx, self.pany = float(pan[0]), float(pan[1])
        self.roll = float(roll)             # 弧度
        self.parallax = float(parallax)
        self.dof = dof                      # (sigma, soft) 或 None
        self.vignette = float(vignette)


def pan_bounds(cv, ax, ay, zoom, margin=70.0):
    """给定锚点与 zoom, 返回 (panx_min, panx_max, pany_min, pany_max), 保证视口在画布内。"""
    hx, hy = (cv.vw / 2.0) / zoom, (cv.vh / 2.0) / zoom
    px0, px1 = hx - ax + margin, cv.ww - ax - hx - margin
    py0, py1 = hy - ay + margin, cv.hh - ay - hy - margin
    if px0 > px1: px0 = px1 = (px0 + px1) / 2
    if py0 > py1: py0 = py1 = (py0 + py1) / 2
    return px0, px1, py0, py1


# ---------------- 帧渲染 ----------------

def render_frame(cv, cam, dof_wgt=None):
    """单帧渲染：相机姿态 → 画布采样坐标 → 逐像素重采样（含深度视差）。

    性能（2026-09-29 优化，输出 1080×1920，仅改实现不改几何口径）：
      · 无旋转镜头（roll == 0，现用镜头全部如此）走一维广播路径，
        省掉四次全尺寸旋转乘加（实测 293 ms → 54 ms）；
      · 深度图改用 cv2.remap 采样，替代 numpy 花式索引的随机访存（102 ms → 29 ms）；
      · 深度按 parallax 预缩放缓存，逐帧只做 d*pan 两次乘加（130 ms → ~45 ms）；
      · 采样坐标缓冲复用它处预分配，省掉每帧 16 MB 的分配/回收。
    采样点与优化前逐像素一致；唯一差别是深度取值由截断改为四舍五入（≤1px，深度图已平滑）。
    """
    ww, hh = cv.ww, cv.hh
    lim_x, lim_y = ww - 1.001, hh - 1.001
    map_x, map_y = cv.maps()

    if cam.roll == 0.0:
        # sx 只随列变化、sy 只随行变化 → 一维算好再广播填充
        row = (cv.xs - cv.cx) / cam.zoom + (cam.ax + cam.panx)
        col = (cv.ys - cv.cy) / cam.zoom + (cam.ay + cam.pany)
        map_x[:] = row
        map_y[:] = col[:, None]
    else:
        th = -cam.roll
        ct, st = math.cos(th), math.sin(th)
        rx = (cv.xx - cv.cx) * ct - (cv.yy - cv.cy) * st
        ry = (cv.xx - cv.cx) * st + (cv.yy - cv.cy) * ct
        np.add(rx, cam.ax + cam.panx, out=map_x)
        np.divide(map_x, cam.zoom, out=map_x)
        np.add(ry, cam.ay + cam.pany, out=map_y)
        np.divide(map_y, cam.zoom, out=map_y)

    # dof 权重需在基础坐标（视差修正前）上取，与优化前一致
    wgt = None
    if dof_wgt is not None:
        wgt = cv2.remap(dof_wgt, map_x, map_y, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_REPLICATE)[..., None]

    # 深度视差：在基础采样位置取深度（前景位移更大），再修正采样坐标
    if cam.panx or cam.pany:
        d = cv2.remap(cv.depth_scaled(cam.parallax), map_x, map_y,
                      cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        if cam.panx:
            np.add(map_x, d * cam.panx, out=map_x)
        if cam.pany:
            np.add(map_y, d * cam.pany, out=map_y)

    np.clip(map_x, 0, lim_x, out=map_x)
    np.clip(map_y, 0, lim_y, out=map_y)

    out = cv2.remap(cv.img, map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    if wgt is not None:
        outb = cv2.remap(cv.blur, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        out = (out.astype(np.float32) * wgt + outb.astype(np.float32) * (1 - wgt))
    if cam.vignette > 0:
        # [2026-10-10 性能修复] 原实现每帧重算整张浮点暗角权重并做浮点广播乘 —— 9:16
        #   实测 286 ms/帧，使 cm07（微距镜）成为默认序列最慢镜头（380 ms/帧）。
        #   改为「每强度只算一次」的 uint8 掩膜定点乘（286 ms → 11.5 ms，逐像素等价 ±1 LSB）。
        out = cv.vignette_apply(out, cam.vignette)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def encode(path, frames_iter, n, size=None):
    # -preset veryfast: 90 帧实测 13.3s → 3.6s，crf 口径不变（画质主体由 crf 决定）
    # size：成片尺寸 (w, h)。缺省沿用 9:16（1080×1920）；1:1 时由调用方传 (vw, vh)。
    wr = imageio_ffmpeg.write_frames(path, size or (W, H), fps=FPS,
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=1,
        output_params=["-crf", "17", "-preset", "veryfast"])
    wr.send(None)
    for i, fr in enumerate(frames_iter):
        if fr.dtype != np.uint8:
            fr = np.clip(fr, 0, 255).astype(np.uint8)
        wr.send(fr)
        if i % 10 == 0:
            progress_hook(i + 1, n)
    wr.close()


progress_hook = lambda done, total: None   # 由服务端注入


def drop_temps(paths, tries=3):
    """删渲染中间产物（.partN.mp4 / .list.txt）。
    Windows 上 ffmpeg 刚写完的片段常被杀软/索引器短暂占用，删一次会抛 WinError 32；
    实测多图渲染因此把 .part0..N 全留在磁盘上（既占地方，也让会话目录判定"渲染中"）。
    返回仍然没删掉的路径。

    ⚠ 必须走 safedel：本机可能存在「安全删除钩子」，删除超阈值时抛 SystemExit
    （BaseException，except OSError 抓不住）→ 线程静默死亡、任务永远 running。"""
    from . import safedel
    left = safedel.unlink_all(paths, tries=tries)
    if left:
        print("[cammove] %d temp file(s) left on disk: %s" % (len(left), left[:3]))
    return left


# ---------------- 自动锚点 ----------------

def auto_anchors(cv, k=3, suppress=520.0):
    """内容密集处: 深度能量图取 top-k 局部极大值。返回 [(x,y)...] 画布坐标。"""
    energy = np.abs(cv.depth)
    e = cv2.GaussianBlur(energy, (0, 0), 45)
    pts = []
    em = e.copy()
    for _ in range(k):
        _, _, _, loc = cv2.minMaxLoc(em)
        x, y = loc
        pts.append((float(x), float(y)))
        cv2.circle(em, (x, y), int(suppress), -1, -1)
    return pts


def texture_anchors(cv, k=3, suppress=520.0):
    """纹理/边缘密集处: 高频细节能量图取 top-k 局部极大值。返回 [(x,y)...] 画布坐标。

    [2026-10-09 用户口径] "运镜切不到重点，尤其是蕾丝/镂空/烫金/浮雕/雕刻字"。
    当视觉模型没能给出工艺框时，微距镜改退到这个锚点（烫金、浮雕、镂空、印刷字
    都有强边缘/高频纹理），比"深度能量峰值"更可能落在真实工艺上。
    """
    try:
        gray = cv2.cvtColor(cv.img, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        energy = np.abs(lap)
    except Exception:
        return []
    e = cv2.GaussianBlur(energy, (0, 0), 45)
    pts = []
    em = e.copy()
    for _ in range(k):
        _, _, _, loc = cv2.minMaxLoc(em)
        x, y = loc
        pts.append((float(x), float(y)))
        cv2.circle(em, (x, y), int(suppress), -1, -1)
    return pts


# ---------------- 工艺区域锚点（2026-09-30 用户口径） ----------------
# 「细节运镜，截取的画面优先是工艺类的（浮雕/凹凸/烫金等）」
# 视觉模型在质检时给出工艺区域框（相对原图归一化 0-1），这里映射成画布坐标，
# 交给微距类镜头（CM-06/CM-07/CM-08）当取景锚点。拿不到框就返回 None，
# 调用方一律回落到原有的内容锚点 —— 保证「识别不到工艺」时行为与改动前完全一致。

def craft_anchor(cv, box01):
    """归一化工艺框 [x,y,w,h]（相对原图 0-1）→ 该框中心在画布上的坐标；非法返回 None。"""
    if not box01 or not isinstance(box01, (list, tuple)) or len(box01) < 4:
        return None
    try:
        x, y, w, h = (float(box01[0]), float(box01[1]), float(box01[2]), float(box01[3]))
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) or w <= 0 or h <= 0:
        return None
    sw, sh = getattr(cv, "src_wh", (cv.ww, cv.hh))
    s = float(getattr(cv, "scale", 1.0))
    padx = float(getattr(cv, "padx", 0))
    pady = float(getattr(cv, "pady", 0))
    px = (x + w / 2.0) * sw * s + padx
    py = (y + h / 2.0) * sh * s + pady
    return (float(px), float(py))


def craft_anchors(cv, boxes):
    """归一化工艺框（单个 [x,y,w,h]，或 [[..],[..]] 多个）→ 画布坐标点列表。

    [2026-10-09 用户口径] 一张卡可能同时有烫金/蕾丝/镂空/浮雕等多处工艺，
    视觉模型可给多个框；微距镜按序各取一个，让多处工艺都能被取景照顾到。
    非法框被过滤；全部非法或为空时返回 []（调用方回落到纹理/内容锚点）。
    """
    if not boxes:
        return []
    if (isinstance(boxes, (list, tuple)) and len(boxes) >= 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in boxes[:4])):
        boxes = [boxes]                      # 单个框 → 包成一元素列表
    pts = []
    for b in boxes:
        p = craft_anchor(cv, b)
        if p:
            pts.append(p)
    return pts


def _frame_box(cv, box, fill=0.80, z_max=2.4):
    """归一化「单卡框」→ (z, ax, ay)：使该框约占视口 fill 比例，并钳制到合法取景。

    [2026-10-12 学习案例] 案例运镜会从「整套」切到「单独一张卡」再切工艺细节；
    单图 set/group（平铺套装图）之前缺这层。cm15/cm16 把单卡框当作取景目标，
    框住某张独立卡（主卡/信封/RSVP/Menu），形成 整套→单卡→工艺 的层次。
    box 非法或缺失时回落中心、1.15×。"""
    p = craft_anchor(cv, box)
    if not p:
        return 1.15, cv.ww * 0.5, cv.hh * 0.5
    sw, sh = getattr(cv, "src_wh", (cv.ww, cv.hh))
    s = float(getattr(cv, "scale", 1.0))
    bw = max(1.0, box[2] * sw * s)
    bh = max(1.0, box[3] * sh * s)
    vw, vh = cv.vw, cv.vh
    z = min(z_max, fill * vw / bw, fill * vh / bh)
    ax, ay = _clamp_center(cv, p[0], p[1], z)
    return z, ax, ay



# ---------------- 镜头编排 ----------------
# 参考 product_10s_25d: S1上移 S2下移 S3横移 S4特写 S5细节
# 6-7s 精简结构:
#   layout "compact" (6s): 上移2.0 + 横移2.0 + 特写2.0
#   layout "full"    (7s): 上移1.8 + 横移1.8 + 特写1.7 + 细节1.7

LAYOUTS = {
    "reference": {"secs": [2.2, 2.2, 2.4], "names": ["Slide push", "Pan", "Settle"]},
    "compact": {"secs": [2.0, 2.0, 2.0], "names": ["Move up", "Pan", "Close-up"]},
    "full":    {"secs": [1.8, 1.8, 1.7, 1.7], "names": ["Move up", "Pan", "Close-up", "Detail"]},
}

# ---------------- 标准基础运镜镜头库 CM-01 ~ CM-10（单镜头，一镜一支片）----------------
# 口径来源：用户给定「标准基础运镜指令」+ CM 对照表（2026-09-23）
#   单镜头、2–3s、匀速缓慢、产品完整优先、不俯仰/旋转/环绕、同背景同光线。
#   zoom/行程按「产品完整可见」优先取值；竖向/斜向行程需要更高 zoom 才有位移余量。
MOVES = {
    "cm01": {"name": "CM-01 Diagonal sweep (top-left → bottom-right)", "secs": 3.0, "suits": "Whole set laid out", "stability": "Very high", "zh": "Diagonal sweep"},
    "cm02": {"name": "CM-02 Horizontal sweep (left → right)",       "secs": 3.0, "suits": "Horizontal suite, multiple components", "stability": "Very high", "zh": "Pan"},
    "cm03": {"name": "CM-03 Vertical sweep (top → bottom)",     "secs": 3.0, "suits": "Long menu, long card, suite", "stability": "Very high", "zh": "Vertical sweep"},
    "cm04": {"name": "CM-04 Slow push in",    "secs": 3.0, "suits": "Single main card, hero shot", "stability": "Very high", "zh": "Slow push in"},
    "cm05": {"name": "CM-05 Slow pull out",   "secs": 3.0, "suits": "Detail to whole set", "stability": "Very high", "zh": "Slow pull out"},
    "cm06": {"name": "CM-06 Diagonal sweep (bottom-left → top-right)",   "secs": 3.0, "suits": "Whole set, second angle", "stability": "High", "zh": "Reverse diagonal sweep"},
    "cm07": {"name": "CM-07 Macro pan",        "secs": 2.5, "suits": "Foil, emboss, letterpress", "stability": "High", "zh": "Macro pan"},
    "cm08": {"name": "CM-08 Macro push in",    "secs": 3.0, "suits": "Logo, names, craft detail", "stability": "High", "zh": "Macro push in"},
    "cm09": {"name": "CM-09 Gentle arc move",    "secs": 3.0, "suits": "Hero product", "stability": "Medium", "zh": "Arc move"},
    "cm10": {"name": "CM-10 Locked shot + micro push", "secs": 2.5, "suits": "Handheld or card-inserting action", "stability": "Very high", "zh": "Locked + micro push"},
    # 2026-10-09 用户口径：平移镜头补齐相反方向，保证一条片子里扫向不重复
    "cm11": {"name": "CM-11 Diagonal sweep (bottom-right → top-left)", "secs": 3.0, "suits": "Whole set laid out, opposite diagonal", "stability": "Very high", "zh": "Diagonal sweep (reverse)"},
    "cm12": {"name": "CM-12 Horizontal sweep (right → left)", "secs": 3.0, "suits": "Horizontal suite, opposite direction", "stability": "Very high", "zh": "Pan (reverse)"},
    "cm13": {"name": "CM-13 Vertical sweep (bottom → top)", "secs": 3.0, "suits": "Long menu, long card, suite, opposite direction", "stability": "Very high", "zh": "Vertical sweep (reverse)"},
    # 2026-10-10 案例学习口径：多产品/整套片的开场要先给一个「场景全景」再逐个怼细节。
    #   本镜从最小倍率（≈全画幅）极缓推入，无横扫，专做开场建立镜。
    "cm14": {"name": "CM-14 Establishing wide push (reveal whole set, slow push in)", "secs": 3.0, "suits": "Multi-product opener, full suite reveal", "stability": "Very high", "zh": "Establishing push"},
    # 2026-10-12 学习案例（社媒热帖）新增：案例里高频出现两类工作台原来没有的手法 ——
    #   ① 真人手持轻晃（稳定展示，读起来"真实高级"）；② 纯静止定格（让观众读卡/工艺）。
    #   两者都是 2.5D 可复现、零 AI 算力、不破坏"文字零变形"红线。
    "cm15": {"name": "CM-15 Handheld micro-sway", "secs": 3.0, "suits": "Card held / product showcase, subtle life-like motion", "stability": "High", "zh": "Handheld sway"},
    "cm16": {"name": "CM-16 Static hold (freeze frame)", "secs": 2.0, "suits": "Read-the-card beat, let viewer absorb text/craft", "stability": "Very high", "zh": "Static hold"},
    # 2026-10-12 修正：原 cm16 只是"定格停住"（机位静止、不算运镜），不满足用户"识别单卡后再运镜"。
    #   cm17 改为从较低倍率真实推入并放大单卡（推镜，锚点锁在单卡中心），是真正的相机运动；
    #   用于单图整套图：开场扫镜建立全貌后，逐个把相机推到每张独立卡上，形成 整套→单卡→工艺 层次。
    "cm17": {"name": "CM-17 Push into card (real dolly onto individual card)", "secs": 3.0, "suits": "Recognized individual card in a whole-set image; real camera move onto it", "stability": "Very high", "zh": "Push into card"},
    # 2026-10-13 用户口径：模版库要覆盖「平移横扫 / 微距 / 旋转」三类纯运镜，补一个真正的旋转镜头。
    #   画框缓慢 roll 旋转（正弦往返，起止归零 → 末尾自动定格不歪），同时极缓推近。
    #   幅度 ±7°、峰速 ≈7°/s，克制不晕；取景放大到 1.30+ 保证旋转视口仍在画布内（去拖影）。
    "cm18": {"name": "CM-18 Slow roll rotate (frame rotation)", "secs": 3.0, "suits": "Hero card / craft detail; add rotation life without panning", "stability": "High", "zh": "Roll rotate"},
}
# 速度口径（画宽/s，视口宽 1080 计）：参考片峰值 35%，用户口径「不用太快」→
#   全部单镜头峰值控制在 ≤20%，即比参考片慢约一半；行程比例 frac 按此反推。


# ---------------- 扫向去重（2026-10-09 用户口径）----------------
# 要求：同一条运镜视频里，平移/扫类镜头（横扫/竖扫/斜扫）不得出现重复的扫向；
#   否则一条片子里会有多个「左上→右下」这种肉眼一样的镜头（用户在线上反馈过）。
# SWEEP_DIR：每个扫类镜头的「扫向」唯一键；非扫类（推/拉/微距）不在表内、不参与去重。
SWEEP_DIR = {
    "cm01": "TL-BR", "cm11": "BR-TL",
    "cm02": "L-R",   "cm12": "R-L",
    "cm03": "T-B",   "cm13": "B-T",
    "cm06": "BL-TR", "cm09": "ARC",
}
# 出现重复扫向时的替补池（顺序 = 优先替换顺序；方向都用尽就保持原样）。
SWEEP_POOL = ["cm01", "cm02", "cm03", "cm06", "cm11", "cm12", "cm13", "cm09"]


def dedupe_sweeps(moves):
    """把镜头序列里重复的「扫向」替换成未使用过的扫向（只动扫类，不动推拉/微距）。

    保持列表长度不变（sources 与 moves 严格 1:1，引擎有硬校验），只做原地换镜。
    """
    used = set()
    out = []
    for m in moves:
        d = SWEEP_DIR.get(m)
        if d is None:
            out.append(m)          # 非扫类：不参与去重
            continue
        if d not in used:
            used.add(d)
            out.append(m)
            continue
        repl = None
        for cand in SWEEP_POOL:
            cd = SWEEP_DIR.get(cand)
            if cd and cd not in used:
                repl = cand
                break
        if repl:
            used.add(SWEEP_DIR[repl])
            out.append(repl)
        else:
            out.append(m)          # 扫向已用尽：保持原样，避免死循环
    return out


def _sweep(cv, anchor, z, dx, dy, frac=0.8, jitter=1.6, roll=0.0,
           parallax=180.0, dof=None, vignette=0.0, arc=0.0, on_anchor=False,
           curve=even, speed_scale=1.0):
    """通用平移镜头：在 zoom=z 的合法取景范围内，沿 (dx,dy) 方向缓扫。
    frac：行程占可用范围的比例（越大越"横扫"，越小越"克制"）。
    arc：轻微弧形偏移（水平行程叠加一条正弦弧线，纵向位移）。
    curve：速度曲线（even 近匀速 / curve_in 缓起 / curve_out 缓收）。
      [2026-10-10 用户口径] 给不同扫镜分配不同 curve/parallax/arc，破解
      "所有平移都是同一种滑法"的重复感。
    on_anchor：把视口基准压在锚点上（细节镜要落在工艺区域时用）。注意默认行为的
      基准是 pan 可行范围的中点，等价于画布正中 —— 锚点只影响行程包络、不影响取景；
      所以要"真的对着锚点取景"必须开这个开关。
      行程按「该侧余量」分配：工艺区居中时与常规完全等价，工艺区贴近画面边缘时
      自动退化为单侧行程（工艺区仍在画面内），保证不会出现"镜头几乎不动"的死镜头
      （2026-09-29 用户反馈过这个坑）。
    speed_scale：位移缩放系数（= 每镜时长 / 基准 2s）。时长越短位移同比缩小，
      使峰速恒定在 ≤20% 画宽/s 红线内，不会因提速而"甩镜"。"""
    # 位移按 speed_scale 缩放：frac/jitter/arc 直接决定位移幅度
    frac = frac * speed_scale
    jitter = jitter * speed_scale
    arc = arc * speed_scale
    ax, ay = _clamp_center(cv, anchor[0], anchor[1], z)
    b = pan_bounds(cv, ax, ay, z)
    cx, cy = (b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0
    hx, hy = (b[1] - b[0]) / 2.0 * frac, (b[3] - b[2]) / 2.0 * frac
    if on_anchor:
        # 取景要在锚点上，就要求 pan=0 合法 → 用放宽的边距重算可行范围
        b2 = pan_bounds(cv, ax, ay, z, margin=ANCHOR_MARGIN)
        for axis in (0, 1):
            lo, hi = b2[axis * 2], b2[axis * 2 + 1]
            if hi <= lo:
                continue
            base = min(max(0.0, lo), hi)      # 锚点（尽量；被边界挡到就退到边界）
            A = hx if axis == 0 else hy       # 振幅口径与常规一致（速度不变）
            up, dn = min(A, base - lo), min(A, hi - base)
            c_, h_ = base + (dn - up) / 2.0, (up + dn) / 2.0
            if axis == 0:
                cx, hx = c_, h_
            else:
                cy, hy = c_, h_

    def fn(t):
        e = curve(t)
        u = -1.0 + 2.0 * e
        px = cx + hx * u * dx + jitter * math.sin(t * math.pi * 2 * 0.8)
        py = cy + hy * u * dy + jitter * math.sin(t * math.pi * 2 * 0.65 + 0.9)
        if arc:
            py += arc * math.sin(math.pi * e) * (-1.0 if dx >= 0 else 1.0)
        return Cam(zoom=z, ax=ax, ay=ay, pan=(px, py), roll=roll,
                   parallax=parallax, dof=dof, vignette=vignette)
    return fn


def _dolly(cv, anchor, z0, z1, pan_px=0.0, pan_py=0.0, roll=0.0,
           parallax=180.0, dof=None, vignette=0.0, curve=even, speed_scale=1.0):
    """推/拉镜头：zoom 从 z0 按 curve 变化到 z1（尺寸变化 ≈ (z1/z0-1)），叠加极小位移。
    curve：默认 even（近匀速）；cm07 工艺近景用 curve_out 缓收落定，让推入"停稳"在细节上。
    speed_scale：位移缩放系数（= 每镜时长 / 基准 2s）。时长越短位移同比缩小，
      使峰速恒定在 ≤20% 画宽/s 红线内，不会因提速而"甩镜"。"""
    # 位移按 speed_scale 缩放：pan 与 zoom 增量都围绕中点同比缩小
    pan_px = pan_px * speed_scale
    pan_py = pan_py * speed_scale
    z_mid = (z0 + z1) / 2.0
    z_half = (z1 - z0) / 2.0 * speed_scale
    z0, z1 = z_mid - z_half, z_mid + z_half
    ax, ay = _clamp_center(cv, anchor[0], anchor[1], max(z0, z1))

    def fn(t):
        e = curve(t)
        z = z0 + (z1 - z0) * e
        p0 = _clamp_center(cv, anchor[0], anchor[1], z)
        b = pan_bounds(cv, p0[0], p0[1], z)
        px = min(max(pan_px * (2 * e - 1), b[0]), b[1])
        py = min(max(pan_py * (2 * e - 1), b[2]), b[3])
        return Cam(zoom=z, ax=ax, ay=ay, pan=(px, py), roll=roll,
                   parallax=parallax, dof=dof, vignette=vignette)
    return fn


def build_move_shot(cv, mid, aa, secs=None, craft=None):
    """单镜头运镜：CM-01 ~ CM-10。返回 [(name, cam_fn, secs)]。

    craft: 工艺区域中心（画布坐标）——仅微距类镜头（CM-06/07/08）优先取此点取景，
      让细节镜的画面落在烫金/浮雕/压凹上；其余镜头不受影响（仍用内容锚点）。
    """
    a1 = aa[0] if aa else (cv.ww * 0.5, cv.hh * 0.5)
    if craft and mid in ("cm06", "cm07", "cm08"):
        a1 = craft
    # cm15/cm16 把 craft 当作「单卡框」(归一化 [x,y,w,h]) 来取景，框住某张独立卡；
    #   craft 是 4 元组（框）时才走框住逻辑，否则回落内容锚点（与微距锚点的 2 元组区分）。
    card_box = craft if (mid in ("cm15", "cm16", "cm17") and craft and len(craft) >= 4) else None
    framed = _frame_box(cv, card_box) if card_box else None   # (z, ax, ay)
    center = (cv.ww * 0.5, cv.hh * 0.5)
    s = _int_secs(secs if secs is not None else MOVES[mid]["secs"])
    # 位移缩放：以 BASE_SECS=2s 为基准调校，时长越短位移同比缩小，峰速维持 ≤20% 画宽/s 红线
    speed_scale = (float(secs) if secs is not None else float(MOVES[mid]["secs"])) / BASE_SECS

    if mid == "cm01":      # 左上 → 右下斜向横扫（整套平铺）
        # [2026-09-24 模板校准] 模板镜1 峰速 16% 画宽/s，原 frac=0.55 实测 23.7% → 放缓至 ~16%
        # [2026-10-10] 轻弧线 + 中档 parallax：滑法不再是纯直线，破"同一种平移"观感
        fn = _sweep(cv, center, 1.22, +1.0, +1.0, frac=0.38, jitter=1.6,
                    parallax=200.0, arc=20.0, speed_scale=speed_scale)
    elif mid == "cm02":    # 左 → 右横扫（横向套装）
        # [2026-10-10] 缓起曲线 + 较低 parallax：起手慢、前景浮层浅
        fn = _sweep(cv, center, 1.10, +1.0, 0.0, frac=0.60, jitter=1.4,
                    parallax=170.0, curve=curve_in, speed_scale=speed_scale)
    elif mid == "cm03":    # 上 → 下纵向扫（长卡 / 菜单）
        # [2026-10-10] 缓收曲线 + 较高 parallax：收尾稳、前景浮层深
        fn = _sweep(cv, center, 1.26, 0.0, +1.0, frac=0.75, jitter=1.4,
                    parallax=220.0, curve=curve_out, speed_scale=speed_scale)
    elif mid == "cm04":    # 缓慢 Push In（单张主卡 / Hero）: 尺寸 +8%
        fn = _dolly(cv, a1, 1.06, 1.145, pan_px=6.0, pan_py=-5.0, parallax=190.0,
                    speed_scale=speed_scale)
    elif mid == "cm05":    # 缓慢 Pull Out（细节 → 整套）: 尺寸 −7%
        # [2026-09-24 模板校准] 模板镜5 有明显回位漂移（峰 6.5%）→ 加大 pan 行程，配合末镜定格
        fn = _dolly(cv, a1, 1.17, 1.09, pan_px=-16.0, pan_py=18.0, parallax=190.0,
                    speed_scale=speed_scale)
    elif mid == "cm06":    # 左下 → 右上斜扫（第二角度）
        # craft 已知（图里有工艺区域）时对着工艺区取景，否则维持原有的画布中心起手
        fn = _sweep(cv, a1 if craft else center, 1.22, +1.0, -1.0, frac=0.55,
                    jitter=1.5, parallax=240.0, on_anchor=bool(craft),
                    speed_scale=speed_scale)
    elif mid == "cm07":    # 微距推入定格（烫金 / 浮雕 / 雕刻字等工艺细节）
        # [2026-10-10 用户口径] 工艺细节要"切到近景、展示细节"：
        #   不再用 near-static 微距漂移，改为从宽幅(1.6x)缓推到工艺区高倍(2.2x)、
        #   缓收落定（curve_out 末速归零），让细节真正占满屏并停稳展示。
        #   无工艺框时退到内容锚点、略降倍率(2.0x)。
        z1 = 2.20 if craft else 2.00
        fn = _dolly(cv, a1, 1.60, z1, pan_px=4.0, pan_py=-4.0,
                    parallax=150.0, vignette=0.12, curve=curve_out,
                    speed_scale=speed_scale)
    elif mid == "cm08":    # 微距 Push In（Logo / 姓名 / 工艺）
        fn = _dolly(cv, a1, 1.75, 1.95, pan_px=4.0, pan_py=-4.0,
                    parallax=140.0, vignette=0.10, speed_scale=speed_scale)
    elif mid == "cm09":    # 轻微弧形移动（Hero）
        fn = _sweep(cv, center, 1.15, +1.0, 0.28, frac=0.45, jitter=1.2, arc=42.0,
                    parallax=200.0, speed_scale=speed_scale)
    elif mid == "cm10":    # 固定镜头 + 微推（手持 / 插卡等动作的静态底）
        fn = _dolly(cv, a1, 1.05, 1.075, pan_px=6.0, pan_py=4.0, parallax=170.0,
                    speed_scale=speed_scale)
    elif mid == "cm11":    # 右下 → 左上斜扫（CM-01 反向；同 zoom/行程，仅方向相反）
        # [2026-10-10] 轻弧线 + 缓收曲线：与 cm01 同方向反向、运动性格也不同
        fn = _sweep(cv, center, 1.22, -1.0, -1.0, frac=0.38, jitter=1.6,
                    parallax=180.0, arc=18.0, curve=curve_out, speed_scale=speed_scale)
    elif mid == "cm12":    # 右 → 左横扫（CM-02 反向）
        # [2026-10-10] 缓起曲线 + 较高 parallax
        fn = _sweep(cv, center, 1.10, -1.0, 0.0, frac=0.60, jitter=1.4,
                    parallax=210.0, curve=curve_in, speed_scale=speed_scale)
    elif mid == "cm13":    # 下 → 上竖扫（CM-03 反向）
        # [2026-10-10] 低 parallax：与 cm03 的"深浮层"形成深浅反差
        fn = _sweep(cv, center, 1.26, 0.0, -1.0, frac=0.75, jitter=1.4,
                    parallax=160.0, speed_scale=speed_scale)
    elif mid == "cm14":    # 全景建立缓推（多产品/整套片的开场镜）
        # [2026-10-10 用户口径·案例学习] 参考案例（多张图场景片）以「场景全景缓推」开场，
        #   先给全貌、再硬切到各个产品的细节近景，形成"全景建立 → 逐个细节"的层次。
        #   本镜从最小倍率 1.02（≈全画幅、几乎无位移余量）缓推到 1.13，末速归零停稳；
        #   无横扫（不进 SWEEP_DIR），与后续细节镜性格区分开。
        fn = _dolly(cv, center, 1.02, 1.13, pan_px=3.0, pan_py=-2.0,
                    parallax=170.0, curve=curve_out, speed_scale=speed_scale)
    elif mid == "cm15":    # 手持轻晃：小幅多轴低频摆动（锚点可落在单卡框上），读作"手持展示"
        # 不破坏速度红线：振幅按画宽 5%、频率 0.5Hz，峰速 ≈ 16% 画宽/s；微滚 ≤0.6° 不伤文字。
        z, ax, ay = framed if framed else (1.12, a1[0], a1[1])
        A = 0.05 * cv.vw
        def fn(t):
            ph = 2.0 * math.pi * 0.5 * t
            px = A * math.sin(ph) + 0.45 * A * math.cos(ph * 0.7)
            py = 0.6 * A * math.sin(ph * 0.9 + 1.3)
            roll = math.radians(0.6) * math.sin(ph * 0.8)
            return Cam(zoom=z, ax=ax, ay=ay, pan=(px, py), roll=roll,
                       parallax=150.0, dof=None, vignette=0.06)
        return [(MOVES[mid]["zh"], fn, s)]
    elif mid == "cm16":    # 定格保持：机位完全静止停留（锚点可落在单卡框上，作"读卡"瞬间）
        # 案例里会故意让一张卡定住 1–2s 给观众读文字/工艺；比 cm10「锁定+微推」更像这种克制感。
        z, ax, ay = framed if framed else (1.10, a1[0], a1[1])
        def fn(t):
            return Cam(zoom=z, ax=ax, ay=ay, pan=(0.0, 0.0), roll=0.0,
                       parallax=150.0, dof=None, vignette=0.04)
        return [(MOVES[mid]["zh"], fn, s)]
    elif mid == "cm17":    # 推入单卡：从较低倍率真实推入并放大单卡（运镜，锚点锁在单卡中心）
        # framed = (z_card, card_cx, card_cy)：单卡约占视口 80%（_frame_box）。
        # 速度红线（锚点固定、纯推镜）：视口边缘峰值屏幕速度 ≈ Δz·1.5/(2·z0·secs) ≤ 20% 画宽/s。
        #   反向解出"峰值恰达红线"的起始倍率 z0 = zc/(1+0.267·s)，并抬到全画幅下限 z_fit 以上，
        #   保证是真正的"推镜到卡"且永不破红线；若卡片本就大于全画幅（zc≤z_fit）则改为极缓推近(10%)。
        z_fit = min(cv.vw / float(cv.ww), cv.vh / float(cv.hh))  # ≤此值即可见整套
        zc, ccx, ccy = framed if framed else (1.40, a1[0], a1[1])
        if zc <= z_fit:
            z0 = zc * 0.90                       # 卡片已大于全画幅：极缓推近停稳在卡上
        else:
            z0 = max(z_fit, zc / (1.0 + 0.15 * s))
        ax, ay = _clamp_center(cv, ccx, ccy, zc)
        def fn(t):
            e = curve_out(t)
            z = z0 + (zc - z0) * e
            p = _clamp_center(cv, ax, ay, z)
            return Cam(zoom=z, ax=p[0], ay=p[1], pan=(0.0, 0.0), roll=0.0,
                       parallax=150.0, dof=None, vignette=0.05)
        return [(MOVES[mid]["zh"], fn, s)]
    elif mid == "cm18":    # 画框旋转：正弦往返 roll（起止归零），同时极缓推近
        # [2026-10-13 用户口径] 纯运镜「旋转」镜头。9:16 画布上下本无边距，只有先把取景
        #   放大到 1.3x 再旋转，旋转后的视口才仍落在画布内，避免 BORDER_REPLICATE 在四角
        #   拉出拖影（与 card_pan 的 overscan 同理，但这里用「放大取景」实现）。
        #   锁画布中心、±7° 正弦往返（起止归零 → 序列末尾自动定格画面是正的）。
        z0, z1 = 1.30, 1.38
        A = math.radians(7.0)
        def fn(t):
            z = z0 + (z1 - z0) * even(t)
            p = _clamp_center(cv, cv.ww * 0.5, cv.hh * 0.5, z)
            return Cam(zoom=z, ax=p[0], ay=p[1], pan=(0.0, 0.0),
                       roll=A * math.sin(math.pi * t),
                       parallax=150.0, dof=None, vignette=0.06)
        return [(MOVES[mid]["zh"], fn, s)]
    else:
        raise ValueError("Unknown camera move: %s" % mid)
    return [(MOVES[mid]["zh"], fn, s)]


# ---------------- 多镜头序列（多选镜头 → 一次渲染成一条长片）----------------
# 口径：按勾选顺序依次执行，镜头间**硬切**（与参考运镜一致的剪辑语汇），
#       每镜沿用自身运镜规则与速度红线（≤20% 画宽/s），总时长 = 各镜时长之和。
MAX_SEQ_SHOTS = 9          # 单次拼接的镜头上限（快档 1s/镜 ×9 ≈ 10s；中档 2s/镜 ×5/6 仍 ≤15s）

# 运镜位移的基准时长：镜头 frac/pan/zoom 幅度按 2s 调校，使峰速落在 ≤20% 画宽/s 红线内。
# speed_scale = 每镜时长 / BASE_SECS：时长越短位移同比缩小，保证提速不"甩镜"。
BASE_SECS = 2.0

# 序列末尾定格（2026-09-24 模板校准）：模板视频 14.4–15.0s 完全静止 0.6s 收尾，
# "缓移 → 落定"的收束感是该模板的标志性节奏，多镜序列最后一镜后自动追加。
FREEZE_SECS = 1.0

# 标准五镜序列（用户口径 2026-09-23 固定，不再勾选）：
#   CM-01 斜扫建全貌 → CM-04 缓推主卡 → CM-02 横扫套装 → CM-07 微距工艺 → CM-05 缓拉收尾
# 节奏校准（2026-09-24，对齐模板视频 5653fa67：5 镜 × 3s 硬切 = 15s）：
#   模板各镜峰速 = 16% / 3.4% / 20.5% / 2.1% / 6.5% 画宽/s，钟形速度（缓起缓收、末速归零），
#   末镜后 0.6s 完全定格。cm01/cm07/cm05 参数已按此放缓。
STANDARD_SEQ = ["cm01", "cm04", "cm02", "cm07", "cm05"]
STANDARD_SEQ_NAME = "Standard five-shot (diagonal sweep → slow push on main card → horizontal sweep across the set → macro on craft detail → slow pull out)"


def normalize_moves(layout):
    """把多种入参归一化为镜头 id 列表；**非序列**（单镜 id / 组合布局）返回 []。

    接受：["cm01","cm03"] / "cm01,cm03" / "cm01+cm03" / "cm01 cm03"
    注意：单镜 "cm01"、组合 "reference" 均不含分隔符 → 返回 []，走原有单镜/布局分支。
    """
    if isinstance(layout, (list, tuple)):
        items = [str(x).strip().lower() for x in layout if str(x).strip()]
    elif isinstance(layout, str) and re.search(r"[,\+\s]", layout):
        items = [s.strip().lower() for s in re.split(r"[,\+\s]+", layout) if s.strip()]
    else:
        return []
    return [m for m in items if m][:MAX_SEQ_SHOTS]


def layout_slug(layout):
    """文件名用短标识：序列 → "cm01-cm03"，其余原样返回。"""
    seq = normalize_moves(layout)
    return "-".join(seq) if seq else str(layout)


def build_move_seq(cv, mids, aa, secs=None, crafts=None):
    """多镜序列：按给定顺序逐镜生成（镜头间硬切），secs 可统一覆盖每镜时长。
    锚点按镜头性质轮转取用，避免连续两镜落点雷同：
      · 微距类（cm06/cm07/cm08）优先用工艺区域点，其次纹理密集点；
      · 横扫/弧移类用视口中心 + 内容锚点轮转，保证位移余量。
    crafts: 工艺区域点列表（画布坐标）。微距类镜头按序各取一个 —— 多处工艺都能被照顾到；
      没有工艺框时退到 texture_anchors（纹理密集处），仍然"切向重点"。
    """
    aa = list(aa) if aa else auto_anchors(cv, 3)
    center = (cv.ww * 0.5, cv.hh * 0.5)
    a1 = aa[0] if len(aa) > 0 else center
    a2 = aa[1] if len(aa) > 1 else (cv.ww * 0.62, cv.hh * 0.45)
    a3 = aa[2] if len(aa) > 2 else (cv.ww * 0.38, cv.hh * 0.62)
    crafts = list(crafts) if crafts else []
    det = (a2, a3, a1)              # 微距用：内容点
    if not crafts:                  # 没识别到工艺框 → 纹理/边缘密集处兜底
        t = texture_anchors(cv, 3)
        if t:
            det = tuple(t)
    wide = (center, a1, center, a3)  # 横扫/弧移用：中心起手更稳
    shots, i, mi = [], 0, 0
    for mid in mids:
        if mid not in MOVES:
            raise ValueError("Unknown camera move: %s" % mid)
        if mid in ("cm06", "cm07", "cm08"):
            if crafts:
                anchor = crafts[mi % len(crafts)]
                shots += build_move_shot(cv, mid, [anchor], secs=secs, craft=anchor)
            else:
                shots += build_move_shot(cv, mid, [det[i % len(det)]], secs=secs, craft=None)
            mi += 1
        else:
            shots += build_move_shot(cv, mid, [wide[i % len(wide)]], secs=secs, craft=None)
        i += 1
    # 末镜定格（模板收尾手法：完全静止 0.6s）
    if shots:
        shots.append(_freeze_shot(cv, shots[-1][1]))
    return shots


def _freeze_shot(cv, last_fn):
    """取上一镜 t=1.0 的最终相机参数，生成完全静止的定格镜头。"""
    c = last_fn(1.0)
    def fn(t):
        return Cam(zoom=c.zoom, ax=c.ax, ay=c.ay, pan=(c.panx, c.pany),
                   roll=c.roll, parallax=c.parallax, dof=c.dof, vignette=c.vignette)
    return ("Freeze", fn, FREEZE_SECS)


# ---------------- 按画面构图自动选镜 ----------------

def _content_stats(cv):
    """画面内容统计：宽高比 / 主体组件数 / 烫金占比。"""
    ar = cv.ww / float(cv.hh)
    e = np.abs(cv.depth).astype(np.float32)
    e = cv2.GaussianBlur(e, (0, 0), 25)
    thr = float(e.mean() + e.std() * 0.6)
    m = (e > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(m, 8)
    area = float(cv.ww * cv.hh)
    comps = [s for s in stats[1:] if s[4] > 0.015 * area]
    f = cv.img.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    gold = np.clip((r - b) * 1.4 + (g - b) * 0.9 - 0.18, 0, 1) * np.clip(gray * 1.1, 0, 1)
    gold_ratio = float((gold > 0.25).mean())
    return ar, len(comps), gold_ratio


def recommend_move(cv):
    """按标准指令的适配规则自动选镜。返回 (move_id, reason)。"""
    ar, ncomp, gold = _content_stats(cv)
    if ar >= 1.35:
        return "cm02", "Framing is wide (aspect %.2f), horizontal suite → left-to-right sweep" % ar
    if ar <= 0.75:
        return "cm03", "Framing is tall (aspect %.2f), long card or menu → top-to-bottom pan" % ar
    if ncomp >= 3:
        return "cm01", "Identified %d product components (whole set laid out) → diagonal sweep from top-left to bottom-right" % ncomp
    if gold >= 0.10 and ncomp <= 2:
        return "cm07", "Foil coverage %.0f%% with a concentrated subject — craft detail dominates → macro pan" % (gold * 100)
    if ncomp <= 1:
        return "cm04", "Single hero product (%d components) → very slow camera push" % ncomp
    return "cm01", "Mixed composition (%d components) → diagonal sweep to establish the whole set" % ncomp


def _clamp_center(cv, x, y, z, margin=60.0):
    """把视口中心钳制在 zoom=z 时画布的合法取景范围内。"""
    hx, hy = cv.vw / 2.0 / z + margin, cv.vh / 2.0 / z + margin
    return (min(max(x, hx), cv.ww - hx), min(max(y, hy), cv.hh - hy))


def build_ref_shots(cv, aa):
    """「参考运镜」布局 —— 复刻 2026-09-23 参考片（桌面 2.mp4, 720x1280, 23.6s）
    机位段的量化节奏：
      · 单镜头做一件事，钟形速度（smootherstep：缓起缓收、末速归零 = 每镜收稳）
      · 参考片峰值平移约 35% 画宽/s，本 preset 放慢到 ~19% 画宽/s（用户口径：不用太快）
      · 镜头间硬切，zoom 逐镜递进 1.06 → 1.15 → 1.23，叙事 = 滑推建全貌 → 横移 → 落定特写
      · 叠加 ±4px 低频"呼吸感"，避免机械直线感
    """
    a1 = aa[0]
    a2 = aa[1] if len(aa) > 1 else (cv.ww * 0.5 + 120, cv.hh * 0.55)
    a3 = aa[2] if len(aa) > 2 else (cv.ww * 0.4, cv.hh * 0.6)
    S = 180.0
    shots = []

    # L1 滑推（2.2s）：全景中心 → 主锚点，zoom 1.06 → 1.15
    def s_glide(t, a1=a1):
        e = smoother(t)
        z = 1.06 + 0.09 * e
        ax, ay = _clamp_center(cv, a1[0], a1[1], z)
        sx, sy = _clamp_center(cv, cv.ww / 2.0, cv.hh / 2.0, 1.06)
        b = pan_bounds(cv, ax, ay, z)
        px = min(max((sx - ax) * (1 - e), b[0]), b[1])
        py = min(max((sy - ay) * (1 - e), b[2]), b[3])
        px += 4.0 * math.sin(t * math.pi * 2 * 1.1)
        py += 3.0 * math.sin(t * math.pi * 2 * 0.8 + 1.2)
        return Cam(zoom=z, ax=ax, ay=ay, pan=(px, py), parallax=S)
    shots.append(("glide", s_glide, LAYOUTS["reference"]["secs"][0]))

    # L2 横移（2.2s）：a1 → a2 缓扫，zoom 1.15 恒定
    def s_pan(t, a1=a1, a2=a2):
        e = smoother(t)
        z = 1.15
        mx, my = (a1[0] + a2[0]) / 2, (a1[1] + a2[1]) / 2
        ax, ay = _clamp_center(cv, mx, my, z)
        b = pan_bounds(cv, ax, ay, z)
        p0 = (min(max(a1[0] - ax, b[0]), b[1]), min(max(a1[1] - ay, b[2]), b[3]))
        p1 = (min(max(a2[0] - ax, b[0]), b[1]), min(max(a2[1] - ay, b[2]), b[3]))
        px = p0[0] + (p1[0] - p0[0]) * e + 3.0 * math.sin(t * math.pi * 2 * 0.9)
        py = p0[1] + (p1[1] - p0[1]) * e + 2.5 * math.sin(t * math.pi * 2 * 0.7 + 0.8)
        return Cam(zoom=z + 0.008 * e, ax=ax, ay=ay, pan=(px, py), parallax=S)
    shots.append(("drift", s_pan, LAYOUTS["reference"]["secs"][1]))

    # L3 落定（2.4s）：缓推 1.15 → 1.23 + 微滚转 + 浅景深，末段静止落定
    def s_settle(t, a3=a3):
        e = smoother(t)
        z = 1.15 + 0.08 * e
        ax, ay = _clamp_center(cv, a3[0], a3[1], z)
        b = pan_bounds(cv, ax, ay, z)
        px = min(max(-14 * e + 6 * (1 - e), b[0]), b[1]) + 3.0 * math.sin(t * math.pi * 2 * 0.7)
        py = min(max(10 * e - 5 * (1 - e), b[2]), b[3]) + 2.5 * math.sin(t * math.pi * 2 * 0.55 + 1.0)
        return Cam(zoom=z, ax=ax, ay=ay, pan=(px, py),
                   roll=math.radians(-0.8 + 1.2 * e),
                   parallax=160.0, vignette=0.06)
    shots.append(("settle", s_settle, LAYOUTS["reference"]["secs"][2]))
    return shots


def layout_secs(layout, secs=None):
    """布局/镜头的总时长（秒）。
    单镜可被 secs 覆盖；**多镜序列时 secs 为「每镜时长」**，合计 = secs × 镜数。"""
    seq = normalize_moves(layout)
    if seq:
        return _int_secs(sum(float(secs or MOVES[m]["secs"]) for m in seq) + FREEZE_SECS)
    if layout in MOVES:
        return _int_secs(secs if secs is not None else MOVES[layout]["secs"])
    if layout == "auto":
        return _int_secs(secs if secs is not None else 2.5)
    return _int_secs(sum(LAYOUTS[layout]["secs"]))


def layout_label(layout):
    seq = normalize_moves(layout)
    if seq:
        names = [MOVES[m]["name"].split(" ")[0] for m in seq]
        return "%s (%d shots)" % (" + ".join(names), len(names))
    if layout in MOVES:
        return MOVES[layout]["name"]
    return {"reference": "Reference camera moves (3 shots)", "compact": "Compact 3 shots",
            "full": "Full 4 shots"}.get(layout, layout)


def build_shots(cv, layout, anchors=None, anchors_pct=None, secs=None, crafts=None):
    """返回 [(name, cam_fn, secs)]
    layout: cm01~cm10 / auto（自动选镜） / reference / compact / full
            / 多镜序列（["cm01","cm03"] 或 "cm01,cm03"，按序拼接、硬切）
    anchors: 画布像素 [[x,y]...]; anchors_pct: 画布百分比 [[0-100,0-100]...], 优先。
    crafts: 工艺区域点列表（画布坐标）—— 微距类镜头优先取此点（细节镜取景优先工艺）。
            兼容旧用法：传单个点或单个框也行。"""
    if anchors_pct:
        anchors = [[p[0] / 100.0 * cv.ww, p[1] / 100.0 * cv.hh] for p in anchors_pct]
    aa = auto_anchors(cv, 3) if not anchors else anchors
    crafts = list(crafts) if crafts else []
    seq = normalize_moves(layout)
    if seq:
        return build_move_seq(cv, seq, aa, secs=secs, crafts=crafts)
    if layout in MOVES:
        return build_move_shot(cv, layout, aa, secs=secs,
                               craft=(crafts[0] if crafts else None))
    L = LAYOUTS[layout]
    if layout == "reference":
        return build_ref_shots(cv, aa)
    a1 = aa[0]
    a2 = aa[1] if len(aa) > 1 else (cv.ww * 0.5, cv.hh * 0.55)
    a3 = aa[2] if len(aa) > 2 else (cv.ww * 0.35, cv.hh * 0.65)
    S = 200.0
    shots = []

    # S1 上移: 纵向扫过, 以允许范围中点对称往返
    z = 1.30
    ax, ay = min(max(a1[0], cv.vw / 2 / z + 40), cv.ww - cv.vw / 2 / z - 40), a1[1]
    b = pan_bounds(cv, ax, a1[1], z)
    midy = (b[2] + b[3]) / 2
    amp = min(360.0, (b[3] - b[2]) * 0.45)
    def s_up(t, ax=ax, ay=a1[1], midy=midy, amp=amp):
        e = ease_io(t)
        return Cam(zoom=z + 0.02 * e, ax=ax, ay=ay,
                   pan=(10 - 20 * e, midy + amp - 2 * amp * e), parallax=S)
    shots.append(("up", s_up, L["secs"][0]))

    # S2 横移
    z = 1.26
    ax, ay = min(max(a2[0], cv.vw / 2 / z + 40), cv.ww - cv.vw / 2 / z - 40), min(max(a2[1], cv.vh / 2 / z + 40), cv.hh - cv.vh / 2 / z - 40)
    b = pan_bounds(cv, ax, ay, z)
    midx = (b[0] + b[1]) / 2
    ampx = min(300.0, (b[1] - b[0]) * 0.45)
    def s_pan(t, ax=ax, ay=ay, midx=midx, ampx=ampx):
        e = ease_io(t)
        return Cam(zoom=z + 0.03 * e, ax=ax, ay=ay,
                   pan=(midx - ampx + 2 * ampx * e, 0), parallax=S)
    shots.append(("pan", s_pan, L["secs"][1]))

    # S3 特写: 缓推 + 微滚转 + 浅景深
    z0 = 1.30
    ax, ay = min(max(a2[0], cv.vw / 2 / 1.5 + 40), cv.ww - cv.vw / 2 / 1.5 - 40), min(max(a2[1], cv.vh / 2 / 1.5 + 40), cv.hh - cv.vh / 2 / 1.5 - 40)
    def s_close(t, ax=ax, ay=ay):
        e = ease_io(t)
        return Cam(zoom=z0 + 0.12 * e, ax=ax, ay=ay,
                   pan=(-15 * e, 10 * e), roll=math.radians(-1.5 + 1.8 * e),
                   parallax=180.0, dof=None, vignette=0.07)
    shots.append(("closeup", s_close, L["secs"][2]))

    # S4 细节 (仅 full 布局): 微距缓推
    if layout == "full":
        z0 = 1.45
        ax, ay = min(max(a3[0], cv.vw / 2 / (z0 + 0.12) + 40), cv.ww - cv.vw / 2 / (z0 + 0.12) - 40), \
                 min(max(a3[1], cv.vh / 2 / (z0 + 0.12) + 40), cv.hh - cv.vh / 2 / (z0 + 0.12) - 40)
        def s_detail(t, ax=ax, ay=ay):
            e = ease_io(t)
            return Cam(zoom=z0 + 0.10 * e, ax=ax, ay=ay,
                       pan=(10 * e, -12 * e), roll=math.radians(0.6 - 0.9 * e),
                       parallax=170.0, dof=None, vignette=0.08)
        shots.append(("detail", s_detail, L["secs"][3]))
    return shots


# ---------------- 渲染入口 ----------------

def render_cameramove(src_path, out_path, layout="cm01",
                      anchors=None, progress=None, anchors_pct=None, secs=None,
                      craft=None, ratio=DEFAULT_RATIO):
    """渲染基础运镜成片（单镜头 CM-01~CM-10 / 组合布局 reference·compact·full /
    auto 自动选镜 / 多镜序列 ["cm01","cm03"]，按序硬切拼接）。
    craft: 归一化工艺框 [x,y,w,h]（相对原图 0-1）—— 微距类镜头的取景优先落在此区域。
    progress(done,total) 回调。返回 (耗时, 分镜信息, meta)。"""
    global progress_hook
    progress_hook = progress or (lambda d, t: None)
    t0 = time.time()
    cv = build_canvas(src_path, ratio)
    craft_pts = craft_anchors(cv, craft)
    seq = normalize_moves(layout)
    used, reason = layout, ""
    if layout == "auto":
        used, reason = recommend_move(cv)
    elif seq:
        used = seq
        reason = "Manual selection of %d shots, hard-cut in the chosen order" % len(seq)
    shots = build_shots(cv, used, anchors, anchors_pct, secs=secs, crafts=craft_pts)
    info = []
    # 预览信息: 每镜锚点
    for name, fn, secs in shots:
        c = fn(0.0)
        info.append({"name": name, "secs": secs,
                     "anchor": [round(c.ax), round(c.ay)], "zoom": round(c.zoom, 2)})
    import tempfile
    parts = []
    total = sum(s[2] for s in shots)
    done = 0
    total_frames = int(sum(s[2] * FPS for s in shots))
    for i, (name, fn, secs) in enumerate(shots):
        p = out_path + ".part%d.mp4" % i
        n = int(FPS * secs)
        c0 = fn(0.0)
        dw = cv.dof_layer((c0.ax, c0.ay), c0.dof[0], c0.dof[1]) if c0.dof else None
        def gen(fn=fn, dw=dw, n=n):
            for j in range(n):
                yield render_frame(cv, fn(j / (n - 1)), dof_wgt=dw)
        encode(p, gen(), n, (cv.vw, cv.vh))
        done += n
        parts.append(p)
    # concat
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    lst = out_path + ".list.txt"
    with open(lst, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '%s'\n" % os.path.abspath(p).replace("'", "'\\''"))
    import subprocess
    from . import ffproc
    rc, err = ffproc.run_ffmpeg([ff, "-y", "-f", "concat", "-safe", "0", "-i", lst,
                                 "-c", "copy", out_path], timeout=300, err_tail=400)
    left = drop_temps(parts + [lst])
    if left:
        print("[cammove] %d temp file(s) left behind: %s" % (len(left), left[:3]))
    if rc != 0:
        raise RuntimeError("ffmpeg stitching failed: %s" % err)
    total_secs = sum(s[2] for s in shots)
    meta = {"layout": used if isinstance(used, str) else layout_slug(used),
            "requested": layout_slug(layout), "reason": reason, "ratio": ratio,
            "secs": int(round(total_secs)), "label": layout_label(used),
            "moves": (seq if seq else [used]),
            "shots": [s[0] for s in shots]}
    return time.time() - t0, info, meta


def render_cameramove_multi(src_list, mids, out_path, secs=None, progress=None,
                            crafts=None, ratio=DEFAULT_RATIO):
    """多张照片各跑一段运镜，硬切拼接成一条长片（2026-09-28 用户需求）。
    src_list 与 mids 一一对应：第 i 张照片用 mids[i] 镜头（单镜，时长 secs 或该镜缺省）。
    crafts: 与 src_list 等长的工艺框列表（每个元素是归一化 [x,y,w,h]，或 [ [..],[..] ] 多个框，
      None 表示无）—— 该图跑微距类镜头（CM-06/07/08）时取景优先落在工艺区域
      （2026-09-30 用户口径；2026-10-09 支持一张图给多个工艺框）。
    末尾自动追加 FREEZE_SECS 定格（与多镜序列同口径）。
    返回 (耗时, info, meta)。"""
    global progress_hook
    progress_hook = progress or (lambda d, t: None)
    t0 = time.time()
    crafts = list(crafts) if crafts else []
    if crafts and len(crafts) != len(src_list):
        crafts = []
    if len(src_list) != len(mids):
        raise ValueError("Source image count (%d) does not match shot count (%d)" % (len(src_list), len(mids)))
    for m in mids:
        if m not in MOVES:
            raise ValueError("Unknown camera move: %s" % m)
    parts, info = [], []
    last_cv = last_cam = None
    for i, (src, mid) in enumerate(zip(src_list, mids)):
        cv = build_canvas(src, ratio)
        box = crafts[i] if i < len(crafts) else None
        cpts = craft_anchors(cv, box)
        # cm15/cm16/cm17 需要「单卡框」(4 元组) 来框住某张独立卡；其余镜头只吃锚点(2 元组)。
        craft_arg = box if (mid in ("cm15", "cm16", "cm17") and box) else (cpts[0] if cpts else None)
        shots = build_move_shot(cv, mid, auto_anchors(cv, 3), secs=secs,
                                craft=craft_arg)
        name, fn, s = shots[0]
        c0 = fn(0.0)
        info.append({"src": src, "name": name, "secs": s,
                     "anchor": [round(c0.ax), round(c0.ay)], "zoom": round(c0.zoom, 2),
                     "craft": bool(cpts)})
        p = out_path + ".part%d.mp4" % i
        n = int(FPS * s)
        def gen(cv=cv, fn=fn, n=n):
            for j in range(n):
                yield render_frame(cv, fn(j / (n - 1) if n > 1 else 0.0))
        encode(p, gen(), n, (cv.vw, cv.vh))
        parts.append(p)
        last_cv, last_cam = cv, fn(1.0)   # 保留末张画布供定格帧渲染
    # 末尾定格（模板收尾手法：完全静止 FREEZE_SECS）
    p = out_path + ".part%d.mp4" % len(src_list)
    n = int(FPS * FREEZE_SECS)
    fr = np.clip(render_frame(last_cv, last_cam), 0, 255).astype(np.uint8)
    def gen_f():
        for _ in range(n):
            yield fr
    encode(p, gen_f(), n, (last_cv.vw, last_cv.vh))
    parts.append(p)
    # concat（与 render_cameramove 同：concat demuxer + stream copy）
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    lst = out_path + ".list.txt"
    with open(lst, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '%s'\n" % os.path.abspath(p).replace("'", "'\\''"))
    import subprocess
    from . import ffproc
    rc, err = ffproc.run_ffmpeg([ff, "-y", "-f", "concat", "-safe", "0", "-i", lst,
                                 "-c", "copy", out_path], timeout=300, err_tail=400)
    left = drop_temps(parts + [lst])      # 先清中间产物，再判成败（否则失败时残渣永远留在盘上）
    if left:
        print("[cammove] %d temp file(s) left behind: %s" % (len(left), left[:3]))
    if rc != 0:
        raise RuntimeError("ffmpeg stitching failed: %s" % err)
    total_secs = sum(x["secs"] for x in info) + FREEZE_SECS
    meta = {"layout": "multi", "requested": "multi", "ratio": ratio,
            "reason": "Multi-image sequence: %d photos, one camera move each, hard cuts between them" % len(src_list),
            "secs": round(total_secs, 1),
            "label": "Multi-image, %d photos (%s)" % (len(src_list), "+".join(MOVES[m]["zh"] for m in mids)),
            "moves": list(mids), "shots": [x["name"] for x in info] + ["Freeze"],
            "sources": list(src_list)}
    return time.time() - t0, info, meta


def render_preview_sheet(src_path, out_jpg, layout="compact", anchors=None, anchors_pct=None,
                         ratio=DEFAULT_RATIO):
    """联络表: 每镜 首/中/末 三帧 + 锚点标记。返回 (耗时, info)。"""
    import time as _t
    t0 = _t.time()
    cv = build_canvas(src_path, ratio)
    shots = build_shots(cv, layout, anchors, anchors_pct)
    tiles = []
    info = []
    for name, fn, secs in shots:
        c = fn(0.0)
        info.append({"name": name, "secs": secs, "anchor": [round(c.ax), round(c.ay)]})
        dw = cv.dof_layer((c.ax, c.ay), c.dof[0], c.dof[1]) if c.dof else None
        for t in (0.02, 0.5, 0.98):
            fr = render_frame(cv, fn(t), dof_wgt=dw)
            im = Image.fromarray(np.clip(fr, 0, 255).astype(np.uint8)).resize((240, 427))
            tiles.append(im)
    cols = 3
    rows = len(shots)
    sheet = Image.new("RGB", (cols * 244 + 4, rows * 431 + 4), (245, 245, 246))
    for i, im in enumerate(tiles):
        r_, c_ = divmod(i, cols)
        sheet.paste(im, (4 + c_ * 244, 4 + r_ * 431))
    sheet.save(out_jpg, quality=88)
    return _t.time() - t0, info
