# -*- coding: utf-8 -*-
"""装饰现场展示的 2.5D 运镜引擎（Decor Showcase 专用）。

来源
----
2026-11-02 用户上传的参考成片「6图运镜模仿_口播字幕.mp4」用的就是这套几何位移 +
深度视差的运镜（原型 shotlib.py）。用户要求 Decor Showcase 模版的运镜**沿用这条片子
的手法**：水平平移 / 垂直平移 / 对角平移 / 缓推近 / 拉远展开，全部靠像素搬移实现，
画面不出旋转、不拉伸，竖幅与纸品的直线不会被掰弯。

视口模型
--------
源图按高度铺到一块「画布」（画布高 = 视口高 × CANVAS_SCALE），输出视口在画布上取景。
放大 SCALE 是为了给平移留行程余量（画布只比视口大一点点时，pan 会一路顶到边界，
画面看起来像卡住不动 —— 这个坑此前踩过）。

    sample = anchor + R(-roll)·(p - center)/zoom + pan + parallax(depth, pan)

用法
----
    cam = build_canvas(path)            # -> Canvas(img, depth, w, h)
    fn  = make_move("h", canvas, W, H)  # -> cam_fn(t in 0..1) -> Cam
    frame = warp(canvas, fn(t), W, H)
"""
import math

import numpy as np
import cv2
from PIL import Image, ImageFilter

# 画布相对视口高度的放大系数。1.06 → 9:16 时 1920 高的视口在 2035 高的画布上取景，
# 画面能露出原图约 89%（与参考片 1.0 画布 92% 的取景尺度基本一致），同时留出
# 水平 ±180px / 垂直 ±145px 的平移余量，够五种运镜走满行程。
CANVAS_SCALE = 1.06


# ---------------------------------------------------------------------------
# 读图 / 画布 / 深度
# ---------------------------------------------------------------------------
def _imread_rgb(path):
    """cv2 读图（非 ASCII 路径安全）→ RGB float32。"""
    buf = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cannot read image: %s" % path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)


class CanvasMeta(object):
    """画布几何信息（无像素）：规划运镜只需这些。"""

    __slots__ = ("w", "h", "aspect", "edge")

    def __init__(self, w, h, aspect, edge):
        self.w, self.h = int(w), int(h)
        self.aspect, self.edge = aspect, edge

    # 给定 zoom，视口在画布上的可平移余量（画布 px）
    def slack(self, zoom, vw, vh):
        sx = max(0.0, (self.w - vw / zoom) / 2.0)
        sy = max(0.0, (self.h - vh / zoom) / 2.0)
        return sx, sy


class Canvas(CanvasMeta):
    """铺满画布后的源图 + 相对深度图。"""

    __slots__ = ("img", "depth")

    def __init__(self, img, depth, aspect, edge):
        CanvasMeta.__init__(self, img.shape[1], img.shape[0], aspect, edge)
        self.img, self.depth = img, depth


def probe(path, vw, vh, scale=CANVAS_SCALE):
    """轻量探测：只算画布尺寸 / 宽高比 / 细节强度，不保留像素（省内存）。"""
    rgb = _imread_rgb(path)
    h, w = rgb.shape[:2]
    ch = int(round(vh * scale))
    cw = max(int(round(w * ch / float(h))), int(vw * 1.06))
    return CanvasMeta(cw, ch, w / float(h), _edge_score(rgb))


def _edge_score(rgb):
    """细节强度：缩小到 320 宽后的梯度均值（纸品/文字/花艺细节高，大场景低）。"""
    g = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    h, w = g.shape[:2]
    s = 320.0 / max(w, 1)
    g = cv2.resize(g, (320, max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.clip(mag.mean() / 60.0, 0.0, 1.0))


def build_depth_relief(img, gain_edge=0.045, gain_gold=0.018, dome=0.012):
    """平面浅浮雕深度：基准 0.5 + 穹面 + 边缘浮雕 + 暖金层（值为 ±0.5 的相对深度）。"""
    hh, ww = img.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    nx, ny = (xx / ww - 0.5) * 2, (yy / hh - 0.5) * 2
    dm = dome * np.exp(-(nx * nx + ny * ny) * 1.1)
    gray = img @ np.array([0.299, 0.587, 0.114], np.float32)
    blur = np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(6)), np.float32) / 255.0
    edge = np.clip(np.abs(gray - blur) * 2.0, 0, 1)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    gold = np.clip((r - b) * 1.4 + (g - b) * 0.9 - 0.18, 0, 1) * np.clip(gray * 1.1, 0, 1)
    d = 0.5 + dm + edge * gain_edge + gold * gain_gold
    d = np.asarray(Image.fromarray((np.clip(d, 0, 1) * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(1.2)), np.float32) / 255.0 - 0.5
    return d.astype(np.float32)


def build_canvas(path, vh, vw, scale=CANVAS_SCALE):
    """源图 → 画布（高 = vh*scale，宽按原比例，至少比视口宽 6%）。"""
    rgb = _imread_rgb(path)
    h, w = rgb.shape[:2]
    ch = int(round(vh * scale))
    cw = max(int(round(w * ch / float(h))), int(vw * 1.06))
    big = np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize(
        (cw, ch), Image.LANCZOS), dtype=np.float32)
    return Canvas(big, build_depth_relief(big), w / float(h), _edge_score(rgb))


# ---------------------------------------------------------------------------
# 相机
# ---------------------------------------------------------------------------
class Cam(object):
    __slots__ = ("zoom", "ax", "ay", "panx", "pany", "parallax")

    def __init__(self, zoom, ax, ay, panx=0.0, pany=0.0, parallax=180.0):
        self.zoom, self.ax, self.ay = zoom, ax, ay
        self.panx, self.pany = panx, pany
        self.parallax = parallax


RAMP = 0.12      # 起停斜坡占整镜时长的比例


def _ramp(t):
    """近匀速缓动（梯形速度曲线）：只在首尾各 RAMP 段从静止加速到匀速。

    位置曲线 C1 连续 —— 速度不会突然跳变，中段速度恒定，因此不会出现
    「smoothstep 首尾速度归零」导致的卡顿观感（2026-10-29 定量复盘的结论）。
    """
    r = RAMP
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    norm = 1.0 - r          # 梯形面积（匀速段 + 两个斜坡）归一化
    if t < r:
        return (t * t / (2.0 * r)) / norm
    if t > 1.0 - r:
        return (norm - (1.0 - t) ** 2 / (2.0 * r)) / norm
    return (r / 2.0 + (t - r)) / norm


_GRID = {}


def _grid(vw, vh):
    """视口坐标网格（只算一次）：返回 (x-中心, y-中心) float32。"""
    key = (vw, vh)
    g = _GRID.get(key)
    if g is None:
        yy, xx = np.mgrid[0:vh, 0:vw].astype(np.float32)
        g = (xx - np.float32(vw * 0.5), yy - np.float32(vh * 0.5))
        _GRID[key] = g
    return g


def warp(canvas, cam, vw, vh):
    """按相机参数重采样一帧（双线性 + 深度视差），输出 RGB float32。

    2026-11-02：手写四邻域 gather 实测 1.68s/帧（float64 提升 + 4 次大数组散射），
    12 图要跑半小时；改用 cv2.remap（C 实现，双线性 + 边缘复制）后约 30ms/帧。
    边缘复制与本引擎原本的 clip 到画布边界语义一致。
    """
    img, d_rel = canvas.img, canvas.depth
    gx, gy = _grid(vw, vh)
    z = np.float32(cam.zoom)
    mx = gx / z + np.float32(cam.ax + cam.panx)
    my = gy / z + np.float32(cam.ay + cam.pany)
    # 深度回采（最近邻）后按平移行程叠加视差：前景随镜头反向多走一点
    d_out = cv2.remap(d_rel, mx, my, cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_REPLICATE)
    kx = np.float32(cam.parallax * (cam.panx / 800.0))
    ky = np.float32(cam.parallax * (cam.pany / 800.0))
    if kx or ky:
        mx = mx + d_out * kx
        my = my + d_out * ky
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# 五种运镜（参数取自参考成片）
# ---------------------------------------------------------------------------
FAMILIES = ("h", "v", "diag", "push", "pull")


def make_move(family, canvas, vw, vh, direction=1, zoom=None, strength=1.0,
              diag_cross=False):
    """构造 cam_fn(t) —— 平移行程自动按画布余量收敛，绝不顶到边界卡住。

    diag_cross=True 时对角平移改为「反 Y」轨迹：
      direction>0  → 左下→右上 (BL→TR)
      direction<0  → 右下→左上 (BR→TL)
    默认 diag_cross=False 维持旧口径（direction>0=左上→右下，direction<0=右下→左上），
    Decor Showcase 不受影响。
    """
    ax, ay = canvas.w * 0.5, canvas.h * 0.5

    if family in ("h", "v", "diag"):
        z = zoom or {"h": 1.12, "v": 1.18, "diag": 1.12}[family]
        sx, sy = canvas.slack(z, vw, vh)
        want = {"h": (120.0, 0.0), "v": (0.0, 85.0), "diag": (120.0, 55.0)}[family]
        amp_x = min(want[0] * strength, sx * 0.86)
        amp_y = min(want[1] * strength, sy * 0.86)
        # 行程太小会看起来像静止 → 缩一点 zoom 换取余量（保持中心构图）
        tries = 0
        while tries < 3 and family == "h" and amp_x < 45.0 and z < 1.30:
            z = min(1.30, z + 0.05)
            sx, sy = canvas.slack(z, vw, vh)
            amp_x = min(want[0] * strength, sx * 0.86)
            tries += 1
        while tries < 3 and family == "v" and amp_y < 40.0 and z > 1.02:
            z = max(1.02, z - 0.05)
            sx, sy = canvas.slack(z, vw, vh)
            amp_y = min(want[1] * strength, sy * 0.86)
            tries += 1
        par = 180.0 if family != "v" else 170.0

        def fn(t):
            e = _ramp(t)
            px = -amp_x + 2.0 * amp_x * e
            py = -amp_y + 2.0 * amp_y * e
            if direction < 0:
                px, py = -px, -py
            if family == "diag" and diag_cross and direction > 0:
                py = -py                       # 反 Y：左上→右下 变 左下→右上
            return Cam(z, ax, ay, panx=px, pany=py, parallax=par)
        return fn

    if family == "push":
        z0 = zoom or 1.06
        z1 = z0 + 0.16 * strength
        par = 160.0

        def fn(t):
            e = _ramp(t)
            return Cam(z0 + (z1 - z0) * e, ax, ay,
                       pany=26.0 * strength * e, parallax=par)
        return fn

    # pull：先贴近再退开，露出完整场景
    z1 = zoom or 1.06
    z0 = z1 + 0.15 * strength
    par = 160.0

    def fn(t):
        e = _ramp(t)
        return Cam(z0 + (z1 - z0) * e, ax, ay,
                   pany=-20.0 * strength * e, parallax=par)
    return fn


# ---------------------------------------------------------------------------
# 运镜规划：按图片特征匹配，并且相邻两镜不重复、近期用过的不优先
# ---------------------------------------------------------------------------
def plan_moves(canvases, vw, vh):
    """返回 [(family, direction), ...]，与 canvases 一一对应。

    评分 = 内容契合 + 行程可得性 - 近期重复惩罚：
      内容：横幅→水平平移、竖幅→垂直平移、细节多（纸品/花艺近景）→缓推近、
            开阔少细节→拉远、方图两边都有余量→对角平移；
      行程：平移家族必须真有余量，否则直接判不可用（避免「画面卡住」）；
      去重：上一镜同家族重罚、近 4 镜出现过的按次数递减，保证整片运镜不雷同。
    """
    plan = []
    recent = []
    for c in canvases:
        a, edge = c.aspect, c.edge
        sx, sy = c.slack(1.12, vw, vh)
        score = {
            "h": 1.0 + 0.8 * min(1.0, sx / 240.0) + (0.5 if a >= 1.12 else 0.0),
            "v": 1.0 + 0.8 * min(1.0, sy / 240.0) + (0.5 if a <= 0.85 else 0.0),
            "diag": 0.9 + 0.5 * min(1.0, sx / 240.0) + 0.5 * min(1.0, sy / 240.0),
            "push": 1.0 + 1.3 * edge + (0.2 if a >= 0.9 else 0.0),
            "pull": 1.0 + 1.3 * (1.0 - edge),
        }
        # 硬门槛：没有行程的平移家族不用
        if sx < 55.0:
            score["h"] = -99.0
        if sy < 50.0:
            score["v"] = -99.0
        if sx < 55.0 or sy < 50.0:
            score["diag"] = -99.0

        for f in recent[-4:]:
            score[f] -= 1.6
        if plan:
            score[plan[-1][0]] -= 2.5

        best = max(FAMILIES, key=lambda f: score[f])
        # 平移方向：同一家族连用时反向；否则按镜头序交替，避免整片同向扫
        direction = 1
        if plan and plan[-1][0] == best:
            direction = -plan[-1][1]
        elif best in ("h", "v", "diag"):
            direction = -1 if (len(plan) % 2 == 1) else 1
        plan.append((best, direction))
        recent.append(best)
    return plan
