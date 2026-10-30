# -*- coding: utf-8 -*-
"""单图裁卡运镜（内容驱动 + 按需旋转 + overscan 去条纹）。

与 core.cammove 的差异
----------------------
cammove 用 cv2.remap + 深度视差做 2.5D，镜头仅平移（源码注释「现用镜头全部
roll==0」），无 Yaw/Pitch 透视旋转、无 overscan。本模块用 cv2.warpPerspective +
homography 实现单图仿 3D 的全三维旋转（Yaw/Pitch/Roll），并走内部
OVERSCAN 1.8 放大渲染再缩回输出、旋转镜头额外留 ROT_MARGIN 源图边距，消除
单图仿 3D 旋转时透视取样飞出原图、被 BORDER_REPLICATE 补成条纹的问题。

按需旋转决策树（2026-10-14 终版）
----------------------------------------------------------------------
· 工艺微距（kind=craft，烫金 / 雕刻字）**不旋转** —— 用户反馈旋转「有点晃」，
  改为「切到细节」的微距推近（纯缩放、无 Yaw/Pitch/Roll，细节更大更清楚）；
· 开场只保留两种（用户 2026-10-14 收紧）：
  - hook_scan 横扫整体 = 保留的自线上开场，整图取景匀速横移（稳、全能兜底）；
  - hook_orbit 旋转整体 = 整图级小幅三维转动，**已调稳**（匀速缓动、去 pitch、
    ±2.5°、3.4s，见下方参数注释），全片旋转额度不受它影响；
  （推进 / 揭示 / 甩镜 / 切细节入镜 已移除，不再作为开场可选）
· 外壳纸材 tilt（俯仰 / 横滚）仍保留，与开场旋转合计仍走各自动作路径；
· 其余镜头（整体扫描 / 内页横扫 / 信封推近 / 小卡对角 / 收尾拉远）一律纯平移。
旋转幅度都很小（≤4°），并留源图边距，防透视取样飞出原图被补成条纹。

依赖：仅 cv2 / numpy / imageio_ffmpeg，不 import 工作台其他 core 模块，避免循环依赖。
"""
import os, json, subprocess, tempfile, time
from collections import deque
import cv2
import numpy as np
import imageio_ffmpeg

OVERSCAN = 1.8       # 内部渲染放大倍数，再 resize 回输出
ROT_MARGIN = 1.12    # 有旋转的镜头，源窗口再放大 12% 留边距防飞出
FOC = 1.15
# 叠化时长（2026-10-29 用户反馈「运镜太卡、不顺畅」）：原 0.45s 叠化期间
# 前后两镜都处于各自缓动的首/尾（速度为 0），等于每两镜之间插了半秒"定格"，
# 5-6 个镜头就是 5-6 次顿挫。收到 0.30s，且全片改用近匀速缓动后，叠化期
# 两镜都在匀速运动中交叉，衔接变顺。
XFADE = 0.30
SPEED_LIMIT = 0.26   # 速度红线（窗口宽/秒）；0.20 偏保守，平移像"微挪"，放宽到 0.26
MAX_ROT_SHOTS = 3    # 全片最多 3 个旋转移镜（2026-10-29 放开：用户要求 Product Spotlight
                      # 也要能用「旋转」等多样运镜；仍走下方已调稳的小角度三维旋转，
                      # 边缘框自动降级为纯横滚，不晃）
MAX_TOTAL = 15.0     # 成片总时长上限（秒，含叠化重叠）
MIN_DUR = 1.6        # 单镜最短时长（秒）

# ---------------- 输出画幅（真源） ----------------
# 客户端第三步「Video shape」可选，成片尺寸与取景比例都从这张表取，
# 不再散落在各处的 if ratio == "9x16" else ... 硬编码。
# 取景比例 R = OW / OH 直接参与 _win_for_box / reveal 的窗口计算，
# 所以同一套模版运镜换任何画幅都会自动重新构图（不是把成片裁一刀）。
RATIOS = {
    "9x16": (1080, 1920),   # Portrait · Reels / TikTok / Shorts（模版默认）
    "4x5":  (1080, 1350),   # Portrait · Instagram feed（最高占比）
    "1x1":  (1080, 1080),   # Square   · feed / marketplace
    "16x9": (1920, 1080),   # Landscape· website / YouTube / email header
}
DEFAULT_RATIO = "9x16"


def norm_ratio(r):
    """把客户端传来的画幅写法归一（"9:16" / "9X16" / " 9x16 " → "9x16"）。"""
    s = str(r or "").strip().lower().replace(":", "x").replace("*", "x")
    s = s.replace(" ", "")
    if s in RATIOS:
        return s
    # 兼容 "1080x1920" 这类尺寸写法
    for k, (w, h) in RATIOS.items():
        if s == "%dx%d" % (w, h):
            return k
    return DEFAULT_RATIO


def ratio_size(r):
    """画幅 id → 输出像素尺寸 (OW, OH)。未识别的一律回落默认 9:16。"""
    return RATIOS[norm_ratio(r)]


FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# ---------------- 缓动 ----------------
def smoothstep(t): return t * t * (3.0 - 2.0 * t)
def trap(t, r=0.25):
    r = min(r, 0.49); v = 1.0 / (1.0 - r)
    if t < r: return t * t * v / (2.0 * r)
    if t > 1.0 - r:
        u = 1.0 - t; return 1.0 - u * u * v / (2.0 * r)
    return v * (t - r / 2.0)
def ease_out(t): return 1.0 - (1.0 - t) ** 3    # 快起慢收（起手速度=均速3倍）
def ease_out2(t): return 1.0 - (1.0 - t) ** 2   # 快起慢收·柔版（起手速度=均速2倍）
def ease_in(t): return t ** 3                    # 慢起快收


def _flow_table(ramp=0.05, n=401):
    """近匀速缓动表（2026-10-29 用户反馈「运镜太卡、不顺畅」的核心修复）。

    原来的 smoothstep 在 t=0 / t=1 处速度都为 0：每个镜头都是「静止起步 →
    加速 → 减速 → 停住」，3 秒一镜、5-6 镜连起来就是不停"起步-刹车"，再加上
    叠化期两镜都静止，观感就是一顿一顿的。

    flow 的速度曲线：只在首尾各 ramp(5%) 时长内用 sin² 做平滑加减速（加速度
    连续，没有抽搐），中间 80% 严格匀速。这样镜头中段像轨道滑行，镜头之间
    靠短叠化接力，整片速度感连贯。
    """
    t = np.linspace(0.0, 1.0, n)
    v = np.ones_like(t)
    m0 = t < ramp
    v[m0] = np.sin(0.5 * np.pi * (t[m0] / ramp)) ** 2
    m1 = t > 1.0 - ramp
    v[m1] = np.sin(0.5 * np.pi * ((1.0 - t[m1]) / ramp)) ** 2
    acc = np.cumsum(v)
    acc = (acc - acc[0]) / (acc[-1] - acc[0])
    return t, acc


_FLOW_T, _FLOW_E = _flow_table()


def flow(t):
    """近匀速缓动（首尾各 10% 平滑加减速，中段恒速）。"""
    return float(np.interp(float(t), _FLOW_T, _FLOW_E))


EASE = {"smooth": smoothstep, "trap": lambda t: trap(t, 0.25),
        "out": ease_out, "out2": ease_out2, "in": ease_in, "flow": flow}

# ---------------- 窗口 ----------------
def clamp_win(cx, cy, ww, wh, W, H, ang=0.0):
    """把窗口（连同它旋转后的轴对齐包围盒）收进原图内。

    ang 为该帧最大旋转角(度)。窗口绕中心旋转 θ 后包围盒变为
    (ww·cosθ + wh·sinθ) × (ww·sinθ + wh·cosθ)；先按需等比收缩窗口，再按包围盒
    尺寸夹中心，保证旋转时四角不会飞出原图被 BORDER_REPLICATE 补成条纹。
    """
    a = np.radians(ang); c, s = abs(np.cos(a)), abs(np.sin(a))
    bw, bh = ww * c + wh * s, ww * s + wh * c
    if bw > W or bh > H:
        sc = min(W / float(bw), H / float(bh))
        ww *= sc; wh *= sc
        bw, bh = ww * c + wh * s, ww * s + wh * c
    cx = min(max(cx, bw / 2.0), W - bw / 2.0)
    cy = min(max(cy, bh / 2.0), H - bh / 2.0)
    return cx, cy, ww, wh

def mk(kind, p0, p1, dur, ease="smooth", yaw=(0, 0), pitch=(0, 0), roll=(0, 0),
       tag="", margin=ROT_MARGIN):
    return dict(kind=kind, p0=list(p0), p1=list(p1), dur=dur, ease=ease,
                yaw=yaw, pitch=pitch, roll=roll, tag=tag, margin=margin)

def has_rot(m):
    return abs(m["yaw"][0]) + abs(m["yaw"][1]) + abs(m["pitch"][0]) + abs(m["pitch"][1]) + abs(m["roll"][0]) + abs(m["roll"][1]) > 1e-6

def peak_speed(m, fps=90):
    n = max(4, int(m["dur"] * fps)); f = EASE[m["ease"]]
    prev = None; peak = 0.0
    for k in range(n + 1):
        t = k / n; e = f(t)
        cx = m["p0"][0] + (m["p1"][0] - m["p0"][0]) * e
        cy = m["p0"][1] + (m["p1"][1] - m["p0"][1]) * e
        ww = m["p0"][2] + (m["p1"][2] - m["p0"][2]) * e
        if has_rot(m):
            ww *= ROT_MARGIN
        yw = np.radians(m["yaw"][0] + (m["yaw"][1] - m["yaw"][0]) * e)
        if prev is not None:
            dt = m["dur"] / n
            vx = abs(cx - prev[0]) / dt
            vy = abs(cy - prev[1]) / dt
            vz = abs(ww - prev[2]) / dt * 0.5
            vyaw = abs(yw - prev[3]) / dt * (ww / (2.0 * FOC))
            peak = max(peak, (vx + vy + vz + vyaw) / max(ww, 1.0))
        prev = (cx, cy, ww, yw)
    return peak

def fit_motion2(m, limit=0.26, dmin=2.0, dmax=3.5):
    base0 = list(m["p0"]); base1 = list(m["p1"])
    while peak_speed(m) > limit and m["dur"] < dmax - 1e-6:
        m["dur"] = min(dmax, m["dur"] + 0.05)
    sc = 1.0
    while peak_speed(m) > limit and sc > 0.15:
        sc -= 0.05
        m["p0"] = base0[:]; m["p1"] = [base0[i] + (base1[i] - base0[i]) * sc for i in range(3)]
    m["scale"] = round(sc, 2)
    return m


# ---------------- 透视变换（单图仿 3D） ----------------
def homography(cx, cy, ww, OW, OH, yaw_deg, pitch_deg, roll_deg, W, H):
    """输出像素 (OW×OH) → 源图坐标的单应矩阵（反投影，含相机旋转）。"""
    wh = ww * OH / float(OW)
    A = np.array([[ww / OW, 0.0, cx - ww / 2.0],
                  [0.0, wh / OH, cy - wh / 2.0],
                  [0.0, 0.0, 1.0]])
    f = ww * FOC
    K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1.0]])
    Kinv = np.array([[1.0 / f, 0, 0], [0, 1.0 / f, 0], [0, 0, 1.0]])
    ax, ay, az = np.radians(pitch_deg), np.radians(yaw_deg), np.radians(roll_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    Rot = Rz @ Rx @ Ry
    Hh = K @ Rot @ Kinv
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1.0]])
    H_out2src = T2 @ Hh @ T1 @ A
    return np.linalg.inv(H_out2src)


def gen_shot(src, OW, OH, fps, m, W, H):
    """逐帧单图仿 3D。

    性能红线（2026-10-14 修复）：原实现**所有镜头**都先渲染到 OVERSCAN 1.8 倍
    （9x16 即 1944x3456）再 LANCZOS4 缩回输出，每帧 ~0.095s；但绝大多数镜头
    （整体扫描 / 平移 / 推近 / 拉远 / 对角 / 收尾 / 非旋转开场）**没有旋转**，
    Rot=I 时单应退化为仿射、窗口已被 clamp_win 收进原图，根本不会取样飞出原图、
    也就不存在条纹问题 —— 这些镜头直接按输出分辨率 (OW,OH) 渲染即可，每帧 ~0.009s
    （约 10 倍提速）。只有带 Yaw/Pitch/Roll 的旋转镜头才保留 OVERSCAN+LANCZOS4 路径
    （旋转时透视取样可能飞出原图，需要留边距 + 超采样消除条纹）。
    """
    rot = has_rot(m)
    if rot:
        OWin, OHin = int(round(OW * OVERSCAN)), int(round(OH * OVERSCAN))
    else:
        OWin, OHin = OW, OH
    n = max(2, int(round(m["dur"] * fps)))
    f = EASE.get(m["ease"], smoothstep)
    for k in range(n):
        # 2026-10-29：t 用 k/n（原 k/(n-1)）。原写法每镜的最后一帧 t=1、与下一镜
        # 首帧 t=0 之间位移差为 0（两端都在缓动端点、速度≈0），叠化时两帧都是
        # "停住的画面"，等于每镜尾部多停一帧；同时整片被拉长 1 帧/镜。
        # 改 k/n 后帧间位移均匀，末帧不重复、不静止。
        t = k / float(n); e = f(t)
        cx = m["p0"][0] + (m["p1"][0] - m["p0"][0]) * e
        cy = m["p0"][1] + (m["p1"][1] - m["p0"][1]) * e
        ww = m["p0"][2] + (m["p1"][2] - m["p0"][2]) * e
        yaw = m["yaw"][0] + (m["yaw"][1] - m["yaw"][0]) * e
        pit = m["pitch"][0] + (m["pitch"][1] - m["pitch"][0]) * e
        rol = m["roll"][0] + (m["roll"][1] - m["roll"][0]) * e
        if rot:
            ww *= m.get("margin", ROT_MARGIN)
        # 旋转镜头按「旋转包围盒也能装进原图」收缩窗口 + 夹中心，避免四角飞出补成条纹
        ang = max(abs(yaw), abs(pit), abs(rol)) if rot else 0.0
        cx, cy, ww, wh = clamp_win(cx, cy, ww, ww * OH / float(OW), W, H, ang)
        M = homography(cx, cy, ww, OWin, OHin, yaw, pit, rol, W, H)
        if rot:
            frame_big = cv2.warpPerspective(src, M, (OWin, OHin), flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_REPLICATE)
            frame = cv2.resize(frame_big, (OW, OH), interpolation=cv2.INTER_LANCZOS4)
        else:
            frame = cv2.warpPerspective(src, M, (OW, OH), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
        yield frame


def render(src, OW, OH, fps, out_path, plan):
    W, H = src.shape[1], src.shape[0]
    # 低清源预超分（2026-10-23 用户反馈：微距拉近对不高清的素材很模糊）。
    # macro 推近本质是放大裁切，低清源一拉就露出像素/糊。LANCZOS 先放大到长边
    # 2400（与 Three-Screen 模版同策略），plan 的像素坐标同步等比缩放。
    LONG = 2400
    if max(H, W) < LONG:
        s = LONG / float(max(H, W))
        src = cv2.resize(src, (int(round(W * s)), int(round(H * s))),
                         interpolation=cv2.INTER_LANCZOS4)
        plan = [dict(m, p0=[v * s for v in m["p0"]], p1=[v * s for v in m["p1"]])
                for m in plan]
        W, H = src.shape[1], src.shape[0]
    D = max(1, int(round(XFADE * fps)))
    cmd = [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
           "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "18", "-preset", "veryfast", "-movflags", "+faststart", out_path]
    err_f = tempfile.TemporaryFile()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=err_f, stderr=err_f)
    t0 = time.time(); kept = 0
    def wf(fr):
        nonlocal kept
        p.stdin.write(fr.tobytes()); kept += 1
    try:
        carry = []
        for si, m in enumerate(plan):
            g = gen_shot(src, OW, OH, fps, m, W, H)
            head = []
            for _ in range(D):
                try: head.append(next(g))
                except StopIteration: break
            if carry and head:
                n = min(len(carry), len(head))
                for i in range(n):
                    w = (i + 1) / float(n + 1)
                    wf(cv2.addWeighted(carry[i], 1.0 - w, head[i], w, 0.0))
            elif head:
                for fr in head: wf(fr)
            buf = deque(maxlen=D + 1)
            for fr in g:
                buf.append(fr)
                if len(buf) > D:
                    wf(buf.popleft())
            carry = list(buf)
        for fr in carry:
            wf(fr)
        p.stdin.close(); p.wait(timeout=900)
    except Exception:
        try: p.kill()
        except Exception: pass
        p.wait(timeout=10); raise
    finally:
        err_f.seek(0); tail_txt = err_f.read(1500).decode("utf-8", "ignore"); err_f.close()
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg rc={p.returncode}\n{tail_txt}")
    print(f"  -> {out_path}  frames={kept}  {time.time()-t0:.1f}s")


def make_strip(src, OW, OH, W, H, plan, out_path, cols=7):
    fps = 30
    cells = []
    for m in plan:
        g = gen_shot(src, OW, OH, fps, m, W, H)
        fr = [f for f in g]
        cells.append(fr[len(fr) // 2])
    pad = 6
    cw = OW // 3; ch = OH // 3
    rows = (len(cells) + cols - 1) // cols
    strip = np.full((rows * (ch + pad), cols * (cw + pad), 3), 18, np.uint8)
    for i, c in enumerate(cells):
        r, col = divmod(i, cols)
        c = cv2.resize(c, (cw, ch))
        y0, x0 = r * (ch + pad), col * (cw + pad)
        strip[y0:y0+ch, x0:x0+cw] = c
    ok = cv2.imencode(".jpg", strip)[1].tofile(out_path)
    print(f"  -> {out_path}  ok={ok}")


# ---------------- 按需旋转决策树 ----------------
def refine_focus_pt(img, box):
    """把「切细节」的焦点从识别框中心精确到框内细节最密的点（2026-10-26）。

    问题：识别框常常比真正的重点（烫金字、雕花、花体字句）大一圈，框中心
    未必落在重点上——微距推近后重点偏在画面边缘甚至出画。修法：在框内做
    边缘密度（Sobel 幅度）分块投票，取「细节密度 × 居中权重」最高的块，
    再在该块内用边缘像素质心收一点，返回归一化焦点 [fx, fy]。

    保守护栏（宁可不动也不切飞）：
    · 框太小（<64px）或整框平坦（边缘密度方差极低）→ 返回 None，保持框心；
    · 焦点只允许在原框内移动（先内缩 12% 防贴边），不会跑出识别框；
    · 密度最高的块若紧贴框缘，居中权重会把它压回去。

    img   : BGR ndarray（整图）
    box   : [x, y, w, h] 归一化（0-1），与 regions 的 box 语义一致
    返回  : [fx, fy] 归一化整图坐标；无法精确时 None
    """
    try:
        H, W = img.shape[:2]
        x, y, bw, bh = [float(v) for v in box]
        px0, py0 = max(0, int(x * W)), max(0, int(y * H))
        px1, py1 = min(W, int((x + bw) * W)), min(H, int((y + bh) * H))
        if px1 - px0 < 64 or py1 - py0 < 64:
            return None
        roi = img[py0:py1, px0:px1]
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        # 金色偏置（2026-10-27）：框内同时有正文文字与烫金徽记/花纹时，纯边缘密度
        # 会偏向笔画更密的正文，把推近焦点带偏（诊断实锤：烫金.jpg 的 macro 切到了
        # 正文而不是金色 CJ 徽记）。对金色像素（烫金特征：H 10-38/S>60/V>70）加权，
        # 焦点优先落在金色细节上；框内无金色时权重退化为 0.35×密度，排序不受影响。
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            gm = ((hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 38) &
                  (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 70))
            if gm.any():
                gf = cv2.GaussianBlur(gm.astype(np.float32), (0, 0), 9)
                mag = mag * (0.35 + 0.65 * gf)
        except BaseException:
            pass
        # 6x6 分块：密度 = 块内平均边缘幅度；居中权重 = 1 - 归一化块心到框心距离
        nb = 6
        bh_, bw_ = mag.shape
        best, bfx, bfy = -1.0, None, None
        for iy in range(nb):
            for ix in range(nb):
                y0, y1 = int(iy * bh_ / nb), int((iy + 1) * bh_ / nb)
                x0, x1 = int(ix * bw_ / nb), int((ix + 1) * bw_ / nb)
                dens = float(mag[y0:y1, x0:x1].mean())
                # 块心在框内的相对位置（0-1）
                ry = (iy + 0.5) / nb
                rx = (ix + 0.5) / nb
                cen = 1.0 - (abs(rx - 0.5) + abs(ry - 0.5))   # 0~1，越靠框心越大
                score = dens * (0.35 + 0.65 * cen)
                if score > best:
                    # 块内边缘像素质心（再往细节聚一步）
                    sub = mag[y0:y1, x0:x1]
                    thr = sub.mean() + sub.std()
                    ys, xs = np.nonzero(sub > max(thr, 1e-3))
                    if len(xs) >= 12:
                        cxi = x0 + float(xs.mean())
                        cyi = y0 + float(ys.mean())
                    else:
                        cxi, cyi = x0 + (x1 - x0) / 2.0, y0 + (y1 - y0) / 2.0
                    best, bfx, bfy = score, cxi, cyi
        if bfx is None or best <= 0:
            return None
        # 焦点收在原框内（内缩 12%），防推近后重点贴边出画
        fx = (px0 + bfx) / float(W)
        fy = (py0 + bfy) / float(H)
        ix0, ix1 = x + 0.12 * bw, x + 0.88 * bw
        iy0, iy1 = y + 0.12 * bh, y + 0.88 * bh
        fx = min(max(fx, ix0), ix1)
        fy = min(max(fy, iy0), iy1)
        return [round(fx, 4), round(fy, 4)]
    except BaseException:
        return None


def _edge_margin(box):
    """取景框中心距四边最近的归一化距离（0-0.5）。"""
    cx_n = box[0] + box[2] / 2.0
    cy_n = box[1] + box[3] / 2.0
    return min(cx_n, 1.0 - cx_n, cy_n, 1.0 - cy_n)


def _classify_box(box):
    """无视觉模型时，按归一化框的位置/尺寸启发式推断内容种类（自动识别兜底）。

    有火山 API key 时由 craftdetect 直接给出 craft 标签，本函数仅在 CV 卡片
    框（best_card_boxes 不分类）上做位置/面积推断：大块→内页主体、顶部→信封封蜡、
    底部→外壳纸材、极小→小卡。
    """
    x, y, w, h = box
    cy = y + h / 2.0
    area = w * h
    if area >= 0.22:
        return "inner"          # 大块主体 / 内页
    if cy < 0.30:
        return "envelope"       # 顶部信封 / 封蜡
    if cy > 0.72:
        return "material"       # 底部外壳 / 纸材
    if area <= 0.06:
        return "card"           # 小卡
    return "inner"              # 其余视为内页局部


def _auto_move_for(kind, idx):
    """未显式指定 move 时，按内容种类给定默认运镜。

    2026-10-29 用户口径收紧：自动运镜只保留三类 —— **旋转**（orbit / tilt）、
    **平移（不同角度方向）**（pan 左右 / pan_rl 右左 / pan_v 上下 / pan_vr 下上 /
    diag 对角）、**拉远**（pull / reveal）。不再自动产 macro 推近镜头
    （macro 代码保留，仅显式 regions 传 move="macro" 时可用）。
    """
    if kind in ("whole",):
        return "scan"
    if kind in ("craft",):
        # 工艺（烫金/雕刻/浮雕/贴花）→ 小角度旋转展示金属反光随角度的变化
        # （2026-10-29 用户：「就保留旋转，平移，拉远」；原「工艺只推近不旋转」
        # 口径以本条为准）
        return "orbit"
    if kind in ("hero",):
        return "orbit"
    if kind in ("text", "inner", "envelope"):
        # 内页 / 信封：平移横扫保可读性，方向按序轮换（左右 / 右左 / 上下）
        return ("pan", "pan_rl", "pan_v", "pan_vr")[idx % 4]
    if kind in ("material",):
        return "tilt"           # 外壳纸材 → 俯仰/横滚看纸纹（旋转类）
    # 通用卡片：平移各方向 / 对角 / 旋转 / 拉远 轮转（不出推近）
    pool = ["pan", "diag", "orbit", "pan_v", "pull"]
    return pool[idx % len(pool)]


def _win_for_box(box, fill, R, W, H, focus=None):
    """box 归一化语义为 [x, y, w, h]（相对原图 0-1），与 regions 文档 / _edge_margin /
    _classify_box 保持一致。早期版本误按 [x1,y1,x2,y2] 解析，导致 x2<x1 时宽度为负、
    取景整片错乱（黑框/半卡），已修正。

    focus: 可选 [fx, fy] 归一化焦点（refine_focus_pt 产出）——窗口尺寸仍由 box 决定，
    但中心移到焦点上（再 clamp 进原图）。「切细节切到重点」的关键：识别框偏大时
    框心≠重点，推近的镜头必须对准框内细节最密处。
    """
    x, y, bw, bh = box
    cw, ch = bw * W, bh * H
    ww = max(cw / fill, ch / fill * R)
    wh = ww / R
    if focus and len(focus) == 2:
        try:
            cx = float(focus[0]) * W
            cy = float(focus[1]) * H
        except Exception:
            cx, cy = (x + bw / 2.0) * W, (y + bh / 2.0) * H
    else:
        cx, cy = (x + bw / 2.0) * W, (y + bh / 2.0) * H
    cx, cy, ww, wh = clamp_win(cx, cy, ww, wh, W, H)
    return cx, cy, ww


def build_plan_from_regions(regions, OW, OH, W, H, target_secs=None):
    """把内容区域列表编排成单图裁卡运镜计划（运镜由内容 kind 决定，不做强制兜底）。

    regions: list of dict，每个：
        box   : [x,y,w,h] 归一化（相对原图 0-1）
        kind  : "whole"(整体) / "text"(内页文字) / "craft"(工艺微距) / "hero"(主体) /
                "card"(小卡) / "envelope"(信封封蜡) / "material"(外壳纸材) / "inner"(内页局部)
        move  : "scan" / "pan" / "macro"(推) / "pull"(拉) / "diag"(对角) /
                "orbit"(轨道旋转) / "tilt"(俯仰) / "reveal"(收尾拉远) /
                "hook_scan"(横扫整体，保留的自线上开场版本) /
                "hook_orbit"(旋转整体，整图级小幅旋转，已调稳) /
                注：开场钩子仅 hook_scan / hook_orbit 两种（2026-10-14 收紧，
                推进/揭示/甩镜/切细节入镜已移除，不再作为开场可选）。
                —— 不指定时按 kind 分配（工艺->orbit，信封->推近，内页->平移，
                多张卡在 推/平移/对角/拉 间轮换）；不硬加运动类型。
                hook_* 只用于第一镜（开场钩子，由编排层按图片内容挑选，见
                server._pick_hook）：统一 ease-out 把速度前置制造起势，
                幅度仍受 20% 速度红线约束（稳而不飘）。
        label : 展示名（进 tag）
        fill  : 可选，取景窗口占框比例，默认 0.9
    返回 shot plan 列表；末尾自动补一个 reveal 拉远（除非已有 move="reveal"）。
    """
    R = OW / float(OH)
    # ---- 1) 解析 + 自动分配运镜 ----
    specs = []          # (box, kind, label, fill, move)
    for idx, rg in enumerate(regions):
        box = rg["box"]; kind = rg.get("kind", "card")
        label = rg.get("label", ""); fill = rg.get("fill", 0.9)
        focus = rg.get("focus")
        move = rg.get("move")
        if move is None:
            move = _auto_move_for(kind, idx)
        specs.append([box, kind, label, fill, move, focus])
    # 旋转额度（2026-10-29 用户口径：「就保留旋转，平移（不同角度方向），拉远」）：
    # 旋转是主力运镜之一，最多放行 MAX_ROT_SHOTS(3) 个，超出才降级平移。
    # 排序优先级：烫金/雕刻字等工艺（反光随角度变化最上镜）> 其他工艺 > 外壳纸材。
    rot_idx = [i for i, s in enumerate(specs) if s[4] in ("orbit", "tilt")]
    if rot_idx:
        craft_rot = [i for i in rot_idx if specs[i][1] == "craft"]
        pref = [i for i in craft_rot
                if any(k in specs[i][2] for k in ("雕刻", "烫", "金"))]
        order = (pref + [i for i in craft_rot if i not in pref]
                 + [i for i in rot_idx if i not in craft_rot])
        keep = set(order[:MAX_ROT_SHOTS])
        for i in rot_idx:
            if i not in keep:
                specs[i][4] = "pan"
    # ---- 2) 生成镜头：运镜完全由内容 kind 决定，不再硬凑运动类型 ----
    # 用户明确否掉「每个镜头都旋转 / 为凑类型硬加运动」，这里不做任何强制兜底：
    # 识别到工艺(craft)才用 orbit 旋转，信封用推近，内页用平移，收尾拉远。
    plan = []
    rot_used = 0
    for box, kind, label, fill, move, focus in specs:
        # 全局旋转限额：旋转移镜（orbit/tilt）最多 MAX_ROT_SHOTS 个，超出降级为平移，
        # 避免同一区域重复出多个工艺镜头、或每个镜头都在晃。
        if move in ("orbit", "tilt") and rot_used >= MAX_ROT_SHOTS:
            move = "pan"
        cx, cy, ww = _win_for_box(box, fill, R, W, H, focus=focus)
        tag = label
        if move in ("hook_scan", "hook_orbit"):
            # 开场钩子镜头（2026-10-14 收紧为 2 种：横扫整体 / 旋转整体）。
            # 两者都走匀速缓动 (smooth) + 20% 速度红线，冲而不晃、稳而不飘。
            if move == "hook_scan":           # 保留的原线上开场：整图视角匀速横移
                # 与发布过的那版开场一致（整图取景、窗口恒定、±25% 横移、匀速缓动）。
                # 用户 2026-10-14 明确要求"保留之前线上的开篇"，作为稳、全能的兜底开场。
                # 2026-10-27 修复「第一幕完全静止」：整图窗口经 clamp_win 收缩后恰好
                # 贴满原图边界，gen_shot 逐帧把中心夹死 → ±25% 行程全部被裁掉，横扫
                # 退化成静帧。把窗口缩到 ≤ W/1.4，clamp 后仍留 ≈0.29W 真实横移行程
                # （约 1/3 屏宽，肉眼明确可见），观感仍是"整图开场扫过"。
                sx, sy, sw = _win_for_box([0.0, 0.0, 1.0, 1.0], 0.9, R, W, H)
                sw = min(sw, W / 1.4)
                sx, sy, sw, _wh = clamp_win(W / 2.0, H / 2.0, sw, sw / R, W, H)
                plan.append(mk("hook_scan", [sx - sw * 0.32, sy, sw], [sx + sw * 0.32, sy, sw],
                               3.6, "flow", tag=tag + " · 整体扫描"))
            else:                             # hook_orbit：整图小幅环绕（带旋转，已调稳版）
                # 用户 2026-10-14：「开场旋转的镜头也保留」。原版晃动来自三处叠加：
                # ① ease-out 起手速度是均速的 2~3 倍（起步猛）；② 起点先放大到 1.06x
                # 再推到 0.90x（先拉远再推近，尺度反向）；③ yaw+pitch+roll 三轴同动，
                # pitch 最容易被看成"上下摆"。稳版改为：匀速缓动、起点不放大、去掉
                # pitch、幅度 ±2.5°、时长 3.4s（补齐后约 4.3s）——稳而不飘。
                # 本镜是"整图级"小幅转动，与内容镜头的旋转额度互不抢占（不占 rot_used）。
                if _edge_margin(box) >= 0.18:
                    # 2026-10-27：±2.5°/7% 推近在 3.4s 里太弱（用户反馈"基本没运镜"），
                    # 加强到 ±3.2° yaw + 10% 推近，仍远低于晃动阈值。
                    plan.append(mk("hook_orbit", [cx, cy, ww], [cx, cy, ww * 0.90],
                                   3.4, "flow", yaw=(3.2, -3.2), roll=(3.2, -3.2),
                                   margin=1.6, tag=tag + " · 旋转整体"))
                else:
                    plan.append(mk("hook_orbit", [cx, cy, ww], [cx, cy, ww * 0.93],
                                   3.4, "flow", roll=(3.2, -3.2), tag=tag + " · 旋转整体"))
        elif move in ("tour_h", "tour_v"):
            # ---- 开场巡览（2026-10-29 用户口径：一开始先各种平移扫描，把卡片都
            # 走一遍，再进小卡/内页细节）----
            # tour_h：横向满幅扫（从卡片并集左缘扫到右缘），把平铺的卡片依次带进画面；
            # tour_v：斜向回扫（右→左 + 上→下），换个方向再走一遍。
            # 窗口大小自适应：竖屏输出下方图/竖图几乎一屏装下整图，横向行程会被压到
            # 20% 图宽以下，"扫"看起来像轻微抖动 —— 这时把窗口收小（近景巡览），
            # 换取 26%~30% 图宽的真实行程。
            _ux0 = max(0.0, min(1.0, box[0])) * W
            _ux1 = max(0.0, min(1.0, box[0] + box[2])) * W
            _uy0 = max(0.0, min(1.0, box[1])) * H
            _uy1 = max(0.0, min(1.0, box[1] + box[3])) * H

            def _tour_win(frac):
                _wh = H * frac
                _ww = min(W, _wh * R)
                return _ww, _ww / R

            if move == "tour_h":
                _tg, _f, _need = "巡览 →", 0.92, 0.30
            else:
                _tg, _f, _need = "巡览 ←", 0.78, 0.26
            ww_c, wh_c = _tour_win(_f)
            while (W - ww_c) < W * _need and _f > 0.44:
                _f -= 0.08
                ww_c, wh_c = _tour_win(_f)
            if move == "tour_h":
                if _ux1 - _ux0 <= ww_c:                 # 一屏装得下 → 围绕中心扫
                    _cx = min(max((_ux0 + _ux1) / 2.0, ww_c / 2.0), W - ww_c / 2.0)
                    x0, x1 = _cx - ww_c * 0.15, _cx + ww_c * 0.15
                else:
                    x0 = min(max(_ux0 + ww_c * 0.5, ww_c / 2.0), W - ww_c / 2.0)
                    x1 = min(max(_ux1 - ww_c * 0.5, ww_c / 2.0), W - ww_c / 2.0)
                y0 = y1 = min(max((_uy0 + _uy1) / 2.0, wh_c / 2.0), H - wh_c / 2.0)
            else:
                if _ux1 - _ux0 <= ww_c:
                    _cx = min(max((_ux0 + _ux1) / 2.0, ww_c / 2.0), W - ww_c / 2.0)
                    x0, x1 = _cx + ww_c * 0.13, _cx - ww_c * 0.13
                else:
                    x0 = min(max(_ux1 - ww_c * 0.5, ww_c / 2.0), W - ww_c / 2.0)
                    x1 = min(max(_ux0 + ww_c * 0.5, ww_c / 2.0), W - ww_c / 2.0)
                if _uy1 - _uy0 <= wh_c:
                    _cy = min(max((_uy0 + _uy1) / 2.0, wh_c / 2.0), H - wh_c / 2.0)
                    y0, y1 = _cy - wh_c * 0.10, _cy + wh_c * 0.10
                else:
                    y0 = min(max(_uy0 + wh_c * 0.5, wh_c / 2.0), H - wh_c / 2.0)
                    y1 = min(max(_uy1 - wh_c * 0.5, wh_c / 2.0), H - wh_c / 2.0)
            # 两边都夹进原图，且保证真的在动（否则退化成静帧）
            _c0 = clamp_win(x0, y0, ww_c, wh_c, W, H)
            _c1 = clamp_win(x1, y1, ww_c, wh_c, W, H)
            x0, y0 = _c0[0], _c0[1]
            x1, y1 = _c1[0], _c1[1]
            _run = max(abs(x1 - x0), abs(y1 - y0)) / max(1e-6, ww_c)
            if _run < 0.06:                            # 夹完后几乎没行程 → 给最小位移
                _dc = ww_c * 0.12 * (1.0 if move == "tour_h" else -1.0)
                x0, x1 = x0 - _dc, x1 + _dc
                _run = 0.24
            # 时长按行程定（目标速度 ≈0.24 窗宽/秒）：短行程不会慢到"像没动"，
            # 长行程也不会快到发飘。
            _d = min(3.20, max(2.80, _run / 0.24))
            plan.append(mk("tour", [x0, y0, ww_c], [x1, y1, ww_c], round(_d, 2), "flow",
                           tag=tag + " · " + _tg))
        elif move == "scan":
            # 整体扫描：纯平移、窗口尺寸恒定（不边扫边推近，避免单图 warp 下机位
            # 晃动观感）；横移 ±25% 视窗宽 + 匀速缓动 + 速度压到 20% 红线内，
            # 让扫动方向感与幅度都明显，同时保留稳定感。
            p0 = [cx - ww * 0.32, cy, ww]; p1 = [cx + ww * 0.32, cy, ww]
            plan.append(mk("scan", p0, p1, 3.6, "flow", tag=tag + " · 扫"))
        elif move in ("pan", "pan_rl", "pan_v", "pan_vr"):
            # 平移（不同角度方向，2026-10-29 用户口径）：窗口尺寸恒定、纯位移，
            # 方向分为 左→右 / 右→左 / 上→下 / 下→上，相邻镜头方向去重见下。
            # 2026-10-29b 行程鲁棒化：窗口贴满整图时 clamp 会把行程吃成 0（实测
            # 方图纵向平移整镜静止）——先按可动空间换轴/缩窗，再按
            # 「行程 = min(2×幅度, 可动空间×0.9)、起点尽量覆盖框心」生成端点。
            _dx = ww * 0.22
            _dy = ww / max(1e-6, R) * 0.13
            _hx = max(0.0, W - ww)
            _hy = max(0.0, H - ww / R)
            if move in ("pan_v", "pan_vr") and _hy < _dy * 1.2:
                move = "pan" if move == "pan_v" else "pan_rl"
            if move in ("pan", "pan_rl") and _hx < _dx * 1.2:
                move = "pan_v" if move == "pan" else "pan_vr"
            if _hx < _dx * 1.2 and _hy < _dy * 1.2:
                ww *= 0.72                      # 两轴都没空间 → 缩窗近景化
                cx, cy, ww, _tw = clamp_win(cx, cy, ww, ww / R, W, H)
                _dx = ww * 0.22
                _dy = ww / max(1e-6, R) * 0.13
            if move in ("pan", "pan_rl"):
                _lo, _hi = ww / 2.0, W - ww / 2.0
                _run = min(2.0 * _dx, (W - ww) * 0.9)
            else:
                _lo, _hi = (ww / R) / 2.0, H - (ww / R) / 2.0
                _run = min(2.0 * _dy, (H - ww / R) * 0.9)
            _c = cx if move in ("pan", "pan_rl") else cy
            _s = min(max(_c - _run / 2.0, _lo), max(_lo, _hi - _run))
            _e = _s + _run
            if move == "pan":
                p0 = [_s, cy, ww]; p1 = [_e, cy, ww]; _tg = "平移 →"
            elif move == "pan_rl":
                p0 = [_e, cy, ww]; p1 = [_s, cy, ww]; _tg = "平移 ←"
            elif move == "pan_v":
                p0 = [cx, _s, ww]; p1 = [cx, _e, ww]; _tg = "平移 ↓"
            else:
                p0 = [cx, _e, ww]; p1 = [cx, _s, ww]; _tg = "平移 ↑"
            plan.append(mk("pan", p0, p1, 2.6, "flow", tag=tag + " · " + _tg))
        elif move == "macro":
            # 2026-10-28：聚焦细节（工艺微距）不再切太近 —— 推近终点放宽，
            # 高清源 0.85x(≈1.18 倍) / 中清 0.88x / 低清 0.90x，保留"看清工艺"
            # 的意图，但避免把细节放得过大、边缘失焦发糊（用户反馈切太近）。
            _long = float(max(W, H))
            if kind == "craft":
                # 工艺细节（烫金/雕刻字）：用户 2026-10-14「烫金不用旋转，可以切到细节
                # 展示」——去掉 Roll 微晃，改成微距推近。纯缩放，零旋转、零条纹。
                _end = 0.85 if _long >= 3200 else (0.88 if _long >= 2600 else 0.90)
                plan.append(mk("macro", [cx, cy, ww], [cx, cy, ww * _end], 3.0, "flow",
                               tag=tag + " · 细节推近"))
            else:
                # 普通推近（信封等）维持原口径（2026-10-27）
                _end = 0.70 if _long >= 3200 else (0.78 if _long >= 2600 else 0.85)
                plan.append(mk("macro", [cx, cy, ww], [cx, cy, ww * _end], 2.2, "flow",
                               tag=tag + " · 推近"))
        elif move == "pull":
            p0 = [cx, cy, ww * 0.78]; p1 = [cx, cy, ww]
            plan.append(mk("pull", p0, p1, 2.2, "flow", tag=tag + " · 拉远"))
        elif move == "diag":
            # 对角平移：方向按框在画面左右侧自动反向（左半 → 右下扫，右半 → 左上扫），
            # 相邻两个对角镜头自然不同向；窗口恒定、纯位移（2026-10-29 口径）。
            _sg = 1.0 if (box[0] + box[2] / 2.0) < 0.5 else -1.0
            p0 = [cx - ww * 0.13 * _sg, cy + ww * R * 0.10, ww]
            p1 = [cx + ww * 0.13 * _sg, cy - ww * R * 0.10, ww]
            plan.append(mk("diag", p0, p1, 2.4, "flow", tag=tag + " · 对角扫"))
        elif move == "tilt":
            p0 = [cx, cy, ww]; p1 = [cx, cy, ww * 0.88]
            if _edge_margin(box) >= 0.18:
                # 中心纸材：小角度俯仰 + 微横滚，留 1.5x 源边距防飞出条纹
                plan.append(mk("tilt", p0, p1, 2.2, "flow", pitch=(3.5, -3.5),
                               roll=(2.5, -2.5), margin=1.5, tag=tag + " · 俯仰"))
            else:
                # 边缘只给横滚，避免 Yaw/Pitch 飞出原图
                plan.append(mk("tilt", p0, p1, 2.2, "flow", roll=(5.0, -5.0),
                               tag=tag + " · 横滚"))
            rot_used += 1
        elif move == "orbit":
            p0 = [cx, cy, ww]; p1 = [cx, cy, ww * 0.88]
            if _edge_margin(box) >= 0.18:
                # 中心工艺/主体：小角度环绕 + 留 1.6x 源边距
                yaw, pit, rol, tg = (3.0, -3.0), (2.0, -2.0), (4.0, -4.0), "轨道旋转"
                plan.append(mk("orbit", p0, p1, 2.4, "flow", yaw=yaw, pitch=pit,
                               roll=rol, margin=1.6, tag=tag + " · " + tg))
            else:
                # 边缘只给横滚（安全）
                plan.append(mk("orbit", p0, p1, 2.4, "flow", roll=(4.0, -4.0),
                               tag=tag + " · 横滚"))
            rot_used += 1
        elif move == "reveal":
            ww_full = H * R if R < 1.0 else W
            r0 = [W * 0.50, H * 0.50, ww * 0.35]
            r1 = [W * 0.50, H * 0.50, ww_full]
            plan.append(mk("reveal", r0, r1, 2.4, "flow", tag=tag + " · 拉远"))
    # ---- 旋转兜底（2026-10-29）：用户要求 Product Spotlight 也要能用「旋转」等多样运镜。
    # 当内容镜头数≥2 且当前一个旋转都没有时，把第一个「窗口居中、非工艺/非钩子/非收尾」
    # 的内容镜改写为稳定的小角度 orbit（yaw±3° + roll±4°，留 1.6x 源边距；若该框贴边，
    # 下方 mk 的 build 分支会自动降级为纯横滚，不晃）。保证成片至少出现一次旋转。
    _content = [m for m in plan if not str(m["kind"]).startswith("hook")
                and m["kind"] not in ("reveal", "tour")]
    if len(_content) >= 2 and not any(has_rot(m) for m in _content):
        # 2026-10-29 修正：改写对象绝不选 macro ——「细节推近」是工艺（烫金/雕刻字/
        # 压纹）的专属展示手法（2026-10-14 口径：工艺微距不旋转、只推近），把它改写成
        # orbit 会变成「又推又转」，且工艺镜头失去推近展示。优先把 pan 改 orbit
        # （平移改旋转位移最自然），没有 pan 再退 diag/pull。
        for cand_kind in ("pan", "diag", "pull"):
            _hit = next((m for m in _content if m["kind"] == cand_kind
                         and "细节推近" not in str(m.get("tag", ""))
                         and "推近" not in str(m.get("tag", ""))), None)
            if _hit is None:
                continue
            cx0, cy0, ww0 = _hit["p0"][0], _hit["p0"][1], _hit["p0"][2]
            _hit["kind"] = "orbit"
            _hit["p0"] = [cx0, cy0, ww0]
            _hit["p1"] = [cx0, cy0, ww0 * 0.90]
            _hit["yaw"] = (3.0, -3.0)
            _hit["pitch"] = (0.0, 0.0)
            _hit["roll"] = (4.0, -4.0)
            _hit["margin"] = 1.6
            # tag 重拼：去掉原「平移 →」等方向后缀，只保留内容名 + 旋转
            _base = str(_hit.get("tag", "") or "内容").split(" · ")[0]
            _hit["tag"] = _base + " · 旋转"
            break
    # ---- 相邻平移方向去重（2026-10-29）：巡览占掉前两个轮转位后，内容镜的
    # 方向轮转可能撞车（实测连续两镜都是「平移 ↓」）。后一个反向（交换 p0/p1），
    # 保证「→ ← ↑ ↓」观感上交替出现。----
    for _i in range(1, len(plan)):
        _a, _b = plan[_i - 1], plan[_i]
        if _a["kind"] != "pan" or _b["kind"] != "pan":
            continue
        _ax = _a["p1"][0] - _a["p0"][0]
        _ay = _a["p1"][1] - _a["p0"][1]
        _bx = _b["p1"][0] - _b["p0"][0]
        _by = _b["p1"][1] - _b["p0"][1]
        if abs(_bx) >= abs(_by):
            _same = abs(_ax) > 1e-9 and (_ax > 0) == (_bx > 0)
        else:
            _same = abs(_ay) > 1e-9 and (_ay > 0) == (_by > 0)
        if _same:
            _b["p0"], _b["p1"] = _b["p1"], _b["p0"]
            _tg = str(_b.get("tag", ""))
            for _x, _y in (("→", "←"), ("←", "→"), ("↓", "↑"), ("↑", "↓")):
                if _x in _tg:
                    _b["tag"] = _tg.replace(_x, _y)
                    break
    if not any(m["kind"] == "reveal" for m in plan):
        ww_full = H * R if R < 1.0 else W
        plan.append(mk("reveal", [W * 0.50, H * 0.50, W * 0.35 if R < 1.0 else W * 0.50],
                       [W * 0.50, H * 0.50, ww_full], 2.4, "flow",
                       tag="收尾拉远 · 平移"))
    # ---- 总时长控制：压到 MAX_TOTAL 以内（含叠化重叠），保证成片 ≤15s ----
    def _total(pl):
        return sum(m["dur"] for m in pl) - XFADE * max(0, len(pl) - 1)
    if _total(plan) > MAX_TOTAL and plan:
        # 迭代收敛（2026-10-29）：单次缩放后 round(,2)+MIN_DUR 下限可能让总时长
        # 仍略超 MAX_TOTAL（实测 15.27s），迭代压到真正 ≤15s 再进补齐段。
        for _ in range(4):
            if _total(plan) <= MAX_TOTAL:
                break
            target = MAX_TOTAL + XFADE * max(0, len(plan) - 1)
            cur = sum(m["dur"] for m in plan)
            sc = target / cur if cur > 0 else 1.0
            for m in plan:
                m["dur"] = max(MIN_DUR, round(m["dur"] * sc, 2))
    # 目标时长补齐：识别件少导致成片偏短时（如模版展示视频要求 15s），
    # 等比放慢每镜节奏补到目标时长；单镜上限 4.5s 防止拖沓，迭代收敛。
    if target_secs and plan and _total(plan) < target_secs - 0.05:
        for _ in range(6):
            sc = target_secs / _total(plan)
            for m in plan:
                m["dur"] = min(4.5, round(m["dur"] * sc, 2))
            if _total(plan) >= target_secs - 0.05:
                break
        # 补齐可能把总时长顶过 MAX_TOTAL（单镜 4.5s 上限卡住等比缩放），再收一轮
        for _ in range(4):
            if _total(plan) <= MAX_TOTAL:
                break
            target = MAX_TOTAL + XFADE * max(0, len(plan) - 1)
            cur = sum(m["dur"] for m in plan)
            sc = target / cur if cur > 0 else 1.0
            for m in plan:
                m["dur"] = max(MIN_DUR, round(m["dur"] * sc, 2))
    # 速度收紧：dmax 锁到当前时长（只允许缩位移，不允许把时长加回来撑破 15s）
    for m in plan:
        cap = m["dur"]
        if m["kind"] in ("scan", "hook_scan", "hook_orbit", "tour"):
            fit_motion2(m, limit=0.26, dmax=cap)   # 开场横扫/巡览/旋转给到红线上限
        elif m["kind"] == "reveal":
            fit_motion2(m, limit=0.22, dmax=cap)
        else:
            fit_motion2(m, limit=0.24, dmax=cap)
    return plan


def build_synthetic_variety(W, H, OW, OH):
    """无任何识别结果时的兜底：用整图启发式子区合成「开篇推进 + 推/拉/旋转」的运镜。

    保证即便没有 API key、也没有卡片识别，自动模式也能给出完整的开场 + 内容镜头，
    而不是退化成「整体扫描 + 收尾拉远」两镜。
    """
    subs = [
        ([0.10, 0.28, 0.46, 0.46], "inner", "主体"),
        ([0.46, 0.52, 0.34, 0.34], "craft", "工艺细节"),
        ([0.56, 0.18, 0.32, 0.34], "card", "局部"),
        ([0.18, 0.68, 0.42, 0.28], "material", "材质"),
    ]
    # 开场用「横扫整体」钩子（无识别时的稳定兜底；开场库现仅 横扫整体/旋转整体 两种）
    regions = [{"box": [0.10, 0.28, 0.46, 0.46], "kind": "hook",
                "label": "开场", "move": "hook_scan"}]
    regions += [{"box": b, "kind": k, "label": l} for b, k, l in subs]
    return build_plan_from_regions(regions, OW, OH, W, H)


# ---------------- 0W5A0561 按图定制基准（2026-10-13 确认稿，回归用） ----------------
def plan_0W5A0561(W, H, OW, OH):
    """2026-10-13 在测试目录确认的单图裁卡方案，固化为历史回归基准。

    注意：本基准为**历史稿**，S3（OA 金色雕刻字微距）仍保留 Roll±4° 以便与旧输出
    逐帧比对；线上自动路径已按 2026-10-14 口径改为「工艺微距不旋转、只推近」。
    """
    R = OW / float(OH)
    plan = []
    # S1 整体扫描（纯平移、窗口尺寸恒定、横移 ±25% 视窗宽，避免单图 warp 机位晃动）
    cx = W * 0.50
    if R < 1.0:
        ww0 = H * R
        p0, p1 = [cx - ww0 * 0.25, H * 0.50, ww0], [cx + ww0 * 0.25, H * 0.50, ww0]
    else:
        ww0 = W * 0.82
        p0, p1 = [cx - ww0 * 0.25, H * 0.50, ww0], [cx + ww0 * 0.25, H * 0.50, ww0]
    plan.append(mk("scan", p0, p1, 3.6, "flow", tag="整体扫描 · 平移"))
    # S2 内页请柬（平移横扫）
    p0 = [W * 0.39, H * 0.44, 700.0]; p1 = [W * 0.53, H * 0.44, 620.0]
    plan.append(mk("pan", p0, p1, 3.0, "flow", tag="内页请柬 · 平移横扫"))
    # S3 OA 字母微距（唯一旋转 Roll±4°）
    seal = (0.43, 0.71, 0.60, 0.93)
    cw, ch = (seal[2] - seal[0]) * W, (seal[3] - seal[1]) * H
    ww_s = max(cw / 0.88, ch / 0.88 * R)
    ws = list(clamp_win(((seal[0]+seal[2])/2)*W, ((seal[1]+seal[3])/2)*H,
                        ww_s, ww_s / R, W, H)[:3])
    wt = [ws[0] - ws[2] * 0.02, ws[1] + ws[2] * R * 0.01, ws[2] * 0.80]
    plan.append(mk("macro", ws, wt, 2.4, "flow", roll=(4.0, -4.0),
                   tag="OA字母 · Roll微晃（唯一旋转）"))
    # S4 信封封蜡（平移推近）
    p0 = [W * 0.46, H * 0.14, 560.0]; p1 = [W * 0.485, H * 0.11, 380.0]
    plan.append(mk("macro", p0, p1, 2.2, "flow", tag="信封封蜡 · 平移推近"))
    # S5 RSVP（平移对角）
    bx = (0.58, 0.62, 0.28, 0.24)   # [x,y,w,h]，等价于旧 xyxy(0.58,0.62,0.86,0.86)
    r0 = list(_win_for_box(bx, 0.95, R, W, H))
    r0 = r0[:3]
    r1 = list(r0); r1[0] += r0[2] * 0.10; r1[1] -= r0[2] * R * 0.08; r1[2] = r0[2] * 0.93
    plan.append(mk("diag", r0, r1, 2.0, "flow", tag="RSVP · 平移对角"))
    # S6 外壳（平移）
    b0 = list(clamp_win(W * 0.415, H * 0.73, 560.0, 560.0 / R, W, H)[:3])
    b1 = [b0[0] + b0[2] * 0.06, b0[1] - b0[2] * R * 0.05, b0[2] * 0.90]
    plan.append(mk("pan", b0, b1, 1.8, "flow", tag="外壳 · 平移"))
    # S7 收尾拉远（平移）
    ww_full = H * R if R < 1.0 else W
    plan.append(mk("reveal", [W * 0.50, H * 0.50, W * 0.35 if R < 1.0 else W * 0.50],
                   [W * 0.50, H * 0.50, ww_full], 2.4, "flow", tag="收尾拉远 · 平移"))
    for m in plan:
        if m["kind"] == "scan":
            fit_motion2(m, limit=0.20, dmax=4.0)
        elif m["kind"] == "reveal":
            fit_motion2(m, limit=0.18, dmax=4.0)
        else:
            fit_motion2(m, limit=0.18, dmax=3.5)
    # 压回 15s 以内（fit_motion2 可能把单镜 dur 拉上限，导致总片超时）
    def _total(pl):
        return sum(m["dur"] for m in pl) - XFADE * max(0, len(pl) - 1)
    if _total(plan) > MAX_TOTAL and plan:
        for _ in range(4):
            if _total(plan) <= MAX_TOTAL:
                break
            target = MAX_TOTAL + XFADE * max(0, len(plan) - 1)
            cur = sum(m["dur"] for m in plan)
            sc = target / cur if cur > 0 else 1.0
            for m in plan:
                m["dur"] = max(MIN_DUR, round(m["dur"] * sc, 2))
    return plan


# ---------------- 高层接口 ----------------
def card_pan_render(src, plan, out_path, ratio="9x16", fps=30, strip_path=None):
    """渲染单图裁卡运镜。

    src       : 图片路径或 BGR ndarray
    plan      : shot 列表（build_plan_from_regions / plan_0W5A0561 产出）
    out_path  : 输出 mp4
    ratio     : 画幅 id，"9x16" / "4x5" / "1x1" / "16x9"（也接受 "9:16" 写法，
                见 RATIOS 真源；未识别的回落 DEFAULT_RATIO）
    strip_path: 可选，抽帧条输出路径

    注意：plan 必须是用同一个 (OW, OH) 编排出来的 —— 窗口尺寸含画幅比例 R，
    换画幅要重新编排（server 的 /api/card-pan 已按所选画幅先算 plan 再渲染）。
    """
    if isinstance(src, str):
        img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(src)
        from core.imgproc import normalize_luma   # 2026-10-27 亮度归一（用户需求）
        img = normalize_luma(img)
    else:
        img = src
    H, W = img.shape[:2]
    OW, OH = ratio_size(ratio)
    render(img, OW, OH, fps, out_path, plan)
    if strip_path:
        make_strip(img, OW, OH, W, H, plan, strip_path)
    return out_path


if __name__ == "__main__":
    pass
