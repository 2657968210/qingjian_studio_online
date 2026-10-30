# -*- coding: utf-8 -*-
"""电影感运镜模版（Cinematic Cut）—— N 张请柬图 → 卡点快切 + 旋转/平移混排。

用户 2026-10-22 口径（v10 样片定稿）：
  * 第 1→2 幕切点用 **whoosh（急推变焦）**，且必须**卡在 BGM 鼓点**上；
  * 全片只保留 **2~3 个旋转镜头**（整图绕中心刚性顺时针转 11~15°，
    等效镜头小幅环绕，正面/俯拍产品图都适用）；
  * 其余镜头改为 **平移**（上/下/左/右），相邻平移方向必须不同、四向轮换；
  * 其余切点用**交叉溶解**（正常转场）；
  * 满幅 cover，旋转/平移全程不露黑边；原图直出，不加滤镜调色。

节拍：运行时对 BGM 做 energy-flux onset 检测（numpy 即可，无需 librosa），
把每个镜头切点吸附到最近的鼓点（±0.18s），因此 whoosh 天然卡拍。

接口
----
render_cinematic(images, out_path, ratio="9x16", fps=30, bgm_path=None, progress=None)
  images : 图片路径列表（6~12 张，各出场 1 次）
  bgm_path: 用于卡点的 BGM（只读取做节拍检测；本函数**不混音**）
  返回 dict：{video, secs, images, ratio, shots, whoosh_cut, rot_shots, pan_dirs}
"""
import os
import subprocess
import tempfile

import numpy as np
import cv2

from core import card_pan as cp
from core.fastcut import _build_text_overlay, DEFAULT_TEXT as CAPTION_DEFAULT

MIN_PHOTOS, MAX_PHOTOS = 6, 12
PAN_AMP = 0.045          # 平移幅度（占画幅比例，单程）
PAN_SCALE = 1.14         # 放大留边，保证平移不露黑边
ROT_DEG = [14, 12, 15, 13, 11]   # 旋转角度（顺时针，度）
WHOOSH_IDX = 1           # 切到第 2 幕的那道切点
BEAT_SNAP = 0.18         # 切点吸附鼓点的最大偏差（秒）
MIN_SHOT_SECS = 0.60
CRF, PRESET = 19, "slow"

# 动态总时长（2026-10-22 用户：根据图片数量匹配合适的时长，不固定同一时长）：
#   total = SHOT_BASE + SHOT_PER * N   → N=6→8.2s / 10→10.4s(与定稿样片一致) / 12→11.5s
#   单镜时长 1.36s → 0.96s，图越多节奏越快、成片越长。
SHOT_BASE, SHOT_PER = 4.8, 0.56


# ---------------------------------------------------------------------------
# BGM → 节拍
# ---------------------------------------------------------------------------
def _decode_mono(path, sr=22050):
    """ffmpeg 解码成单声道 float32（支持 mp3/wav，无需额外依赖）。"""
    cmd = [cp.FFMPEG, "-v", "error", "-i", path,
           "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError("cannot decode bgm: %s" % (p.stderr[:200],))
    return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768.0, sr


def detect_beats(bgm_path):
    """返回 (duration_secs, beat_times[])，energy-flux onset + 峰值挑选。"""
    x, sr = _decode_mono(bgm_path)
    dur = len(x) / float(sr)
    x = x / (np.max(np.abs(x)) + 1e-9)

    win = int(0.023 * sr)
    hop = int(0.011 * sr)
    env, t_env = [], []
    for s in range(0, max(0, len(x) - win), hop):
        seg = x[s:s + win]
        env.append(float(np.sqrt(np.mean(seg ** 2))))
        t_env.append((s + win / 2.0) / sr)
    env = np.asarray(env, np.float32)
    t_env = np.asarray(t_env, np.float32)

    flux = np.zeros_like(env)
    flux[1:] = np.maximum(env[1:] - env[:-1], 0.0)
    k = max(1, int(0.04 * sr / hop))
    if k > 1:
        flux = np.convolve(flux, np.ones(k) / k, mode="same")
    flux = flux / (flux.max() + 1e-9)

    min_gap = int(0.18 * sr / hop)
    beats, last = [], -min_gap
    for i in range(1, len(flux) - 1):
        if flux[i] > 0.30 and flux[i] >= flux[i - 1] and flux[i] > flux[i + 1]:
            if i - last >= min_gap:
                beats.append(float(t_env[i]))
                last = i
    return dur, beats


def plan_bounds(total, n, beats):
    """切点时间（含 0 与 total）：等分后每道切点吸附最近鼓点，保证 whoosh 卡拍。

    total 为按图片数量动态算出的目标总时长（见 target_secs），与 BGM 原长无关
    （BGM 由 build_bgm 循环/裁切到同一长度）；beats 已按循环周期扩展。"""
    bounds = [0.0]
    for i in range(1, n):
        ideal = total * i / float(n)
        near = min(beats, key=lambda b: abs(b - ideal)) if beats else ideal
        if i == 1:
            # whoosh 切点（1→2 幕）必须卡鼓点：强制吸附最近一拍，不受容差限制
            # （单镜偏长导致容差外时依然落拍；过近的拍点由下方 MIN_SHOT 兜底）。
            b = near if near >= MIN_SHOT_SECS else ideal
        else:
            b = near if abs(near - ideal) <= BEAT_SNAP else ideal
        b = max(b, bounds[-1] + MIN_SHOT_SECS)
        b = min(b, total - MIN_SHOT_SECS * (n - i))
        bounds.append(b)
    bounds.append(total)
    return bounds


def target_secs(n):
    """按图片数量动态计算总时长（秒）。"""
    return SHOT_BASE + SHOT_PER * n


def _extend_beats(beats, bgm_dur, upto):
    """BGM 循环时把鼓点表按循环周期扩展到 upto 秒。"""
    out = list(beats)
    k = 1
    while bgm_dur * k < upto:
        out += [b + bgm_dur * k for b in beats if b + bgm_dur * k <= upto]
        k += 1
    return out


def build_bgm(bgm_path, total_secs, out_wav):
    """把 BGM 循环/裁切成与成片等长的 wav（供混音，成片可能长于 BGM 原长）。"""
    src = bgm_path.replace("\\", "/")
    cmd = [cp.FFMPEG, "-y", "-v", "error",
           "-stream_loop", "-1", "-i", src,
           "-t", "%.3f" % total_secs,
           "-ac", "2", "-ar", "48000", out_wav]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not os.path.isfile(out_wav):
        raise RuntimeError("build_bgm failed: %s" % p.stderr[:200])
    return out_wav


# ---------------------------------------------------------------------------
# 运动计划
# ---------------------------------------------------------------------------
def plan_motion(n):
    """旋转镜 3 个（n>=8）或 2 个，位置分散且不含第 1 幕；其余平移四向轮换。"""
    k = 3 if n >= 8 else 2
    raw = [int(round(n * 0.32)), int(round(n * 0.62)), int(round(n * 0.90))]
    rot = []
    for r in raw:
        r = min(max(r, 1), n - 1)
        while r in rot and r < n - 1:
            r += 1
        if r not in rot:
            rot.append(r)
        if len(rot) == k:
            break
    while len(rot) < k:
        for r in range(1, n):
            if r not in rot:
                rot.append(r)
                break
    rot = sorted(rot)[:k]

    dirs = ["down", "up", "left", "right"]
    pan, p = {}, 0
    for i in range(n):
        if i in rot:
            continue
        pan[i] = dirs[p % len(dirs)]
        p += 1
    return rot, pan


# ---------------------------------------------------------------------------
# 画布 / 单帧
# ---------------------------------------------------------------------------
def _imread_bgr(p):
    arr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("cannot read %s" % p)
    from core.imgproc import normalize_luma   # 2026-10-27 亮度归一（用户需求）
    return normalize_luma(arr)


def _cover_scale_rot(w, h, theta_deg, W, H):
    t = np.radians(abs(theta_deg))
    c, s = np.cos(t), np.sin(t)
    return max((W * c + H * s) / float(w), (W * s + H * c) / float(h)) * 1.012


def _build_layer(path, mode, theta, W, H):
    im = _imread_bgr(path)
    h, w = im.shape[:2]
    if mode == "rotate":
        cs = _cover_scale_rot(w, h, theta, W, H)
    else:
        cs = max(W / float(w), H / float(h)) * PAN_SCALE
    lw, lh = int(round(w * cs)), int(round(h * cs))
    layer = cv2.resize(im, (lw, lh), interpolation=cv2.INTER_LANCZOS4)
    return layer, lw, lh


def _rot_matrix(ang, lw, lh, W, H):
    R = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, 1.0)
    ox, oy = (W - lw) / 2.0, (H - lh) / 2.0
    T = np.array([[1.0, 0.0, ox], [0.0, 1.0, oy]], np.float64)
    M = np.vstack([R, [0.0, 0.0, 1.0]]) @ np.vstack([T, [0.0, 0.0, 1.0]])
    return np.ascontiguousarray(M[:2])


def _zoom_blur(img, strength, zoom, W, H):
    if strength <= 0.001 and abs(zoom - 1.0) < 0.001:
        return img
    K = 7
    acc = np.zeros((H, W, 3), np.float32)
    for k in range(K):
        sc = 1.0 + strength * 0.20 * (k / float(K - 1))
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), 0.0, sc)
        acc += cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32)
    acc /= K
    Mz = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), 0.0, zoom)
    return cv2.warpAffine(acc, Mz, (W, H), flags=cv2.INTER_LINEAR).astype(np.uint8)


def _iter_shot(path, mode, theta, direction, sf, W, H):
    layer, lw, lh = _build_layer(path, mode, theta, W, H)
    for f in range(sf):
        e = (f + 0.5) / float(sf)
        if mode == "rotate":
            M = _rot_matrix(-theta * e, lw, lh, W, H)   # 负角 => 顺时针
            fr = cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        else:
            d = PAN_AMP * (2.0 * e - 1.0)
            dx = dy = 0.0
            if direction == "right":
                dx = d * W
            elif direction == "left":
                dx = -d * W
            elif direction == "down":
                dy = d * H
            elif direction == "up":
                dy = -d * H
            M = np.array([[1.0, 0.0, (W - lw) / 2.0 + dx],
                          [0.0, 1.0, (H - lh) / 2.0 + dy]], np.float64)
            fr = cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        yield fr


def _transition(prev, cur, kind, W, H):
    """就地修改 prev 尾部 / cur 头部（各最多 2 帧）。"""
    if kind == "whoosh":
        if len(prev) >= 2:
            prev[-2] = _zoom_blur(prev[-2], 0.55, 1.05, W, H)
        prev[-1] = _zoom_blur(prev[-1], 1.00, 1.12, W, H)
        cur[0] = _zoom_blur(cur[0], 1.00, 1.12, W, H)
        if len(cur) >= 2:
            cur[1] = _zoom_blur(cur[1], 0.55, 1.05, W, H)
    else:
        A = prev[-1].astype(np.float32)
        cur[0] = (0.35 * A + 0.65 * cur[0].astype(np.float32)).astype(np.uint8)
        if len(cur) >= 2:
            cur[1] = (0.15 * A + 0.85 * cur[1].astype(np.float32)).astype(np.uint8)
        if len(prev) >= 2:
            prev[-2] = (0.85 * A + 0.15 * prev[-2].astype(np.float32)).astype(np.uint8)
        prev[-1] = (0.60 * A + 0.40 * prev[-1].astype(np.float32)).astype(np.uint8)
    return prev, cur


# ---------------------------------------------------------------------------
# 主渲染
# ---------------------------------------------------------------------------
def render_cinematic(images, out_path, ratio="9x16", fps=30,
                     bgm_path=None, progress=None,
                     text=None, show_text=False, font=None):
    """text/show_text: 持久斜体字幕（2026-10-22 用户：第 6 模版也提供字幕/口播选择）。
    show_text=True 且 text 非空时全程叠加（同 Fast Cut 的 44% 屏高斜体白字）。
    font: 客户自选字体键（core.fonts；None = 斜体 CG 默认）。"""
    paths = [p for p in (images or []) if p and os.path.isfile(p)]
    if len(paths) < MIN_PHOTOS:
        raise ValueError("Cinematic Cut needs at least %d photos" % MIN_PHOTOS)
    if len(paths) > MAX_PHOTOS:
        paths = paths[:MAX_PHOTOS]
    n = len(paths)

    W, H = cp.ratio_size(ratio)
    W, H = int(W), int(H)

    if progress:
        progress(6, 100, "Reading the music")
    if not bgm_path or not os.path.isfile(bgm_path):
        raise ValueError("bgm missing: %s" % bgm_path)
    bgm_dur, beats = detect_beats(bgm_path)
    if not beats:
        raise RuntimeError("no beats detected in bgm")

    # 动态总时长（按图片数量），BGM 循环扩展鼓点表保证每道切点仍可吸附
    total = target_secs(n)
    beats = _extend_beats(beats, bgm_dur, total)
    bounds = plan_bounds(total, n, beats)
    cuts = [int(round(b * fps)) for b in bounds]
    total_frames = int(round(total * fps))
    cuts[-1] = total_frames
    for i in range(1, len(cuts)):
        if cuts[i] - cuts[i - 1] < int(MIN_SHOT_SECS * fps):
            cuts[i] = cuts[i - 1] + int(MIN_SHOT_SECS * fps)
    cuts[-1] = total_frames

    rot, pan = plan_motion(n)

    # 持久字幕 overlay（一次生成，全程叠加；写帧前 blend，转场模糊之上更清晰）
    txt_rgb = txt_a = None
    if show_text:
        from core import fonts as _fonts
        cap = (text or "").strip() or CAPTION_DEFAULT
        txt_rgb, txt_a = _build_text_overlay(cap, W, H, font_path=_fonts.ttf(font))

    if progress:
        progress(12, 100, "Rendering")

    cmd = [cp.FFMPEG, "-y", "-hide_banner",
           "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", "%dx%d" % (W, H), "-r", str(fps), "-i", "-",
           "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", str(CRF), "-preset", PRESET, "-movflags", "+faststart",
           out_path]
    err_f = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=err_f, stderr=err_f)

    written = 0
    tail = []          # 上一镜最后 <=2 帧（尚未写出）

    def emit(fr):
        # 写帧统一出口：先叠持久字幕（alpha blend），再交给 ffmpeg
        if txt_rgb is not None:
            a = txt_a
            fr = (fr.astype(np.float32) * (1.0 - a) + txt_rgb * a).astype(np.uint8)
        proc.stdin.write(fr.tobytes())

    try:
        for i in range(n):
            sf = cuts[i + 1] - cuts[i]
            mode = "rotate" if i in rot else "pan"
            theta = ROT_DEG[rot.index(i) % len(ROT_DEG)] if mode == "rotate" else 0
            gen = _iter_shot(paths[i], mode, theta, pan.get(i, "down"), sf, W, H)
            head = [next(gen) for _ in range(min(2, sf))]

            if i > 0:
                kind = "whoosh" if i == WHOOSH_IDX else "dissolve"
                tail, head = _transition(tail, head, kind, W, H)
                for fr in tail:
                    emit(fr)
                    written += 1

            if i == n - 1:
                for fr in head:
                    emit(fr)
                    written += 1
                for fr in gen:
                    emit(fr)
                    written += 1
                tail = []
            else:
                for fr in head:
                    emit(fr)
                    written += 1
                hold = []
                for fr in gen:
                    hold.append(fr)
                    if len(hold) > 2:
                        emit(hold.pop(0))
                        written += 1
                tail = hold

            if progress and i % 2 == 0:
                progress(int(15 + 80 * (i + 1) / max(n, 1)), 100, "Rendering")
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
        err_f.close()

    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed rc=%s" % proc.returncode)

    if progress:
        progress(100, 100, "Done")

    return {"video": out_path, "secs": round(written / float(fps), 2),
            "images": n, "ratio": ratio, "shots": n,
            "whoosh_cut": round(bounds[1], 3),
            "rot_shots": [r + 1 for r in rot],
            "pan_dirs": {k + 1: v for k, v in sorted(pan.items())}}