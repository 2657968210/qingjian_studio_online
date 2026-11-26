# -*- coding: utf-8 -*-
"""装饰类现场展示模版（Decor Showcase）。

用户 2026-11-02 需求（第二版）：模版成片的运镜要沿用参考片
「6图运镜模仿_口播字幕.mp4」的手法 —— 逐张满幅 2.5D 运镜（水平平移 / 垂直平移 /
对角平移 / 缓推近 / 拉远展开，带深度视差，无旋转），口播与字幕都不要，配合适的 BGM。

与 four_grid（Grid Showcase，纯切换不运镜）**完全独立**：本模版不做四宫格开场、
不叠任何文字、不出配音，只把上传的 N 张图逐张做满幅运镜展示，硬切衔接，每张
DECOR_CUT 秒；总时长随图数线性增长。

运镜匹配规则见 core.decor_cam.plan_moves：横幅大场景→水平平移、竖幅长条→垂直平移、
方图细节多→缓推近、开阔场景→拉远、其余→对角平移；并保证相邻两镜不重样。
平移行程按画布余量自动收敛，不会顶到边界出现「画面卡住」。

BGM 由 server 混（默认 Wedding_01_657 婚礼竖琴向，不与其他模版重复）。

接口
----
render_decor_show(images, out_path, ratio="9x16", fps=30, progress=None,
                  cut_secs=None)
  images : 图片路径列表（3~12 张）
  返回 dict：{video, secs, images, ratio, moves}
（只出画面，不含配乐）
"""
import os
import subprocess
import tempfile

import numpy as np

from core import card_pan as cp
from core import decor_cam as dc

# ---------------------------------------------------------------------------
# 节奏 / 编码
# ---------------------------------------------------------------------------
DECOR_CUT = 3.0        # 每张图运镜展示时长（s）
CRF = 18
PRESET = "slow"


def _encode_frames(out_path, frames, vw, vh, fps):
    cmd = [cp.FFMPEG, "-y", "-hide_banner",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (vw, vh), "-r", str(fps), "-i", "-",
           "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", str(CRF), "-preset", PRESET, "-movflags", "+faststart",
           out_path]
    err_f = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=err_f, stderr=err_f)
    try:
        for fr in frames:
            proc.stdin.write(np.clip(fr, 0, 255).astype(np.uint8).tobytes())
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
    return proc.returncode


def render_decor_show(images, out_path, ratio="9x16", fps=30, progress=None,
                      cut_secs=None):
    """N(3~12) 张装饰/请柬图 → 逐张满幅 2.5D 运镜展示成片（仅画面，无音轨）。"""
    paths = [p for p in (images or []) if p and os.path.isfile(p)]
    if len(paths) < 3:
        raise ValueError("Decor Showcase needs at least 3 photos")
    if len(paths) > 99:
        paths = paths[:99]

    OW, OH = cp.ratio_size(ratio)
    vw, vh = int(OW), int(OH)
    cut = float(cut_secs or DECOR_CUT)

    if progress:
        progress(6, 100, "Preparing photos")

    # 1) 轻量探测 + 运镜规划（不预载全部像素，12 图也不会吃爆内存）
    metas = []
    for k, p in enumerate(paths):
        try:
            metas.append(dc.probe(p, vw, vh))
        except Exception as exc:            # 单张坏图不该拖垮整条任务
            raise ValueError("cannot read photo %d: %s" % (k + 1, exc))
        if progress:
            progress(6 + int(18 * (k + 1) / len(paths)), 100, "Analyzing photos")
    plan = dc.plan_moves(metas, vw, vh)

    cut_frames = int(round(cut * fps))
    total = cut_frames * len(paths)
    total_secs = total / float(fps)

    if progress:
        progress(25, 100, "Rendering")

    def frames():
        for idx, path in enumerate(paths):
            cv_ = dc.build_canvas(path, vh, vw)
            fam, direction = plan[idx]
            cam_fn = dc.make_move(fam, cv_, vw, vh, direction=direction)
            for i in range(cut_frames):
                t = i / max(cut_frames - 1, 1)
                yield dc.warp(cv_, cam_fn(t), vw, vh)
                if progress and i % 45 == 0:
                    done = idx * cut_frames + i
                    progress(25 + int(72 * done / max(total, 1)), 100, "Rendering")

    rc = _encode_frames(out_path, frames(), vw, vh, fps)
    if rc != 0:
        raise RuntimeError("ffmpeg failed rc=%s" % rc)

    if progress:
        progress(100, 100, "Done")
    return {"video": out_path, "secs": round(total_secs, 2),
            "images": len(paths), "ratio": ratio,
            "moves": [{"move": f, "dir": d} for f, d in plan]}
