# -*- coding: utf-8 -*-
"""
拼接 + BGM 合成 + 导出
========================
- 多段视频统一到同一画幅（默认跟随首段实际尺寸，缺失时回退 1080x1920）/30fps 后 concat 重编码
- BGM: 本地无版权库（Mixkit License 免费商用无需署名 / CC BY 需署名）
- 混音: loudnorm I=-18 + 淡入淡出, 音画等长
"""
import os, re, subprocess
import imageio_ffmpeg
from . import ffproc

FF = imageio_ffmpeg.get_ffmpeg_exe()


def _duration(path):
    return ffproc.duration_of(path)


def _probe_size(path, timeout=120):
    """从视频流解析 (w, h)；解析失败返回 None。用于自动判定拼接目标画幅。"""
    info = ffproc.probe_info(path, timeout=timeout)
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", info)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _probe_video(path, timeout=120):
    """解析 (w, h, fps, pix_fmt)；缺失项为 None。用于判断单段可否免重编码。"""
    info = ffproc.probe_info(path, timeout=timeout)
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", info)
    w = int(m.group(1)) if m else None
    h = int(m.group(2)) if m else None
    f = re.search(r"(\d+(?:\.\d+)?)\s*fps", info)
    fps = float(f.group(1)) if f else None
    pf = re.search(r"Video:.*?,\s*(yuvj?\d+p|rgba?|bgra?|gray)\b", info)
    return w, h, fps, (pf.group(1) if pf else None)


def _single_needs_no_reencode(path, size):
    """单段是否「画幅 / 帧率 / 像素格式」已达标 —— 达标就不必整片重编码。
    渲染器产出的段本就是 H.264/30fps/yuv420p，只有尺寸需要与目标画幅一致。"""
    w, h, fps, pf = _probe_video(path)
    if size is not None and (w, h) != (int(size[0]), int(size[1])):
        return False
    if fps is None or abs(fps - 30.0) > 0.01:
        return False
    if pf not in ("yuv420p", "yuvj420p"):
        return False
    return True


def _remux_faststart(src, out_path, timeout=300):
    """无损重封装：-c copy + faststart。不加滤镜、不重编码（实测 ~1s）。"""
    cmd = [FF, "-y", "-i", src, "-c", "copy", "-movflags", "+faststart", out_path]
    rc, err = ffproc.run_ffmpeg(cmd, timeout=timeout, err_tail=500)
    if rc != 0:
        raise RuntimeError("Remux failed: %s" % err)
    return out_path


def concat_segments(segments, out_path, progress=None, size=None):
    """segments: [video_path...] -> out_path (同画幅/30fps/H.264)

    size: (w, h) 目标成片尺寸；为 None 时按**第一段的实际画幅**自动判定
          （各段由同一 ratio 渲染，画幅本应一致）——避免把 1:1 段塞回 9:16。
    """
    n = len(segments)
    if size is None:
        size = _probe_size(segments[0]) or (1080, 1920)
    tw, th = int(size[0]), int(size[1])
    # [2026-10-10 性能修复] 单段且画幅/帧率已达标 → 直接无损重封装，免整片重编码
    if n == 1 and _single_needs_no_reencode(segments[0], (tw, th)):
        return _remux_faststart(segments[0], out_path)
    inputs = []
    for s in segments:
        inputs += ["-i", s]
    fl = []
    for i in range(n):
        fl.append(
            "[%d:v]scale=%d:%d:force_original_aspect_ratio=decrease,"
            "pad=%d:%d:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1[v%d]" % (i, tw, th, tw, th, i))
    concat_in = "".join("[v%d]" % i for i in range(n))
    fl.append("%sconcat=n=%d:v=1:a=0[outv]" % (concat_in, n))
    cmd = [FF, "-y"] + inputs + [
        "-filter_complex", ";".join(fl), "-map", "[outv]",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    rc, err = ffproc.run_ffmpeg(cmd, timeout=600, err_tail=500)
    if rc != 0:
        raise RuntimeError("Stitching failed: %s" % err)
    return out_path


def _has_audio(path):
    return "Audio:" in ffproc.probe_info(path)


def mix_bgm(video_path, bgm_path, out_path,
            gain_db=-3.0, fade_in=1.2, fade_out=2.6, bgm_start=0.0):
    """BGM 混音: 响度标准化 + 淡入淡出 + 音画等长。视频无音轨时只保留 BGM。"""
    dur = _duration(video_path)
    if dur <= 0:
        raise RuntimeError("Could not read the video duration")
    fo_start = max(0.0, dur - fade_out)
    afilter = (
        "atrim=0:%.2f,asetpts=PTS-STARTPTS,"
        "volume=%.1fdB,"
        "loudnorm=I=-18:TP=-2:LRA=11,"
        "aresample=48000,"
        "afade=t=in:st=0:d=%.1f,"
        "afade=t=out:st=%.2f:d=%.1f" % (
            dur, gain_db, fade_in, fo_start, fade_out))
    if _has_audio(video_path):
        fg = ("[1:a]%s[bgm];[0:a][bgm]amix=inputs=2:duration=first:"
              "dropout_transition=0[aout]" % afilter)
    else:
        fg = "[1:a]%s[aout]" % afilter
    cmd = [FF, "-y",
           "-i", video_path,
           "-ss", str(max(0.0, bgm_start)), "-i", bgm_path,
           "-filter_complex", fg,
           "-map", "0:v", "-map", "[aout]",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-shortest", "-movflags", "+faststart", out_path]
    rc, err = ffproc.run_ffmpeg(cmd, timeout=600, err_tail=500)
    if rc != 0:
        raise RuntimeError("Audio mixing failed: %s" % err)
    return out_path


def list_bgm(bgm_dir):
    """返回曲库 [{name, file, license, credit, mood, light, duration}]。

    只暴露 manifest.json 中**授权已核实**的曲目（当前均为 Kevin MacLeod
    incompetech CC-BY 4.0，可商用、需署名）。目录里未登记的散落文件
    （含从第三方样片/Canva 抽取、未知来源）一律不对外暴露，避免把无授权
    的曲子误标成「可商用免署名」再分发给客户。
    """
    import json
    tracks = []
    man = os.path.join(bgm_dir, "manifest.json")
    data = {}
    if os.path.isfile(man):
        try:
            data = json.load(open(man, encoding="utf-8")).get("music", {})
        except Exception:
            data = {}
    for key, t in data.items():
        f = t.get("file")
        if not f:
            continue
        fp = os.path.join(bgm_dir, f)
        if not os.path.isfile(fp):
            continue
        tracks.append({
            "name": key,
            "file": fp,
            # 2026-10-27：尊重 manifest 登记的 license（用户上传曲标 User upload，
            # 不再一律冒充 CC-BY 免版权曲）
            "license": t.get("license") or "CC BY 4.0 (free for commercial use; credit in the caption)",
            "credit": t.get("credit", ""),
            "mood": t.get("mood", ""),
            "light": t.get("light", False),
            "duration": t.get("duration", 0),
        })
    return tracks


def export_final(segments, bgm_path, out_path, progress=None, size=None, **kw):
    """全流程: 拼接 -> 混音 -> 导出。size 同 concat_segments（None=按首段自动判定画幅）。

    [2026-10-10 性能修复] 只有一段且画幅/帧率已达标时，跳过整片重编码：
      前端始终只把「渲染器已拼好的那一条」交给这里，原来仍会走一遍
      scale+pad+fps+concat 的 crf18/medium 重编码 —— 实测 16s 成片白花 70s
      （占「加 BGM」总耗时的 ~90%），且多一次有损重编码。
      改为：无损重封装 + 直接混音（70s → ~10s，画质更好）。"""
    if progress: progress(5, 100)
    if len(segments) == 1 and _single_needs_no_reencode(segments[0], size):
        tmp = out_path + ".pre.mp4"
        _remux_faststart(segments[0], tmp)
        if progress: progress(70, 100)
        mix_bgm(tmp, bgm_path, out_path, **kw)
        from . import safedel
        safedel.unlink(tmp)
        if progress: progress(100, 100)
        return out_path
    tmp = out_path + ".tmp.mp4"
    concat_segments(segments, tmp, size=size)
    if progress: progress(70, 100)
    mix_bgm(tmp, bgm_path, out_path, **kw)
    from . import safedel
    safedel.unlink(tmp)          # 见 safedel：清理绝不允许抛 BaseException 带走任务线程
    if progress: progress(100, 100)
    return out_path
