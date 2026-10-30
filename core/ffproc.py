# -*- coding: utf-8 -*-
"""Windows 安全的 ffmpeg/ffprobe 子进程调用
=============================================
背景（2026-09-30 实测）：`subprocess.run(..., capture_output=True)` 在本机会无限挂死——
杀软（360 等）会在 ffmpeg 退出后仍持有其继承的 stdout/stderr 管道句柄，
PIPE 永远读不到 EOF，communicate() 卡死。症状：任务线程永远 running、
成片其实已完整落盘、前端无限轮询（"更长时间的视频无法加载"的根因）。

对策：stdout/stderr 落临时文件（不用 PIPE），stdin 接 DEVNULL；
wait(timeout) 超时则 taskkill /F /T 杀掉整棵进程树。
所有新代码一律经此模块调用 ffmpeg，禁止再直接用 capture_output=True。
"""
import re
import subprocess
import tempfile


def run_ffmpeg(cmd, timeout=300, err_tail=500):
    """运行 ffmpeg 命令，返回 (returncode, stderr尾部)。
    挂死风险已消除：不用 PIPE，超时强杀进程树。"""
    out_f = tempfile.TemporaryFile()
    err_f = tempfile.TemporaryFile()
    try:
        p = subprocess.Popen(cmd, stdout=out_f, stderr=err_f,
                             stdin=subprocess.DEVNULL)
    except OSError:
        raise
    try:
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(p.pid)
        try:
            p.wait(timeout=10)
        except Exception:
            pass
        raise RuntimeError(
            "ffmpeg timed out after %ss (process tree killed): %s"
            % (timeout, " ".join(str(c) for c in cmd)[:200]))
    err_f.seek(0)
    err = err_f.read().decode("utf-8", "replace")[-err_tail:]
    out_f.close()
    err_f.close()
    return rc, err


def _kill_tree(pid):
    """taskkill /T /F：连子进程一起杀（Windows 专用）。"""
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       stdin=subprocess.DEVNULL, timeout=15)
    except Exception:
        pass


def probe_info(path, timeout=120):
    """返回 ffmpeg -i 的 stderr 全文（供 Duration/Stream 正则解析）。"""
    import os
    ff = _default_ff()
    rc, err = run_ffmpeg([ff, "-i", path, "-hide_banner"],
                         timeout=timeout, err_tail=100000)
    return err


_ff_cache = None


def _default_ff():
    global _ff_cache
    if _ff_cache is None:
        import imageio_ffmpeg
        _ff_cache = imageio_ffmpeg.get_ffmpeg_exe()
    return _ff_cache


def duration_of(path, timeout=120):
    """视频时长（秒）；解析失败返回 0.0。"""
    err = probe_info(path, timeout=timeout)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if not m:
        return 0.0
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
