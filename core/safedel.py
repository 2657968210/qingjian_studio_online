# -*- coding: utf-8 -*-
"""安全删除：任何清理动作都不许把任务线程带走。

背景（2026-09-30 实测，真实事故）：
    本机运行环境可能装了「安全删除钩子」（sitecustomize 把 os.remove / shutil.rmtree
    换成"移入回收站"，并在一个回合内删除超过阈值（50）个文件时直接
    `raise SystemExit(1)` 拒绝执行）。
    SystemExit 属于 BaseException，普通 `except OSError` / `except Exception`
    都抓不住 —— 线程会静默死亡：任务状态永远停在 running，前端无限轮询，
    用户看到的就是「点了修复没反应」「视频渲染完却加载不出来」。

所以：
    * 所有「删临时文件 / 清中间产物」的地方一律走本模块，不再直接 os.remove；
    * 本模块对外只返回 True/False，永远不向上抛异常（KeyboardInterrupt 除外）；
    * 删不掉就删不掉——临时文件留在盘上最多占点空间，绝不能挡住成片交付。
"""
import os
import shutil
import sys
import time


def _note(action, path, ok, detail=""):
    try:
        sys.stderr.write("[safedel] %s %-60s %s %s\n"
                         % (action, os.path.basename(str(path))[:60],
                            "ok" if ok else "left", detail))
        sys.stderr.flush()
    except Exception:
        pass


def unlink(path, tries=2, quiet=True):
    """删单个文件。返回 True=已不存在（删掉或本来就没有），False=还留在盘上。"""
    if not path:
        return True
    last = None
    for i in range(max(1, tries)):
        try:
            if not os.path.lexists(path):
                return True
            os.remove(path)
            return True
        except KeyboardInterrupt:
            raise
        except BaseException as e:          # SystemExit / PermissionError / OSError …
            last = e
            time.sleep(0.12 * (i + 1))
    if not quiet:
        _note("unlink", path, False, repr(last)[:120])
    return False


def unlink_all(paths, tries=2):
    """批量删除，返回仍然留在盘上的路径列表（供调用方如实打日志）。"""
    left = []
    for p in paths or []:
        if not unlink(p, tries=tries):
            left.append(p)
    return left


def rmtree(path, tries=3, quiet=False):
    """删目录树。返回 True=已不存在，False=还留在盘上（被占用/被钩子拒绝）。"""
    if not path:
        return True
    last = None
    for i in range(max(1, tries)):
        try:
            if not os.path.isdir(path):
                return True
            shutil.rmtree(path)
            if not os.path.isdir(path):
                return True
            last = "still present"
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            last = e
        time.sleep(0.25 * (i + 1))
    if not quiet:
        _note("rmtree", path, False, repr(last)[:120])
    return False


def swap_into_place(tmp, final):
    """把临时产物挪到最终位置；跨设备/被占用时退化为复制。
    返回 True 表示 final 已就绪。"""
    try:
        os.replace(tmp, final)
        return True
    except KeyboardInterrupt:
        raise
    except BaseException:
        pass
    try:
        shutil.copy2(tmp, final)
        unlink(tmp)
        return os.path.isfile(final)
    except KeyboardInterrupt:
        raise
    except BaseException:
        return False
