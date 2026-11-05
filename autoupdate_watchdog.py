# -*- coding: utf-8 -*-
"""内网自动更新看门狗（2026-11-05 用户需求：内网自动更新）。

用法（内网服务器，替代 `python server.py`）：
    python autoupdate_watchdog.py

职责：
  1. 保活：server.py 未监听端口时自动拉起（崩溃/误关自动重启）；
  2. 自动更新：每 interval_secs 秒 `git fetch` 一次，发现远端 main 有新提交 →
     停掉 server → `git fetch + reset --hard origin/main`（**只动 git 跟踪的文件**，
     config.json / data/ / bgm 新增文件等未跟踪内容一律不受影响）→ 重新拉起 server；
  3. 全程写 autoupdate.log，可随时查更新了什么、为什么重启。

一次性安装（开机自启，可选）：
    schtasks /Create /TN "StudioWatchdog" /SC ONSTART /RU SYSTEM
             /TR "cmd /c cd /d E:\\studio && python autoupdate_watchdog.py >> autoupdate_boot.log 2>&1"
    （路径按内网实际仓库位置修改）

安全边界：
  · 仓库里没有 .git（如线上部署副本）或找不到 git 命令 → 更新功能自动禁用，
    仅保留保活；
  · config.json 的 auto_update.enabled = false → 只保活、不更新；
  · reset --hard 只回退「被 git 跟踪」的文件——内网同事不要直接改这些文件，
    要改配置请改 config.json（未跟踪，永远安全）。
"""
import os
import socket
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "autoupdate.log")
CFG = os.path.join(ROOT, "config.json")

DEFAULTS = {
    "enabled": True,        # false = 只保活不更新
    "interval_secs": 300,   # 检查更新的间隔（秒）
    "remote": "origin",
    "branch": "main",
    "proxy": "",            # 内网访问 GitHub 需要代理时填，如 http://127.0.0.1:7890
    "git_path": "",         # git 不在 PATH 时手动指定，如 C:\\Program Files\\Git\\cmd\\git.exe
}

_GIT_CANDIDATES = (
    "git",  # PATH 里的优先
    r"C:\Program Files\Git\cmd\git.exe",
    r"C:\Program Files (x86)\Git\cmd\git.exe",
    os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs", "Git", "cmd", "git.exe"),
)


def find_git(cfg):
    """定位 git 可执行文件：config.auto_update.git_path > PATH > 常见安装位置。"""
    p = (cfg.get("git_path") or "").strip()
    if p and os.path.isfile(p):
        return p
    import shutil
    w = shutil.which("git")
    if w:
        return w
    for c in _GIT_CANDIDATES[1:]:
        if os.path.isfile(c):
            return c
    return None


def log(msg):
    line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    try:
        print(line, flush=True)
    except OSError:
        pass


def load_cfg():
    """每次循环重读 config.json —— 改配置无需重启看门狗。"""
    cfg = dict(DEFAULTS)
    try:
        import json
        with open(CFG, encoding="utf-8") as f:
            data = json.load(f)
        cfg.update({k: v for k, v in (data.get("auto_update") or {}).items()
                    if k in DEFAULTS})
    except BaseException as e:        # 配置缺失/损坏不影响保活
        log("config.json 读取失败(%s)，用默认自动更新配置" % e)
    return cfg


def server_port():
    try:
        import json
        with open(CFG, encoding="utf-8") as f:
            return int(json.load(f).get("port") or 8765)
    except BaseException:
        return 8765


def port_open(port):
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def run_git(args, cfg):
    """执行 git 命令，返回 (rc, stdout+stderr)。代理按配置注入；找不到 git 返回 rc=127。"""
    env = dict(os.environ)
    if cfg.get("proxy"):
        env["HTTPS_PROXY"] = env["HTTP_PROXY"] = cfg["proxy"]
    git = find_git(cfg)
    if not git:
        return 127, "git 可执行文件未找到（可在 config.json auto_update.git_path 指定）"
    try:
        p = subprocess.run([git] + args, cwd=ROOT, capture_output=True,
                           text=True, timeout=180, env=env,
                           errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except BaseException as e:
        return 126, "git 执行异常: %s" % e


def git_ok(cfg):
    if not os.path.isdir(os.path.join(ROOT, ".git")):
        log("本目录不是 git 仓库（无 .git）→ 自动更新禁用，仅保活")
        return False
    rc, out = run_git(["--version"], cfg)
    if rc != 0:
        log("找不到可用的 git 命令 → 自动更新禁用，仅保活（%s）" % out.strip()[:120])
        return False
    return True


def start_server():
    """新控制台窗口启动 server.py（内网同事能直接看到服务日志）。"""
    flags = 0
    kwargs = {}
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_CONSOLE
    kwargs["cwd"] = ROOT
    if flags:
        kwargs["creationflags"] = flags
    log("启动 server.py …")
    return subprocess.Popen([sys.executable, "server.py"], **kwargs)


def remote_head(cfg):
    rc, out = run_git(["rev-parse", "--verify",
                       "%s/%s" % (cfg["remote"], cfg["branch"])], cfg)
    return out.strip() if rc == 0 else None


def local_head():
    try:
        with open(os.path.join(ROOT, ".git", "HEAD"), encoding="utf-8") as f:
            ref = f.read().strip()
        if ref.startswith("ref:"):
            p = os.path.join(ROOT, ".git", *ref[4:].split("/"))
            with open(p, encoding="utf-8") as f:
                return f.read().strip()
        return ref
    except BaseException:
        return None


def check_update(cfg, child):
    """返回 True 表示发生了「停服→更新→重启」。"""
    remote = cfg["remote"]
    branch = cfg["branch"]
    rc, out = run_git(["fetch", remote, branch], cfg)
    if rc != 0:
        log("git fetch 失败（网络/凭据？）：%s" % out.strip()[:200])
        return False
    rh, lh = remote_head(cfg), local_head()
    if not rh or not lh or rh == lh:
        return False
    log("发现新版本：%s -> %s，开始更新" % (lh[:8], rh[:8]))
    if child and child.poll() is None:
        log("停止旧 server.py（pid=%s）" % child.pid)
        try:
            child.terminate()
        except OSError:
            pass
        for _ in range(30):                     # 最多等 15s 退出
            if child.poll() is not None:
                break
            time.sleep(0.5)
        if child.poll() is None:
            log("旧进程未退出，强制结束")
            try:
                child.kill()
                child.wait(timeout=10)
            except BaseException:
                pass
        time.sleep(2)                           # 留端口释放时间
    rc, out = run_git(["reset", "--hard", "%s/%s" % (remote, branch)], cfg)
    log("git reset --hard %s/%s -> rc=%d %s" % (remote, branch, rc, out.strip()[:200]))
    if rc != 0:
        log("更新失败！保持旧版本运行，下轮重试")
        return False
    return True


def main():
    log("=== autoupdate_watchdog 启动，仓库=%s ===" % ROOT)
    cfg = load_cfg()
    can_git = git_ok(cfg)
    child = None
    last_check = 0.0
    while True:
        try:
            cfg = load_cfg()                    # 每轮重读，改配置即生效
            port = server_port()
            alive = child is not None and child.poll() is None and port_open(port)
            if not alive:
                if child is not None and child.poll() is not None and \
                        child.returncode != 0:
                    log("server.py 异常退出（rc=%s）→ 自动拉起" % child.returncode)
                child = start_server()
                # 等服务起来再进入下一轮判断
                for _ in range(30):
                    if port_open(port):
                        break
                    time.sleep(1)
            now = time.time()
            if can_git and cfg.get("enabled") and now - last_check >= \
                    float(cfg.get("interval_secs", 300)):
                last_check = now
                if check_update(cfg, child):
                    log("更新完成，server.py 已在新代码上重启")
        except BaseException:
            log("看门狗循环异常（不影响下轮）：%s" % traceback.format_exc()[-500:])
        time.sleep(10)


if __name__ == "__main__":
    main()
