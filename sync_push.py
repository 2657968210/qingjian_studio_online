# -*- coding: utf-8 -*-
"""把 工作台_online 的最新版同步推送到 GitHub（供内网 git pull）。
双击 sync_push.bat 运行；token 也可作为命令行参数传入。

v2 (2026-11-04):
- 实时显示 git 进度（不再吞输出，避免"假死"）
- fetch 用 --depth 1 只拉同事最新一层提交，大幅减少下载量
- fetch 失败自动重试 2 次
- 浅层仓库 push 失败时自动 --unshallow 补拉后重试
- 自动读取 Windows 系统代理（开了全局模式的代理软件即生效）
"""
import os
import subprocess
import sys
try:
    import winreg
except ImportError:
    winreg = None

GIT = r"C:\Users\Admin\.workbuddy\binaries\PortableGit\versions\1.2.0\cmd\git.exe"
REPO = r"E:\推广素材\工作台_online"
PUSH_URL_FMT = "https://2657968210:{token}@github.com/2657968210/qingjian_studio_online.git"
MERGE_MSG = ("sync 2026-11-04 latest "
             "(Product Spotlight 2.5D, hide setup, 16-color palette, EN copy)")


def get_system_proxy():
    """读取 Windows 系统代理（Internet 设置）。返回 'http://host:port' 或 None。"""
    if winreg is None:
        return None
    try:
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                           r"Software\Microsoft\Windows\CurrentVersion\Internet Settings")
        enable, _ = winreg.QueryValueEx(k, "ProxyEnable")
        server, _ = winreg.QueryValueEx(k, "ProxyServer")
        winreg.CloseKey(k)
        if not enable or not server:
            return None
        if "=" in server:  # 按协议分写的形式，取 http/https
            for part in server.split(";"):
                if part.lower().startswith(("http=", "https=")):
                    server = part.split("=", 1)[1]
                    break
        if not server.startswith("http"):
            server = "http://" + server
        return server
    except OSError:
        return None


PROXY = get_system_proxy()


def run_stream(args, display=None):
    """运行 git 并把输出/进度实时打印出来（自动附带系统代理）。"""
    cmd = [GIT, "-C", REPO]
    if PROXY:
        cmd += ["-c", "http.proxy=" + PROXY, "-c", "https.proxy=" + PROXY]
    cmd += args
    print("> git %s" % (display or " ".join(args)), flush=True)
    if PROXY:
        print("  (走系统代理 %s)" % PROXY, flush=True)
    elif winreg is not None:
        print("  (未检测到开启的系统代理，直连)", flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in p.stdout:
        line = line.rstrip()
        if line:
            print("  " + line, flush=True)
    p.wait()
    return p.returncode


def main():
    if not os.path.isfile(GIT):
        print("[X] 找不到 git.exe：%s" % GIT)
        return 1
    if not os.path.isdir(os.path.join(REPO, ".git")):
        print("[X] %s 不是 git 仓库（缺少 .git）" % REPO)
        return 1

    token = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    if not token:
        try:
            token = input("请粘贴 GitHub Token (ghp_...) 后回车: ").strip()
        except EOFError:
            token = ""
    if not token:
        print("[X] 未输入 token，已取消。")
        return 1

    print("\n[1/3] 拉取 GitHub 上同事的最新提交（只拉一层，下面会显示实时进度）……")
    rc = -1
    for attempt in (1, 2, 3):
        rc = run_stream(["fetch", "origin", "main", "--depth", "1", "--progress"],
                        display="fetch origin main --depth 1 (第 %d 次尝试)" % attempt)
        if rc == 0:
            break
        if attempt < 3:
            print("  拉取中断，10 秒后自动重试……（若连续失败请检查能否打开 github.com）")
            import time
            time.sleep(10)
    if rc != 0:
        print("\n[X] fetch 三次都失败：本机当前无法直连 github.com（公司防火墙/网络不稳）。"
              "\n    可换网络（手机热点）后重跑本脚本，或配好代理软件后告诉我加代理参数。")
        return 1

    print("\n[2/3] 合并：以本地 11-04 最新版为准（冲突自动保留我方文件）……")
    rc = run_stream(["merge", "origin/main", "--allow-unrelated-histories",
                     "-X", "ours", "-m", MERGE_MSG])
    if rc != 0:
        print("\n[X] merge 失败：请在 %s 下运行 git status 查看冲突并手动处理，"
              "\n    或直接截图发我。" % REPO)
        return 1

    print("\n[3/3] 推送到 GitHub（上传约 280MB，进度实时显示，请耐心等）……")
    push_url = PUSH_URL_FMT.format(token=token)

    def do_push(extra=()):
        return run_stream(["push"] + list(extra) + [push_url, "main", "--progress"],
                          display="push origin main%s (token hidden)"
                                  % (" -f" if "-f" in extra else ""))

    rc = do_push()
    if rc != 0:
        # 浅层仓库可能导致 push 失败：补全历史后重试
        print("\n  push 失败，尝试补全历史后重推（--unshallow，会再下载一些数据）……")
        run_stream(["fetch", "origin", "main", "--unshallow", "--progress"],
                   display="fetch --unshallow")
        run_stream(["merge", "origin/main", "--allow-unrelated-histories",
                    "-X", "ours", "-m", MERGE_MSG + " (after unshallow)"])
        rc = do_push()
    if rc != 0:
        ans = ""
        try:
            ans = input("\n推送仍被拒。是否改用【强制推送】覆盖 GitHub 上的旧版本？"
                        "\n（覆盖同事之前的提交历史，本地内容是其超集，一般可接受；"
                        "确认请输入 y）: ").strip().lower()
        except EOFError:
            pass
        if ans == "y":
            rc = do_push(("-f",))
            if rc != 0:
                print("\n  强制推送也被仓库规则拒绝，尝试推送到新分支 sync-2026-11-04 ……")
                rc = run_stream(["push", push_url,
                                 "main:refs/heads/sync-2026-11-04", "--progress"],
                                display="push main -> sync-2026-11-04 (token hidden)")
                if rc == 0:
                    print("\n[OK] 已推送到新分支 sync-2026-11-04（main 被仓库规则保护，"
                          "无法直接覆盖）。")
                    print("请同事在 GitHub 网页上做一次性合并，两种方式任选：")
                    print("  方式1: 打开仓库 -> Pull requests -> New pull request ->")
                    print("         base: main  compare: sync-2026-11-04 -> Create -> Merge")
                    print("  方式2: 打开仓库 -> Settings -> Branches/Rules，临时删掉 main 的")
                    print("         保护规则，告诉我后我再重推 main。")
                    print("合并完成后内网即可 git pull。")
                    return 0
                print("\n[X] 新分支也推送失败，请把上面的报错截图发我。")
                return 1
        else:
            print("已取消。请把上面的报错截图发我。")
            return 1

    print("\n[OK] 推送成功！请通知同事在内网服务器执行：git pull 然后重启服务（端口 8765）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
