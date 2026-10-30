# -*- coding: utf-8 -*-
"""
请柬推广素材工作台 - 本地服务
================================
零外部依赖（Python 标准库 + numpy/PIL/cv2/imageio-ffmpeg）。
启动: python server.py   访问: http://localhost:8765
"""
import os, sys, json, time, uuid, threading, re, traceback, base64, hmac, shutil, hashlib
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from core import imgproc, cammove, compose, videoapi, imgapi, elemapi, prompts, showcase, smart, autofix, safedel, craftdetect
from core import metering as _metering

DATA = os.path.join(BASE, "data")
STATIC = os.path.join(BASE, "static")
BGM_DIR = os.path.join(BASE, "bgm")
CONFIG_PATH = os.path.join(BASE, "config.json")

os.makedirs(DATA, exist_ok=True)

# ---------------- 配置 ----------------

def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"cloud_token": "", "port": 8765, "video_api": {}, "storyboard_api": {}}

CONFIG = load_config()

# AI 调用计量（成本失控防护）：按会话 + 每日计数，超限闸门在 AI 端点拦截。
_metering.init(DATA)

# ---------------- 运行环境（本地 / 云端） ----------------
# 本地：直接 python server.py，绑 127.0.0.1、无口令，行为与以前完全一致。
# 云端：平台注入 PORT 环境变量 → 绑 0.0.0.0 并向平台端口汇报。
# 口令：设置了 STUDIO_PASSWORD（环境变量）或 config 里的 access_password 时启用访问门禁。
#       凭据可走四种通道（平台反代会吃掉 Authorization 头，所以不能只依赖 Basic）：
#         1) URL 令牌  ?k=<password>   → 校验通过后种 Cookie 并跳到干净地址（登录入口）
#         2) Cookie    studio_auth=<token>
#         3) 请求头    X-Studio-Key: <password>   （脚本/接口调用）
#         4) 请求头    Authorization: Basic ...    （本地或直连场景）
CLOUD_PORT = os.environ.get("PORT")
ACCESS_PASSWORD = os.environ.get("STUDIO_PASSWORD") or str(CONFIG.get("access_password") or "")
COOKIE_NAME = "studio_auth"
COOKIE_DAYS = 30
# 每个浏览器会话一个随机 sid，用于把上传/成片数据按客户端隔离，
# 防止共享口令下客户 A 看到/读取客户 B 的文件。
SID_NAME = "studio_sid"
SID_MAX_AGE = 180 * 86400  # 180 天


def auth_token(password):
    """由口令派生的稳定令牌，避免把口令本身写进浏览器 Cookie。"""
    return hmac.new(b"invitation-promo-studio", password.encode("utf-8"), hashlib.sha256).hexdigest()

def save_config(cfg):
    global CONFIG
    CONFIG = cfg
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

# ---------------- 任务系统 ----------------

JOBS = {}
JOBS_LOCK = threading.Lock()

def new_job(jtype):
    jid = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[jid] = {"id": jid, "type": jtype, "status": "running",
                     "progress": 0, "msg": "", "result": None, "error": None,
                     "created": time.time()}
    return jid

def job_progress(jid, done, total=100, msg=""):
    with JOBS_LOCK:
        j = JOBS.get(jid)
        if j:
            j["progress"] = round(done * 100 / max(total, 1))
            if msg: j["msg"] = msg

def _job_timeout(jid, minutes):
    """看门狗：任务超时（线程可能被外部因素卡死）→ 置 error，让前端停止轮询。"""
    with JOBS_LOCK:
        j = JOBS.get(jid)
        if j and j["status"] == "running":
            j["status"] = "error"
            j["error"] = ("The task ran longer than %d minutes and was stopped. "
                          "Please try again." % minutes)

def run_job(jid, fn, timeout_min=25):
    """在后台线程跑任务，并保证任务一定会落到终态（done / error）。

    ⚠ 2026-09-30 真实事故：清理临时文件时若被外部「安全删除钩子」拒绝，
    钩子抛的是 SystemExit（BaseException），`except Exception` 抓不住 ——
    线程静默死亡，任务状态永远停在 running，前端无限轮询：
    用户看到「点了修复没反应」「视频渲染完却加载不出来」。
    因此这里必须捕获 BaseException，并在 finally 里兜底收尾。"""
    def finish(status, result=None, error=None):
        with JOBS_LOCK:
            j = JOBS.get(jid)
            if not j:
                return
            j["status"] = status
            if status == "done":
                j["progress"] = 100
                j["result"] = result
            else:
                j["error"] = error or "The task stopped unexpectedly."

    def wrap():
        wd = threading.Timer(timeout_min * 60, _job_timeout, args=(jid, timeout_min))
        wd.daemon = True
        wd.start()
        try:
            result = fn()
            finish("done", result=result)
        except KeyboardInterrupt:
            finish("error", error="The task was interrupted.")
            raise
        except BaseException as e:            # SystemExit / MemoryError / 一切
            if not isinstance(e, Exception):
                traceback.print_exc()
            else:
                traceback.print_exc()
            with JOBS_LOCK:
                j = JOBS.get(jid)
                already = bool(j and j["status"] != "running")
            if not already:
                finish("error", error=str(e) or e.__class__.__name__)
        finally:
            wd.cancel()
            # 兜底：绝不允许有任务永远留在 running
            with JOBS_LOCK:
                j = JOBS.get(jid)
                if j and j["status"] == "running":
                    j["status"] = "error"
                    j["error"] = ("The task stopped unexpectedly (no result was produced). "
                                  "Please try again.")
    threading.Thread(target=wrap, daemon=True).start()

# ---------------- 项目存储 ----------------

def project_path(pid):
    return os.path.join(DATA, pid)

def load_project(pid):
    p = os.path.join(project_path(pid), "project.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_project(proj):
    p = os.path.join(project_path(proj["id"]), "project.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(proj, f, ensure_ascii=False, indent=2)

# 单图裁卡 auto 识别缓存：src 绝对路径 -> 内容区列表（进程内）。
# 同一张图重复渲染不重复烧视觉 API（craft_detect 一次约 10-20 次模型调用）。
_CARD_PAN_CACHE = {}

_LABEL_CN = {"gold_foil": "烫金", "emboss": "压纹", "lace": "蕾丝", "deboss": "凹印",
             "envelope": "信封", "main_card": "主卡", "uncertain": "细节",
             "rsvp": "RSVP回执卡", "reply": "RSVP回执卡", "insert": "内页插卡",
             "menu": "菜单卡", "band": "腰封", "seal": "蜡封",
             "sleeve": "封套", "shell": "外壳", "pocket": "口袋卡套", "other": "组件"}


def _kind_from_label(lab):
    """把视觉模型给的组件标签映射成 card_pan 的内容 kind。

    套装平铺图常见组件：主卡/内页 -> inner（平移横扫保可读），信封 -> envelope（推近），
    RSVP/菜单/腰封/蜡封 -> card（小卡轮换），深色外壳/封套/卡套 -> material（俯仰看纸材）。
    """
    s = str(lab or "").lower()
    if any(k in s for k in ("main_card", "invitation", "invite", "insert", "inner")):
        return "inner"          # 请柬内页：平移横扫，保文字可读
    if "envelope" in s:
        return "envelope"
    if any(k in s for k in ("sleeve", "shell", "outer", "folder", "pocket", "box", "clutch", "wrap")):
        return "material"
    if any(k in s for k in ("rsvp", "reply", "response", "menu", "band", "belly",
                            "seal", "wax", "tag", "card")):
        return "card"
    return "inner"


# ---- 切镜优先级（用户 2026-10-14 确认的顺序）----
#   外壳 → 请柬内页 → 回执卡 → 信封 → tag/配件
#   每个成片最多展示 MAX_DETAIL_SHOTS 个内容细节：识别到几件就切几件，不做品类补齐。
#   数值越小越靠前；同档按画面上下位置（从上到下）排。
_PRI_ORDER = (
    (("sleeve", "shell", "outer", "folder", "pocket", "box", "clutch", "wrap"), 10, "外壳"),
    (("main_card", "invitation", "invite", "insert", "inner"), 20, "请柬内页"),
    (("rsvp", "reply", "response"), 30, "回执卡"),
    (("envelope",), 40, "信封"),
)
_PRI_CRAFT = 45      # 工艺细节（烫金/雕刻/压纹微距）排在信封之后
_PRI_TAIL = 50       # menu / tag / 腰封 / 蜡封 / 其他
MAX_DETAIL_SHOTS = 4


def _content_pri(raw):
    """组件标签 → (优先级, 中文展示名)。"""
    s = str(raw or "").lower()
    if s == "craft":
        return _PRI_CRAFT, "工艺"
    for keys, pri, cn in _PRI_ORDER:
        if any(k in s for k in keys):
            return pri, cn
    return _PRI_TAIL, "配件"


def _detail_sort_key(r):
    """细节镜头排序键：切镜优先级 → 同档按画面从上到下。"""
    return (r.get("pri", _PRI_TAIL), r["box"][1] + r["box"][3] / 2.0)


def _cap_details(content, limit=MAX_DETAIL_SHOTS):
    """细节镜头截断，但给「工艺」保留固定席位（用户 2026-10-14 选定 A 方案）。

    规则：识别到几件就切几件、最多 limit 个；但只要有工艺细节（烫金 / 雕刻字的
    微距展示镜头），就一定留在成片里 —— 从末位回退挤掉一个非工艺项（配件优先让位），
    保证每支片都有一段金属工艺的细节特写。
    """
    if len(content) <= limit:
        return list(content)
    kept = list(content[:limit])
    craft = next((r for r in content if r.get("kind") == "craft"), None)
    if craft is None or any(k.get("kind") == "craft" for k in kept):
        return kept
    for i in range(len(kept) - 1, -1, -1):
        if kept[i].get("kind") != "craft":
            kept[i] = craft
            break
    kept.sort(key=_detail_sort_key)
    return kept


def _gold_emblem_box(roi):
    """在 ROI（BGR）里找「金在黑底」的雕刻字徽记（如黑封套上的金色字母章）。

    判定依据（用户确认的取景需求：切雕刻字，不切平面空白的封套）：
      · 金色像素（R/G 高、B 明显低）；金色在白底上的元素（烫金正文、二维码）
        会被「组件周边要暗」条件排除——只有刻在深色外壳上的才算雕刻字。
    返回相对 ROI 的归一化 [x,y,w,h]（含 22% 呼吸边距）；找不到返回 None。
    """
    import cv2
    import numpy as np
    try:
        roi16 = roi.astype(np.int16)
        b, g, r = roi16[:, :, 0], roi16[:, :, 1], roi16[:, :, 2]
        gold = ((r > 110) & (g > 85) & (b < np.minimum(r, g) - 30)).astype(np.uint8)
        H, W = gold.shape
        if W < 16 or H < 16 or gold.mean() < 0.004:
            return None
        n, lab, stats, _ = cv2.connectedComponentsWithStats(gold, 8)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        keep = np.zeros_like(gold, dtype=np.uint8)
        kept_stats = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < max(30, 0.0004 * W * H):
                continue
            x0, y0 = max(0, x - 15), max(0, y - 15)
            x1, y1 = min(W, x + w + 15), min(H, y + h + 15)
            ring = gray[y0:y1, x0:x1][lab[y0:y1, x0:x1] == 0]
            if ring.size and float(ring.mean()) < 95.0:     # 周边是暗底 → 雕在深色外壳上
                keep[lab == i] = 1
                kept_stats.append((x, y, w, h, area))
        if not keep.any():
            return None
        # 2026-10-29 珠链闸：暗底上的金色**离散小圆珠**（珠链/金珠串绕盘摆放）
        # 同样满足「金在暗底」，此前被当成雕刻字徽记切进微距（封面实锤）。
        # 雕刻字徽记是少数几个大连通域；珠链是大量小圆珠。kept≥8 且最大域
        # 占金面积 <30% → 判珠链，返回 None。
        if len(kept_stats) >= 8:
            _areas = sum(s[4] for s in kept_stats)
            _largest = max(s[4] for s in kept_stats)
            if _areas > 0 and _largest / float(_areas) < 0.30:
                return None
        # 近圆珠形态闸（2026-10-29 补）：珠子入框常常只有 3-7 颗（数量闸不触发）。
        # 珠子特征：bbox 接近方形（|w/h-1|<0.4）且填充度高（圆 fill≈0.785，>0.5）；
        # 雕刻字字母/花纹形状多样、fill 通常低于 0.5。kept≥3 且 ≥60% 近圆珠
        # 且没有一个大域占绝对主导（<50%）→ 判珠链，返回 None。
        if len(kept_stats) >= 3:
            _tot = sum(s[4] for s in kept_stats)
            _big = max(s[4] for s in kept_stats)
            _round = sum(1 for s in kept_stats
                         if abs(s[2] / max(1, s[3]) - 1.0) < 0.4
                         and s[4] / float(max(1, s[2] * s[3])) > 0.5)
            if _tot > 0 and _round >= max(3, int(0.6 * len(kept_stats))) and _big / float(_tot) < 0.5:
                return None
        # 单/成对实心圆珠闸（2026-10-29 补）：入框只有 1-2 颗珠时数量/占比闸全部
        # 不触发，单颗大金珠被当成徽记（封面实锤：micro 切到一颗金珠特写）。
        # 字母章是「多个字母域」或带内部镂空的域；单个近圆实心域（|w/h-1|<0.35
        # 且 fill>0.6）是珠子/圆钉，不是雕刻字。
        if 0 < len(kept_stats) <= 2:
            _s = max(kept_stats, key=lambda t: t[4])
            if (abs(_s[2] / max(1, _s[3]) - 1.0) < 0.35
                    and _s[4] / float(max(1, _s[2] * _s[3])) > 0.6):
                return None
        # 核心区几何闸（2026-10-29 终闸）：雕刻字徽记印/雕在封套**表面深处**，
        # 四周是封套纸面；而珠链/串饰只会**贴边或横穿**组件框（绕盘摆放）。
        # 调用方传入的 ROI = 组件框外扩 10%，把 ROI 内缩 12% 即组件核心区。
        # 金色域的包围盒与核心区的相交面积 / 包围盒面积 < 0.55 → 金域主要落在
        # 框边带（= 道具横穿），不是雕在封套上的字，返回 None。
        # （形态学闸在珠子反光粘连时会失效——粘连域 fill/圆度全变，几何闸不受影响）
        ys, xs = np.where(keep)
        gx0, gx1 = xs.min() / float(W), (xs.max() + 1) / float(W)
        gy0, gy1 = ys.min() / float(H), (ys.max() + 1) / float(H)
        _cw0, _ch0 = 0.12, 0.12
        _iw0 = max(0.0, min(gx1, 1.0 - _cw0) - max(gx0, _cw0))
        _ih0 = max(0.0, min(gy1, 1.0 - _ch0) - max(gy0, _ch0))
        _abox = (gx1 - gx0) * (gy1 - gy0)
        if _abox > 1e-6 and (_iw0 * _ih0) / _abox < 0.55:
            return None
        ew, eh = (gx1 - gx0) * 0.22, (gy1 - gy0) * 0.22
        x0, y0 = max(0.0, gx0 - ew), max(0.0, gy0 - eh)
        x1, y1 = min(1.0, gx1 + ew), min(1.0, gy1 + eh)
        return [x0, y0, x1 - x0, y1 - y0]
    except BaseException:
        return None


def _box_color_scores(img, box):
    """框内颜色画像：(gold_frac, green_frac)，均归一化 0-1（2026-10-27）。

    诊断实锤（用户反馈「切重点细节还是不太行」）：
      · 工艺候选没有颜色把关 —— 绿植/木盒等道具纹理被当成 pattern 工艺，
        macro 切到了绿植而不是请柬细节；
      · 多候选并存时正文文字块（笔画密）比金色徽记更容易拿高分。
    gold_frac: 烫金/金色像素占比（H 10-38 / S>60 / V>70，白底金字也算）
    green_frac: 绿色像素占比（H 35-90 / S>50 / V>40）—— 过高 = 花艺绿植，淘汰
    """
    import cv2 as _cv
    import numpy as _np
    try:
        H, W = img.shape[:2]
        x, y, bw, bh = [float(v) for v in box]
        px0, py0 = max(0, int(x * W)), max(0, int(y * H))
        px1, py1 = min(W, int((x + bw) * W)), min(H, int((y + bh) * H))
        if px1 - px0 < 24 or py1 - py0 < 24:
            return 0.0, 0.0
        roi = img[py0:py1, px0:px1]
        hsv = _cv.cvtColor(roi, _cv.COLOR_BGR2HSV)
        h_ = hsv[:, :, 0].astype(_np.int16)
        s_ = hsv[:, :, 1].astype(_np.int16)
        v_ = hsv[:, :, 2].astype(_np.int16)
        gold = ((h_ >= 10) & (h_ <= 38) & (s_ > 60) & (v_ > 70))
        green = ((h_ >= 35) & (h_ <= 90) & (s_ > 50) & (v_ > 40))
        return float(gold.mean()), float(green.mean())
    except BaseException:
        return 0.0, 0.0


def _bead_chain_like(img, box):
    """判别框内金色是否为「珠链/珍珠串」类装饰道具（2026-10-29）。

    诊断实锤（用户反馈「微距找的内容不对」）：封面图的桌面金珠链被金色
    占比评分当成烫金工艺，macro 切到了配饰珠子而不是请柬细节。珠链的
    视觉特征与烫金字/花纹完全不同：金色呈**大量离散小圆珠**状——连通域
    多、单域小、形状接近圆、彼此分离。烫金文字/徽记/花纹则相反：连通域
    少（几个字母/一朵花纹）、最大域占金面积比例高。
    判为珠链返回 True（工艺候选淘汰）；判断失败返回 False（不误杀）。
    """
    import cv2 as _cv
    import numpy as _np
    try:
        H, W = img.shape[:2]
        x, y, bw, bh = [float(v) for v in box]
        px0, py0 = max(0, int(x * W)), max(0, int(y * H))
        px1, py1 = min(W, int((x + bw) * W)), min(H, int((y + bh) * H))
        if px1 - px0 < 32 or py1 - py0 < 32:
            return False
        roi = img[py0:py1, px0:px1]
        hsv = _cv.cvtColor(roi, _cv.COLOR_BGR2HSV)
        gold = ((hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 38) &
                (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 70)).astype(_np.uint8)
        if gold.mean() < 0.01:
            return False
        n, _lab, stats, _ = _cv.connectedComponentsWithStats(gold, 8)
        comps = [s for s in stats[1:] if s[4] >= max(20, 0.0002 * (px1 - px0) * (py1 - py0))]
        if len(comps) < 8:
            return False                      # 域少 → 更像文字/徽记/花纹
        areas = _np.array([s[4] for s in comps], dtype=_np.float64)
        largest = float(areas.max()) / float(areas.sum())
        # 圆珠：单域填充度高（接近 1 的圆 → π/4≈0.785；字母/花纹通常远低）
        fills = []
        for s in comps[:24]:
            w_, h_ = int(s[2]), int(s[3])
            if w_ <= 0 or h_ <= 0:
                continue
            fills.append(float(s[4]) / float(w_ * h_))
        roundish = (sum(1 for f in fills if f > 0.6) / float(len(fills))) if fills else 0.0
        return len(comps) >= 8 and largest < 0.25 and roundish > 0.5
    except BaseException:
        return False


def _tighten_box_cv(cv, np, gray, box, W, H):
    """按边缘能量把组件框向内容密集区收缩（内容紧缩）。

    模型框偶尔把旁边的道具（花/珍珠链）或空白背景一起框进来，取景会切到
    非卡片内容（用户反馈：切到的必须是卡片）。文字/花纹边缘能量高，
    花瓣内部/空白背景能量低 → 从框四端向内去掉低能量端。
    收缩幅度限制：新宽/高不小于原框 55%，防收太狠；回补 8% 边距保持构图。
    返回归一化 [x, y, w, h]；失败/太小时返回 None（保留原框）。
    """
    try:
        x0, y0, w, h = [float(v) for v in box]
        px0, py0 = max(0, int(x0 * W)), max(0, int(y0 * H))
        px1, py1 = min(W, int((x0 + w) * W)), min(H, int((y0 + h) * H))
        if px1 - px0 < 40 or py1 - py0 < 40:
            return None
        roi = gray[py0:py1, px0:px1].astype(np.float32)
        gx = cv.Sobel(roi, cv.CV_32F, 1, 0, ksize=3)
        gy = cv.Sobel(roi, cv.CV_32F, 0, 1, ksize=3)
        e = gx * gx + gy * gy
        colp = e.mean(axis=0)
        rowp = e.mean(axis=1)

        def _span(p, keep=0.90):
            tot = float(p.sum())
            if tot <= 0:
                return 0, len(p) - 1
            i, j = 0, len(p) - 1
            lo, hi = 0.0, 0.0
            while i < j and (lo + hi) < (1.0 - keep) * tot:
                if lo <= hi:
                    lo += float(p[i])
                    i += 1
                else:
                    hi += float(p[j])
                    j -= 1
            return i, j

        ci0, ci1 = _span(colp)
        ri0, ri1 = _span(rowp)
        bw, bh = px1 - px0, py1 - py0
        ci0 = int(min(ci0, 0.225 * bw))
        ci1 = int(max(ci1, 0.775 * bw))
        ri0 = int(min(ri0, 0.225 * bh))
        ri1 = int(max(ri1, 0.775 * bh))
        padx, pady = int(0.08 * bw), int(0.08 * bh)
        qx0, qy0 = max(0, px0 + ci0 - padx), max(0, py0 + ri0 - pady)
        qx1, qy1 = min(W, px0 + ci1 + padx), min(H, py0 + ri1 + pady)
        if qx1 - qx0 < 40 or qy1 - qy0 < 40:
            return None
        return [qx0 / float(W), qy0 / float(H),
                (qx1 - qx0) / float(W), (qy1 - qy0) / float(H)]
    except BaseException:
        return None


# ---- 开场钩子镜头选择（用户 2026-10-14：开头要有钩子、按图片匹配、稳定性优先）----
_HOOK_LABEL = "开场"      # tag 由 card_pan 的开场分支补全（"开场 · 开篇推进"），别写重
# 开场手法全集（客户端不传时按图自动匹配；可用 hook 参数强制固定，如模版视频）
# 开场手法库（2026-10-14 收紧为 2 种：横扫整体 / 旋转整体）。
# 推进 / 揭示 / 甩镜 / 切细节入镜 已全部移除，不再作为开场钩子可选。
_HOOK_MOVES = ("hook_scan", "hook_orbit")
_HOOK_SEED_CACHE = {}


def _hook_seed(src):
    """源图指纹：同一张图 → 同一个开场；换图才换（用户要求"同图固定、不同图不同"）。"""
    k = os.path.abspath(src)
    v = _HOOK_SEED_CACHE.get(k)
    if v is None:
        try:
            with open(src, "rb") as f:
                v = int(hashlib.md5(f.read()).hexdigest()[:8], 16)
        except BaseException:
            v = 0
        _HOOK_SEED_CACHE[k] = v
    return v


def _pick_hook(content, OW, OH, src, force=None):
    """按图片内容挑开场镜头（确定性，不随机）。

    老版本开场写死「整体扫描」，每条视频、每张图开头都一样（该手法已保留为
    hook_scan 候选）。这里分两步：
      1) 用内容特征筛出「这张图成立的开场」——有烫金/蜡封才有东西可"揭示"；
         主体大小合适才推得动；竖或方画幅才有横向空间甩镜；中心留白足够
         （edge≥0.18）才用旋转开场，防转出原图条纹；细节推近与整体扫描恒可用；
      2) 在成立集合里用源图指纹分散选取 —— 语义合理，同时换图就换开场。
    force 传入 _HOOK_MOVES 之一时跳过自动匹配、固定用该手法（模版视频等场合）。
    """
    R = OW / float(OH)
    main = content[0]                     # content 已按切镜优先级排好，[0] 为最优先细节
    bx, by, bw, bh = main["box"]
    cx_n, cy_n = bx + bw / 2.0, by + bh / 2.0
    off = abs(cx_n - 0.5)                 # 主体偏离画面中线的程度
    area = bw * bh
    edge = min(cx_n, 1.0 - cx_n, cy_n, 1.0 - cy_n)   # 主体中心到最近画边的余量
    has_craft = any(k.get("kind") == "craft" for k in content)

    # 开场手法收紧为 2 种（用户 2026-10-14）：横扫整体(hook_scan) / 旋转整体(hook_orbit)。
    # 横扫整体恒可用（稳、全能兜底）；旋转整体仅当主体中心留白足够
    # （edge>=0.18，转不飞出原图、不出条纹）才可选。推进 / 揭示 / 甩镜 / 切细节入镜
    # 已全部移除，不再作为开场钩子候选。
    score = {"hook_scan": 2}
    if edge >= 0.18:
        score["hook_orbit"] = 2

    usable = sorted(score.keys())          # 排序保证确定性
    pick = usable[_hook_seed(src) % len(usable)]
    if force in _HOOK_MOVES:               # 显式指定：固定用该开场手法（模版视频等）
        pick = force
    return {"box": list(main["box"]), "kind": "hook",
            "label": _HOOK_LABEL, "move": pick}


def _synthetic_macro_recenter(cp, plan, src, W, H, OW, OH):
    """兜底 plan（build_synthetic_variety，不看图）的 craft 微距窗口重对焦。

    2026-10-27 诊断实锤：识别为空走兜底时，微距窗是固定启发式位置——
    对烫金.jpg 会切到正文、对带绿植的请柬会切到道具，用户看到的
    「切重点细节不行」多半是这个。读图后重排：
      1) 金色（烫金特征 H10-38/S>60/V>70）最大聚类质心优先；
      2) 图内无金色则退回整图细节密度焦点（refine_focus_pt）。
    只动 macro 镜窗口中心（p0/p1 同步），尺寸/时长/缓动不变，窗口 clamp 进原图。
    """
    try:
        import cv2 as _c
        import numpy as _n
        img = _c.imdecode(_n.fromfile(src, dtype=_n.uint8), _c.IMREAD_COLOR)
        if img is None:
            return
        from core.imgproc import normalize_luma as _nl
        img = _nl(img)
        ih, iw = img.shape[:2]
        R = OW / float(OH)
        # 缩小到长边 1200 做打分（精度足够、速度快）
        _s = 1200.0 / max(ih, iw)
        small = _c.resize(img, (int(iw * _s), int(ih * _s)),
                          interpolation=_c.INTER_AREA) if _s < 1 else img
        sh, sw = small.shape[:2]

        def _win_score(fx, fy, ww_px):
            """候选中心的窗口得分：金占比×3 + 边缘密度 − 绿占比×1.5。"""
            wh_px = ww_px / R
            x0 = int(max(0, (fx - ww_px / 2.0) * sw / iw))
            y0 = int(max(0, (fy - wh_px / 2.0) * sh / ih))
            x1 = int(min(sw, (fx + ww_px / 2.0) * sw / iw))
            y1 = int(min(sh, (fy + wh_px / 2.0) * sh / ih))
            if x1 - x0 < 8 or y1 - y0 < 8:
                return -9.0
            roi = small[y0:y1, x0:x1]
            # 只对窗口中央 50% 打分：9x16 微距窗高达图高 60%+，全窗打分时
            # 各候选大量重叠、得分趋同（实测 0.5837 vs 0.5853 分不出好坏）；
            # 中央半窗才是「推近后真正落在画面中心的内容」。
            _h2, _w2 = roi.shape[:2]
            roi = roi[_h2 // 4:_h2 - _h2 // 4, _w2 // 4:_w2 - _w2 // 4]
            hsv = _c.cvtColor(roi, _c.COLOR_BGR2HSV)
            gold = float(((hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 38) &
                          (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 70)).mean())
            green = float(((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) &
                           (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 40)).mean())
            g = _c.cvtColor(roi, _c.COLOR_BGR2GRAY)
            edge = float(_c.Sobel(g, _c.CV_32F, 1, 0, ksize=3).std()) + \
                float(_c.Sobel(g, _c.CV_32F, 0, 1, ksize=3).std())
            return gold * 3.0 + edge / 300.0 - green * 1.5

        # 候选中心：金色最大聚类质心 / 细节密度焦点 / 原窗口中心
        cands = []
        hsv = _c.cvtColor(small, _c.COLOR_BGR2HSV)
        gm = ((hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 38) &
              (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 70)).astype(_n.uint8)
        if float(gm.mean()) >= 0.006:
            gm2 = _c.morphologyEx(gm, _c.MORPH_CLOSE,
                                  _n.ones((15, 15), _n.uint8))
            nlab, _lab, stats, cents = _c.connectedComponentsWithStats(gm2)
            best, bi = 0, None
            for i in range(1, nlab):
                a = int(stats[i, _c.CC_STAT_AREA])
                if a > best and a >= max(200, 0.0006 * sw * sh):
                    best, bi = a, i
            if bi is not None:
                cands.append((float(cents[bi][0]) / sw, float(cents[bi][1]) / sh))
        fpt = cp.refine_focus_pt(img, [0.05, 0.05, 0.9, 0.9])
        if fpt:
            cands.append((float(fpt[0]), float(fpt[1])))
        if not cands:
            return
        for m in plan:
            if m.get("kind") != "macro":
                continue
            cx0, cy0, ww = m["p0"]
            ocx, ocy = cx0 / float(iw), cy0 / float(ih)
            scored = []
            for fx, fy in cands + [(ocx, ocy)]:
                _fx = min(max(fx, ww / 2.0 / iw), 1.0 - ww / 2.0 / iw)
                _wh = ww / R
                _fy = min(max(fy, _wh / 2.0 / ih), 1.0 - _wh / 2.0 / ih)
                scored.append((_win_score(_fx, _fy, ww), _fx, _fy))
            scored.sort(reverse=True)
            _sc, bfx, bfy = scored[0]
            if _sc <= -8.0:
                continue
            for pk in ("p0", "p1"):
                cx, cy, ww2 = m[pk]
                m[pk] = [bfx * iw, bfy * ih, ww2]
    except BaseException:
        return


def _fill_variety_shots(cp, src, W, H, OW, OH, n_extra):
    """内容稀疏时（识别件少），用整图启发式子区补出 n_extra 个运镜镜头，
    保证 旋转 / 多方向平移 / 拉远 齐全（2026-10-29 用户口径：不出推近）。

    子区刻意错开位置（四角 + 中央）以制造取景变化；move 直接指定，各走
    card_pan 已调稳的实现（旋转边缘框自动降级为纯横滚，不晃）。
    """
    subs = [
        ([0.08, 0.10, 0.40, 0.42], "pan"),     # 左上 · 平移 →
        ([0.52, 0.12, 0.40, 0.42], "orbit"),   # 右上 · 旋转
        ([0.10, 0.54, 0.40, 0.40], "pan_v"),   # 左下 · 平移 ↓
        ([0.55, 0.58, 0.40, 0.40], "pull"),    # 右下 · 拉远
        ([0.30, 0.30, 0.42, 0.42], "diag"),    # 中央 · 对角扫
    ]
    regions = []
    for i, (b, mv) in enumerate(subs[:max(1, n_extra)]):
        regions.append({"box": list(b), "kind": "card", "label": "细节%d" % (i + 1),
                        "move": mv})
    return cp.build_plan_from_regions(regions, OW, OH, W, H)


def _converge_duration(cp, plan, src, W, H, OW, OH, target_secs=15.0):
    """把单图裁卡成片时长收敛到 ~target_secs（默认 15s），并保证运镜多样性。

    1) 内容镜头不足（识别件少）-> 用整图子区补出多样运镜镜头（pan/旋转/对角/拉/推）；
    2) 不够长 -> 等比拉伸每镜节奏到 target（单镜上限 4.5s，防拖沓）；
    3) 超过 MAX_TOTAL(15s) -> 整体压回 15s 内（含叠化重叠）。
    返回新 plan（原 plan 列表被复用修改）。
    """
    def _tot(pl):
        return sum(m["dur"] for m in pl) - cp.XFADE * max(0, len(pl) - 1)

    # 1) 内容稀疏补镜：先确保成片起码有几个运镜，再凑时长
    if plan and _tot(plan) < target_secs - 0.5:
        need = int((target_secs - _tot(plan)) // 2.3) + 1
        extra = _fill_variety_shots(cp, src, W, H, OW, OH, max(1, min(need, 4)))
        for m in extra:
            if str(m["kind"]).startswith("hook") or m["kind"] == "reveal":
                continue
            plan.append(m)
    # 2) 拉伸到 target（单镜封顶 4.5s）
    if plan and _tot(plan) < target_secs - 0.05:
        for _ in range(8):
            sc = target_secs / _tot(plan)
            for m in plan:
                m["dur"] = min(4.5, round(m["dur"] * sc, 2))
            if _tot(plan) >= target_secs - 0.05:
                break
    # 3) 压回 MAX_TOTAL 内
    if plan and _tot(plan) > cp.MAX_TOTAL:
        target = cp.MAX_TOTAL + cp.XFADE * max(0, len(plan) - 1)
        cur = sum(m["dur"] for m in plan)
        sc = target / cur if cur > 0 else 1.0
        for m in plan:
            m["dur"] = max(cp.MIN_DUR, round(m["dur"] * sc, 2))
    return plan


def _auto_card_pan_plan(cp, src, proj, cfg, W, H, OW, OH, target_secs=None, force_hook=None):
    """单图裁卡自动编排（真识别版）。

    识别链路（视觉模型，cfg 用 storyboard_api 的 key）：
      1) detect_cards  -> 独立组件框（主卡 / 信封 / RSVP 回执卡 / 菜单 / 腰封 /
                          深色外壳封套…）
      2) craft_detect  -> 工艺细节框（烫金/压纹/雕刻，is_text/is_logo/is_pattern；
                          lace 等道具装饰不进镜头）
    编排：开场钩子镜头（按图片内容匹配，整体扫描/推进/揭示/甩镜/细节推近/环绕 六选一，见 _pick_hook）
    + 内容镜头（**按切镜优先级排序**：外壳 → 请柬内页 → 回执卡 →
    信封 → 工艺 → 配件，同档按画面从上到下；运镜由内容 kind 决定：内页平移 / 信封推近 /
    小卡轮换 / 外壳俯仰 / **工艺改微距推近（不旋转）**）+ 收尾拉远。
    约束（用户 2026-10-14 口径）：
      · 识别到几件就切几件，**不做品类补齐**；内容细节最多 MAX_DETAIL_SHOTS(4) 个；
      · **工艺镜头保留席位**：检到工艺就一定进片，末位非工艺项（配件优先）让位；
      · 工艺候选按 IoU 去重，全片最多 1 个工艺微距镜头（不重复工艺镜头）；
      · 工艺框与整卡框几乎重合(IoU>0.85)时丢工艺（等于整卡没有独立细节）；
      · 总时长由 card_pan 压到 15s 内。
    识别全空时退回整图合成套（保证 扫/推/拉/细节 齐全，不再退化两镜）。
    """
    key = os.path.abspath(src)
    content = _CARD_PAN_CACHE.get(key)
    if content is None:
        def _iou(a, b):
            ax2, ay2 = a[0] + a[2], a[1] + a[3]
            bx2, by2 = b[0] + b[2], b[1] + b[3]
            iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
            ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
            u = a[2] * a[3] + b[2] * b[3] - iw * ih
            return (iw * ih / u) if u > 0 else 0.0

        # 1) + 2) 独立组件检测 与 工艺细节检测 是**两个互不依赖的视觉调用**，
        #    原串行要 ~2 倍单调用时；用线程池并行把这段墙钟时间砍半（2026-10-14 性能修复）。
        def _fetch_cards():
            out = []
            try:
                cd = craftdetect.detect_cards(src, cfg, depth="fast", max_cards=6)
                for c in (cd.get("cards") or []):
                    b = c.get("bbox")
                    if not (isinstance(b, (list, tuple)) and len(b) == 4):
                        continue
                    try:
                        conf = float(c.get("confidence") or 0.0)
                    except Exception:
                        conf = 0.0
                    lab = str(c.get("label") or "card").lower()
                    out.append({"box": [float(v) for v in b], "kind": _kind_from_label(lab),
                                "label": _LABEL_CN.get(lab, "组件"), "raw": lab, "conf": conf})
            except BaseException:
                pass
            return out

        def _fetch_crafts():
            out = []
            try:
                cr = craftdetect.craft_detect(src, cfg, depth="fast", max_candidates=4)
                for c in (cr.get("candidates") or []):
                    b = c.get("bbox")
                    if not (isinstance(b, (list, tuple)) and len(b) == 4):
                        continue
                    if c.get("label") == "lace":
                        continue
                    # is_text/is_logo/is_pattern 全 false 不再直接丢（2026-10-29）：
                    # 浮雕边框 / 立体贴花 / 压凹底纹可能三者都不标，直接丢会把
                    # 「浮雕类工艺」整类漏掉——改为降权 0.5，让有明确标记的优先。
                    mark = 1.0 if (c.get("is_text") or c.get("is_logo")
                                   or c.get("is_pattern")) else 0.5
                    out.append((float(c.get("micro_score") or 0.0) * mark,
                                [float(v) for v in b],
                                _LABEL_CN.get(c.get("label"), "工艺"),
                                bool(c.get("is_text")), bool(c.get("is_logo"))))
            except BaseException:
                pass
            return out

        with ThreadPoolExecutor(max_workers=2) as _ex:
            cards = _ex.submit(_fetch_cards).result()
            raw = _ex.submit(_fetch_crafts).result()
        # 组件去重：模型偶尔把同一件报两次（如两个 RSVP）——
        # 同名标签只留置信度最高的一个，再按 IoU>0.5 去掉高度重叠的框
        cards.sort(key=lambda r: r["conf"], reverse=True)
        _kept = []
        for r in cards:
            # 按「原始标签」去重（不能按中文展示名：同级品类共用同一个名字）
            if any(r.get("raw") == k.get("raw") or _iou(r["box"], k["box"]) > 0.5 for k in _kept):
                continue
            _kept.append(r)
        cards = _kept

        # 1.2) 打优先级：外壳 → 请柬内页 → 回执卡 → 信封 → 配件（见 _content_pri）。
        #   旧版只留「请柬本体」、把外壳/信封一律丢掉；用户 2026-10-14 已给出新的切镜顺序，
        #   改为全部保留、按优先级选出前 N 个（不再按品类整类丢弃）。
        for r in cards:
            r["pri"], r["label"] = _content_pri(r.get("raw"))

        # 1.5) 外壳/封套：内部若有「金在黑底」的雕刻字徽记，该镜就切成徽记特写
        #   （金字母章随角度反光变化最上镜，用户此前确认）；没有徽记则保留整件外壳本身。
        try:
            import cv2 as _cv
            import numpy as _np
            _img = _cv.imdecode(_np.fromfile(src, dtype=_np.uint8), _cv.IMREAD_COLOR)
            if _img is not None:
                from core.imgproc import normalize_luma as _nl
                _img = _nl(_img)   # 2026-10-27 亮度归一：暗图金徽记/焦点分析更稳
        except BaseException:
            _img = None
        if _img is not None:
            _IH, _IW = _img.shape[:2]
            for r in cards:
                if r.get("kind") != "material":
                    continue
                b = r["box"]
                px0, py0 = max(0, int(b[0] * _IW)), max(0, int(b[1] * _IH))
                px1, py1 = min(_IW, int((b[0] + b[2]) * _IW)), min(_IH, int((b[1] + b[3]) * _IH))
                if px1 - px0 < 16 or py1 - py0 < 16:
                    continue
                # 徽记搜索范围比组件框外扩 10%，避免金字被组件框边缘截断
                _mx, _my = int(0.10 * (px1 - px0)), int(0.10 * (py1 - py0))
                qx0, qy0 = max(0, px0 - _mx), max(0, py0 - _my)
                proi = _img[qy0:min(_IH, py1 + _my), qx0:min(_IW, px1 + _mx)]
                emb = _gold_emblem_box(proi)
                if emb is not None:
                    r["box"] = [(qx0 + emb[0] * proi.shape[1]) / float(_IW),
                                (qy0 + emb[1] * proi.shape[0]) / float(_IH),
                                emb[2] * proi.shape[1] / float(_IW),
                                emb[3] * proi.shape[0] / float(_IH)]
                    r["kind"] = "craft"             # 雕刻字：烫金工艺细节（走微距推近展示）
                    r["label"] = (r.get("label") or "外壳") + " · 雕刻字"

        # 2) 工艺细节：raw 已由上方并发取回；同一区域常被识别成多个候选
        #    （uncertain + gold_foil 覆盖同一块烫金字）。2026-10-27 加「颜色把关」：
        #    绿占比 >0.30 的候选直接淘汰（诊断实锤：花艺绿植/木盒纹理被当成
        #    pattern 工艺，macro 切到绿植而不是请柬细节）；2026-10-29 再加四道
        #    把关（用户反馈「微距找的内容不对，应该找烫金/浮雕/雕刻字等细节」）：
        #    a) 珠链判别：金色呈大量离散小圆珠 → 配饰珠链，淘汰；
        #    b) 面积把关：工艺框 >0.30 整图 → 不是局部工艺细节，淘汰；
        #    c) 形状把关：极端细长条（横跨画面的链/带）→ 淘汰；
        #    d) 中心落卡：工艺框中心必须在某张卡片组件框内（工艺是卡片上的工艺，
        #       中心在卡外的基本是道具/配饰）。其余按
        #    micro_score ×（0.35 + 0.65×金占比）× 中心先验 择优——金色徽记/
        #    烫金花纹优先于正文文字块。仍按 IoU 去重、全片只留 1 个工艺微距。
        _cands = []
        for _sc, b, lab, _is_txt, _is_logo in raw:
            gold, green = (0.0, 0.0)
            if _img is not None:
                gold, green = _box_color_scores(_img, b)
            if green > 0.30:
                continue
            if _img is not None and _bead_chain_like(_img, b):
                continue
            if b[2] * b[3] > 0.30:
                continue
            # 2026-10-29 小饰物闸：极小面积（<3% 图幅）的纯 pattern 金色框
            # 基本是珠子/小饰物/金属点缀（诊断实锤：盘边金珠链被框成
            # gold_foil/pattern 切进微距），真工艺细节（烫金字母/徽记/花纹）
            # 会标 is_text/is_logo。小框必须挂文字/标志属性才保留。
            if b[2] * b[3] < 0.03 and not (_is_txt or _is_logo):
                continue
            _ar = b[2] / max(1e-6, b[3])
            if _ar > 4.0 or _ar < 0.25:
                continue
            if cards and not any(
                    (b[0] + b[2] / 2.0) >= c["box"][0] and (b[0] + b[2] / 2.0) <= c["box"][0] + c["box"][2]
                    and (b[1] + b[3] / 2.0) >= c["box"][1] and (b[1] + b[3] / 2.0) <= c["box"][1] + c["box"][3]
                    for c in cards):
                continue
            cxn = b[0] + b[2] / 2.0
            cyn = b[1] + b[3] / 2.0
            center = max(0.4, 1.0 - ((cxn - 0.5) ** 2 + (cyn - 0.5) ** 2) * 2.0)
            _cands.append((float(_sc) * (0.35 + 0.65 * gold) * center, b, lab))
        _cands.sort(key=lambda t: t[0], reverse=True)
        crafts = []
        for _cs, b, lab in _cands:
            if any(_iou(b, k["box"]) > 0.3 for k in crafts):   # 同区域重复候选
                continue
            crafts.append({"box": b, "kind": "craft", "label": lab, "pri": _PRI_CRAFT})
        # 整卡与卡内工艺几乎重合时丢工艺（等于整卡没有独立细节）
        crafts = [k for k in crafts
                  if all(_iou(k["box"], c["box"]) < 0.85 for c in cards)]
        crafts = crafts[:1]        # 全片最多 1 个工艺微距镜头（不要重复工艺镜头）
        # 2026-10-28：工艺特写优先请柬主卡本体 —— 剔除落在独立配饰/道具
        # （非请柬部件，如丝带/花朵/小物件）上的工艺候选，避免把「搭配类配饰」
        # 的细节当卡片工艺微距切进去。落在主卡/信封/外壳(含金徽记)上的保留。
        if cards and any(c.get("kind") == "inner" for c in cards):
            def _ovc(a, b):
                ax2, ay2 = a[0] + a[2], a[1] + a[3]
                bx2, by2 = b[0] + b[2], b[1] + b[3]
                iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
                ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
                return (iw * ih) / (a[2] * a[3] + 1e-9)

            _kept = [k for k in crafts
                     if any(_ovc(k["box"], c["box"]) > 0.10 for c in cards)]
            if _kept:
                crafts = _kept

        # 1.6) 框内容紧缩：把含道具/空白背景的框收缩到卡片实际内容区（见函数说明）
        if _img is not None:
            _gray = _cv.cvtColor(_img, _cv.COLOR_BGR2GRAY)
            for r in cards + crafts:
                nb = _tighten_box_cv(_cv, _np, _gray, r["box"], _IW, _IH)
                if nb:
                    r["box"] = nb
            # 1.7) 焦点精确化（2026-10-26：用户反馈「切细节没切到重点」）：
            #   识别框偏大时框心≠重点，微距/推近必须对准框内细节最密处
            #   （烫金字、雕花、花体字句）。对每个内容框算 refine_focus_pt，
            #   card_pan._win_for_box 会把取景中心移到该焦点上（框太小/平坦
            #   时返回 None，保持框心，绝不硬挪）。
            for r in cards + crafts:
                fpt = cp.refine_focus_pt(_img, r["box"])
                # 2026-10-27 金色质心直取：框内金色（烫金特征）占比达标时，
                # 焦点直接取金色像素质心——probe3 实锤：密度投票即便加了金色
                # 偏置，macro 终窗金占比仍只有 0.05-0.07（被笔画密的正文带走），
                # 而金色质心才是徽记/烫金花纹真正所在。金色不达标时保留密度焦点。
                try:
                    _H2, _W2 = _img.shape[:2]
                    _bx, _by, _bw2, _bh2 = [float(v) for v in r["box"]]
                    _px0, _py0 = max(0, int(_bx * _W2)), max(0, int(_by * _H2))
                    _px1, _py1 = min(_W2, int((_bx + _bw2) * _W2)), min(_H2, int((_by + _bh2) * _H2))
                    if _px1 - _px0 >= 48 and _py1 - _py0 >= 48:
                        _roi = _img[_py0:_py1, _px0:_px1]
                        _hsv = _cv.cvtColor(_roi, _cv.COLOR_BGR2HSV)
                        _gm = ((_hsv[:, :, 0] >= 10) & (_hsv[:, :, 0] <= 38) &
                               (_hsv[:, :, 1] > 60) & (_hsv[:, :, 2] > 70))
                        if float(_gm.mean()) >= 0.02 and int(_gm.sum()) >= 200:
                            _gf = _cv.GaussianBlur(_gm.astype(_np.float32), (0, 0), 7)
                            _ys, _xs = _np.nonzero(_gf > _gf.max() * 0.5)
                            if len(_xs) >= 50:
                                _gx = (_px0 + float(_xs.mean())) / float(_W2)
                                _gy = (_py0 + float(_ys.mean())) / float(_H2)
                                _ix0, _ix1 = _bx + 0.12 * _bw2, _bx + 0.88 * _bw2
                                _iy0, _iy1 = _by + 0.12 * _bh2, _by + 0.88 * _bh2
                                fpt = [min(max(_gx, _ix0), _ix1),
                                       min(max(_gy, _iy0), _iy1)]
                except BaseException:
                    pass
                if fpt:
                    r["focus"] = fpt

        content = cards + crafts
        # 2026-10-29 用户口径：自动运镜只保留「旋转 / 平移（多方向）/ 拉远」。
        # 一张图没检出工艺时，把第一个小卡（card）或内页（inner）显式指定为旋转
        # 展示（原轮转可能给它分到拉远，成片就没有细节展示了）；只指定一个，
        # 其余镜头仍保持多方向平移/旋转/拉远轮转。不再产 macro 推近。
        if not any(r.get("kind") == "craft" for r in content):
            _m = next((r for r in content if r.get("kind") == "card"),
                      next((r for r in content if r.get("kind") == "inner"), None))
            if _m is not None:
                _m["move"] = "orbit"
        # 排序：切镜优先级（外壳 → 内页 → 回执 → 信封 → 工艺 → 配件）→ 同档按画面从上到下
        content.sort(key=_detail_sort_key)
        # 细节镜头上限：识别到几件就切几件（不做品类补齐），但不超过 MAX_DETAIL_SHOTS；
        # 工艺镜头（微距展示）保留席位，配件让位（用户 2026-10-14 选定 A 方案）
        content = _cap_details(content)
        _CARD_PAN_CACHE[key] = content
    if not content:
        plan = cp.build_synthetic_variety(W, H, OW, OH)
        # 2026-10-27：兜底微距窗重对焦（金色聚类质心/细节密度），治「切重点
        # 细节切到正文/道具」——固定启发式窗不看图，是最常见的偏靶原因。
        _synthetic_macro_recenter(cp, plan, src, W, H, OW, OH)
        return plan
    # 开场（2026-10-29 用户口径：「一开始要各种平移扫描，展示所有的卡片，然后再
    # 进行小卡内页等展示」）：
    #   · 默认 = 两段巡览扫描（tour_h 横向满幅 + tour_v 斜向回扫），行程按**全部
    #     卡片并集框**计算，保证每件卡片都亮过相，再进细节镜头；
    #   · force_hook 显式指定时仍用旧钩子（hook_scan / hook_orbit），不影响其他台。
    if force_hook in _HOOK_MOVES:
        tour = [_pick_hook(content, OW, OH, src, force=force_hook)]
        detail_max = MAX_DETAIL_SHOTS
    else:
        _bx0 = min(float(r["box"][0]) for r in content)
        _by0 = min(float(r["box"][1]) for r in content)
        _bx1 = max(float(r["box"][0]) + float(r["box"][2]) for r in content)
        _by1 = max(float(r["box"][1]) + float(r["box"][3]) for r in content)
        _bx0, _by0 = max(0.0, _bx0), max(0.0, _by0)
        _ub = [_bx0, _by0,
               max(1e-3, min(1.0, _bx1) - _bx0),
               max(1e-3, min(1.0, _by1) - _by0)]
        tour = [{"box": list(_ub), "kind": "hook", "label": _HOOK_LABEL, "move": "tour_h"},
                {"box": list(_ub), "kind": "hook", "label": _HOOK_LABEL, "move": "tour_v"}]
        # 两段巡览已占 ~6s，细节镜头收到 3 个，保证每镜有足够时长走完行程
        # （镜头太多会把每镜压到 1.6s 下限，位移被速度红线裁成"微动"，又变卡）。
        detail_max = 3
    # 截断走 _cap_details：工艺（烫金/雕刻字）保留席位，不会因为排第 5 被砍
    _detail = content if force_hook in _HOOK_MOVES else _cap_details(content, detail_max)
    regions = tour + _detail
    return cp.build_plan_from_regions(regions, OW, OH, W, H, target_secs=target_secs)


def safe_name(name):
    name = os.path.basename(name or "file")
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    return name or "file"


def _bgm_pick(body, default_name):
    """生成请求的 BGM 选择：body.bgm 优先（曲库/用户上传曲名，safe_name 防目录穿越，
    可带可不带 .mp3），文件存在才采用；否则回退该模版的固定默认曲。
    2026-10-27 用户需求：生成后 BGM 可选择、也可上传自己的。"""
    want = body.get("bgm")
    if isinstance(want, str) and want.strip():
        n = want.strip()
        if not n.lower().endswith(".mp3"):
            n += ".mp3"
        n = safe_name(n)
        # 2026-10-27 产品口径：只允许客户用「模版默认曲」或「自己上传的曲」
        # （user_ 前缀）。曲库曲目不再对客户开放，传了也回退模版默认。
        if n.startswith("user_"):
            cand = os.path.join(BGM_DIR, n)
            if os.path.isfile(cand):
                return cand
    return os.path.join(BGM_DIR, default_name)

# ---------------- 一次性客户端会话的回收 ----------------
# 客户端每次打开/刷新页面都开一个新会话（Studio session），旧的照片不入下一次。
# 这里把旧会话目录删掉，顺带回收超过一天的历史会话，避免磁盘无限增长。
SESSION_NAME = "Studio session"
SESSION_TTL = 24 * 3600
# .part 只保护"最近半小时内动过的"：正在渲染的要留着，
# 崩掉之后留下的残渣不能让目录永远删不掉。
SESSION_PART_GRACE = 30 * 60
SESSION_LOG = os.path.join(DATA, "_session.log")
SESSION_PID_RE = re.compile(r"^(?:p\d{4}_\d{6}_[0-9a-f]+|[0-9a-f]{32})$")


def session_log(msg):
    """会话清理的流水账。服务跑在后台，print 出去没人看得到，落文件才查得动。"""
    try:
        with open(SESSION_LOG, "a", encoding="utf-8") as f:
            f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))
    except OSError:
        pass


def _session_busy(d):
    """目录里是否还有正在写入的中间产物（.part*）。递归找，但只看最近动过的。"""
    now = time.time()
    try:
        for root, _dirs, files in os.walk(d):
            for f in files:
                if ".part" not in f:
                    continue
                try:
                    if now - os.path.getmtime(os.path.join(root, f)) < SESSION_PART_GRACE:
                        return True         # 刚在写 = 渲染进行中，别动
                except OSError:
                    return True
    except OSError:
        return True
    return False


def _session_dir_deletable(name, item):
    """只删自动命名的客户端会话，且目录里没有正在生成的 .part 文件。"""
    if (item.get("name") or "") != SESSION_NAME:
        return False
    d = os.path.join(DATA, name)
    if not os.path.isdir(d):
        return False
    return not _session_busy(d)


def rm_tree_retry(d, tries=4):
    """删目录并确认结果。Windows 上刚写完的文件常被杀软/索引器/ffmpeg 短暂占用
    （WinError 32/5），rmtree 当场抛错但文件其实已经删掉了 —— 所以不能只看异常，
    要重试几次、最后按"目录是否真的没了"判定。返回 (ok, last_error)。

    ⚠ 必须走 safedel：本机可能存在「安全删除钩子」，删除超阈值时抛 SystemExit
    （BaseException，except OSError/Exception 都抓不住）→ 清理线程静默死亡。"""
    from core import safedel
    if safedel.rmtree(d, tries=tries):
        return True, None
    return (not os.path.exists(d)), "still present after %d tries" % tries


def discard_session(pid):
    """删除指定的一次性会话目录，并回收过期会话。
    返回 (removed, kept)：removed = 确认删掉的目录名；kept = [(目录名, 原因)]。"""
    removed, kept = [], []
    if pid and SESSION_PID_RE.match(pid):
        try:
            proj = load_project(pid)
        except Exception:
            proj = None
        if not proj:
            kept.append((pid, "no project.json"))
        elif not _session_dir_deletable(pid, proj):
            kept.append((pid, "not a disposable session, or a render is running"))
        else:
            ok, err = rm_tree_retry(project_path(pid))
            if ok:
                removed.append(pid)
            else:
                kept.append((pid, str(err)))
    # 历史会话回收（超过 24 小时未动的自动会话目录）
    now = time.time()
    try:
        names = os.listdir(DATA)
    except OSError as e:
        session_log("list %s failed: %s" % (DATA, e))
        names = []
    for name in names:
        if name in removed or not os.path.isdir(os.path.join(DATA, name)):
            continue
        pj = os.path.join(DATA, name, "project.json")
        try:
            if not os.path.isfile(pj) or now - os.path.getmtime(pj) < SESSION_TTL:
                continue
            with open(pj, "r", encoding="utf-8") as f:
                proj = json.load(f)
        except Exception:
            continue
        if _session_dir_deletable(name, proj):
            ok, err = rm_tree_retry(os.path.join(DATA, name), tries=2)
            if ok:
                removed.append(name)
            else:
                kept.append((name, str(err)))
    return removed, kept

def _img_size(path):
    """返回 (w, h)；读不出时返回 (0, 0)。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        try:
            import cv2
            import numpy as np
            im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                h, w = im.shape[:2]
                return w, h
        except Exception:
            pass
    return 0, 0


def _asset_paths(proj):
    """素材候选分组，按优先级从高到低返回：
      1) 元素素材 · 整套参考（步骤2 产物中 granularity=set 的图）——「整套逐张展示」口径的首选
      2) 元素素材 · 单个元素（granularity=item）
      3) 步骤1 上传原图 / 本地处理产物（备选）
    """
    ok = lambda p: p and os.path.isfile(p)
    els = proj.get("elements") or []
    set_els = [e.get("path") for e in els
               if e.get("granularity") == "set" and ok(e.get("path"))]
    item_els = [e.get("path") for e in els
                if e.get("granularity") != "set" and ok(e.get("path"))]
    raw = [e.get("path") for e in (proj.get("uploads") or [])]
    raw += [e.get("path") for e in (proj.get("processed") or {}).values()]
    raw = [p for p in raw if ok(p)]
    return [set_els, item_els, raw]


def _best(cands, exclude=None):
    """在候选中挑分辨率最高的一张，返回 (path, w, h)。"""
    ex = os.path.abspath(exclude) if exclude else None
    best, bw, bh = None, -1, -1
    for p in cands:
        if ex and os.path.abspath(p) == ex:
            continue
        w, h = _img_size(p)
        if w * h > max(bw, 0) * max(bh, 0) or best is None:
            best, bw, bh = p, w, h
    if best is None:
        return None, 0, 0
    return best, max(bw, 0), max(bh, 0)


def pick_first_frame(proj):
    """自动识别视频首帧：**元素素材优先**（步骤2 AI 产物，整套参考 > 单个元素），
    无元素素材时回落到上传原图 / 本地处理产物。

    返回 (path, w, h)；无可用素材时返回 (None, 0, 0)。
    """
    for group in _asset_paths(proj):
        p, w, h = _best(group)
        if p:
            return p, w, h
    return None, 0, 0

def pick_last_frame(proj, exclude_path):
    """自动识别尾帧：优先级同上，排除首帧后取分辨率最高的一张。
    返回 (path, w, h)；无可用素材时返回 (None, 0, 0)。"""
    for group in _asset_paths(proj):
        p, w, h = _best(group, exclude=exclude_path)
        if p:
            return p, w, h
    return None, 0, 0


# ---------------- HTTP ----------------

class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    # ---- 会话隔离（sid）----
    # 浏览器客户端各自带一个随机 sid；admin（X-Studio-Key / Basic）不带，拥有全量权限。
    def _read_sid(self):
        cookie = self.headers.get("Cookie") or ""
        for part in cookie.split(";"):
            part = part.strip()
            if part.startswith(SID_NAME + "="):
                v = part.split("=", 1)[1].strip().strip('"')
                if v:
                    return v
        return None

    def send_response(self, code, message=None):
        super().send_response(code, message)
        # 浏览器首次来访（有 studio_auth 但无 sid）时，借本次响应补发 sid Cookie，
        # 此后该浏览器所有请求都带 sid，数据按 sid 隔离。
        nsid = getattr(self, "_new_sid", None)
        if nsid:
            self.send_header("Set-Cookie",
                             "%s=%s; Path=/; Max-Age=%d; SameSite=Lax" % (SID_NAME, nsid, SID_MAX_AGE))

    # ---- helpers ----
    # ---- access gate ----
    LOGIN_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign in - Invitation Promo Studio</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600&family=Inter:wght@400;500&display=swap">
<style>
  :root{--paper:#F5F1EA;--card:#FFFDF9;--ink:#221F1B;--ink2:#6B655C;--line:#DED7CA;--sage:#7D8471}
  *{box-sizing:border-box}
  body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
       background:var(--paper);color:var(--ink);font:15px/1.6 Inter,system-ui,sans-serif;padding:24px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:4px;
        padding:44px 40px;width:100%;max-width:380px;box-shadow:0 1px 2px rgba(34,31,27,.04)}
  .mark{font:600 13px/1 Inter,sans-serif;letter-spacing:.22em;text-transform:uppercase;color:var(--sage)}
  h1{font-family:'Cormorant Garamond',Georgia,serif;font-weight:500;font-size:30px;margin:14px 0 6px}
  p{color:var(--ink2);margin:0 0 22px;font-size:13.5px}
  label{display:block;font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink2);margin-bottom:8px}
  input{width:100%;padding:11px 13px;border:1px solid var(--line);border-radius:3px;background:#fff;
        font:15px Inter,sans-serif;color:var(--ink);outline:none}
  input:focus{border-color:var(--sage)}
  button{margin-top:18px;width:100%;padding:12px;border:1px solid var(--ink);background:var(--ink);
         color:#F7F4EE;border-radius:3px;font:500 14px Inter,sans-serif;cursor:pointer}
  button:hover{background:#3A352E}
  .err{margin-top:14px;font-size:12.5px;color:#A44A3F;display:none}
  .foot{margin-top:26px;font-size:11.5px;color:var(--ink2);text-align:center;letter-spacing:.04em}
</style></head>
<body>
  <div class="card">
    <div class="mark">Invitation Promo Studio</div>
    <h1>Sign in</h1>
    <p>This studio is private. Enter the access password to continue.</p>
    <form id="f" autocomplete="off">
      <label for="p">Access password</label>
      <input id="p" type="password" autofocus>
      <button type="submit">Enter studio</button>
      <div class="err" id="e">Wrong password. Try again.</div>
    </form>
    <div class="foot">Protected workspace &middot; invitation promo pipeline</div>
  </div>
<script>
  document.getElementById("f").addEventListener("submit", function (ev) {
    ev.preventDefault();
    var v = document.getElementById("p").value.trim();
    if (!v) return;
    location.href = location.pathname + "?k=" + encodeURIComponent(v);
  });
  if (/[?&]k=/.test(location.search)) document.getElementById("e").style.display = "block";
</script>
</body></html>"""

    def guard(self):
        """Access gate. Inactive unless a password is configured (cloud deploys)."""
        # 初始化会话隔离状态
        self._sid = self._read_sid()
        self._new_sid = None
        self._auth_method = None
        if not ACCESS_PASSWORD:
            return True
        want = auth_token(ACCESS_PASSWORD)
        u = urlparse(self.path)
        q = parse_qs(u.query)

        # 1) URL token -> set cookie, then redirect to the clean URL
        if q.get("k"):
            if hmac.compare_digest(q["k"][0], ACCESS_PASSWORD):
                rest = "&".join(seg for seg in u.query.split("&") if not seg.startswith("k="))
                loc = u.path + (("?" + rest) if rest else "")
                sid = self._sid or uuid.uuid4().hex
                self.send_response(302)
                self.send_header("Location", loc or "/")
                self.send_header("Set-Cookie",
                                 "%s=%s; Path=/; Max-Age=%d; SameSite=Lax" % (COOKIE_NAME, want, COOKIE_DAYS * 86400))
                self.send_header("Set-Cookie",
                                 "%s=%s; Path=/; Max-Age=%d; SameSite=Lax" % (SID_NAME, sid, SID_MAX_AGE))
                self.send_header("Content-Length", "0")
                self.end_headers()
                self._auth_method = "browser"
                self._sid = sid
            else:
                self.send_login_page()
            return False

        # 2) Cookie
        cookie = self.headers.get("Cookie") or ""
        for part in cookie.split(";"):
            part = part.strip()
            if part.startswith(COOKIE_NAME + "="):
                if hmac.compare_digest(part.split("=", 1)[1], want):
                    # 浏览器客户端：首次无 sid 则借本次响应补发
                    if self._sid is None:
                        self._sid = uuid.uuid4().hex
                        self._new_sid = self._sid
                    self._auth_method = "browser"
                    return True

        # 3) X-Studio-Key header (scripts / API clients) —— admin，拥有全量权限
        xk = self.headers.get("X-Studio-Key")
        if xk and hmac.compare_digest(xk, ACCESS_PASSWORD):
            self._auth_method = "admin"
            return True

        # 4) Basic auth (local / direct connections) —— admin
        hdr = self.headers.get("Authorization") or ""
        if hdr.startswith("Basic "):
            try:
                raw = base64.b64decode(hdr[6:]).decode("utf-8", "replace")
                pwd = raw.split(":", 1)[1] if ":" in raw else ""
            except Exception:
                pwd = ""
            if pwd and hmac.compare_digest(pwd, ACCESS_PASSWORD):
                self._auth_method = "admin"
                return True

        self.send_login_page()
        return False

    def _is_admin(self):
        return getattr(self, "_auth_method", None) == "admin"

    def _owner_ok(self, pid):
        """浏览器客户端只能访问自己 sid 下的项目；admin 放行。"""
        if self._is_admin():
            return True
        try:
            proj = load_project(pid)
        except Exception:
            return False
        return (proj.get("owner_sid") or None) == (self._sid or None)

    def _owns_path(self, path):
        """浏览器客户端只能操作自己 sid 项目目录下的文件；admin 放行。"""
        if self._is_admin():
            return True
        ap = os.path.abspath(path).replace("\\", "/")
        dp = os.path.abspath(DATA).replace("\\", "/")
        if not ap.startswith(dp + "/"):
            return False
        pid = ap[len(dp) + 1:].split("/")[0]
        return self._owner_ok(pid)

    def _ai_gate(self, kind):
        """AI 调用额度闸门。admin 不受限；浏览器会话超限返回错误 dict，否则返回 None。"""
        if self._is_admin():
            return None
        m = _metering.get_meter()
        if not m:
            return None
        ok, reason, used, cap = m.count(self._sid, kind)
        if not ok:
            if reason == "session":
                return {"error": "This session's AI quota is used up "
                                  "(%d/%d). Please contact the studio owner to top up." % (used, cap),
                        "code": 402}
            return {"error": "The studio's daily AI budget is used up "
                              "(%d/%d). Please try again tomorrow." % (used, cap),
                    "code": 402}
        return None

    def send_login_page(self):
        body = self.LOGIN_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def read_body_bytes(self, max_mb=200):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > max_mb * 1024 * 1024:
            raise ValueError("The request body is empty or exceeds %dMB" % max_mb)
        return self.rfile.read(n)

    def serve_file(self, path, content_type, download_name=None):
        try:
            f = open(path, "rb")
        except OSError:
            self.send_json({"error": "file not found"}, 404)
            return
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)\s*$", rng.strip())
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    # 普通范围 bytes=S-E / bytes=S-
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    # 后缀范围 bytes=-N =「最后 N 字节」。
                    # 浏览器（尤其 Safari）seek 到片尾 / 读 moov 时会发这个，
                    # 以前被当成 0..N 答错 → 线上网关校验不过直接 502，视频就播不了。
                    start = max(0, size - int(m.group(2)))
                    end = size - 1
                end = min(end, size - 1)
                if start >= size or start > end:
                    f.close()
                    self.send_response(416)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", "bytes */%d" % size)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                code = 206
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        # 前端静态资源禁用缓存：改完 css/js 刷新即生效，避免旧版残留
        if "/static/" in path.replace("\\", "/") or path.replace("\\", "/").endswith("index.html"):
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        if code == 206:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        if download_name:
            self.send_header("Content-Disposition",
                             "attachment; filename*=UTF-8''%s" % download_name)
        self.end_headers()
        f.seek(start)
        remain = end - start + 1
        while remain > 0:
            chunk = f.read(min(65536, remain))
            if not chunk: break
            self.wfile.write(chunk)
            remain -= len(chunk)
        f.close()

    # ---- GET ----
    def do_GET(self):
        if not self.guard():
            return
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = unquote(u.path)
        try:
            if p == "/" or p == "/index.html":
                return self.serve_file(os.path.join(STATIC, "index.html"), "text/html; charset=utf-8")
            if p.startswith("/static/"):
                fp = os.path.join(STATIC, p[len("/static/"):])
                if not os.path.abspath(fp).startswith(os.path.abspath(STATIC)):
                    return self.send_json({"error": "forbidden"}, 403)
                # 模版封面 mp4 走这里：必须是 video/mp4。以前一律 octet-stream，
                # Chrome 勉强能播、Safari/内嵌浏览器会直接拒绝加载。
                ct = {"css": "text/css", "js": "application/javascript", "json": "application/json",
                      "mp4": "video/mp4", "webm": "video/webm", "m4v": "video/mp4",
                      "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                      "webp": "image/webp", "gif": "image/gif", "svg": "image/svg+xml",
                      "ico": "image/x-icon", "woff": "font/woff", "woff2": "font/woff2",
                      "txt": "text/plain", "md": "text/plain"}.get(
                    p.rsplit(".", 1)[-1].lower(), "application/octet-stream")
                if ct.startswith("text/") or ct.endswith(("javascript", "json", "svg+xml")):
                    ct += "; charset=utf-8"
                return self.serve_file(fp, ct)
            if p.startswith("/media/"):
                rel = p[len("/media/"):]
                fp = os.path.join(DATA, rel)
                if not os.path.abspath(fp).startswith(os.path.abspath(DATA)):
                    return self.send_json({"error": "forbidden"}, 403)
                # 浏览器客户端只能读取自己 sid 下的项目文件；admin 放行
                if not self._is_admin():
                    seg = rel.split("/")[0]
                    if seg and not self._owner_ok(seg):
                        return self.send_json({"error": "forbidden"}, 403)
                ext = p.rsplit(".", 1)[-1].lower()
                ct = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                      "mp4": "video/mp4", "m4v": "video/mp4", "webm": "video/webm",
                      "mov": "video/quicktime", "gif": "image/gif", "svg": "image/svg+xml",
                      "webp": "image/webp", "json": "application/json"}.get(ext, "application/octet-stream")
                # ?dl=<name> → 触发浏览器下载（客户端「下载导出」按钮用）
                dl = q.get("dl", [None])[0]
                return self.serve_file(fp, ct, download_name=dl if dl else None)
            if p.startswith("/bgm/"):
                fp = os.path.join(BGM_DIR, safe_name(p[len("/bgm/"):]))
                return self.serve_file(fp, "audio/mpeg")
            if p == "/api/projects":
                items = []
                if os.path.isdir(DATA):
                    for d in sorted(os.listdir(DATA)):
                        pj = os.path.join(DATA, d, "project.json")
                        if not os.path.isfile(pj):
                            continue
                        try:
                            proj = load_project(d)
                        except Exception:
                            continue
                        # 浏览器客户端只看自己 sid 下的项目；admin 看全量（管理用）
                        if not self._is_admin() and (proj.get("owner_sid") or None) != (self._sid or None):
                            continue
                        items.append(proj)
                items.sort(key=lambda x: x.get("created", 0), reverse=True)
                return self.send_json({"projects": items})
            if p == "/api/project":
                pid = q["id"][0]
                if not self._owner_ok(pid):
                    return self.send_json({"error": "forbidden"}, 403)
                return self.send_json({"project": load_project(pid)})
            if p == "/api/job":
                with JOBS_LOCK:
                    j = JOBS.get(q["id"][0])
                return self.send_json({"job": j})
            if p == "/api/bgm":
                return self.send_json({"tracks": compose.list_bgm(BGM_DIR)})
            if p == "/api/templates":
                try:
                    with open(os.path.join(STATIC, "templates.json"), encoding="utf-8") as _tf:
                        return self.send_json(json.load(_tf))
                except FileNotFoundError:
                    return self.send_json({"categories": [], "templates": []})
                except Exception as _e:
                    return self.send_json({"error": str(_e)}, 500)
            if p == "/api/config":
                c = dict(CONFIG)
                c["cloud_token"] = bool(c.get("cloud_token"))
                v = dict(c.get("video_api") or {})
                v["api_key_set"] = bool(v.get("api_key"))
                v["effective_endpoint"] = videoapi.api_base(v)
                v["effective_model"] = videoapi.model_id(v)
                v["meta"] = videoapi.meta(v["effective_model"])
                c["video_api"] = v
                c["video_models"] = videoapi.MODELS
                c["video_ratios"] = videoapi.RATIOS
                # 分镜图（Seedream）
                sb = dict(c.get("storyboard_api") or {})
                sb["api_key_set"] = bool(sb.get("api_key"))
                sb["effective_endpoint"] = imgapi.api_base(sb)
                sb["effective_model"] = imgapi.model_id(sb)
                sb["meta"] = imgapi.meta(sb["effective_model"])
                c["storyboard_api"] = sb
                c["image_models"] = imgapi.IMAGE_MODELS
                # 步骤2 · AI 处理元素素材（模式 / 输出粒度）
                c["element_modes"] = elemapi.MODES
                c["element_granularities"] = elemapi.GRANULARITIES
                c["element_default_mode"] = elemapi.DEFAULT_MODE
                c["element_default_granularity"] = elemapi.DEFAULT_GRAN
                c["local_ops"] = imgproc.LOCAL_OPS
                # 基础运镜镜头库（CM-01~CM-10 单镜头 + 组合布局）
                c["camera_moves"] = [dict(id=k, **v) for k, v in cammove.MOVES.items()]
                c["camera_layouts"] = [
                    {"id": k, "name": {"reference": "Reference camera moves (3 shots, following the reference film's pacing)",
                                       "compact": "Compact 3 shots (push in + pan + close-up)",
                                       "full": "Full 4 shots（+ Detail）"}[k],
                     "secs": round(sum(v["secs"]), 1)} for k, v in cammove.LAYOUTS.items()]
                c["camera_seq_max"] = cammove.MAX_SEQ_SHOTS   # 多镜自选上限
                return self.send_json({"config": c})
            if p == "/api/prompts":
                return self.send_json({"prompts": prompts.DEFAULTS})
            if p == "/api/videoapi/models":
                return self.send_json({"models": videoapi.MODELS, "ratios": videoapi.RATIOS,
                                       "image_models": imgapi.IMAGE_MODELS})
            return self.send_json({"error": "not found"}, 404)
        except Exception as e:
            traceback.print_exc()
            return self.send_json({"error": str(e)}, 500)

    # ---- POST ----
    def do_POST(self):
        if not self.guard():
            return
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = unquote(u.path)
        try:
            if p == "/api/project":
                body = self.read_body_json()
                pid = uuid.uuid4().hex
                os.makedirs(project_path(pid), exist_ok=True)
                proj = {"id": pid, "name": body.get("name") or ("Project " + pid),
                        "created": time.time(), "uploads": [], "processed": {},
                        "elements": [], "cameramove": [], "storyboards": [],
                        "handheld_videos": [], "exports": [], "smart_photos": {},
                        "owner_sid": self._sid or None}
                save_project(proj)
                return self.send_json({"project": proj})

            if p == "/api/upload":
                pid = q["project"][0]
                name = safe_name(q.get("name", ["photo.jpg"])[0])
                kind = q.get("kind", ["upload"])[0]
                sub = {"upload": "uploads", "storyboard": "storyboards"}.get(kind, "uploads")
                d = os.path.join(project_path(pid), sub)
                os.makedirs(d, exist_ok=True)
                _, ext = os.path.splitext(name)
                ext = (ext.lower()[:10]) or ".jpg"
                # 落盘文件名随机化（UUID），原始文件名只存元数据，杜绝按名枚举
                fname = uuid.uuid4().hex + ext
                fp = os.path.join(d, fname)
                data = self.read_body_bytes()
                with open(fp, "wb") as f:
                    f.write(data)
                proj = load_project(pid)
                entry = {"path": fp, "file": fname, "fname": name}
                if kind == "storyboard":
                    proj["storyboards"].append(entry)
                else:
                    proj["uploads"].append(entry)
                save_project(proj)
                return self.send_json({"entry": entry, "project": proj})

            if p == "/api/smart/assess":
                g = self._ai_gate("vision")
                if g:
                    return self.send_json({"error": g["error"]}, g["code"])
                """客户端精简流程 · 照片质检。

                本地技术体检（分辨率/清晰度/曝光）+ 视觉大模型判定（文字是否清楚、
                产品是否完整、画质是否够高清）→ 逐张给 pass/warn/fail 与英文提示。
                不合格的照片由前端提示客户重新上传；结论写回项目的 smart_photos。
                """
                body = self.read_body_json()
                try:
                    proj = load_project(body["project"])
                except OSError:
                    return self.send_json({"error": "Project not found: %s" % body.get("project")}, 404)
                files = body.get("files") or [e["path"] for e in (proj.get("uploads") or [])]
                files = [f for f in files if f and os.path.isfile(f)]
                if not files:
                    return self.send_json({"error": "No photo to check - upload at least one photo first"}, 400)
                cfg = dict(CONFIG.get("storyboard_api") or {})
                use_ai = bool((cfg.get("api_key") or "").strip())
                depth = body.get("depth") or "fast"
                jid = new_job("smart_assess")

                def work():
                    res = smart.assess_batch(
                        cfg, files, depth=depth, use_ai=use_ai,
                        progress=lambda d, t, m: job_progress(jid, d, t, m))
                    p2 = load_project(proj["id"])
                    store = p2.setdefault("smart_photos", {})
                    for r in res:
                        loc = r.get("local") or {}
                        store[r["path"]] = {
                            "verdict": r.get("verdict"), "content_kind": r.get("content_kind"),
                            "issues": r.get("issues") or [], "tip": r.get("tip") or "",
                            "subject": r.get("subject") or "",
                            # 工艺识别（2026-09-30）：细节微距镜的取景优先落在工艺区域上
                            "craft": r.get("craft") or [], "craft_box": r.get("craft_box") or [],
                            "craft_boxes": r.get("craft_boxes") or [],
                            "long_edge": loc.get("long_edge"), "sharpness": loc.get("sharpness"),
                            "brightness": loc.get("brightness"), "aspect": loc.get("aspect"),
                            "ai": bool(r.get("ai")), "checked": time.time(),
                        }
                    save_project(p2)
                    return {"results": res, "ai": use_ai}

                run_job(jid, work)
                return self.send_json({"job": jid, "ai": use_ai})

            if p == "/api/craft-detect":
                """工艺细节识别（多尺度扫描 + 先检测后分类）。

                独立按需调用：传入单张请柬卡片产品图路径，返回 Top-5 工艺候选框 JSON。
                不在上传质检(assess)里自动跑，避免拖慢那条快路径。
                输出结构见 core/craftdetect.craft_detect：bbox / label / confidence /
                evidence / micro_score / is_text / is_logo / is_pattern + 纪律约束。
                """
                body = self.read_body_json()
                path = body.get("path")
                if not path or not os.path.isfile(path):
                    return self.send_json({"error": "Missing or invalid image path"}, 400)
                proj = None
                pid = body.get("project")
                if pid:
                    try:
                        proj = load_project(pid)
                    except OSError:
                        proj = None
                cfg = dict(CONFIG.get("storyboard_api") or {})
                depth = body.get("depth") or "fast"
                use_grid = bool(body.get("use_grid", True))
                jid = new_job("craft_detect")

                def work():
                    res = craftdetect.craft_detect(path, cfg, depth=depth, use_grid=use_grid)
                    if proj is not None:
                        store = proj.setdefault("craft_detail", {})
                        store[path] = res
                        save_project(proj)
                    return {"path": path, "result": res}

                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/card-detect":
                """卡片分区检测（学习案例：运镜会从整套切到单独卡片）。

                独立按需调用：传入单张请柬产品图路径，返回「独立的单卡/组件」bbox 列表 JSON，
                供单图 set/group 模式在渲染时让 cm15/cm16 框住某张独立卡。
                不在上传质检(assess)里自动跑，避免拖慢那条快路径。
                输出结构见 core.craftdetect.detect_cards：cards[{bbox,label,confidence,reason}]。
                """
                body = self.read_body_json()
                path = body.get("path")
                if not path or not os.path.isfile(path):
                    return self.send_json({"error": "Missing or invalid image path"}, 400)
                proj = None
                pid = body.get("project")
                if pid:
                    try:
                        proj = load_project(pid)
                    except OSError:
                        proj = None
                cfg = dict(CONFIG.get("storyboard_api") or {})
                depth = body.get("depth") or "fast"
                jid = new_job("card_detect")

                def work():
                    res = craftdetect.detect_cards(path, cfg, depth=depth)
                    if proj is not None:
                        store = proj.setdefault("card_layout", {})
                        store[path] = res
                        save_project(proj)
                    return {"path": path, "result": res}

                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/smart/recipe":
                """客户端精简流程 · 运镜配方匹配。

                按质检结论（内容类型 set/single/detail/group）+ 本地构图分析自动选镜头、
                给出每镜时长方案（客户只看到时长档位，看不到镜头编号）。
                运镜细节只写在 note 里供服务端排查，界面不显示。
                """
                body = self.read_body_json()
                try:
                    proj = load_project(body["project"])
                except OSError:
                    return self.send_json({"error": "Project not found: %s" % body.get("project")}, 404)
                store = proj.get("smart_photos") or {}
                files = body.get("files") or [e["path"] for e in (proj.get("uploads") or [])]
                items = []
                for f in files:
                    if not f or not os.path.isfile(f):
                        continue
                    info = store.get(f) or {}
                    if info.get("verdict") == "fail":
                        continue        # 未通过质检的照片不参与生成
                    items.append({"path": f, "content_kind": info.get("content_kind"),
                                  "verdict": info.get("verdict") or "unknown",
                                  "subject": info.get("subject") or "",
                                  "craft": info.get("craft") or [],
                                  "craft_box": info.get("craft_box") or [],
                                  "craft_boxes": info.get("craft_boxes") or [],
                                  "card_boxes": smart.best_card_boxes(proj, f)})
                if not items:
                    return self.send_json(
                        {"error": "None of the photos passed the quality check - upload a sharper, "
                                  "well-lit photo first"}, 400)
                r = smart.recipe(items)
                recipe = {k: v for k, v in r.items() if k != "note"}
                # 界面 Step 2 的比例选项由后端下发（与 cammove.RATIOS 永远一致）。
                recipe["ratios"] = [{"id": k, "w": v[0], "h": v[1]}
                                    for k, v in cammove.RATIOS.items()]
                print("[smart] recipe mode=%s shots=%d moves=%s" %
                      (r["mode"], r["shots"], "-".join(r["moves"])))
                return self.send_json({"recipe": recipe, "note": r["note"]})

            if p == "/api/smart/autofix":
                g = self._ai_gate("image")
                if g:
                    return self.send_json({"error": g["error"]}, g["code"])
                """质检不过 → 自动修复 → 复检。

                按本地体检结论映射算子（超分/提亮/锐化/去噪/白平衡），
                真糊片在配置了 cloud_token 时走 AI enhance；修完自动复检，
                修好的图落入项目 uploads/ 并补跑一次完整质检（供运镜配方用）。
                修不好就如实返回 improved=false，由前端提示重传。
                """
                body = self.read_body_json()
                try:
                    proj = load_project(body["project"])
                except OSError:
                    return self.send_json({"error": "Project not found: %s" % body.get("project")}, 404)
                files = body.get("files") or ([body["file"]] if body.get("file") else [])
                files = [f for f in files if f and os.path.isfile(f)]
                if not files:
                    return self.send_json({"error": "No photo to repair"}, 400)
                cfg = dict(CONFIG.get("storyboard_api") or {})
                use_ai = bool((cfg.get("api_key") or "").strip())
                token = str(CONFIG.get("cloud_token") or "")
                updir = os.path.join(project_path(proj["id"]), "uploads")
                os.makedirs(updir, exist_ok=True)
                jid = new_job("smart_autofix")

                def work():
                    results = []
                    total = len(files)
                    for k, f in enumerate(files):
                        job_progress(jid, k * 100 // total, 100,
                                     "Repairing photo %d of %d" % (k + 1, total))
                        stem = safe_name(os.path.splitext(os.path.basename(f))[0]) or "photo"
                        out_path = os.path.join(
                            updir, "%s_fixed_%s.jpg" % (stem, time.strftime("%m%d%H%M%S")))
                        try:
                            r = autofix.fix(
                                f, out_path, ai_available=use_ai, token=token,
                                progress=lambda pct, _t, msg: job_progress(
                                    jid, min(99, k * 100 // total + pct // max(1, _t)), 100, msg))
                        except BaseException as e:
                            # 单张失败不能拖垮整批（BaseException：外部钩子可能抛 SystemExit）
                            traceback.print_exc()
                            results.append({"source": f, "file": None, "out": None,
                                            "verdict_before": "fail", "verdict_after": "fail",
                                            "flipped": False, "improved": False, "ops": [],
                                            "ai_used": None,
                                            "notes": ["Automatic repair failed on this file (%s)."
                                                      % (str(e) or e.__class__.__name__)]})
                            continue
                        entry_out = {"source": f, "file": None, "out": None,
                                     "verdict_before": r["verdict_before"],
                                     "verdict_after": r["verdict_after"],
                                     "flipped": r["flipped"], "improved": r["improved"],
                                     "ops": r["ops"], "ai_used": r["ai_used"],
                                     "notes": r["notes"],
                                     "before": {kk: r["before"].get(kk) for kk in
                                                ("long_edge", "sharpness", "brightness", "clipped")},
                                     "after": {kk: r["after"].get(kk) for kk in
                                               ("long_edge", "sharpness", "brightness", "clipped")}}
                        if r["out"] and (r["improved"] or r["flipped"]):
                            # 修好的图登记进 uploads（下游与 /media 都走这里），并补跑完整质检
                            p2 = load_project(proj["id"])
                            e = {"path": r["out"], "file": os.path.basename(r["out"]),
                                 "fixed_from": f, "fixed": time.time()}
                            p2.setdefault("uploads", []).append(e)
                            save_project(p2)
                            re = smart.assess(cfg, r["out"], use_ai=use_ai)
                            st = p2.setdefault("smart_photos", {})
                            st[r["out"]] = {
                                "verdict": re.get("verdict"), "content_kind": re.get("content_kind"),
                                "issues": re.get("issues") or [], "tip": re.get("tip") or "",
                                "subject": re.get("subject") or "",
                                "long_edge": (re.get("local") or {}).get("long_edge"),
                                "sharpness": (re.get("local") or {}).get("sharpness"),
                                "brightness": (re.get("local") or {}).get("brightness"),
                                "aspect": (re.get("local") or {}).get("aspect"),
                                "ai": bool(re.get("ai")), "checked": time.time(),
                            }
                            save_project(p2)
                            entry_out.update(file=e["file"], out=r["out"],
                                             verdict_after=re.get("verdict"),
                                             issues=re.get("issues") or [],
                                             content_kind=re.get("content_kind"),
                                             subject=re.get("subject"))
                        results.append(entry_out)
                    return {"results": results}

                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/session/discard":
                """客户端每次刷新都会开新会话，这里把上一次的一次性会话目录清掉
                （顺带回收超过 24 小时的历史会话），避免磁盘上越堆越多。"""
                body = self.read_body_json()
                t0 = time.time()
                removed, kept = discard_session(str(body.get("id") or ""))
                session_log("discard pid=%s removed=%s kept=%s took=%.1fs" %
                            (body.get("id") or "-", removed or "-",
                             ["%s(%s)" % (k[0], k[1]) for k in kept] or "-",
                             time.time() - t0))
                if removed:
                    print("[session] discarded %d dir(s): %s" %
                          (len(removed), ", ".join(removed[:6])))
                return self.send_json({"discarded": removed,
                                       "kept": [{"id": k[0], "why": k[1]} for k in kept]})

            if p == "/api/elements/generate":
                """步骤2 · AI 处理元素素材：上传实拍图 → 拆解元素 → 按模式输出修复后的参考素材。

                复用 storyboard_api 的 ARK 凭据（同一账号同一把 Key）。
                产物写入项目 elements/，成为下游默认来源。
                """
                body = self.read_body_json()
                cfg = dict(CONFIG.get("storyboard_api") or {})
                if not elemapi.is_ready(cfg):
                    return self.send_json(
                        {"error": "ARK_API_KEY is not configured. Go to Settings → Image generation API and enter your Volcengine Ark long-lived key."}, 400)
                try:
                    proj = load_project(body["project"])
                except OSError:
                    return self.send_json({"error": "Project not found: %s" % body.get("project")}, 404)
                refs = body.get("images") or ([body["image"]] if body.get("image") else [])
                if not refs:
                    # 默认使用步骤1 全部上传图（最多 3 张，兼顾参考图配额）
                    refs = [e["path"] for e in (proj.get("uploads") or [])][:3]
                refs = [p for p in refs if p and os.path.isfile(p)]
                if not refs:
                    return self.send_json(
                        {"error": "Upload product photos in step 1 to use as references first"}, 400)
                mode, gran = elemapi.normalize(body.get("mode"), body.get("granularity"))
                prompt = (body.get("prompt") or "").strip() or None
                if prompt is None:
                    prompt = elemapi.build_prompt(mode, gran, body.get("extra"))
                size = body.get("size") or cfg.get("size") or "4K"
                max_images = body.get("max_images") or elemapi.suggest_max_images(gran, len(refs))
                out_fmt = body.get("output_format") or cfg.get("output_format") or "jpeg"
                watermark = bool(body.get("watermark", cfg.get("watermark", False)))
                optimize = body.get("optimize_prompt", cfg.get("optimize_prompt"))
                out_dir = os.path.join(project_path(proj["id"]), "elements")
                jid = new_job("element_gen")
                tag_prefix = "element_%s_%s" % (mode, time.strftime("%m%d%H%M%S"))

                def work():
                    res = elemapi.generate(
                        cfg, out_dir, refs, mode=mode, granularity=gran, prompt=prompt,
                        size=size, max_images=max_images, output_format=out_fmt,
                        watermark=watermark, optimize_prompt=optimize,
                        tag=tag_prefix,
                        progress=lambda pct, msg: job_progress(jid, pct, 100, msg))
                    p2 = load_project(proj["id"])
                    entries = []
                    for fp in res["files"]:
                        e = {"path": fp, "file": os.path.basename(fp),
                             "kind": "ai_seedream", "mode": mode,
                             "granularity": gran, "model": res["model"],
                             "size": res["size"], "created": time.time()}
                        p2.setdefault("elements", []).append(e)
                        entries.append(e)
                    p2.setdefault("ai_generations", []).append(
                        {"kind": "element", "mode": mode, "granularity": gran,
                         "model": res["model"], "size": res["size"], "prompt": prompt,
                         "refs": refs, "count": res["count"], "files": res["files"],
                         "urls": res["urls"], "elapsed": res["elapsed"],
                         "created": time.time()})
                    save_project(p2)
                    return {"entries": entries, "files": res["files"], "count": res["count"],
                            "model": res["model"], "size": res["size"],
                            "elapsed": res["elapsed"], "note": res["note"],
                            "mode": mode, "mode_name": res["mode_name"],
                            "granularity": gran, "prompt": prompt,
                            "used_refs": [os.path.basename(p) for p in refs],
                            "thumbs": ["/media/%s/elements/%s"
                                       % (proj["id"], os.path.basename(f))
                                       for f in res["files"]]}

                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/process":
                """降级通道：本地算子（零 AI 算力）补救素材 —— 仅在 AI 不可用或素材本身就够用时使用。
                不做任何质检判定；AI 处理请走 /api/elements/generate。"""
                body = self.read_body_json()
                proj = load_project(body["project"])
                src = body["file"]
                ops = [o for o in (body.get("ops") or []) if o in imgproc.LOCAL_OPS]
                if not ops:
                    return self.send_json({"error": "Select at least one local operator"}, 400)
                out_dir = os.path.join(project_path(proj["id"]), "processed")
                os.makedirs(out_dir, exist_ok=True)
                r = imgproc.process(src, ops, out_dir,
                                    crop_percent=body.get("crop_percent"))
                entry = {"path": r["out"], "file": os.path.basename(r["out"])}
                proj["processed"][src] = entry
                save_project(proj)
                return self.send_json({"result": r, "entry": entry, "project": proj})

            if p == "/api/cameramove/preview":
                body = self.read_body_json()
                jid = new_job("cm_preview")
                src = body["src"]
                def work():
                    out = os.path.join(project_path(body["project"]),
                                       "cm_preview_%s.jpg" % jid)
                    el, info = cammove.render_preview_sheet(
                        src, out, body.get("layout", "compact"),
                        body.get("anchors"))
                    return {"sheet": "/media/%s/%s" % (body["project"], os.path.basename(out)),
                            "info": info, "elapsed": round(el, 1)}
                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/cameramove/render":
                g = self._ai_gate("caption")
                if g:
                    return self.send_json({"error": g["error"]}, g["code"])
                body = self.read_body_json()
                jid = new_job("cm_render")
                proj = load_project(body["project"])
                # 2026-10-10：成片画幅由界面选（9:16 / 1:1）；非法值回落默认，绝不因脏入参 500。
                ratio = str(body.get("ratio") or cammove.DEFAULT_RATIO)
                if ratio not in cammove.RATIOS:
                    ratio = cammove.DEFAULT_RATIO
                # ---- 多图拼接模式（2026-09-28）：sources 列表 ≥2 张，每张一段运镜 ----
                sources = body.get("sources")
                if sources:
                    sources = [s for s in sources if s and os.path.isfile(s)]
                    if len(sources) < 2:
                        return self.send_json({"error": "A multi-image sequence needs at least 2 valid assets — upload more photos in step 1 first"}, 400)
                    mids = body.get("moves") or []
                    if isinstance(mids, str):
                        mids = cammove.normalize_moves(mids)
                    mids = [str(m).strip().lower() for m in mids if str(m).strip()]
                    if not mids:
                        # 未指定 → 按固定五镜轮转分配
                        mids = [cammove.STANDARD_SEQ[i % len(cammove.STANDARD_SEQ)]
                                for i in range(len(sources))]
                    elif len(mids) == 1:
                        mids = mids * len(sources)
                    if len(mids) != len(sources):
                        return self.send_json({"error": "Shot count (%d) does not match asset count (%d)" % (len(mids), len(sources))}, 400)
                    bad = [m for m in mids if m not in cammove.MOVES]
                    if bad:
                        return self.send_json({"error": "Unknown shot: %s (available: %s)" % (
                            "、".join(bad), "、".join(sorted(cammove.MOVES)))}, 400)
                    req_secs = body.get("secs")
                    plan_secs = sum(float(req_secs or cammove.MOVES[m]["secs"]) for m in mids) \
                        + cammove.FREEZE_SECS
                    slug = "-".join(mids)
                    out_dir = os.path.join(project_path(proj["id"]), "cameramove")
                    os.makedirs(out_dir, exist_ok=True)
                    stamp = time.strftime("%m%d%H%M%S")
                    out = os.path.join(out_dir, "camera_moves_multi%d_%s_%ss_%s.mp4" % (
                        len(sources), slug, round(plan_secs, 1), stamp))

                    # 工艺区域（2026-09-30）：质检时识别的工艺框，微距镜取景优先落在这里
                    sstore = proj.get("smart_photos") or {}
                    # [2026-10-12 学习案例] cm15/cm16 负责"框住单卡"（切到单独卡片），其余镜头
                    #   仍用工艺框（foil/emboss/lace）取景。按镜头类型逐段选锚点：cm15/16 取
                    #   单卡框（多段时按序轮转不同卡），其它镜头取工艺框。
                    crafts = []
                    for i, (s, m) in enumerate(zip(sources, mids)):
                        if m in ("cm15", "cm16", "cm17"):
                            cb = smart.best_card_boxes(proj, s)
                            crafts.append(cb[i % len(cb)] if cb else None)
                        else:
                            cb = smart.best_craft_boxes(proj, s)
                            crafts.append(cb if cb else None)
                    if any(crafts):
                        print("[smart] craft-framed shots: %s" % [
                            ("%s%s" % (os.path.basename(s), "+craft" if c else ""))
                            for s, c in zip(sources, crafts)])

                    def work():
                        el, info, meta = cammove.render_cameramove_multi(
                            sources, mids, out, secs=req_secs, crafts=crafts, ratio=ratio,
                            progress=lambda d, t: job_progress(jid, d, t, "Rendering"))
                        p2 = load_project(proj["id"])
                        p2.setdefault("cameramove", []).append(
                            {"file": out, "name": "Camera moves %s" % meta["label"],
                             "kind": "cameramove", "dur": meta["secs"]})
                        save_project(p2)
                        return {"video": "/media/%s/cameramove/%s" % (proj["id"], os.path.basename(out)),
                                "file": out, "info": info, "elapsed": round(el, 1),
                                "layout": meta["layout"], "secs": meta["secs"],
                                "label": meta["label"], "moves": meta["moves"],
                                "shots": meta["shots"]}
                    run_job(jid, work)
                    return self.send_json({"job": jid})
                # ---- 单图模式（原有路径）----
                src = body.get("src")
                if not src or not os.path.isfile(src):
                    return self.send_json({"error": "The source file is missing or not found — upload and select an asset in step 1 first"}, 400)
                # 运镜入参：moves 列表 / "cm01,cm03" / 单镜 id / 组合布局；
                # 缺省或 "standard" → 固定标准五镜序列（CM-01→04→02→07→05，用户口径，不再勾选）
                layout = body.get("layout") or "standard"
                moves_in = body.get("moves")
                if moves_in:
                    layout = moves_in
                elif layout == "standard":
                    layout = list(cammove.STANDARD_SEQ)
                seq = cammove.normalize_moves(layout)
                if seq:
                    bad = [m for m in seq if m not in cammove.MOVES]
                    if bad:
                        return self.send_json({"error": "Unknown shot: %s (available: %s)" % (
                            "、".join(bad), "、".join(sorted(cammove.MOVES)))}, 400)
                    layout = seq
                elif layout != "auto" and layout not in cammove.MOVES and layout not in cammove.LAYOUTS:
                    return self.send_json({"error": "Unknown camera move: %s" % layout}, 400)
                req_secs = body.get("secs")
                plan_secs = cammove.layout_secs(layout, req_secs)
                slug = cammove.layout_slug(layout)
                out_dir = os.path.join(project_path(proj["id"]), "cameramove")
                os.makedirs(out_dir, exist_ok=True)
                stamp = time.strftime("%m%d%H%M%S")
                out = os.path.join(out_dir,
                                   "camera_moves_%s_%ss_%s.mp4" % (slug, round(plan_secs, 1), stamp))
                def work():
                    el, info, meta = cammove.render_cameramove(
                        src, out, layout, body.get("anchors"),
                        progress=lambda d, t: job_progress(jid, d, t, "Rendering"),
                        anchors_pct=body.get("anchors_pct"), secs=req_secs, ratio=ratio,
                        craft=smart.best_craft_boxes(proj, src))
                    # auto 模式预估值可能与实际选中镜头不同 / 多镜实际时长可能与预估不同
                    # → 统一按渲染后的实际镜头与时长改名（stamp 复用，保证当次唯一）
                    final_path = out
                    want = "camera_moves_%s_%ss_%s.mp4" % (meta["layout"], round(meta["secs"], 1), stamp)
                    if want != os.path.basename(out):
                        cand = os.path.join(out_dir, want)
                        try:
                            os.replace(out, cand)
                            final_path = cand
                        except OSError:
                            final_path = out
                    p = load_project(proj["id"])
                    p.setdefault("cameramove", []).append(
                        {"file": final_path, "name": "Camera moves %s" % meta["label"],
                         "kind": "cameramove", "dur": meta["secs"]})
                    save_project(p)
                    return {"video": "/media/%s/cameramove/%s" % (proj["id"], os.path.basename(final_path)),
                            "file": final_path, "info": info, "elapsed": round(el, 1),
                            "layout": meta["layout"], "requested": meta["requested"],
                            "reason": meta["reason"], "secs": meta["secs"],
                            "label": meta["label"], "moves": meta.get("moves", []),
                            "shots": meta.get("shots", [])}
                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/bgm/upload":
                """上传自定义 BGM（2026-10-27 用户需求：生成后 BGM 可选择、也可上传）。

                入参(JSON)：{name: 原文件名, data_b64: base64 音频内容}
                支持 mp3/wav/m4a/ogg/flac，解码后 ≤30MB；统一转码 mp3 192k 存入
                bgm/（user_ 前缀防与曲库重名），并登记进 manifest.json —— 这样
                /api/bgm 会把它当正常曲目返回，生成端点 _bgm_pick 也能按名选用。
                license 记为 User upload（版权责任在上传方，不冒充免版权曲）。
                """
                body = self.read_body_json()
                raw_name = str(body.get("name") or "upload.mp3")
                ext = os.path.splitext(raw_name)[1].lower()
                if ext not in (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"):
                    return self.send_json({"error": "Unsupported audio type — use mp3/wav/m4a/ogg/flac"}, 400)
                import base64 as _b64
                try:
                    blob = _b64.b64decode(body.get("data_b64") or "")
                except Exception:
                    return self.send_json({"error": "Bad audio payload"}, 400)
                if not blob:
                    return self.send_json({"error": "Empty audio file"}, 400)
                if len(blob) > 30 * 1024 * 1024:
                    return self.send_json({"error": "Audio too large (max 30MB)"}, 400)
                stem = safe_name(os.path.splitext(raw_name)[0]).strip() or "upload"
                stem = re.sub(r"\s+", "_", stem)[:60]
                slug = "user_" + re.sub(r"[^A-Za-z0-9_-]", "_", stem)
                out_mp3 = os.path.join(BGM_DIR, slug + ".mp3")
                tmp_in = out_mp3 + ".tmp" + ext
                try:
                    with open(tmp_in, "wb") as f:
                        f.write(blob)
                    from core.ffproc import run_ffmpeg, _default_ff
                    rc, err = run_ffmpeg([_default_ff(), "-y", "-v", "error", "-i", tmp_in,
                                          "-vn", "-codec:a", "libmp3lame", "-b:a", "192k",
                                          out_mp3])
                    if rc != 0 or not os.path.isfile(out_mp3) or os.path.getsize(out_mp3) < 1024:
                        return self.send_json({"error": "Could not decode this audio file"}, 400)
                finally:
                    try:
                        if os.path.isfile(tmp_in):
                            safedel.unlink(tmp_in)
                    except BaseException:
                        pass
                # 登记 manifest（list_bgm 只暴露登记过的曲目）
                man_path = os.path.join(BGM_DIR, "manifest.json")
                try:
                    man = json.load(open(man_path, encoding="utf-8")) if os.path.isfile(man_path) else {}
                except Exception:
                    man = {}
                music = man.setdefault("music", {})
                music[slug] = {"file": slug + ".mp3",
                               "license": "User upload (rights owned by uploader)",
                               "credit": raw_name, "mood": "custom", "light": False,
                               "duration": 0}
                tmp_man = man_path + ".tmp"
                with open(tmp_man, "w", encoding="utf-8") as f:
                    json.dump(man, f, ensure_ascii=False, indent=1)
                os.replace(tmp_man, man_path)
                return self.send_json({"ok": True, "name": slug,
                                       "label": raw_name})

            if p == "/api/card-pan":
                """单图裁卡运镜（内容驱动 + 按需旋转 + overscan 去条纹）。

                2026-10-14 口径：工艺微距（kind=craft，烫金/雕刻字）**不旋转**，
                开场那个旋转手法也改为「切到细节」推近；常规旋转只剩外壳俯仰，全片 ≤1 个。
                渲染走 core.card_pan（warpPerspective + OVERSCAN 1.8 放大缩回避条纹）。
                与现有 cammove（仅平移、无 overscan）互不干扰，独立模块。

                入参(JSON)：src 图片路径 / srcs 图片路径列表(2026-10-27 起，
                      1~8 张均可——每张独立识别编排后拼接；兼容旧单张 src，
                      两者都缺才报错) / project 项目id(可选) /
                ratio "9x16"|"4x5"|"1x1"|"16x9"(默认9x16，真源 card_pan.RATIOS；
                      也接受 "9:16" 写法) / fps(默认30) / target_secs 目标时长 /
                silent true 则跳过配乐 / plan "0W5A0561" 用确认回归基准 /
                hook 强制开场手法(hook_scan|hook_orbit，二选一；缺省按图片自动匹配) /
                regions 显式内容区域列表 /
                captions {on, text} step4「Add Captions」：单图模版无原生文字，
                   on=true 时烧一行 text（留空 = AI 看图起稿）在画面中上方、字号加大 /
                voiceover {on, selling, cta} 三段式口播（同 captions 冲突时
                   以 voiceover 为准，字幕即屏幕文字，口播字幕烧在中下）；selling=
                   用户填写的核心卖点(空则 AI 兜底)；cta= view/inquiry/samples 三选一
                返回：异步 job，result 含 {video, strip, file, secs, shots,
                captions(布尔), voiceover(布尔)}
                """
                body = self.read_body_json()
                pid = body.get("project")
                src = body.get("src")
                # 2026-10-27 多图支持：srcs 列表（1~8 张，Product Spotlight 不再
                # 限单图——用户红线「图片上传的数量可以不局限于1个」）；兼容旧单张
                # src。无效路径直接过滤，全无效才报错。
                _srcs_in = [s for s in (body.get("srcs") or ([src] if src else []) or [])
                            if s and os.path.isfile(s)]
                if not _srcs_in:
                    return self.send_json({"error": "The source image is missing — upload and select an asset first"}, 400)
                src = _srcs_in[0]
                # 画幅：客户端第三步所选（真源 core.card_pan.RATIOS）。
                # 必须在这里就归一好 —— 下面的 plan 编排要用 OW/OH 算取景比例，
                # 换画幅等于换构图，不能渲染时再裁一刀。
                from core.card_pan import RATIOS as _OUT_RATIOS, norm_ratio as _norm_ratio
                ratio = _norm_ratio(body.get("ratio"))
                if ratio not in _OUT_RATIOS:
                    ratio = "9x16"
                fps = int(body.get("fps") or 30)
                proj = None
                if pid:
                    try:
                        proj = load_project(pid)
                    except OSError:
                        proj = None
                cfg = dict(CONFIG.get("storyboard_api") or {})
                jid = new_job("card_pan")

                def work():
                    import cv2, numpy as np
                    from core import card_pan as cp
                    OW, OH = cp.ratio_size(ratio)
                    regions = body.get("regions")
                    plan_name = body.get("plan")
                    _tsec = None
                    try:
                        _tsec = float(body.get("target_secs") or 0) or None
                    except Exception:
                        _tsec = None
                    out_dir = os.path.join(project_path(pid) if pid else DATA, "cardpan")
                    os.makedirs(out_dir, exist_ok=True)
                    stamp = time.strftime("%m%d%H%M%S") + uuid.uuid4().hex[:3]
                    out = os.path.join(out_dir, "cardpan_%s_%s.mp4" % (ratio, stamp))
                    strip = out.replace(".mp4", "_strip.jpg")
                    if len(_srcs_in) > 1 and not regions and plan_name != "0W5A0561":
                        # ---- 多图（2026-10-27）：每张独立「识别→编排→渲染」，
                        # compose.concat_segments 硬切拼接（与 Collection Showcase
                        # 同一助手，自动同画幅归一）。第 1 张保留开场钩子、最后 1 张
                        # 保留收尾拉远，中间张只出内容镜头；每张目标 8s——总时长随
                        # 图数增长（2026-10-26 红线）。最多 8 张。
                        _srcs = _srcs_in[:8]
                        segs, tags, secs_sum = [], [], 0.0
                        for i, s in enumerate(_srcs):
                            job_progress(jid, 5 + int(70 * i / max(1, len(_srcs))), 100,
                                         "Rendering photo %d of %d" % (i + 1, len(_srcs)))
                            img_i = cv2.imdecode(np.fromfile(s, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if img_i is None:
                                raise FileNotFoundError(s)
                            from core.imgproc import normalize_luma
                            img_i = normalize_luma(img_i)   # 2026-10-27 亮度归一
                            H_i, W_i = img_i.shape[:2]
                            plan_i = _auto_card_pan_plan(cp, s, proj, cfg, W_i, H_i, OW, OH,
                                                         target_secs=8.0)
                            if i > 0:
                                plan_i = [m for m in plan_i
                                          if not str(m["kind"]).startswith("hook")
                                          and m["kind"] != "tour"]
                            else:
                                # 多图拼接只在首段保留一段巡览（8s 目标装不下两段）
                                _tr = [m for m in plan_i if m["kind"] == "tour"]
                                for _m in _tr[1:]:
                                    plan_i.remove(_m)
                            if i < len(_srcs) - 1:
                                plan_i = [m for m in plan_i if m["kind"] != "reveal"]
                            if not plan_i:
                                plan_i = _auto_card_pan_plan(cp, s, proj, cfg, W_i, H_i,
                                                             OW, OH, target_secs=8.0)
                            seg_i = os.path.join(out_dir, "cardpan_seg%d_%s.mp4" % (i, stamp))
                            cp.card_pan_render(img_i, plan_i, seg_i, ratio=ratio, fps=fps,
                                               strip_path=strip if i == 0 else None)
                            segs.append(seg_i)
                            tags += [m.get("tag", m["kind"]) for m in plan_i]
                            secs_sum += sum(m["dur"] for m in plan_i) \
                                - cp.XFADE * max(0, len(plan_i) - 1)
                        compose.concat_segments(segs, out, size=(OW, OH))
                        plan = [{"kind": "multi", "dur": 0, "tag": t} for t in tags]
                        secs = round(secs_sum, 1)
                    else:
                        img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            raise FileNotFoundError(src)
                        H, W = img.shape[:2]
                        if regions:
                            plan = cp.build_plan_from_regions([dict(r) for r in regions], OW, OH, W, H)
                        elif plan_name == "0W5A0561":
                            plan = cp.plan_0W5A0561(W, H, OW, OH)
                        else:
                            plan = _auto_card_pan_plan(cp, src, proj, cfg, W, H, OW, OH,
                                                       target_secs=_tsec,
                                                       force_hook=body.get("hook"))
                        # 时长收敛：内容稀疏时补多样运镜镜头，整体拉到 ~target_secs 并压回 15s 内
                        plan = _converge_duration(cp, plan, src, W, H, OW, OH,
                                                  target_secs=(_tsec or 15.0))
                        cp.card_pan_render(img, plan, out, ratio=ratio, fps=fps, strip_path=strip)
                        secs = round(sum(m["dur"] for m in plan) - cp.XFADE * max(0, len(plan) - 1), 1)
                    # BGM：渲染器只出画面（成片会是无声的）——在这里补混音。
                    # 用户 2026-10-26 红线：模版用各自固定的 BGM，不随机生成。
                    # 2026-10-28 换曲（用户指定）：Product Spotlight 固定配
                    # crystaleyeofficial（明亮流行电子感，129bpm；I Do 退役，
                    # 不与其它模版 BGM 重复）；混音失败不能让整单失败。
                    _bf = None
                    if not body.get("silent"):
                        try:
                            _bf = _bgm_pick(body, "crystaleyeofficial-.mp3")
                            if _bf and os.path.isfile(_bf):
                                _mix = out + ".mix.mp4"
                                job_progress(jid, 88, 100, "Adding music")
                                compose.mix_bgm(out, _bf, _mix, gain_db=-4.0,
                                                fade_in=1.0, fade_out=2.2)
                                if os.path.isfile(_mix) and os.path.getsize(_mix) > 1024:
                                    os.replace(_mix, out)   # 原子覆盖，不删文件（绕开清理钩子）
                        except BaseException:
                            traceback.print_exc()
                    # 屏幕文字 / 口播（step4 Captions & Voiceover 可选）。
                    # 单图模版没有原生文字元素：captions = 后处理烧一行 AI 起稿的
                    # 产品介绍；voiceover = 三段式配音+字幕（字幕本身就是屏幕文字，
                    # 两者同开时跳过 captions 行，避免叠字）。
                    # 位置（用户 2026-10-20 定稿）：captions 行 → 中上方 + 加大
                    # 字号；口播字幕 → 中下（sub_pos="lower"，四个模版统一）。
                    _vo = body.get("voiceover") or {}
                    _cap = body.get("captions") or {}
                    vo_meta = None
                    cap_meta = None
                    if _cap.get("on") and not _vo.get("on"):
                        try:
                            from core import voiceover as _volib
                            _line = str(_cap.get("text") or "").strip()
                            if not _line:
                                _line = _volib.draft_caption_line(
                                    cfg, list(_srcs_in), sheet_path=os.path.join(out_dir, "_cap_contact.jpg"))
                            _cap_out = out + ".cap.mp4"
                            job_progress(jid, 90, 100, "Adding on-screen text")
                            _volib.apply_caption(out, _cap_out, _line, sub_pos="upper",
                                                 font=(body.get("captions") or {}).get("font"))
                            if os.path.isfile(_cap_out) and os.path.getsize(_cap_out) > 1024:
                                os.replace(_cap_out, out)
                                cap_meta = _line
                        except BaseException:
                            traceback.print_exc()
                    if _vo.get("on"):
                        try:
                            from core import voiceover as _volib

                            def _vprog(done, total_, msg=""):
                                job_progress(jid, 92 + int(done) * 8 // max(1, int(total_ or 100)),
                                             100, msg or "Adding voiceover")

                            _vo_out = out + ".vo.mp4"
                            vo_meta = _volib.voiceover_on_rendered(
                                out, _vo_out, list(_srcs_in), 0.0, secs, secs, [],
                                str(_vo.get("selling") or "").strip(),
                                str(_vo.get("cta") or "view").strip(),
                                cfg, progress=_vprog, fallback_bgm=_bf,
                                beats=_volib.three_act_beats(secs),
                                sub_pos="lower")
                            if os.path.isfile(_vo_out) and os.path.getsize(_vo_out) > 1024:
                                os.replace(_vo_out, out)
                        except BaseException:
                            traceback.print_exc()
                    base = os.path.basename(out)
                    if pid:
                        p2 = load_project(pid)
                        p2.setdefault("cardpan", []).append(
                            {"file": out, "name": "Card pan %ss" % secs, "kind": "cardpan", "dur": secs})
                        save_project(p2)
                        video = "/media/%s/cardpan/%s" % (pid, base)
                        strip_url = "/media/%s/cardpan/%s" % (pid, base.replace(".mp4", "_strip.jpg"))
                    else:
                        video = "/media/cardpan/%s" % base
                        strip_url = "/media/cardpan/%s" % base.replace(".mp4", "_strip.jpg")
                    return {"video": video, "strip": strip_url, "file": out, "secs": secs,
                            "shots": [m.get("tag", m["kind"]) for m in plan],
                            "captions": bool(cap_meta), "voiceover": bool(vo_meta)}
                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/multi-suite":
                """多图合集模版（Collection Showcase）—— N 张合集图 → ≤15s 成片。

                与 /api/card-pan（单图模版 Product Showcase）**完全独立**：
                管线在 core.multi_suite（大标题叠在第一个款式上开场 → 逐款分段 + 名牌），
                不动 card_pan 的编排，两条模版互不影响。

                入参(JSON)：srcs 图片路径列表(必填，2~8 张) /
                labels 每张图的款式名牌(可选，与 srcs 等长；空串 = 该段不出名牌) /
                       shots / focuses 每张图的运镜判别(可选，与 srcs 等长；
                       'detail'/'full'/'rotate'，focuses 为 detail 的归一化焦点框) /
                title 开场大标题(可选；None/缺省=默认 "Our Collection"，
                       "" = 不叠标题——step4「Add Captions」关闭时传空) /
                project 项目id(可选) / ratio 画幅(默认 9x16) / fps(默认 30) /
                silent true 则跳过配乐 /
                voiceover {on, selling, cta} 三段式口播(可选，step4 勾选)：
                   on=true 时 BGM 混音后追加「0-3s 钩子 / 3-10s 卖点 /
                   10-15s 行动引导」英文配音+字幕；selling=用户填写的核心卖点
                   (空则 AI 看图兜底)；cta= view(查看款式,默认)/inquire(咨询定制)/
                   samples(申请样品)。失败不影响交付，仍出无口播成片。
                返回：异步 job，result 含 {video, file, secs, images, labels,
                captions(布尔), voiceover(布尔)}
                """
                body = self.read_body_json()
                pid = body.get("project")
                srcs = [s for s in (body.get("srcs") or []) if s]
                srcs = [s for s in srcs if os.path.isfile(s)]
                if len(srcs) < 2:
                    return self.send_json(
                        {"error": "This template needs at least 2 photos — upload a few collection photos first"}, 400)
                labels = [str(x or "") for x in (body.get("labels") or [])]
                shots = [str(x or "") for x in (body.get("shots") or [])]
                focuses = body.get("focuses") or []
                from core.card_pan import norm_ratio as _norm_ratio
                ratio = _norm_ratio(body.get("ratio"))
                fps = int(body.get("fps") or 30)
                jid = new_job("multi_suite")

                def work():
                    from core import multi_suite as ms
                    out_dir = os.path.join(project_path(pid) if pid else DATA, "multisuite")
                    os.makedirs(out_dir, exist_ok=True)
                    stamp = time.strftime("%m%d%H%M%S") + uuid.uuid4().hex[:3]
                    out = os.path.join(out_dir, "multi_%s_%s.mp4" % (ratio, stamp))

                    def prog(done, total_, msg=""):
                        job_progress(jid, int(done), int(total_ or 100), msg)

                    res = ms.render_multi(srcs, out, labels=labels,
                                          title=str(body.get("title") or ""),
                                          ratio=ratio, fps=fps, progress=prog,
                                          shots=shots, focuses=focuses,
                                          font=body.get("font"))
                    secs = res["secs"]
                    _bf = None
                    # BGM：用户 2026-10-26 红线：模版用各自固定的 BGM，不随机生成。
                    # Collection Showcase 固定配 Heartwarming（温暖钢琴，Kevin MacLeod CC-BY，
                    # 不与其它模版 BGM 重复）；混音失败不影响画面交付。
                    if not body.get("silent"):
                        try:
                            _bf = _bgm_pick(body, "heartwarming.mp3")
                            if _bf and os.path.isfile(_bf):
                                _mix = out + ".mix.mp4"
                                job_progress(jid, 90, 100, "Adding music")
                                compose.mix_bgm(out, _bf, _mix, gain_db=-4.0,
                                                fade_in=1.0, fade_out=2.2)
                                if os.path.isfile(_mix) and os.path.getsize(_mix) > 1024:
                                    os.replace(_mix, out)
                        except BaseException:
                            traceback.print_exc()
                    # 口播后处理（step4 Captions & Voiceover 可选）：三段式配音+字幕，
                    # 接在成片之后。成片此时已含 BGM → apply_voiceover 走「压低 BGM +
                    # 叠人声」路径；若 BGM 混音失败(无音轨)，用选曲文件做兜底背景。
                    # 口播字幕走**中下**（sub_pos="lower" 默认，四模版统一）：款式
                    # 名牌在画面顶部 ~14%，中下位置不叠字（用户 2026-10-20 定稿）。
                    # 任何失败只打印，仍交付无口播成片。
                    vo = body.get("voiceover") or {}
                    vo_meta = None
                    if vo.get("on"):
                        try:
                            from core import voiceover as _volib
                            _cfg = dict(CONFIG.get("storyboard_api") or {})

                            def _vprog(done, total_, msg=""):
                                job_progress(jid, 92 + int(done) * 8 // max(1, int(total_ or 100)),
                                             100, msg or "Adding voiceover")

                            _vo_out = out + ".vo.mp4"
                            vo_meta = _volib.voiceover_on_rendered(
                                out, _vo_out, srcs,
                                res.get("opener") or 0.0, res.get("seg") or secs,
                                secs, res.get("labels") or labels,
                                str(vo.get("selling") or "").strip(),
                                str(vo.get("cta") or "view").strip(),
                                _cfg, progress=_vprog,
                                beats=_volib.three_act_beats(secs),
                                fallback_bgm=_bf if _bf and os.path.isfile(_bf) else None)
                            if os.path.isfile(_vo_out) and os.path.getsize(_vo_out) > 1024:
                                os.replace(_vo_out, out)
                        except BaseException:
                            traceback.print_exc()
                    base = os.path.basename(out)
                    if pid:
                        p2 = load_project(pid)
                        p2.setdefault("multisuite", []).append(
                            {"file": out, "name": "Collection %ss" % secs,
                             "kind": "multisuite", "dur": secs})
                        save_project(p2)
                        video = "/media/%s/multisuite/%s" % (pid, base)
                    else:
                        video = "/media/multisuite/%s" % base
                    return {"video": video, "file": out, "secs": secs,
                            "images": res["images"], "labels": res.get("labels") or [],
                            "shots": res.get("moves") or [],
                            "captions": bool(str(body.get("title") or "").strip())
                                        or bool([x for x in labels if str(x or "").strip()]),
                            "voiceover": bool(vo_meta),
                            "ratio": res.get("ratio")}
                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/scroll-gallery":
                """滚动画廊模版（Scrolling Gallery）—— N 张请柬图 → 左右两列反向慢滚动成片。

                与 /api/multi-suite（Collection Showcase）**完全独立**：
                管线在 core.scroll_gallery（左列前半图向上滚 / 右列后半图向下滚，
                3:4 统一比例、圆角浅灰描边、无文字、纯白背景），不动 multi_suite 的编排。

                入参(JSON)：srcs 图片路径列表(必填，≥2 张；前半给左列、后半给右列) /
                project 项目id(可选) / ratio 画幅(默认 9x16) / fps(默认 30) /
                silent true 则跳过配乐（默认配婚礼竖琴 BGM，用户 2026-10-19 指定） /
                captions {on, text} step4「Add Captions」：滚动画廊无原生文字，
                   on=true 时烧一行 text（留空 = AI 看图起稿）在画面中上方、字号加大 /
                voiceover {on, selling, cta} 三段式口播（同 captions 冲突时
                   以 voiceover 为准，字幕即屏幕文字，口播字幕烧在中下）；selling 空则
                   AI 兜底；cta= view/inquiry/samples 三选一
                返回：异步 job，result 含 {video, file, secs, images, ratio,
                captions(布尔), voiceover(布尔)}
                """
                body = self.read_body_json()
                pid = body.get("project")
                srcs = [s for s in (body.get("srcs") or []) if s]
                srcs = [s for s in srcs if os.path.isfile(s)]
                if len(srcs) < 6:
                    return self.send_json(
                        {"error": "This template needs 6 to 8 photos — upload at least 6 invitation photos first"}, 400)
                from core.card_pan import norm_ratio as _norm_ratio
                ratio = _norm_ratio(body.get("ratio"))
                fps = int(body.get("fps") or 30)
                jid = new_job("scroll_gallery")

                def work():
                    from core import scroll_gallery as sg
                    out_dir = os.path.join(project_path(pid) if pid else DATA, "scrollgallery")
                    os.makedirs(out_dir, exist_ok=True)
                    stamp = time.strftime("%m%d%H%M%S") + uuid.uuid4().hex[:3]
                    out = os.path.join(out_dir, "scroll_%s_%s.mp4" % (ratio, stamp))

                    def prog(done, total_, msg=""):
                        job_progress(jid, int(done), int(total_ or 100), msg)

                    res = sg.render_scroll(srcs, out, ratio=ratio, fps=fps, progress=prog)
                    secs = res["secs"]
                    _bf = None
                    # BGM：用户 2026-10-19 指定婚礼竖琴；失败不影响画面交付。
                    if not body.get("silent"):
                        try:
                            _bf = _bgm_pick(body, "Wedding_Harp_672.mp3")
                            if os.path.isfile(_bf):
                                _mix = out + ".mix.mp4"
                                job_progress(jid, 92, 100, "Adding music")
                                compose.mix_bgm(out, _bf, _mix, gain_db=-4.0,
                                                fade_in=1.0, fade_out=2.2)
                                if os.path.isfile(_mix) and os.path.getsize(_mix) > 1024:
                                    os.replace(_mix, out)
                        except BaseException:
                            traceback.print_exc()
                    # 屏幕文字 / 口播（step4 Captions & Voiceover 可选）。滚动画廊
                    # 没有原生文字元素：captions = 后处理烧一行 AI 起稿的产品介绍；
                    # voiceover = 三段式配音+字幕（字幕即屏幕文字，同开时跳过 captions）。
                    # 位置（用户 2026-10-20 定稿）：captions 行 → 中上方 + 加大
                    # 字号；口播字幕 → 中下（sub_pos="lower"，四个模版统一）。
                    _vo = body.get("voiceover") or {}
                    _cap = body.get("captions") or {}
                    vo_meta = None
                    cap_meta = None
                    if _cap.get("on") and not _vo.get("on"):
                        try:
                            from core import voiceover as _volib
                            _line = str(_cap.get("text") or "").strip()
                            if not _line:
                                _line = _volib.draft_caption_line(
                                    cfg, srcs, sheet_path=os.path.join(out_dir, "_cap_contact.jpg"))
                            _cap_out = out + ".cap.mp4"
                            job_progress(jid, 94, 100, "Adding on-screen text")
                            _volib.apply_caption(out, _cap_out, _line, sub_pos="upper",
                                                 font=_cap.get("font"))
                            if os.path.isfile(_cap_out) and os.path.getsize(_cap_out) > 1024:
                                os.replace(_cap_out, out)
                                cap_meta = _line
                        except BaseException:
                            traceback.print_exc()
                    if _vo.get("on"):
                        try:
                            from core import voiceover as _volib
                            _cfg = dict(CONFIG.get("storyboard_api") or {})

                            def _vprog(done, total_, msg=""):
                                job_progress(jid, 92 + int(done) * 8 // max(1, int(total_ or 100)),
                                             100, msg or "Adding voiceover")

                            _vo_out = out + ".vo.mp4"
                            vo_meta = _volib.voiceover_on_rendered(
                                out, _vo_out, srcs, 0.0,
                                secs / float(max(1, len(srcs))), secs, [],
                                str(_vo.get("selling") or "").strip(),
                                str(_vo.get("cta") or "view").strip(),
                                _cfg, progress=_vprog, fallback_bgm=_bf,
                                beats=_volib.three_act_beats(secs),
                                sub_pos="lower")
                            if os.path.isfile(_vo_out) and os.path.getsize(_vo_out) > 1024:
                                os.replace(_vo_out, out)
                        except BaseException:
                            traceback.print_exc()
                    base = os.path.basename(out)
                    if pid:
                        p2 = load_project(pid)
                        p2.setdefault("scrollgallery", []).append(
                            {"file": out, "name": "Scrolling %ss" % secs,
                             "kind": "scrollgallery", "dur": secs})
                        save_project(p2)
                        video = "/media/%s/scrollgallery/%s" % (pid, base)
                    else:
                        video = "/media/scrollgallery/%s" % base
                    return {"video": video, "file": out, "secs": secs,
                            "images": res["images"], "ratio": res.get("ratio"),
                            "captions": bool(cap_meta), "voiceover": bool(vo_meta)}
                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/four-grid":
                """四宫格开场 + 单图快切模版（Grid Showcase）—— N 张请柬图 → 四宫格(前4) + 全图快切成片。

                与 /api/scroll-gallery（Colour Inspiration）、/api/multi-suite（Collection Showcase）
                **完全独立**：管线在 core.four_grid（前 4 张做 2×2 无缝铺满 + 错峰淡入，再定格 1s，
                然后全部 N 张依次单图快剪、轻微推近；中央可选叠衬线标题 + 花体副标题，无滤镜调色），
                不动其它模版的编排。

                入参(JSON)：srcs 图片路径列表(必填，≥4 张；前 4 张做四宫格，全部 N 张做单图快切) /
                title 衬线中央文字(可选，默认空) / subtitle 花体中央文字(可选，默认空) /
                show_text false 则完全不叠文字(默认 false＝opt-in；step4「Add Captions」关闭时传 false) /
                project 项目id(可选) / ratio 画幅(默认 9x16) / fps(默认 30) /
                silent true 则跳过配乐（默认配 cheery_monday 欢快向 BGM，用户 2026-10-20 指定） /
                voiceover {on, selling, cta} 三段式口播(可选，step4 勾选)：on=true 时
                   BGM 后追加三段式英文配音+字幕（0-2s 钩子 / 2-7s 卖点 / 7-10s 行动引导，
                   按成片时长由 three_act_beats 现算），口播字幕烧在画面中下
                   （sub_pos="lower"，四模版统一，不与中央标题叠）；selling 空则
                   AI 兜底；cta 三选一。失败不影响交付。
                返回：异步 job，result 含 {video, file, secs, images, ratio,
                captions(布尔), voiceover(布尔)}
                """
                body = self.read_body_json()
                pid = body.get("project")
                srcs = [s for s in (body.get("srcs") or []) if s]
                srcs = [s for s in srcs if os.path.isfile(s)]
                if len(srcs) < 4:
                    return self.send_json(
                        {"error": "This template needs 4 to 8 photos — upload at least 4 invitation photos first"}, 400)
                from core.card_pan import norm_ratio as _norm_ratio
                ratio = _norm_ratio(body.get("ratio"))
                fps = int(body.get("fps") or 30)
                show_text = body.get("show_text", False)
                if isinstance(show_text, str):
                    show_text = show_text.lower() not in ("0", "false", "no")
                title = str(body.get("title") or "")
                subtitle = str(body.get("subtitle") or "")
                jid = new_job("four_grid")

                def work():
                    from core import four_grid as fg
                    out_dir = os.path.join(project_path(pid) if pid else DATA, "fourgrid")
                    os.makedirs(out_dir, exist_ok=True)
                    stamp = time.strftime("%m%d%H%M%S") + uuid.uuid4().hex[:3]
                    out = os.path.join(out_dir, "grid_%s_%s.mp4" % (ratio, stamp))

                    def prog(done, total_, msg=""):
                        job_progress(jid, int(done), int(total_ or 100), msg)

                    res = fg.render_four_grid(srcs, out, ratio=ratio, fps=fps,
                                              title=title, subtitle=subtitle,
                                              show_text=show_text, progress=prog,
                                              font=body.get("font"))
                    secs = res["secs"]
                    _bf = None
                    # BGM：用户 2026-10-20 指定 cheery_monday（欢快向）；失败不影响画面交付。
                    if not body.get("silent"):
                        try:
                            _bf = _bgm_pick(body, "cheery_monday.mp3")
                            if os.path.isfile(_bf):
                                _mix = out + ".mix.mp4"
                                job_progress(jid, 92, 100, "Adding music")
                                compose.mix_bgm(out, _bf, _mix, gain_db=-4.0,
                                                fade_in=1.0, fade_out=2.2)
                                if os.path.isfile(_mix) and os.path.getsize(_mix) > 1024:
                                    os.replace(_mix, out)
                        except BaseException:
                            traceback.print_exc()
                    # 口播后处理（step4 Captions & Voiceover 可选）：三段式配音+字幕。
                    # captions 对本模版是渲染器原生中央文字（show_text/title/subtitle），
                    # 不走后处理；口播字幕烧中下（sub_pos="lower"）——口播字幕
                    # 四个模版一律中下（用户 2026-10-20 定稿）。
                    # 任何失败只打印，仍交付无口播成片。
                    _vo = body.get("voiceover") or {}
                    vo_meta = None
                    if _vo.get("on"):
                        try:
                            from core import voiceover as _volib
                            _cfg = dict(CONFIG.get("storyboard_api") or {})

                            def _vprog(done, total_, msg=""):
                                job_progress(jid, 92 + int(done) * 8 // max(1, int(total_ or 100)),
                                             100, msg or "Adding voiceover")

                            _vo_out = out + ".vo.mp4"
                            vo_meta = _volib.voiceover_on_rendered(
                                out, _vo_out, srcs, 0.0, secs, secs, [],
                                str(_vo.get("selling") or "").strip(),
                                str(_vo.get("cta") or "view").strip(),
                                _cfg, progress=_vprog, fallback_bgm=_bf,
                                beats=_volib.three_act_beats(secs),
                                sub_pos="lower")
                            if os.path.isfile(_vo_out) and os.path.getsize(_vo_out) > 1024:
                                os.replace(_vo_out, out)
                        except BaseException:
                            traceback.print_exc()
                    base = os.path.basename(out)
                    if pid:
                        p2 = load_project(pid)
                        p2.setdefault("fourgrid", []).append(
                            {"file": out, "name": "Grid %ss" % secs,
                             "kind": "fourgrid", "dur": secs})
                        save_project(p2)
                        video = "/media/%s/fourgrid/%s" % (pid, base)
                    else:
                        video = "/media/fourgrid/%s" % base
                    return {"video": video, "file": out, "secs": secs,
                            "images": res["images"], "ratio": res.get("ratio"),
                            "captions": bool(show_text and (title.strip() or subtitle.strip())),
                            "voiceover": bool(vo_meta)}
                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/three-screen":
                g = self._ai_gate("caption")
                if g:
                    return self.send_json({"error": g["error"]}, g["code"])
                """三屏故事模版（Three-Screen Story）—— N 张图 -> 三屏错峰揭示开场 + 单屏图像感知运镜。

                与 /api/four-grid（Grid Showcase）等 **完全独立**：管线在 core.three_screen
                （开场三条横带上->中->下错峰揭示 + 分层推近；单屏每张图镜型混合分配
                 ——学习参考视频 Download (1).mp4：非纸品按 REF_MIX 循环 pan/push/pull
                 同型不连排，纸品(请柬/桌号卡/sign/菜单)->pan+1.18 放大，
                 近满幅图->慢推兜底；全片交叉溶解，字幕可选（step4 勾选后烧一行
                 奶油白衬线斜体于画面底部居中），不调色），不动其它模版的编排。

                入参(JSON)：srcs 图片路径列表(必填，5~12 张) /
                project 项目id(可选) / ratio 画幅(默认 9x16) / fps(默认 30) /
                silent true 则跳过配乐（默认配 link_ncs（Jim Yosef - Link，NCS 热门潮流电子，
                2026-10-26 指定：不与其它模版 BGM 重复 + royalty-free 但需署名 NCS）） /
                captions {on, text} step4「Add Captions」：on=true 时烧一行字幕于画面
                  底部居中（留空则 AI 看图起稿）；与 voiceover 同开时以 voiceover 字幕为准 /
                voiceover {on, selling, cta} 三段式口播(可选，step4 勾选)：on=true 时
                  BGM 后追加三段式英文配音+字幕（按成片时长由 three_act_beats 现算），
                  口播字幕烧在画面中下（sub_pos="lower"）；selling 空则 AI 兜底；
                  cta 三选一。失败不影响交付。
                返回：异步 job，result 含 {video, file, secs, images, ratio,
                captions(布尔), voiceover(布尔)}
                """
                body = self.read_body_json()
                pid = body.get("project")
                srcs = [s for s in (body.get("srcs") or []) if s]
                srcs = [s for s in srcs if os.path.isfile(s)]
                if len(srcs) < 5:
                    return self.send_json(
                        {"error": "This template needs 5 to 12 photos — upload at least 5 photos first"}, 400)
                if len(srcs) > 12:
                    srcs = srcs[:12]
                from core.card_pan import norm_ratio as _norm_ratio
                ratio = _norm_ratio(body.get("ratio"))
                fps = int(body.get("fps") or 30)
                jid = new_job("three_screen")

                def work():
                    from core import three_screen as tsc
                    out_dir = os.path.join(project_path(pid) if pid else DATA, "threescreen")
                    os.makedirs(out_dir, exist_ok=True)
                    stamp = time.strftime("%m%d%H%M%S") + uuid.uuid4().hex[:3]
                    out = os.path.join(out_dir, "threescreen_%s_%s.mp4" % (ratio, stamp))

                    def prog(done, total_, msg=""):
                        job_progress(jid, int(done), int(total_ or 100), msg)

                    # 字幕（step4「Add Captions」）：on 时取用户文本，留空则 AI 起稿兜底；
                    # 任何失败只打印，仍出无字幕成片（不阻断交付）。
                    cap_text = ""
                    _caps = body.get("captions") or {}
                    if _caps.get("on"):
                        cap_text = str(_caps.get("text") or "").strip()
                        if not cap_text:
                            try:
                                from core import voiceover as _volib
                                cap_text = _volib.draft_caption_line(
                                    dict(CONFIG.get("storyboard_api") or {}), srcs,
                                    sheet_path=os.path.join(out_dir, "_cap_sheet.jpg"))
                            except BaseException:
                                traceback.print_exc()

                    res = tsc.render_three_screen(srcs, out, ratio=ratio, fps=fps,
                                                   progress=prog,
                                                   captions=cap_text or None,
                                                   font=(_caps.get("font") if isinstance(_caps, dict) else None))
                    secs = res["secs"]
                    _bf = None
                    # BGM：用户 2026-10-26 指定 link_ncs（Jim Yosef - Link，NCS 热门
                    # 潮流电子，royalty-free 但需署名）；不与其它模版 BGM 重复；失败不影响画面交付。
                    if not body.get("silent"):
                        try:
                            _bf = _bgm_pick(body, "link_ncs.mp3")
                            if os.path.isfile(_bf):
                                _mix = out + ".mix.mp4"
                                job_progress(jid, 92, 100, "Adding music")
                                compose.mix_bgm(out, _bf, _mix, gain_db=-4.0,
                                                fade_in=1.0, fade_out=2.2)
                                if os.path.isfile(_mix) and os.path.getsize(_mix) > 1024:
                                    os.replace(_mix, out)
                        except BaseException:
                            traceback.print_exc()
                    # 口播后处理（step4 Captions & Voiceover 可选）：三段式配音+字幕。
                    # 字幕烧中下（sub_pos="lower"）；任何失败只打印，仍交付无口播成片。
                    _vo = body.get("voiceover") or {}
                    vo_meta = None
                    if _vo.get("on"):
                        try:
                            from core import voiceover as _volib
                            _cfg = dict(CONFIG.get("storyboard_api") or {})

                            def _vprog(done, total_, msg=""):
                                job_progress(jid, 92 + int(done) * 8 // max(1, int(total_ or 100)),
                                             100, msg or "Adding voiceover")

                            _vo_out = out + ".vo.mp4"
                            vo_meta = _volib.voiceover_on_rendered(
                                out, _vo_out, srcs, 0.0, secs, secs, [],
                                str(_vo.get("selling") or "").strip(),
                                str(_vo.get("cta") or "view").strip(),
                                _cfg, progress=_vprog, fallback_bgm=_bf,
                                beats=_volib.three_act_beats(secs),
                                sub_pos="lower")
                            if os.path.isfile(_vo_out) and os.path.getsize(_vo_out) > 1024:
                                os.replace(_vo_out, out)
                        except BaseException:
                            traceback.print_exc()
                    base = os.path.basename(out)
                    if pid:
                        p2 = load_project(pid)
                        p2.setdefault("threescreen", []).append(
                            {"file": out, "name": "Three-Screen %ss" % secs,
                             "kind": "threescreen", "dur": secs})
                        save_project(p2)
                        video = "/media/%s/threescreen/%s" % (pid, base)
                    else:
                        video = "/media/threescreen/%s" % base
                    return {"video": video, "file": out, "secs": secs,
                            "images": res["images"], "ratio": res.get("ratio"),
                            "captions": bool(cap_text),
                            "voiceover": bool(vo_meta)}
                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/top5":
                """TOP 5 倒数榜模版（TOP 5 Countdown）—— N 张产品图 -> 标题页 + 每图运镜段
                + 左下酒红编号块 + 右侧款式 SKU。图片数 3-8 张（可调）；每图 sku + num 可填
                （num 留空自动倒数 01..N）；BGM 可选（默认 user_paulyudin）。
                与 /api/three-screen 等 **完全独立**：管线在 core.top5（零 AI 依赖）。"""
                body = self.read_body_json()
                pid = body.get("project")
                srcs = [s for s in (body.get("srcs") or []) if s]
                srcs = [s for s in srcs if os.path.isfile(s)]
                if len(srcs) < 3:
                    return self.send_json(
                        {"error": "This template needs 3 to 8 photos — upload at least 3 photos first"}, 400)
                if len(srcs) > 8:
                    srcs = srcs[:8]
                skus = body.get("skus") or []
                nums = body.get("nums") or []
                # 开场标题页文字（第一幕）可填：main 默认 TOP 5，sub 留空不画
                title_main = (body.get("title_main") or "").strip() or "TOP 5"
                title_sub = (body.get("title_sub") or "").strip() or None
                from core.card_pan import norm_ratio as _norm_ratio
                ratio = _norm_ratio(body.get("ratio"))
                fps = int(body.get("fps") or 30)
                jid = new_job("top5")

                def work():
                    from core import top5 as t5
                    out_dir = os.path.join(project_path(pid) if pid else DATA, "top5")
                    os.makedirs(out_dir, exist_ok=True)
                    stamp = time.strftime("%m%d%H%M%S") + uuid.uuid4().hex[:3]
                    out = os.path.join(out_dir, "top5_%s_%s.mp4" % (ratio, stamp))

                    def prog(done, total_, msg=""):
                        job_progress(jid, int(done), int(total_ or 100), msg)

                    # BGM：默认 user_paulyudin（用户 2026-10-27 指定婚礼曲）；
                    # 用户也可在 step4 选自己上传的曲。失败不影响画面交付。
                    _bf = None
                    if not body.get("silent"):
                        try:
                            _bf = _bgm_pick(body, "user_paulyudin.mp3")
                        except BaseException:
                            traceback.print_exc()
                    res = t5.render_top5(srcs, out, skus=skus, nums=nums, bgm=_bf,
                                         ratio=ratio, fps=fps, progress=prog,
                                         title_main=title_main, title_sub=title_sub)
                    secs = res["secs"]
                    base = os.path.basename(out)
                    if pid:
                        p2 = load_project(pid)
                        p2.setdefault("top5", []).append(
                            {"file": out, "name": "TOP5 %ss" % secs,
                             "kind": "top5", "dur": secs})
                        save_project(p2)
                        video = "/media/%s/top5/%s" % (pid, base)
                    else:
                        video = "/media/top5/%s" % base
                    return {"video": video, "file": out, "secs": secs,
                            "images": res["images"], "ratio": res.get("ratio"),
                            "moves": res.get("moves")}
                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/carousel":
                """Center-Stage Carousel 轮播模版—— N 张产品图(2-8) -> 卡片轮播：
                主卡居中全尺寸、两侧邻卡缩小 0.795 露边，每款停留 ~1.9s 后
                0.5s 快速平滑滑到下一款（2026-10-28 用户口径：一次只突出展示
                一款产品，不长时间游移）。三段文字可填：title2(粗体金标题,
                顶部)/title1(细体副题,标题下)/footer(底部网址,上方自动加
                Explore Now + 圆形箭头徽章)。
                与其它模版**完全独立**：管线在 core.carousel（零 AI 依赖）。"""
                body = self.read_body_json()
                pid = body.get("project")
                srcs = [s for s in (body.get("srcs") or []) if s]
                srcs = [s for s in srcs if os.path.isfile(s)]
                if len(srcs) < 2:
                    return self.send_json(
                        {"error": "This template needs 2 to 8 photos — upload at least 2 photos first"}, 400)
                if len(srcs) > 8:
                    srcs = srcs[:8]
                title1 = (body.get("title1") or "").strip() or "Our Wedding Invitations"
                title2 = (body.get("title2") or "").strip() or "NEW COLLECTION"
                footer = (body.get("footer") or "").strip()
                from core.card_pan import norm_ratio as _norm_ratio
                ratio = _norm_ratio(body.get("ratio"))
                fps = int(body.get("fps") or 30)
                jid = new_job("carousel")

                def work():
                    from core import carousel as car
                    out_dir = os.path.join(project_path(pid) if pid else DATA, "carousel")
                    os.makedirs(out_dir, exist_ok=True)
                    stamp = time.strftime("%m%d%H%M%S") + uuid.uuid4().hex[:3]
                    out = os.path.join(out_dir, "carousel_%s_%s.mp4" % (ratio, stamp))

                    def prog(done, total_, msg=""):
                        job_progress(jid, int(done), int(total_ or 100), msg)

                    # BGM：模版固定默认曲「婚礼背景音乐.mp3」（2026-10-28 用户指定：
                    # bgm 目录里用户放置的婚礼背景音乐，31.1s/≈129BPM，无其它模版使用）；
                    # 客户也可在 step4 选自己上传的曲（user_ 前缀）。
                    _bf = None
                    if not body.get("silent"):
                        try:
                            _bf = _bgm_pick(body, "婚礼背景音乐.mp3")
                        except BaseException:
                            traceback.print_exc()
                    res = car.render_carousel(srcs, out, bgm=_bf, ratio=ratio,
                                              fps=fps, progress=prog,
                                              title1=title1, title2=title2,
                                              footer=footer)
                    secs = res["secs"]
                    base = os.path.basename(out)
                    if pid:
                        p2 = load_project(pid)
                        p2.setdefault("carousel", []).append(
                            {"file": out, "name": "Carousel %ss" % secs,
                             "kind": "carousel", "dur": secs})
                        save_project(p2)
                        video = "/media/%s/carousel/%s" % (pid, base)
                    else:
                        video = "/media/carousel/%s" % base
                    return {"video": video, "file": out, "secs": secs,
                            "images": res["images"], "ratio": res.get("ratio")}
                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/fast-cut":
                """快剪模版（Fast Cut）—— N 张请柬图 → 节拍对齐的等间隔快剪 + 持久斜体字幕。

                与 /api/four-grid（Grid Showcase）、/api/scroll-gallery（Colour Inspiration）、
                /api/multi-suite（Collection Showcase）**完全独立**：管线在 core.fastcut
                （每张图出场 2~3 次（≤6张3次、≥7张2次），取景只保留
                全景(1.00)+中景(1.10)两档、全部居中裁切，去掉旧版上/下锚点漂移
                与特写推近；每镜 1 个音乐拍长 0.592s，镜内完全静止，硬切；
                持久斜体白字盖 44% 屏高；原图直出无滤镜调色），
                不动其它模版的编排。

                入参(JSON)：srcs 图片路径列表(必填，3~12 张) /
                text 持久斜体字幕(可选，默认 "the details, the day"；step4「Add Captions」
                开启且输入框留空时用默认句) /
                show_text false 则完全不叠字幕(默认 false＝opt-in；step4「Add Captions」关闭时传 false) /
                project 项目id(可选) / ratio 画幅(默认 9x16) / fps(默认 30) /
                silent true 则跳过配乐（默认配 cheery_monday 轻快向 BGM，用户 2026-10-21 指定） /
                voiceover {on, selling, cta} 三段式口播(可选，step4 勾选)：on=true 时
                   BGM 后追加三段式英文配音+字幕（按成片时长由 three_act_beats 现算），
                   口播字幕烧在画面中下（sub_pos="lower"，不与 44% 屏高的持久字幕叠）；
                   selling 空则 AI 兜底；cta 三选一。失败不影响交付。
                返回：异步 job，result 含 {video, file, secs, images, ratio,
                captions(布尔), voiceover(布尔)}；secs 随张数自动增加
                （appear·N × 0.592s，appear 动态 2~3：≤6张3次、≥7张2次）。
                """
                body = self.read_body_json()
                pid = body.get("project")
                srcs = [s for s in (body.get("srcs") or []) if s]
                srcs = [s for s in srcs if os.path.isfile(s)]
                if len(srcs) < 5:
                    return self.send_json(
                        {"error": "This template needs 5 to 12 photos — upload at least 5 photos first"}, 400)
                if len(srcs) > 12:
                    return self.send_json(
                        {"error": "This template takes up to 12 photos — please remove a few and try again"}, 400)
                from core.card_pan import norm_ratio as _norm_ratio
                ratio = _norm_ratio(body.get("ratio"))
                fps = int(body.get("fps") or 30)
                show_text = body.get("show_text", False)
                if isinstance(show_text, str):
                    show_text = show_text.lower() not in ("0", "false", "no")
                from core.fastcut import DEFAULT_TEXT as _FC_DEFAULT
                text = str(body.get("text") or "").strip() or (_FC_DEFAULT if show_text else "")
                jid = new_job("fast_cut")

                def work():
                    from core import fastcut as fc
                    out_dir = os.path.join(project_path(pid) if pid else DATA, "fastcut")
                    os.makedirs(out_dir, exist_ok=True)
                    stamp = time.strftime("%m%d%H%M%S") + uuid.uuid4().hex[:3]
                    out = os.path.join(out_dir, "fastcut_%s_%s.mp4" % (ratio, stamp))

                    def prog(done, total_, msg=""):
                        job_progress(jid, int(done), int(total_ or 100), msg)

                    res = fc.render_fastcut(srcs, out, ratio=ratio, fps=fps,
                                            text=text, show_text=show_text,
                                            progress=prog, font=body.get("font"))
                    secs = res["secs"]
                    _bf = None
                    # BGM：用户 2026-10-21 指定 cheery_monday（轻快卡点）；失败不影响画面交付。
                    if not body.get("silent"):
                        try:
                            _bf = _bgm_pick(body, "cheery_monday.mp3")
                            if os.path.isfile(_bf):
                                _mix = out + ".mix.mp4"
                                job_progress(jid, 92, 100, "Adding music")
                                compose.mix_bgm(out, _bf, _mix, gain_db=-4.0,
                                                fade_in=0.8, fade_out=1.6)
                                if os.path.isfile(_mix) and os.path.getsize(_mix) > 1024:
                                    os.replace(_mix, out)
                        except BaseException:
                            traceback.print_exc()
                    # 口播后处理（step4 Captions & Voiceover 可选）：三段式配音+字幕。
                    # 持久字幕是本模版渲染器原生（44% 屏高）；口播字幕烧中下
                    # （sub_pos="lower"，口播字幕四个模版一律中下，用户 2026-10-20 定稿），
                    # 两层文字不重叠。任何失败只打印，仍交付无口播成片。
                    _vo = body.get("voiceover") or {}
                    vo_meta = None
                    if _vo.get("on"):
                        try:
                            from core import voiceover as _volib
                            _cfg = dict(CONFIG.get("storyboard_api") or {})

                            def _vprog(done, total_, msg=""):
                                job_progress(jid, 92 + int(done) * 8 // max(1, int(total_ or 100)),
                                             100, msg or "Adding voiceover")

                            _vo_out = out + ".vo.mp4"
                            vo_meta = _volib.voiceover_on_rendered(
                                out, _vo_out, srcs, 0.0, secs, secs, [],
                                str(_vo.get("selling") or "").strip(),
                                str(_vo.get("cta") or "view").strip(),
                                _cfg, progress=_vprog, fallback_bgm=_bf,
                                beats=_volib.three_act_beats(secs),
                                sub_pos="lower")
                            if os.path.isfile(_vo_out) and os.path.getsize(_vo_out) > 1024:
                                os.replace(_vo_out, out)
                        except BaseException:
                            traceback.print_exc()
                    base = os.path.basename(out)
                    if pid:
                        p2 = load_project(pid)
                        p2.setdefault("fastcut", []).append(
                            {"file": out, "name": "Fast Cut %ss" % secs,
                             "kind": "fastcut", "dur": secs})
                        save_project(p2)
                        video = "/media/%s/fastcut/%s" % (pid, base)
                    else:
                        video = "/media/fastcut/%s" % base
                    return {"video": video, "file": out, "secs": secs,
                            "images": res["images"], "ratio": res.get("ratio"),
                            "captions": bool(show_text and text.strip()),
                            "voiceover": bool(vo_meta)}
                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/mix-cut":
                g = self._ai_gate("caption")
                if g:
                    return self.send_json({"error": g["error"]}, g["code"])
                """电影感运镜模版（Cinematic Cut）—— 第 6 个模版（用户 2026-10-22）。

                N(6~12) 张请柬图 → 卡 BGM 鼓点的快切 + 旋转/平移混排：
                  第 1→2 幕 whoosh（急推变焦），切点吸附鼓点（运行时 onset 检测）；
                  全片仅 2~3 个顺时针旋转镜（11~15°，整图刚性 roll，等效小幅环绕）；
                  其余镜头四向平移（上/下/左/右轮换，相邻方向必不同）；
                  其余切点交叉溶解；满幅 cover 不出黑边；原图直出无滤镜。
                BGM 固定 cinematic_wedding.mp3（与模版库封面 tpl_cinematic.mp4
                同一支曲——2026-10-27 音轨指纹比对 0.82 实锤，用户要求「生成视频
                的 BGM 要和模版一样」），失败回退 dreams_become_real；
                （注：此前端点误用 dreams_become_real，与封面听感不一致，已纠正）
                成片长于 BGM 原长时 BGM 自动循环补齐。
                成片总时长随图片数量动态变化（4.8 + 0.56×N 秒，2026-10-22 用户拍板）。

                入参(JSON)：srcs 图片路径(必填 6~12) / project / ratio(默认 9x16) / fps(默认 30) /
                show_text 持久斜体字幕开关(默认 false＝opt-in；step4「Add Captions」) /
                text 字幕内容(可选，留空用默认句 "the details, the day") /
                voiceover {on, selling, cta} 三段式口播(可选，烧中下不与持久字幕叠)。
                返回异步 job，result 含 {video, file, secs, images, ratio, shots,
                whoosh_cut, rot_shots, pan_dirs, captions(布尔), voiceover(布尔)}。
                """
                body = self.read_body_json()
                pid = body.get("project")
                srcs = [s for s in (body.get("srcs") or []) if s]
                srcs = [s for s in srcs if os.path.isfile(s)]
                if len(srcs) < 6:
                    return self.send_json(
                        {"error": "This template needs 6 to 12 photos — upload at least 6 photos first"}, 400)
                if len(srcs) > 12:
                    return self.send_json(
                        {"error": "This template takes up to 12 photos — please remove a few and try again"}, 400)
                from core.card_pan import norm_ratio as _norm_ratio
                ratio = _norm_ratio(body.get("ratio"))
                fps = int(body.get("fps") or 30)
                show_text = body.get("show_text", False)
                if isinstance(show_text, str):
                    show_text = show_text.lower() not in ("0", "false", "no")
                from core.fastcut import DEFAULT_TEXT as _CIN_DEFAULT
                text = str(body.get("text") or "").strip() or (_CIN_DEFAULT if show_text else "")
                jid = new_job("cinematic_cut")

                def work():
                    from core import cinematic as ci
                    out_dir = os.path.join(project_path(pid) if pid else DATA, "cinematic")
                    os.makedirs(out_dir, exist_ok=True)
                    stamp = time.strftime("%m%d%H%M%S") + uuid.uuid4().hex[:3]
                    out = os.path.join(out_dir, "cinematic_%s_%s.mp4" % (ratio, stamp))

                    def prog(done, total_, msg=""):
                        job_progress(jid, int(done), int(total_ or 100), msg)

                    # 2026-10-27：无 BGM 选项（silent=true）——cinematic 的卡点依赖
                    # BGM 节拍（render_cinematic 必须有可测节拍的音轨，静音/None 都会报
                    # "no beats detected"/"bgm missing"）。所以仍用模版默认证曲做节拍与
                    # 渲染，保证卡点节奏与有 BGM 时完全一致，渲染完成后再整体剥掉音轨。
                    _silent = bool(body.get("silent"))
                    if _silent:
                        _bf = os.path.join(BGM_DIR, "cinematic_wedding.mp3")
                        if not os.path.isfile(_bf):
                            _bf = os.path.join(BGM_DIR, "dreams_become_real.mp3")
                    else:
                        _bf = _bgm_pick(body, "cinematic_wedding.mp3")
                        if not os.path.isfile(_bf):
                            _bf = os.path.join(BGM_DIR, "dreams_become_real.mp3")
                    res = ci.render_cinematic(srcs, out, ratio=ratio, fps=fps,
                                              bgm_path=_bf, progress=prog,
                                              text=text, show_text=show_text,
                                              font=body.get("font"))
                    secs = res["secs"]
                    if _silent:
                        # 剥音轨（-an + 视频流直拷，不重编码）
                        try:
                            _noaud = out + ".noaud.mp4"
                            job_progress(jid, 88, 100, "Removing music track")
                            from core.ffproc import run_ffmpeg as _rff2, _default_ff as _ffexe2
                            rc2, _e2 = _rff2([_ffexe2(), "-y", "-v", "error", "-i", out,
                                              "-an", "-c:v", "copy", _noaud])
                            if rc2 == 0 and os.path.isfile(_noaud) and os.path.getsize(_noaud) > 1024:
                                os.replace(_noaud, out)
                        except BaseException:
                            traceback.print_exc()
                    # BGM 循环/裁切到与成片等长（成片时长随图片数量动态变化）
                    _bgm_use = None if _silent else _bf
                    # 无 BGM 时绝不能进这段：否则会把 _bgm_use 重建回模版曲、又混一遍音轨
                    if not _silent and _bf and os.path.isfile(_bf):
                        try:
                            import tempfile as _tmpmod
                            _td = _tmpmod.mkdtemp(prefix="cinebgm_")
                            _bgm_use = ci.build_bgm(_bf, secs,
                                                    os.path.join(_td, "bgm_loop.wav"))
                        except BaseException:
                            traceback.print_exc()
                            _bgm_use = _bf
                    if _bgm_use and os.path.isfile(_bgm_use):
                        try:
                            _mix = out + ".mix.mp4"
                            job_progress(jid, 90, 100, "Adding music")
                            compose.mix_bgm(out, _bgm_use, _mix, gain_db=-4.0,
                                            fade_in=0.8, fade_out=1.6)
                            if os.path.isfile(_mix) and os.path.getsize(_mix) > 1024:
                                os.replace(_mix, out)
                        except BaseException:
                            traceback.print_exc()
                    # 口播后处理（step4 Add AI Voiceover 可选）：三段式配音+字幕，
                    # 烧中下（sub_pos="lower"），不与 44% 屏高的持久字幕叠。失败不影响交付。
                    _vo = body.get("voiceover") or {}
                    vo_meta = None
                    if _vo.get("on"):
                        try:
                            from core import voiceover as _volib
                            _cfg = dict(CONFIG.get("storyboard_api") or {})

                            def _vprog(done, total_, msg=""):
                                job_progress(jid, 90 + int(done) * 9 // max(1, int(total_ or 100)),
                                             100, msg or "Adding voiceover")

                            _vo_out = out + ".vo.mp4"
                            vo_meta = _volib.voiceover_on_rendered(
                                out, _vo_out, srcs, 0.0, secs, secs, [],
                                str(_vo.get("selling") or "").strip(),
                                str(_vo.get("cta") or "view").strip(),
                                _cfg, progress=_vprog, fallback_bgm=_bf,
                                beats=_volib.three_act_beats(secs),
                                sub_pos="lower")
                            if os.path.isfile(_vo_out) and os.path.getsize(_vo_out) > 1024:
                                os.replace(_vo_out, out)
                        except BaseException:
                            traceback.print_exc()
                    base = os.path.basename(out)
                    if pid:
                        p2 = load_project(pid)
                        p2.setdefault("cinematic", []).append(
                            {"file": out, "name": "Cinematic Cut %ss" % secs,
                             "kind": "cinematic", "dur": secs})
                        save_project(p2)
                        video = "/media/%s/cinematic/%s" % (pid, base)
                    else:
                        video = "/media/cinematic/%s" % base
                    return {"video": video, "file": out, "secs": secs,
                            "images": res["images"], "ratio": res.get("ratio"),
                            "shots": res.get("shots"),
                            "whoosh_cut": res.get("whoosh_cut"),
                            "rot_shots": res.get("rot_shots"),
                            "pan_dirs": res.get("pan_dirs"),
                            "captions": bool(show_text and text.strip()),
                            "voiceover": bool(vo_meta)}
                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/recognize-labels":
                g = self._ai_gate("vision")
                if g:
                    return self.send_json({"error": g["error"]}, g["code"])
                """多图合集模版：上传后自动识别每张图的款式名牌（用户 2026-10-15「两者结合」）。

                优先提取图片上**印有的文字**（款式名 / 品牌名），识别不到再让视觉模型
                按画面语义生成一个简短英文标签，保证每段都至少有一个可读名牌。
                复用 core.showcase.chat_vision（与卡片框检测同一套 key / 模型，零新增依赖）。

                入参(JSON)：srcs 图片路径列表(必填，≤8 张)
                返回：异步 job，result 含 {labels:[..], shots:[..], focuses:[..]}
                      （均与 srcs 等长；labels 空串=该图未识别到；shots 为
                       'detail'/'full'/'rotate'/'auto'；focuses 为 detail 的焦点框）
                """
                body = self.read_body_json()
                srcs = [str(x) for x in (body.get("srcs") or []) if x][:8]
                if not srcs:
                    return self.send_json({"error": "No photos to recognize"}, 400)
                # 浏览器客户端只能识别自己 sid 项目下的图片
                if not all(self._owns_path(s) for s in srcs):
                    return self.send_json({"error": "forbidden"}, 403)
                jid = new_job("recognize_labels")

                def work():
                    from concurrent.futures import ThreadPoolExecutor

                    from core import recog_label
                    cfg = dict(CONFIG.get("storyboard_api") or {})
                    n = len(srcs)

                    # 并行识别（2026-10-15）：串行时每张 8~12s，2 张要 24s+、8 张
                    # 可能超过 1.5 分钟，前端名牌框长时间空白像「功能没上线」。
                    # 视觉调用是纯 IO，线程池并行后总耗时 ≈ 最慢一张（~12s）。
                    def one(src):
                        try:
                            return recog_label.analyze_one(cfg, src)
                        except BaseException:
                            return {"label": "", "shot": "auto", "focus": None}

                    done = {"k": 0}
                    with ThreadPoolExecutor(max_workers=min(4, n)) as pool:
                        futs = [pool.submit(one, s_) for s_ in srcs]
                        labels, shots, focuses = [], [], []
                        for f in futs:
                            _r = f.result() or {}
                            labels.append(str(_r.get("label") or ""))
                            shots.append(str(_r.get("shot") or "auto"))
                            focuses.append(_r.get("focus"))
                            done["k"] += 1
                            try:
                                job_progress(jid, int(100 * done["k"] / n),
                                             msg="Reading design %d/%d" % (done["k"], n))
                            except Exception:
                                pass
                    return {"labels": labels, "shots": shots, "focuses": focuses}
                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/handheld/job":
                """创建手持视频生成任务单（供外部工具 / Agent 接手执行）"""
                body = self.read_body_json()
                proj = load_project(body["project"])
                img = (body.get("image") or "").strip()
                if not img or img == "auto":
                    img, _w, _h = pick_first_frame(proj)
                    if not img:
                        return self.send_json(
                            {"error": "No usable assets in this project — go back to step 1 and upload product photos"}, 400)
                job = {"id": uuid.uuid4().hex[:10], "image": img,
                       "last_image": (body.get("last_image") or "").strip() or None,
                       "prompt": body["prompt"], "duration": body.get("duration", 5),
                       "status": "pending", "created": time.time(),
                       "video": None}
                d = os.path.join(project_path(proj["id"]), "handheld_jobs")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "job_%s.json" % job["id"]), "w",
                          encoding="utf-8") as f:
                    json.dump(job, f, ensure_ascii=False, indent=2)
                proj.setdefault("handheld_jobs", []).append(job)
                save_project(proj)
                return self.send_json({"job": job, "project": proj})

            if p == "/api/handheld/video":
                """上传已生成的手持视频（外部工具/Agent 产出）"""
                pid = q["project"][0]
                name = safe_name(q.get("name", ["handheld.mp4"])[0])
                d = os.path.join(project_path(pid), "handheld")
                os.makedirs(d, exist_ok=True)
                stem, ext = os.path.splitext(name)
                fp = os.path.join(d, name)
                i = 1
                while os.path.exists(fp):
                    fp = os.path.join(d, "%s_%d%s" % (stem, i, ext)); i += 1
                data = self.read_body_bytes()
                with open(fp, "wb") as f:
                    f.write(data)
                proj = load_project(pid)
                entry = {"path": fp, "file": os.path.basename(fp)}
                proj["handheld_videos"].append(entry)
                jid = q.get("job_id", [None])[0]
                if jid:
                    for j in proj.get("handheld_jobs", []):
                        if j["id"] == jid:
                            j["status"] = "done"; j["video"] = entry
                    jd = os.path.join(project_path(pid), "handheld_jobs", "job_%s.json" % jid)
                    if os.path.isfile(jd):
                        with open(jd, "r", encoding="utf-8") as f:
                            jj = json.load(f)
                        jj["status"] = "done"; jj["video"] = entry
                        with open(jd, "w", encoding="utf-8") as f:
                            json.dump(jj, f, ensure_ascii=False, indent=2)
                save_project(proj)
                return self.send_json({"entry": entry, "project": proj})

            if p == "/api/videoapi/test":
                """测试火山方舟凭据可用性（只读调用，不消耗生成额度）"""
                body = self.read_body_json()
                cfg = dict(CONFIG.get("video_api") or {})
                if body.get("api_key"):
                    cfg["api_key"] = body["api_key"].strip()
                if body.get("endpoint"):
                    cfg["endpoint"] = body["endpoint"].strip()
                if not videoapi.is_ready(cfg):
                    return self.send_json({"error": "Enter ARK_API_KEY first"}, 400)
                try:
                    j = videoapi.list_tasks(cfg, page_size=3)
                    items = j.get("items", [])
                    return self.send_json({
                        "ok": True,
                        "base": videoapi.api_base(cfg),
                        "model": videoapi.model_id(cfg),
                        "recent_tasks": [{"id": t.get("id"), "status": t.get("status")}
                                         for t in items]})
                except Exception as e:
                    return self.send_json({"ok": False, "error": str(e)}, 400)

            if p == "/api/imageapi/test":
                """核对 Seedream 凭据与模型开通状态（零成本：只读 /models 列表）"""
                body = self.read_body_json()
                cfg = dict(CONFIG.get("storyboard_api") or {})
                if body.get("api_key"): cfg["api_key"] = body["api_key"].strip()
                if body.get("endpoint"): cfg["endpoint"] = body["endpoint"].strip()
                if body.get("model"): cfg["model"] = body["model"].strip()
                if not imgapi.is_ready(cfg):
                    return self.send_json({"error": "Enter ARK_API_KEY first"}, 400)
                mid = imgapi.model_id(cfg)
                vcfg = dict(CONFIG.get("video_api") or {})
                vcfg["api_key"] = cfg["api_key"]
                vcfg["endpoint"] = cfg.get("endpoint", "")
                vmid = videoapi.model_id(vcfg)
                out = {"base": imgapi.api_base(cfg), "model": mid, "video_model": vmid,
                       "sizes": imgapi.meta(mid)["sizes"]}
                try:
                    out["image_model_check"] = imgapi.check_model(cfg, mid)
                    out["video_model_check"] = imgapi.check_model(cfg, vmid)
                except Exception as e:
                    # /models 不可用时退回非法模型名探测（同样是零成本）
                    try:
                        pr = imgapi.probe(cfg)
                        out["image_model_check"] = {"found": None, "status": "unknown",
                                                    "warning": pr.get("detail", "")}
                    except Exception:
                        return self.send_json({"ok": False, "error": str(e), **out}, 400)
                chk = out["image_model_check"]
                out["ok"] = bool(chk.get("found")) or chk.get("found") is None
                warn = [c.get("warning") for c in (out["image_model_check"],
                                                   out["video_model_check"]) if c.get("warning")]
                out["detail"] = ("；".join(warn) if warn
                                 else "Both the storyboard and video models are enabled and callable")
                if not out["ok"]:
                    return self.send_json(out, 400)
                try:
                    t = videoapi.list_tasks(vcfg, page_size=1)
                    out["recent_tasks"] = len(t.get("items", []))
                except Exception as e:
                    out["credential_warn"] = str(e)
                return self.send_json(out)

            if p == "/api/storyboard/generate":
                g = self._ai_gate("image")
                if g:
                    return self.send_json({"error": g["error"]}, g["code"])
                """Seedream 组图直连：产品图 + 指令词 → 多张手持展示分镜图（4K）"""
                body = self.read_body_json()
                cfg = dict(CONFIG.get("storyboard_api") or {})
                if not imgapi.is_ready(cfg):
                    return self.send_json(
                        {"error": "ARK_API_KEY is not configured. Go to Settings → Storyboard API and enter it."}, 400)
                proj = load_project(body["project"])
                refs = body.get("images") or (
                    [body["image"]] if body.get("image") else [])
                if not refs:
                    # 默认取项目内所有「质检合格或未质检」的上传图，最多 3 张
                    refs = [e["path"] for e in (proj.get("uploads") or [])][:3]
                refs = [p for p in refs if p and os.path.isfile(p)]
                if not refs:
                    return self.send_json({"error": "Upload product photos in step 1 to use as references first"}, 400)
                prompt = (body.get("prompt") or "").strip() or prompts.STORYBOARD_HANDHELD
                # 拍摄清单（按产品结构自定义镜头），附加在官方指令之后
                shot_list = (body.get("shot_list") or "").strip()
                if shot_list:
                    prompt = prompt + "\n\n【本产品拍摄清单】\n" + shot_list
                size = body.get("size") or cfg.get("size") or "4K"
                max_images = int(body.get("max_images") or cfg.get("max_images") or 3)
                sequential = body.get("sequential") or cfg.get("sequential") or "auto"
                out_fmt = body.get("output_format") or cfg.get("output_format") or "jpeg"
                watermark = bool(body.get("watermark", cfg.get("watermark", False)))
                optimize = body.get("optimize_prompt", cfg.get("optimize_prompt"))
                out_dir = os.path.join(project_path(proj["id"]), "storyboards")
                jid = new_job("storyboard_gen")

                def work():
                    res = imgapi.generate(
                        cfg, out_dir, prompt, ref_images=refs, size=size,
                        max_images=max_images, sequential=sequential,
                        output_format=out_fmt, watermark=watermark,
                        optimize_prompt=optimize, tag=time.strftime("storyboard_%m%d%H%M%S"),
                        progress=lambda pct, msg: job_progress(jid, pct, 100, msg))
                    p2 = load_project(proj["id"])
                    entries = []
                    for fp in res["files"]:
                        e = {"path": fp, "file": os.path.basename(fp),
                             "kind": "ai_seedream", "model": res["model"],
                             "size": res["size"], "created": time.time()}
                        p2.setdefault("storyboards", []).append(e)
                        entries.append(e)
                    p2.setdefault("ai_generations", []).append(
                        {"kind": "storyboard", "model": res["model"], "size": res["size"],
                         "prompt": prompt, "refs": refs, "count": res["count"],
                         "files": res["files"], "urls": res["urls"],
                         "elapsed": res["elapsed"], "created": time.time()})
                    save_project(p2)
                    return {"entries": entries, "count": res["count"],
                            "model": res["model"], "size": res["size"],
                            "elapsed": res["elapsed"], "note": res["note"],
                            "thumbs": ["/media/%s/storyboards/%s"
                                       % (proj["id"], os.path.basename(f))
                                       for f in res["files"]]}

                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/handheld/generate":
                g = self._ai_gate("video")
                if g:
                    return self.send_json({"error": g["error"]}, g["code"])
                """火山方舟图生视频直连：首帧（上传播图自动识别 / 指定图）+ 指令词 → 手持展示视频"""
                body = self.read_body_json()
                cfg = dict(CONFIG.get("video_api") or {})
                if not videoapi.is_ready(cfg):
                    return self.send_json(
                        {"error": "Volcengine Ark API key is not configured. Go to Settings → Video generation API and enter ARK_API_KEY."}, 400)
                proj = load_project(body["project"])
                image = (body.get("image") or "").strip()
                auto_picked = False
                if not image or image == "auto":
                    # 现阶段口径：首帧直接取上传的产品实拍图，自动识别分辨率最高的一张
                    image, _w, _h = pick_first_frame(proj)
                    auto_picked = True
                    if not image:
                        return self.send_json(
                            {"error": "No usable assets in this project — go back to step 1 and upload product photos to use as the first frame"}, 400)
                elif not os.path.isfile(image):
                    return self.send_json({"error": "Selected first-frame file not found: %s" % image}, 400)
                # 尾帧（可选）：首尾帧模式，视频从首帧自然过渡到尾帧
                last_image = (body.get("last_image") or "").strip()
                last_auto = False
                if last_image in ("", "none", "None"):
                    last_image = None
                elif last_image == "auto":
                    last_image, _w, _h = pick_last_frame(proj, image)
                    last_auto = True
                    if not last_image:
                        return self.send_json(
                            {"error": "First/last-frame mode needs two different assets — no image other than the first frame exists in this project"}, 400)
                elif not os.path.isfile(last_image):
                    return self.send_json({"error": "Selected last-frame file not found: %s" % last_image}, 400)
                prompt = (body.get("prompt") or "").strip()
                if not prompt:
                    return self.send_json({"error": "The prompt cannot be empty"}, 400)
                duration = int(body.get("duration") or 5)
                duration = max(2, min(30, duration))
                resolution = body.get("resolution") or cfg.get("resolution") or "720p"
                ratio = body.get("ratio") or cfg.get("ratio") or "adaptive"
                gen_audio = bool(body.get("generate_audio", cfg.get("generate_audio", False)))
                watermark = bool(body.get("watermark", cfg.get("watermark", False)))
                camera_fixed = bool(body.get("camera_fixed", cfg.get("camera_fixed", False)))
                out_dir = os.path.join(project_path(proj["id"]), "handheld")
                jid = new_job("handheld_gen")

                def work():
                    res = videoapi.generate(
                        cfg, image, out_dir, prompt,
                        duration=duration, resolution=resolution, ratio=ratio,
                        generate_audio=gen_audio, watermark=watermark,
                        camera_fixed=camera_fixed,
                        last_frame_path=last_image,
                        progress=lambda pct, msg: job_progress(jid, pct, 100, msg))
                    entry = {"path": res["file"], "file": os.path.basename(res["file"]),
                             "kind": "ai_ark", "task_id": res["task_id"],
                             "duration": res["duration"], "resolution": res["resolution"],
                             "model": res["model"]}
                    p2 = load_project(proj["id"])
                    p2.setdefault("handheld_videos", []).append(entry)
                    lj = body.get("job_id")
                    if lj:
                        for j in p2.get("handheld_jobs", []):
                            if j["id"] == lj:
                                j["status"] = "done"
                                j["video"] = entry
                    p2.setdefault("ai_generations", []).append(
                        {"task_id": res["task_id"], "file": res["file"],
                         "prompt": prompt, "duration": res["duration"],
                         "resolution": res["resolution"], "model": res["model"],
                         "video_url": res["video_url"], "elapsed": res["elapsed"],
                         "created": time.time(), "image": image})
                    save_project(p2)
                    return {"video": "/media/%s/handheld/%s" % (proj["id"], os.path.basename(res["file"])),
                            "file": res["file"], "entry": entry,
                            "task_id": res["task_id"], "elapsed": res["elapsed"],
                            "duration": res["duration"], "resolution": res["resolution"],
                            "model": res["model"], "image_note": res.get("image_note", ""),
                            "picked_file": os.path.basename(image),
                            "auto_picked": auto_picked,
                            "last_frame": (os.path.basename(last_image) if last_image else None),
                            "last_auto": last_auto,
                            "frames_mode": res.get("frames", "first"),
                            "param_notes": res.get("param_notes", [])}

                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/showcase/plan":
                """脚本规划层：视觉大模型识别整套图组件并排序（快速 15s / 深度 ~100s）"""
                body = self.read_body_json()
                proj = load_project(body["project"])
                src = (body.get("src") or "").strip()
                if not src or not os.path.isfile(src):
                    return self.send_json({"error": "The whole-set product image is missing or not found — select a source image first"}, 400)
                depth = body.get("depth") or "fast"
                if depth not in showcase.PLANNER_MODELS:
                    return self.send_json(
                        {"error": "Unknown planning depth: %s (available: %s)" % (depth, "、".join(showcase.PLANNER_MODELS))}, 400)
                cfg = dict(CONFIG.get("storyboard_api") or {})
                cfg["api_key"] = cfg.get("api_key") or (CONFIG.get("video_api") or {}).get("api_key") or ""
                jid = new_job("showcase_plan")
                def work():
                    plan = showcase.plan_components(src, cfg, depth=depth)
                    p2 = load_project(proj["id"])
                    p2["showcase"] = plan
                    save_project(p2)
                    return {"plan": plan,
                            "summary": "Identified %d component(s) · quality %s%s" % (
                                len(plan["components"]), plan["image_quality"],
                                " · %d note(s)" % len(plan["issues"]) if plan["issues"] else "")}
                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/showcase/update":
                """保存用户编辑后的分段计划（顺序/时长/剔除）"""
                body = self.read_body_json()
                proj = load_project(body["project"])
                sc = proj.get("showcase")
                if not sc:
                    # 允许先建计划骨架（外部工具 / 前端注入计划时使用）
                    sc = {"src": (body.get("src") or "").strip(),
                          "scene": body.get("scene") or "",
                          "image_quality": body.get("image_quality") or "unknown",
                          "issues": [], "time": time.time()}
                    proj["showcase"] = sc
                comps = body.get("components")
                if not isinstance(comps, list) or not comps:
                    return self.send_json({"error": "components cannot be empty"}, 400)
                for i, c in enumerate(comps):
                    c["order"] = i + 1
                    try:
                        c["dur"] = max(1.5, min(4.0, float(c.get("dur") or 2.5)))
                    except Exception:
                        c["dur"] = 2.5
                sc["components"] = comps
                save_project(proj)
                return self.send_json({"plan": sc, "project": proj})

            if p == "/api/showcase/generate":
                """逐组件分段生成：每组件一次 Seedance 调用（4s）→ 本地裁剪到目标时长"""
                body = self.read_body_json()
                cfg = dict(CONFIG.get("video_api") or {})
                if not videoapi.is_ready(cfg):
                    return self.send_json(
                        {"error": "Volcengine Ark API key is not configured. Go to Settings → Video generation API and enter ARK_API_KEY."}, 400)
                proj = load_project(body["project"])
                sc = proj.get("showcase")
                if not sc or not sc.get("components"):
                    return self.send_json({"error": "No component plan yet — run \"Identify components & order\" first"}, 400)
                src = (body.get("src") or sc.get("src") or "").strip()
                if not src or not os.path.isfile(src):
                    return self.send_json({"error": "The set image does not exist — choose the source image again"}, 400)
                sel_ids = set(body.get("ids") or [])
                comps = [c for c in sc["components"]
                         if c.get("include", True) and (not sel_ids or c["id"] in sel_ids)]
                comps.sort(key=lambda c: c.get("order", 99))
                if not comps:
                    return self.send_json({"error": "No component clips are selected"}, 400)
                resolution = body.get("resolution") or "720p"
                gen_audio = bool(body.get("generate_audio", False))
                out_dir = os.path.join(project_path(proj["id"]), "handheld")
                os.makedirs(out_dir, exist_ok=True)
                jid = new_job("showcase_gen")
                def work():
                    stamp = time.strftime("%m%d%H%M%S")
                    entries, files = [], []
                    total = len(comps)
                    for i, c in enumerate(comps):
                        base_pct = int(100 * i / total)
                        seg_pct = dict()

                        def prog(pct, msg, i=i, base_pct=base_pct, total=total):
                            job_progress(jid, base_pct + int((100 - base_pct) * pct / 100.0), 100,
                                         "Component %d/%d %s: %s" % (i + 1, total, c["label_zh"], msg))

                        prompt = showcase.segment_prompt(c, sc.get("scene") or "")
                        res = videoapi.generate(
                            cfg, src, out_dir, prompt,
                            duration=4, resolution=resolution, ratio="9:16",
                            generate_audio=gen_audio, watermark=False,
                            progress=prog)
                        prog(95, "Trimming to %.1fs…" % c["dur"])
                        final = os.path.join(out_dir, "clip_%02d_%s_%ss_%s.mp4" % (
                            c["order"], re.sub(r"[\\\\/:*?\"<>|]", "", c["label_zh"])[:12],
                            round(c["dur"], 1), stamp))
                        showcase.trim_keep_tail(res["file"], final, c["dur"])
                        if os.path.abspath(final) != os.path.abspath(res["file"]):
                            safedel.unlink(res["file"])   # 钩子拒绝也不影响任务完成
                        entry = {"path": final, "file": os.path.basename(final),
                                 "kind": "showcase_seg", "name": "%02d %s %.1fs" % (
                                     c["order"], c["label_zh"], round(c["dur"], 1)),
                                 "order": c["order"], "comp": c["label_zh"],
                                 "comp_id": c["id"], "dur": round(c["dur"], 1),
                                 "resolution": resolution, "model": res["model"],
                                 "task_id": res["task_id"], "created": time.time()}
                        entries.append(entry)
                        files.append(final)
                        p2 = load_project(proj["id"])
                        p2.setdefault("handheld_videos", []).append(entry)
                        save_project(p2)
                    check = showcase.check_segments(files)
                    return {"segments": entries, "check": check,
                            "total": len(entries),
                            "summary": "%d clip(s) generated and saved to the library (visible in the step 4 stitch pool)" % len(entries)}
                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/showcase/hero":
                """Hero 收尾：整套图几何运镜（CM-10 固定+微推 / CM-05 缓拉），零 AI 费用"""
                body = self.read_body_json()
                proj = load_project(body["project"])
                sc = proj.get("showcase") or {}
                src = (body.get("src") or sc.get("src") or "").strip()
                if not src or not os.path.isfile(src):
                    return self.send_json({"error": "The set image does not exist — choose a source image in the identification step first"}, 400)
                move = body.get("move") or "cm10"
                if move not in cammove.MOVES:
                    return self.send_json({"error": "Unknown Hero camera move: %s (available: cm10 / cm05)" % move}, 400)
                secs = float(body.get("secs") or 2.5)
                out_dir = os.path.join(project_path(proj["id"]), "handheld")
                os.makedirs(out_dir, exist_ok=True)
                out = os.path.join(out_dir, "Hero_%s_%ss_%s.mp4" % (
                    move.upper(), round(secs, 1), time.strftime("%m%d%H%M%S")))
                jid = new_job("showcase_hero")
                def work():
                    el, info, meta = cammove.render_cameramove(
                        src, out, move, None,
                        progress=lambda d, t: job_progress(jid, d, t, "Rendering Hero"),
                        secs=secs)
                    entry = {"path": out, "file": os.path.basename(out),
                             "kind": "showcase_hero", "name": "Hero %s %.1fs" % (
                                 meta.get("label", move.upper()), round(meta.get("secs", secs), 1)),
                             "order": 99, "dur": meta.get("secs", secs),
                             "layout": meta.get("layout")}
                    p2 = load_project(proj["id"])
                    p2.setdefault("handheld_videos", []).append(entry)
                    save_project(p2)
                    return {"entry": entry, "elapsed": round(el, 1),
                            "summary": "Hero ending saved to the library: %s" % entry["name"]}
                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/compose":
                body = self.read_body_json()
                jid = new_job("compose")
                proj = load_project(body["project"])
                pdir = project_path(proj["id"])
                # 路径归一化：兼容绝对路径 / 裸文件名（手持段历史条目）
                segments = []
                for s in body["segments"]:
                    if os.path.isabs(s) and os.path.isfile(s):
                        segments.append(s)
                        continue
                    found = None
                    for sub in ("handheld", "cameramove", ""):
                        cand = os.path.join(pdir, sub, s) if sub else os.path.join(pdir, s)
                        if os.path.isfile(cand):
                            found = cand
                            break
                    if not found:
                        return self.send_json(
                            {"error": "Video segment file not found: %s (searched all project subfolders)" % s}, 400)
                    segments.append(found)
                bgm = body.get("bgm")
                if not segments:
                    return self.send_json({"error": "No video segments to compose"}, 400)
                # 目标画幅：显式 ratio（前端所选）→ RATIOS；缺省由 compose 按首段自动判定
                _r = str(body.get("ratio") or "").strip()
                comp_size = cammove.RATIOS.get(_r) if _r else None
                out = os.path.join(project_path(proj["id"]),
                                   "final_cut_%s.mp4" % time.strftime("%m%d%H%M%S"))
                def work():
                    if bgm:
                        compose.export_final(
                            segments, bgm, out, size=comp_size,
                            progress=lambda d, t: job_progress(jid, d, t, "Composing"),
                            gain_db=body.get("gain_db", -3.0),
                            fade_in=body.get("fade_in", 1.2),
                            fade_out=body.get("fade_out", 2.6))
                    else:
                        job_progress(jid, 10, 100, "Stitching (no BGM)")
                        compose.concat_segments(segments, out, size=comp_size)
                    return {"video": "/media/%s/%s" % (proj["id"], os.path.basename(out)),
                            "file": out}
                run_job(jid, work)
                return self.send_json({"job": jid})

            if p == "/api/project/patch":
                """轻量更新项目元信息（拍摄清单 / 名称等），不触碰任何素材文件"""
                body = self.read_body_json()
                try:
                    proj = load_project(body["project"])
                except OSError:
                    return self.send_json({"error": "Project not found: %s" % body["project"]}, 404)
                for k in ("shot_list", "name", "note"):
                    if k in body:
                        proj[k] = body[k]
                save_project(proj)
                return self.send_json({"project": proj})

            if p == "/api/config":
                body = self.read_body_json()
                cfg = load_config()
                if "cloud_token" in body: cfg["cloud_token"] = body["cloud_token"]
                if "video_api" in body: cfg["video_api"] = body["video_api"]
                if "storyboard_api" in body: cfg["storyboard_api"] = body["storyboard_api"]
                save_config(cfg)
                return self.send_json({"ok": True})

            return self.send_json({"error": "not found"}, 404)
        except ValueError as e:
            return self.send_json({"error": str(e)}, 400)
        except Exception as e:
            traceback.print_exc()
            return self.send_json({"error": str(e)}, 500)


class Server(ThreadingHTTPServer):
    # Windows 上 SO_REUSEADDR 会导致同一端口被多个进程同时绑定, 关闭它；
    # Linux（云端）需要它来避免重启时 TIME_WAIT 占用端口。
    allow_reuse_address = (os.name != "nt")
    daemon_threads = True
    # 默认 backlog 只有 5：浏览器一次开 6+ 个并发请求（页面+静态资源+接口），
    # 忙碌时会挤掉后面的连接（客户端报 RemoteDisconnected）。放宽它。
    request_queue_size = 64


def main():
    # 云端：平台注入 PORT → 绑 0.0.0.0；本地：配置文件里的端口 → 绑 127.0.0.1
    port = int(CLOUD_PORT or CONFIG.get("port", 8765))
    host = os.environ.get("HOST") or ("0.0.0.0" if CLOUD_PORT else "127.0.0.1")
    httpd = Server((host, port), Handler)
    print("Invitation Promo Studio running at http://%s:%d" % (host, port))
    print("Data directory:", DATA)
    print("Access password:", "ON" if ACCESS_PASSWORD else "off (local mode)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
