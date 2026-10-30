# -*- coding: utf-8 -*-
"""
客户端精简流程 · 智能层（2026-09-29）
======================================
面向「客户自助」的两件事，都在这一层完成：

1. 照片质检 assess()：客户上传后自动判断这张图能不能直接用。
   本地技术体检（分辨率 / 清晰度 / 曝光）+ 视觉大模型判定（文字是否清楚、
   产品是否完整清晰）→ 合成 pass / warn / fail 与英文提示，不合格就提示重传。
   标准（用户口径）：文字清楚、产品显示清晰、画质高清。

2. 运镜配方 recipe()：按识别到的图片内容与构图，自动匹配运镜镜头与每镜时长
   （每镜 3s、成片 ≤15s；2026-10-10 起取消时长档位，画幅由界面选 9:16 / 1:1）。
   运镜细节（镜头编号/名称）不返回给界面，只在服务端日志与本文件的 note 中出现。

设计约束：
  · 不新增 AI 生成的任何内容，只做判定与匹配 —— 2.5D 几何运镜零 AI 算力；
  · 视觉调用复用 core.showcase.chat_vision（同一把 ARK Key、同一套错误提示）；
  · 视觉模型不可用时降级为「纯本地体检」，不阻断客户流程（air 缺失时给出提示）。
"""
import hashlib
import json
import os
import random
import time

from core import prompts
from core.cammove import (FREEZE_SECS, MOVES, STANDARD_SEQ, build_canvas, recommend_move,
                          dedupe_sweeps, SWEEP_DIR, SWEEP_POOL, MAX_SEQ_SHOTS)
from core.showcase import PLANNER_MODELS, chat_vision, _parse_json_loose

# 本地体检阈值（长边归一化后计算，保证不同尺寸可比）
MIN_LONG_EDGE_FAIL = 1000     # 长边 < 1000px → 直接不合格（文字必然发虚）
MIN_LONG_EDGE_WARN = 1600     # 长边 < 1600px → 可用但打折
SHARP_FAIL = 60.0             # 拉普拉斯方差 < 60 → 糊片
SHARP_WARN = 160.0            # < 160 → 偏软
DARK_FAIL = 38.0              # 平均亮度（0-255）过低 → 暗到看不清
CLIP_FAIL = 0.25              # 死白像素占比 > 25% → 过曝（比看均值稳，纸张本身就偏亮）

# 时长口径（2026-10-10 用户口径）：取消「快 / 中」两档，统一为
#   「每镜 3s，成片总时长 ≤15s」—— 超出就把每镜时长等比收缩（见 recipe 的 eff）。
PER_SHOT_SECS = 3.0
MAX_TOTAL_SECS = 15.0
MIN_PER_SHOT = 0.9           # 每镜时长下限（图很多时仍保证运镜有位移余量）
# 兼容字段（旧调用方可能仍在读）：现在只有一套方案。
SECS_OPTIONS = [PER_SHOT_SECS]
SECS_LABELS = {PER_SHOT_SECS: "Standard"}
SECS_RECOMMENDED = PER_SHOT_SECS

# BGM 情绪口径（2026-10-10 v2 用户反馈「要轻音乐、欢快一点、适合婚礼、稍微卡点」）：
#   池子 = bright（轻快）+ warm（温暖）两层 —— 排除 edm（四拍电子）与过慢的 calm
#   （60–72 BPM 偏"安眠"，15s 短片里听不出欢快）。曲库 BPM 为音频实测值（见 bgm_meta.json）。
BGM_MOODS = ("bright", "warm")
MOOD_ZH = {"bright": "bright and upbeat", "warm": "warm and romantic",
           "calm": "calm and ceremonial"}


# ---------------- 本地技术体检 ----------------

def _local_checks(path):
    """分辨率 / 清晰度 / 曝光 / 画幅比。任何一项读不出来就标记不可用。"""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"readable": False, "width": 0, "height": 0, "long_edge": 0,
                "aspect": 0.0, "sharpness": 0.0, "brightness": 0.0,
                "issues": ["The file could not be read as an image (it may be corrupted)."]}
    h, w = img.shape[:2]
    long_edge = max(h, w)
    # 统一缩放到长边 1600 再测清晰度，避免尺寸影响阈值
    scale = 1600.0 / long_edge if long_edge > 0 else 1.0
    probe = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA) if abs(scale - 1.0) > 0.01 else img
    gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    clipped = float((gray >= 250).mean())
    issues = []
    if long_edge < MIN_LONG_EDGE_FAIL:
        issues.append("Resolution is too low (%d px on the long edge) - at least %d px is needed "
                      "so the lettering stays readable." % (long_edge, MIN_LONG_EDGE_WARN))
    elif long_edge < MIN_LONG_EDGE_WARN:
        issues.append("Resolution is a little low (%d px on the long edge); %d px or more gives a "
                      "sharper result." % (long_edge, MIN_LONG_EDGE_WARN))
    # 过暗/过曝先说（这类图测清晰度没有意义）
    too_dark = bright < DARK_FAIL
    blown = clipped > CLIP_FAIL
    if too_dark:
        issues.append("The photo is very dark - the card details are hard to make out.")
    elif blown:
        issues.append("The photo is overexposed - %d%% of the frame is blown out to pure white."
                      % round(clipped * 100))
    if not too_dark and not blown:
        if sharp < SHARP_FAIL:
            issues.append("The photo looks blurry or out of focus - please upload a sharper shot.")
        elif sharp < SHARP_WARN:
            issues.append("Focus is a little soft; a sharper photo will render better.")
    return {"readable": True, "width": w, "height": h, "long_edge": long_edge,
            "aspect": round(w / float(h), 3) if h else 0.0,
            "sharpness": round(sharp, 1), "brightness": round(bright, 1),
            "clipped": round(clipped, 4),
            "issues": issues}


# ---------------- 视觉判定 ----------------

# 工艺关键词白名单（视觉模型只允许从这些里选；用于微距运镜优先取景）
CRAFT_TYPES = ("gold_foil", "emboss", "deboss", "letterpress",
               "raised_ink", "foil_stamp", "edge_gilding", "silk_screen",
               "lace", "die_cut", "laser_cut", "engraving", "bevel")


def _norm_craft(v):
    """视觉模型返回的 craft → 白名单内的关键词列表（顺序保留、去重）。"""
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, (list, tuple)):
        return []
    out = []
    for x in v:
        k = str(x).strip().lower().replace(" ", "_").replace("-", "_")
        if k in CRAFT_TYPES and k not in out:
            out.append(k)
    return out


def _norm_box(v):
    """视觉模型返回的 craft_box → 合法的归一化框 [x,y,w,h]（0-1）；不合法返回 []。

    容错：接受 4 个数字；越界裁剪到 0-1；框太小（宽或高 < 2%）视为无效 —— 
    拿太小的框当取景锚点会让微距镜推到没有内容的角落，宁可回落默认锚点。
    """
    if isinstance(v, dict):
        v = [v.get("x"), v.get("y"), v.get("w"), v.get("h")]
    if not isinstance(v, (list, tuple)) or len(v) < 4:
        return []
    try:
        x, y, w, h = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
    except (TypeError, ValueError):
        return []
    # 模型偶尔给像素坐标（>1）→ 直接丢弃，不猜
    if max(abs(x), abs(y), abs(w), abs(h)) > 1.5:
        return []
    x, w = min(max(x, 0.0), 1.0), min(max(w, 0.0), 1.0)
    y, h = min(max(y, 0.0), 1.0), min(max(h, 0.0), 1.0)
    if w < 0.02 or h < 0.02:
        return []
    w = min(w, 1.0 - x)
    h = min(h, 1.0 - y)
    return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]

def norm_craft_boxes(rec):
    """从质检记录里取工艺框列表（新字段 craft_boxes 优先，回落旧 craft_box）。

    [2026-10-09 用户口径] 一张图可有多处工艺（烫金/蕾丝/镂空/浮雕…），微距镜按序各取一个。
    返回合法的归一化框列表；没有则返回 []（调用方回落到纹理锚点）。
    """
    if not rec:
        return []
    out = []
    boxes = rec.get("craft_boxes")
    if isinstance(boxes, (list, tuple)):
        for b in boxes:
            nb = _norm_box(b)
            if nb and nb not in out:
                out.append(nb)
    if not out:
        nb = _norm_box(rec.get("craft_box"))
        if nb:
            out.append(nb)
    return out


def best_craft_boxes(proj, path):
    """细节微距运镜取景用的工艺框：**优先用多尺度 craftdetect 的 Top-5 候选框**，
    没有（未跑 / 无候选）时回落到旧 assess 单遍 craft_boxes。

    [2026-10-12 用户口径] 用新的多尺度工艺识别（core.craftdetect）替代原 assess 单遍
    出的 craft_boxes —— 候选更多（≤5）、取景更准（多尺度放大重识别，置信度<0.7 再 3x 重认）。
    返回归一化框列表（与 norm_craft_boxes 同格式）；无工艺框时返回 []（调用方回落纹理锚点）。
    """
    cd = (proj or {}).get("craft_detail") or {}
    rec = cd.get(path)
    boxes = []
    if isinstance(rec, dict):
        for c in (rec.get("candidates") or []):
            nb = _norm_box(c.get("bbox"))
            if nb and nb not in boxes:
                boxes.append(nb)
    if boxes:
        return boxes
    return norm_craft_boxes(((proj or {}).get("smart_photos") or {}).get(path))


def best_card_boxes(proj, path):
    """切到单独卡片用的单卡框：**优先用 card-detect 的分区结果**（proj["card_layout"]），
    没有（未跑 / 无分区）时返回 []。返回归一化框列表（与 norm_craft_boxes 同格式）。

    [2026-10-12 学习案例] 案例运镜会从「整套」切到「单独一张卡」再切工艺细节；单图 set/group
    （平铺套装图）之前只有 整套↔工艺 两端。这条函数在单图模式里让 cm15/cm16 取景框住某张
    独立卡，复刻「切到单独卡片」的层次。多图模式不需要（每张图本就是一张独立卡）。"""
    cd = (proj or {}).get("card_layout") or {}
    rec = cd.get(path)
    if isinstance(rec, dict):
        boxes = []
        for c in (rec.get("cards") or []):
            nb = _norm_box(c.get("bbox"))
            if nb and nb not in boxes:
                boxes.append(nb)
        if boxes:
            return boxes
    return []


def assess(cfg, path, depth="fast", use_ai=True, timeout=180):
    """单张照片质检。返回可直接给前端用的字典。

    verdict: pass 可直接用 / warn 可用但打折 / fail 必须重传
    """
    t0 = time.time()
    loc = _local_checks(path)
    out = {
        "path": path,
        "file": os.path.basename(path),
        "local": loc,
        "verdict": "pass",
        "content_kind": "unknown",
        "text_legible": None, "product_clear": None, "image_sharp": None,
        "craft": [], "craft_box": [], "craft_boxes": [],
        "subject": "", "tip": "",
        "model": None, "ai": False, "ai_error": "",
    }
    issues = list(loc.get("issues") or [])

    if not loc.get("readable"):
        out.update(verdict="fail", issues=issues, elapsed=round(time.time() - t0, 1))
        return out

    if use_ai and (cfg or {}).get("api_key"):
        model = PLANNER_MODELS.get(depth) or PLANNER_MODELS["fast"]
        try:
            txt, _usage, _el = chat_vision(
                cfg, prompts.SMART_PHOTO_CHECK,
                "请判断这张产品照片能否直接用于制作产品展示视频，按系统约定输出 JSON。",
                path, model=model, max_tokens=900, timeout=timeout)
            v = _parse_json_loose(txt)
            out["ai"] = True
            out["model"] = model
            verdict = str(v.get("verdict") or "").strip().lower()
            out["verdict"] = verdict if verdict in ("pass", "warn", "fail") else "warn"
            ck = str(v.get("content_kind") or "").strip().lower()
            out["content_kind"] = ck if ck in ("set", "single", "detail", "group") else "unknown"
            out["text_legible"] = v.get("text_legible")
            out["product_clear"] = v.get("product_clear")
            out["image_sharp"] = v.get("image_sharp")
            # 工艺识别（2026-09-30）：细节微距镜的取景优先落在工艺区域上
            out["craft"] = _norm_craft(v.get("craft"))
            # 工艺框（2026-10-09）：支持多处工艺 —— craft_boxes 优先，回落单个 craft_box
            _boxes = []
            for _b in (v.get("craft_boxes") or []):
                _nb = _norm_box(_b)
                if _nb and _nb not in _boxes:
                    _boxes.append(_nb)
            _one = _norm_box(v.get("craft_box"))
            if not _boxes and _one:
                _boxes = [_one]
            out["craft_boxes"] = _boxes
            out["craft_box"] = _one or (_boxes[0] if _boxes else [])
            out["subject"] = str(v.get("subject") or "").strip()
            out["tip"] = str(v.get("tip") or "").strip()
            for x in (v.get("issues") or []):
                if str(x).strip():
                    issues.append(str(x).strip())
        except Exception as e:
            out["ai_error"] = str(e)[:300]
            out["verdict"] = "warn"       # 视觉不可用 → 只要本地体检过就给过，不阻断客户
    else:
        out["ai_error"] = "AI check unavailable (no Ark key configured)"

    # 本地硬伤优先：读不出、糊、分辨率不足 → 直接不合格
    hard = any(k in i for i in (loc.get("issues") or [])
               for k in ("too low", "blurry", "very dark", "overexposed", "could not be read"))
    if hard:
        out["verdict"] = "fail"

    # 去重（本地与视觉可能报同一件事）
    seen, uniq = set(), []
    for i in issues:
        k = i.lower()[:40]
        if k not in seen:
            seen.add(k)
            uniq.append(i)
    out["issues"] = uniq
    out["elapsed"] = round(time.time() - t0, 1)
    return out


def assess_batch(cfg, paths, depth="fast", progress=None, workers=3, use_ai=True):
    """并发质检多张（视觉接口是串行耗时大头，3 路并发实测稳定）。"""
    from concurrent.futures import ThreadPoolExecutor
    paths = [p for p in paths if p and os.path.isfile(p)]
    total = len(paths)
    results = [None] * total
    done = [0]

    def one(i, p):
        try:
            r = assess(cfg, p, depth=depth, use_ai=use_ai)
        except Exception as e:                       # 单张异常不拖垮整批
            r = {"path": p, "file": os.path.basename(p), "verdict": "warn",
                 "issues": ["This photo could not be checked automatically: %s" % str(e)[:160]],
                 "ai": False, "content_kind": "unknown", "craft": [], "craft_box": [],
                 "craft_boxes": [],
                 "subject": "", "tip": ""}
        results[i] = r
        done[0] += 1
        if progress:
            progress(done[0], total, "Checked %d of %d photos" % (done[0], total))
        return r

    if not total:
        return []
    if total == 1:
        one(0, paths[0])
    else:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, total))) as ex:
            list(ex.map(lambda ip: one(*ip), list(enumerate(paths))))
    return results


# ---------------- 配乐自动匹配（客户不选） ----------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BGM_META = os.path.join(BASE_DIR, "bgm_meta.json")
BGM_DIR = os.path.join(BASE_DIR, "bgm")


def _bgm_tracks():
    """曲库客观特征（实测 BPM / 响度），来自 bgm_meta.json；读不到则按目录名兜底。

    注意：meta 里记的是**生成时**的绝对路径，换成部署副本后路径会变，
    所以按文件名在当前曲库目录里找回来（线上 /workspace/bgm 与本地同名）。
    """
    try:
        with open(BGM_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
        tracks = []
        for t in (meta.get("tracks") or []):
            f = t.get("file") or ""
            if not os.path.isfile(f):
                alt = os.path.join(BGM_DIR, os.path.basename(f))
                if not os.path.isfile(alt):
                    continue
                t = dict(t, file=alt)
            tracks.append(t)
        if tracks:
            return tracks
    except Exception:
        pass
    out = []
    try:
        for fn in sorted(os.listdir(BGM_DIR)):
            if fn.lower().endswith((".mp3", ".wav", ".m4a")):
                out.append({"file": os.path.join(BGM_DIR, fn),
                            "name": os.path.splitext(fn)[0], "bpm": None,
                            "license": ("CC BY 3.0" if "CC-BY" in fn else "Mixkit")})
    except OSError:
        pass
    return out


def match_bgm(per_shot, seed="", exclude=(), mood=None, prefer=None):
    """按镜头节奏自动选曲（客户无需选择）。

    判据（客观、可复核）：让每一镜正好落在一个乐句上——每镜 4 拍，
    所以目标 BPM = 240 / 每镜秒数：1.5s→160、2.5s→96、3.0s→80 BPM。
    曲库 BPM 是从音频本体实测的（谱通量自相关，见 bgm_meta.json），
    不是按曲名猜的。BPM 接近度相同时按内容类型组合轮换，避免所有片子同一首。
    exclude：已配给其他节奏档的文件路径，本档尽量避开（2026-09-29 用户口径：
    每个节奏档配不同的曲子；曲库不够时才允许复用）。
    """
    tracks = _bgm_tracks()
    if not tracks:
        return None
    rated = [t for t in tracks if t.get("bpm")]
    # 情绪层分流：同一节奏档只在对应的情绪层里挑（层内再按 BPM 卡点），
    # 这样三档不只是"不同的曲子"，而是听感/调性明显不同的三种音乐。
    # 层里没货（比如曲库被 exclude 光了）才回落到全库，保证一定有曲。
    if mood:
        tier = [t for t in rated if t.get("mood") == mood]
        if tier:
            rated = tier

    # 指定曲优先（人工指定的客户口径）：命中且未被别的档占用时直接采用，
    # 不再按 BPM 排序。找不到就安静地回落到下面的 BPM 卡点匹配。
    if prefer:
        for t in rated:
            nm = t.get("name") or os.path.basename(t.get("file") or "")
            if nm == prefer and t["file"] not in exclude:
                return {"file": t["file"], "name": t.get("name") or os.path.basename(t["file"]),
                        "license": t.get("license") or "Royalty-free", "bpm": t.get("bpm"),
                        "target_bpm": round(240.0 / float(per_shot), 1),
                        "why": "%s — studio pick for this pace (%s BPM)" % (
                            MOOD_ZH.get(mood, "hand-picked"), t.get("bpm"))}

    if not rated:
        t = tracks[0]
        return {"file": t["file"], "name": t.get("name") or os.path.basename(t["file"]),
                "license": t.get("license") or "Royalty-free", "bpm": None,
                "why": "Royalty-free track from the studio library"}
    want = 240.0 / float(per_shot)
    ranked = sorted(rated, key=lambda t: abs(float(t["bpm"]) - want))
    tol_rel = 0.05                         # 5% 相对容差（绝对下限 2 BPM）
    fresh = [t for t in ranked if t["file"] not in exclude]
    use = fresh if fresh else ranked       # 曲库被排除光时才复用
    h = int(hashlib.md5(str(seed).encode("utf-8")).hexdigest()[:8], 16)

    # 卡点判据：每镜整拍数落拍。优先 4 拍/镜（BPM=240/secs）；曲库没有就放宽到
    # 3 拍（BPM=180/secs）、2 拍（BPM=120/secs）——剪切点仍落在节拍上，
    # 只是每镜少一拍（2026-09-30 快档 1.5s/镜：4拍=160 曲库无货，3拍=120 有 4 首可轮换）。
    # 容差 5%（2026-09-30 用户要"每次尽量多换曲"）：让每档池子里都有 2 首以上可轮换。
    pool, beats, want_used = [], 4, want
    for k in (4, 3, 2):
        w = 60.0 * k / float(per_shot)
        cand = [t for t in use if abs(float(t["bpm"]) - w) <= max(2.0, w * tol_rel)]
        if len(cand) >= 1:
            pool, beats, want_used = cand, k, w
            if len(cand) >= 2:             # 有得轮换就用这一档
                break
    if pool:
        pick = pool[h % len(pool)] if (seed and len(pool) > 1) else pool[0]
        why = "%s — %s BPM, %d beats per shot" % (
            MOOD_ZH.get(mood, "matched to the pace of your shots"), pick.get("bpm"), beats)
    else:
        # 曲库里没有能整拍对上的曲子 → 在最近的 3 首之间轮换，并如实说明
        near = use[:3]
        pick = near[h % len(near)] if (seed and len(near) > 1) else use[0]
        why = ("closest match in the library (%s BPM; a %.1fs-per-shot pace would ideally suit "
               "about %.0f BPM)" % (pick.get("bpm"), float(per_shot), want))
    return {
        "file": pick["file"],
        "name": pick.get("name") or os.path.basename(pick["file"]),
        "license": pick.get("license") or "Royalty-free",
        "bpm": pick.get("bpm"),
        "target_bpm": round(want, 1),
        "why": why,
    }


# ---------------- BGM 随机换曲（2026-10-10 用户口径）----------------
# 「曲库要有多个音乐、每次生成都换一首不同的」：只在婚礼调性层（calm/warm）里抽签，
#   上一次用过的曲名记在 bgm/bgm_state.json，抽到同一首就重抽，让连续两条视频不同曲。
BGM_STATE = os.path.join(BGM_DIR, "bgm_state.json")


def _bgm_state():
    try:
        with open(BGM_STATE, "r", encoding="utf-8") as f:
            d = json.load(f)
            return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _bgm_state_save(d):
    try:
        with open(BGM_STATE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
    except Exception:
        pass


def pick_bgm(seed=None, per_shot=None):
    """随机挑一首「婚礼轻音乐」配乐，并尽量让剪切点落在节拍上。

    2026-10-10 v2 用户口径：**轻音乐 / 欢快一点 / 适合婚礼 / 稍微卡点**。
      · 池子 = bright（轻快）+ warm（温暖）两层，排除 edm 电子与过慢的 calm；
      · 卡点：per_shot 给定时，优先挑「每镜整数拍」的曲（BPM ≈ 60k/per_shot，k=3..8），
        这样每镜的剪切点都落在节拍上；对齐曲不足 2 首则放宽到全池（仍保「每次换曲」）；
      · 仍避开上一条用过的曲（bgm/bgm_state.json），让连续两条视频不同曲。
    返回体比旧版多一个 "beats"（对齐时=每镜拍数，供 recipe 把每镜时长吸附到精确拍长）。
    """

    def _nm(t):
        return t.get("name") or os.path.basename(t.get("file") or "")

    tracks = [t for t in _bgm_tracks() if t.get("mood") in BGM_MOODS]
    if not tracks:
        tracks = _bgm_tracks()
    if not tracks:
        return None

    last = _bgm_state().get("last")
    pool, beats_locked = list(tracks), None

    # 卡点候选：每镜 k 整拍能对上的曲（k=3..8），容差 6%（下限 2 BPM）
    if per_shot:
        try:
            ps = float(per_shot)
        except Exception:
            ps = 0.0
        if ps > 0:
            aligned, hit_k = [], {}
            for t in tracks:
                b = t.get("bpm")
                if not b:
                    continue
                for k in (3, 4, 5, 6, 8):
                    w = 60.0 * k / ps
                    if abs(float(b) - w) <= max(2.0, w * 0.06):
                        aligned.append(t)
                        hit_k[_nm(t)] = k
                        break
            if len(aligned) >= 2:
                pool = aligned
                beats_locked = hit_k

    rng = random.Random(str(seed)) if seed is not None else random.SystemRandom()
    fresh = [t for t in pool if _nm(t) != last]
    pick = rng.choice(fresh or pool or tracks)
    nm = _nm(pick)
    beats = (beats_locked or {}).get(nm)
    st = _bgm_state()
    st["last"] = nm
    st["picked_at"] = int(time.time())
    _bgm_state_save(st)
    return {
        "file": pick["file"],
        "name": nm,
        "license": pick.get("license") or "Royalty-free",
        "bpm": pick.get("bpm"),
        "mood": pick.get("mood"),
        "beats": beats,
        "why": ("Wedding light track, cut on the beat" if beats
                else "Wedding light track from the studio library")
        + (" (%s)" % pick.get("mood") if pick.get("mood") else ""),
    }


# ---------------- 运镜配方匹配 ----------------

# 内容类型 → 运镜路线（客户看不到镜头号，只服务端记录）
ROUTE_BY_KIND = {
    "set":    list(STANDARD_SEQ),        # 整套平铺：五镜标准序列（斜扫→缓推→横扫→微距→缓拉）
    "group":  ["cm01", "cm04", "cm07", "cm05"],
    "single": ["cm04", "cm07", "cm05"],  # 单卡：缓推 → 微距工艺 → 缓拉收尾
    # 细节微距：2026-09-29 用户拍板首镜换掉近静止的 CM-07；2026-09-30 用户口径再明确——
    #   「细节运镜，截取的画面优先是工艺类的（浮雕/凹凸/烫金）」，故首镜改为 CM-08 微距推进，
    #   取景锚点由工艺区域框决定（core.cammove.craft_anchor），次镜 CM-06 反向斜扫负责位移。
    "detail": ["cm08", "cm06", "cm05"],
}

# 多图模式：**每张照片跑一组（≥2 个）运镜**，不是一张图只给一个镜头。
# 用户口径（2026-10-08）：「根据图片判别运用的运镜，保证每个产品图的运镜视频里
#   不要只有一个运镜方式」→ 单张图的运镜视频内部就必须有多种运镜。
# 2026-10-08 修正：只轮换方向不够——cm01/cm02/cm03/cm06 都是"从画面中心平移"的横扫，
#   在平铺的请柬图上三种平移观感几乎一样，用户反馈"运镜还是没什么区别"。
#   故组内做成「平移 ↔ 变焦 ↔ 微距」的**类型搭配**（见 _photo_route）。
#   detail 组的 cm06/cm07/cm08 取景落在工艺框上（2026-09-30 口径，未变）。
# 每图 2 镜会让成片时长翻倍，故只在照片不多时才给 2 镜（见 MULTI_MOVES_MAX_PHOTOS）。
MULTI_MOVES_MAX_PHOTOS = 3     # 多图时「每图 2 镜」的照片数上限；超过则每图 1 镜

_MULTI_ROUTE = {
    # 2026-10-09：扫类组补齐相反方向（cm11 右下→左上、cm12 右→左、cm13 下→上），
    #   让同一内容类型的多张照片按序号轮换到**方向各不相同**的组；dedupe_sweeps 再兜底。
    "set":    [["cm01", "cm04"], ["cm02", "cm05"], ["cm06", "cm04"], ["cm09", "cm05"],
               ["cm11", "cm04"], ["cm12", "cm05"], ["cm13", "cm04"]],
    "group":  [["cm01", "cm04"], ["cm07", "cm05"], ["cm02", "cm04"], ["cm11", "cm05"], ["cm12", "cm04"]],
    "single": [["cm04", "cm07"], ["cm05", "cm09"], ["cm10", "cm07"]],
    "detail": [["cm08", "cm07"], ["cm06", "cm08"], ["cm07", "cm05"]],
}


# ---------------- 单图路线变体池（2026-10-10 用户口径：加轻量随机）----------------
# 背景：recipe 原是确定性函数 —— 同一批图重复生成时，运镜序列 100% 相同，客户会看出
#   「每次都是同一个模板」。用户口径「加轻量随机」：**内容类型仍严格按图匹配**（运动
#   性格不变），只把「同类型内的具体镜头/扫向」做随机轮换 —— 连生两条不再重样。
# 约束：同一 kind 内所有变体**镜头数必须相等**（sources↔moves 引擎硬校验，且镜数决定
#   每镜时长与成片总长，变体间镜数不一致会让时长口径漂移）。
_ROUTE_VARIANTS = {
    "set": [                       # 整套平铺：扫 → 推 → 扫 → 微距 → 拉（5 镜）
        {"id": "s1", "moves": ["cm01", "cm04", "cm02", "cm07", "cm05"]},
        {"id": "s2", "moves": ["cm11", "cm04", "cm03", "cm07", "cm05"]},
        {"id": "s3", "moves": ["cm02", "cm04", "cm12", "cm07", "cm05"]},
        {"id": "s4", "moves": ["cm13", "cm04", "cm01", "cm07", "cm05"]},
    ],
    "group": [                     # 组合：扫 → 推 → 微距 → 拉（4 镜）
        {"id": "g1", "moves": ["cm01", "cm04", "cm07", "cm05"]},
        {"id": "g2", "moves": ["cm11", "cm04", "cm07", "cm05"]},
        {"id": "g3", "moves": ["cm12", "cm04", "cm07", "cm05"]},
        {"id": "g4", "moves": ["cm02", "cm04", "cm08", "cm05"]},
    ],
    "single": [                    # 单卡：推 → 微距 → 拉（3 镜）
        {"id": "n1", "moves": ["cm04", "cm07", "cm05"]},
        {"id": "n2", "moves": ["cm04", "cm08", "cm05"]},
        {"id": "n3", "moves": ["cm10", "cm07", "cm05"]},
    ],
    "detail": [                    # 工艺细节：微距推进 → 反向扫 → 拉（3 镜，首镜固定 CM-08）
        {"id": "d1", "moves": ["cm08", "cm06", "cm05"]},
        {"id": "d2", "moves": ["cm08", "cm11", "cm05"]},
        {"id": "d3", "moves": ["cm08", "cm13", "cm05"]},
        {"id": "d4", "moves": ["cm08", "cm09", "cm05"]},
    ],
}
# 上一次用过的变体（kind → id），内存态：让「连续生成同一张图」也换一条路线。
_LAST_VARIANT = {}


def _pick_variant(kind):
    """按内容类型的变体池随机挑一条路线，并尽量避开上一次用过的那条。

    未识别的 kind 回落到 set 池（与 ROUTE_BY_KIND 的默认一致）。
    """
    pool = _ROUTE_VARIANTS.get(kind) or _ROUTE_VARIANTS["set"]
    last = _LAST_VARIANT.get(kind)
    cands = [v for v in pool if v["id"] != last] or pool
    v = random.choice(cands)
    _LAST_VARIANT[kind] = v["id"]
    return list(v["moves"])


def _photo_route(kind, used=None):
    """一张照片（内容类型 kind）要跑的镜头组——**一组 2 镜，且两镜类型不同**。

    组内刻意做成「平移 ↔ 变焦 ↔ 微距」的搭配（如 斜扫+缓推、横扫+缓拉、
    微距推进+微距漂移），因为同一张图上，缩放和平移是两种一眼能分辨的运动；
    只换方向（斜扫/横扫/竖扫）在平铺纸品上观感几乎一样。

    2026-10-10 用户口径「加轻量随机」：从该类型的组池里**随机**挑一组（不再按照片序号
    固定轮换）。`used` 传本片内已用过的组序号，优先挑没用过的 —— 同一条片子里每张图
    依然各不相同，但换一批图重新生成时组合会变。
    """
    groups = _MULTI_ROUTE.get(kind) or _MULTI_ROUTE["set"]
    used = used if used is not None else set()
    idxs = [i for i in range(len(groups)) if i not in used] or list(range(len(groups)))
    i = random.choice(idxs)
    used.add(i)
    return list(groups[i])


def _route_for(item):
    """单张图的镜头：优先用视觉识别的内容类型；没有就回落到本地构图推荐。"""
    kind = (item or {}).get("content_kind") or "unknown"
    if kind in ROUTE_BY_KIND:
        return ROUTE_BY_KIND[kind], "content:%s" % kind
    path = (item or {}).get("path")
    if path and os.path.isfile(path):
        try:
            mid, reason = recommend_move(build_canvas(path))
            return [mid, "cm07", "cm05"], reason
        except Exception as e:
            return list(STANDARD_SEQ), "fallback (composition analysis failed: %s)" % str(e)[:80]
    return list(STANDARD_SEQ), "fallback"


# 单张快档加镜用的候选池：按"扫向尽量不重复 + 非扫类穿插"排序，保证 9 镜观感丰富。
_EXTEND_POOL = ["cm03", "cm06", "cm09", "cm11", "cm12", "cm13", "cm08", "cm10",
                "cm04", "cm02", "cm01", "cm05", "cm07"]


def _extend_moves(base, target=9):
    """把单张图的镜头序列从 base 补到 target 个，供快档单图把时长撑到 ~10s（1s/镜）。

    补镜时跳过已出现的扫向（SWEEP_DIR 相同即视为同方向），非扫类镜头（cm04/05/07/08/09/10）
    不受扫向约束、仅去重；不足 target 时用尽候选池也不强行塞重复镜头。"""
    base = list(base)
    if len(base) >= target:
        return base
    used_dirs = {SWEEP_DIR.get(m) for m in base if m in SWEEP_DIR}
    out = list(base)
    for m in _EXTEND_POOL:
        if len(out) >= target:
            break
        if m in out:
            continue
        d = SWEEP_DIR.get(m)
        if d and d in used_dirs:
            continue
        out.append(m)
        if d:
            used_dirs.add(d)
    return out


def recipe(items):
    """按质检通过的图片生成运镜配方。

    items: [{path, content_kind, aspect, subject, ...}]（只传 verdict != fail 的）
    返回：{mode, sources, moves, shots, per_shot, options:[{secs,total,label,recommended}],
           note（服务端内部说明，不下发界面）}

    单图 → 走该图内容对应的完整路线（3–5 镜，一次渲染成片）；
    多图 → 每张图各跑一组运镜（组 2 镜、组内类型不同），照片 >3 张时降为每图 1 镜
           （仍按照片序号轮换不同组，方向/类型逐张错开），硬切拼接。
    """
    items = [i for i in (items or []) if i and i.get("path")]
    if not items:
        return {"mode": "none", "sources": [], "moves": [], "shots": 0,
                "per_shot": SECS_RECOMMENDED, "options": [], "note": "no usable photo"}

    picks = []
    for it in items:
        route, why = _route_for(it)
        craft = _norm_craft(it.get("craft"))
        cbox = _norm_box(it.get("craft_box"))
        picks.append({"path": it["path"], "kind": it.get("content_kind") or "unknown",
                      "route": route, "why": why, "craft": craft, "craft_box": cbox,
                      "card_boxes": it.get("card_boxes") or []})

    if len(items) == 1:
        mode = "single"
        # 单张照片 → 跑该图内容对应的**完整路线**（3–5 镜，一次渲染成片）。
        # 2026-10-10「加轻量随机」：同类型内随机挑一条变体（避开上一次用过的那条），
        #   运动性格不变（仍是该内容的扫/推/微距/拉组合），但具体镜头与扫向会变。
        kind0 = picks[0]["kind"]
        if kind0 in _ROUTE_VARIANTS:
            photo_moves = [list(_pick_variant(kind0))]
        elif list(picks[0]["route"]) == list(STANDARD_SEQ):
            # 完全没识别出内容（回落默认整套路线）→ 也在 set 池内随机，
            #   避免同一张图反复生成出一模一样的片子。
            photo_moves = [list(_pick_variant("set"))]
        else:
            # 兜底：本地构图推荐（recommend_move）的结果，按图而定，直接沿用。
            photo_moves = [list(picks[0]["route"])]
        # [2026-10-12 学习案例·修正] 案例运镜主体会从「整套」切到「单独卡片」再切工艺细节；
        #   单图 set/group（平铺套装图）之前只有 整套↔工艺 两端。若视觉分区检测到单卡框，
        #   在开场扫镜之后插入 1–2 个 cm17「推入单卡」镜头（真实推镜，相机推到单卡上），
        #   形成 整套→单卡→工艺 的真实运镜层次（之前用 cm16 定格只是停住、不算运镜）。
        #   多图模式不需要（每张图本就是一张独立卡）。
        card_boxes = [b for b in (_norm_box(c) for c in (picks[0].get("card_boxes") or [])) if b]
        if card_boxes and len(photo_moves[0]) + min(len(card_boxes), 2) <= MAX_SEQ_SHOTS:
            segs = ["cm17"] * min(len(card_boxes), 2)
            photo_moves[0] = photo_moves[0][:1] + segs + photo_moves[0][1:]
    else:
        mode = "multi"
        # 多图时每张照片跑**一组**运镜（组内类型不同）——用户口径：单张图的运镜视频里
        # 不能只有一个运镜方式。引擎 render_cameramove_multi 要求 (图, 镜) 一一对应，
        # 所以同一张图按它的镜头数重复展开 sources，逐段渲染后硬切拼接。
        # 2026-10-08 补充口径：照片 **≤3 张**时每图给 2 镜；**>3 张**时每图回到 1 镜，
        # 否则镜数翻倍会让成片过长（6 张 Standard 档会从 19s 涨到 37s）。
        # 2026-10-10「加轻量随机」：组从该类型池里随机挑（used 去重），不再按序号固定轮换。
        per = 2 if len(picks) <= MULTI_MOVES_MAX_PHOTOS else 1
        photo_moves = []
        used_groups = set()
        for p in picks:
            grp = _photo_route(p["kind"], used_groups)
            photo_moves.append(list(grp[:per]))
        # [2026-10-10 用户口径·看真片后反馈] 多图开场**不要** cm14「全景建立」极缓推
        #   （用户原话：视频开始的运镜不要用这个，可以用平移的）—— 改为平移（扫类）镜开场：
        #   第一张图的首镜已是扫类（set/group 组）就保持不动；不是扫类（single/detail 组）
        #   才在片头插一个本片未用过的扫向。镜数与原 cm14 方案一致（+1），时长口径不变。
        if photo_moves and sum(len(x) for x in photo_moves) + 1 <= MAX_SEQ_SHOTS \
                and photo_moves[0] and photo_moves[0][0] not in SWEEP_DIR:
            used_m = {m for seg in photo_moves for m in seg}
            opener = next((c for c in SWEEP_POOL if c not in used_m), "cm01")
            photo_moves[0] = [opener] + photo_moves[0]

    pairs = []
    for i, p in enumerate(picks):
        for m in photo_moves[i]:
            pairs.append((p["path"], m))

    pairs = [(s, m) for (s, m) in pairs if m in MOVES]
    if not pairs:                                       # 兜底：标准五镜
        pairs = [(picks[0]["path"], m) for m in STANDARD_SEQ] if picks else []
    moves = [m for _, m in pairs]                       # 与 sources 严格等长（引擎硬校验）
    sources = [s for s, _ in pairs]
    # 2026-10-09 用户口径：同一条片子里扫向不得重复（否则会有多个「左上→右下」）。
    #   去重只换扫类镜头、不改长度，sources/moves 仍一一对应。
    base_moves = dedupe_sweeps(moves)

    notes = []
    _k = 0
    for i, p in enumerate(picks):
        seg = []
        for m in photo_moves[i]:
            if m in MOVES and _k < len(base_moves):
                seg.append(base_moves[_k]); _k += 1
        notes.append("%s → %s (%s%s)" % (
            os.path.basename(p["path"]), "-".join(seg), p["why"],
            "; craft=%s @%s" % ("+".join(p["craft"]), p["craft_box"]) if p["craft"] else ""))

    # 时长（2026-10-10 用户口径）：只有一套 —— 每镜 3s，成片总时长 ≤15s。
    #   镜头多到超 15s 时，等比收缩每镜时长（下限 MIN_PER_SHOT），即"默认匹配合适的时长"。
    shots = len(base_moves)
    eff = max(MIN_PER_SHOT,
              min(PER_SHOT_SECS, (MAX_TOTAL_SECS - FREEZE_SECS) / max(shots, 1)))
    total = round(shots * eff + FREEZE_SECS, 1)
    # BGM：每次生成随机换一首（婚礼轻音乐池，避开上一条用过的曲），并尽量卡点。
    bgm = pick_bgm(per_shot=eff)
    # 卡点对齐（2026-10-10 v2）：选中的曲能整数拍对上时，把每镜时长吸附到精确拍长，
    #   使每个剪切点正好落在节拍上。改动幅度限于 ±0.15s，且总时长不得超 15s 上限。
    if bgm and bgm.get("bpm") and bgm.get("beats"):
        try:
            snap = 60.0 * int(bgm["beats"]) / float(bgm["bpm"])
            if abs(snap - eff) <= 0.15 and shots * snap + FREEZE_SECS <= MAX_TOTAL_SECS + 1e-6:
                eff = round(snap, 3)
                total = round(shots * eff + FREEZE_SECS, 1)
        except Exception:
            pass
    options = [{"secs": round(eff, 2), "total": total,
                "label": SECS_LABELS[PER_SHOT_SECS], "recommended": True,
                "bgm": bgm, "moves": base_moves}]
    return {
        "mode": mode,
        "sources": sources,          # 与 base_moves 一一对应（引擎硬校验）
        "moves": base_moves,
        "shots": shots,
        "per_shot": round(eff, 2),
        "total": total,
        "options": options,
        "bgm": bgm,
        "note": " | ".join(notes),
    }
