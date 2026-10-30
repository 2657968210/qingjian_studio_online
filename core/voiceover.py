# -*- coding: utf-8 -*-
"""口播后处理（模板无关）：成片 + 分镜时间线 + 口播稿 → 配音/字幕成片。

定位：接在**视频渲染之后**用（不碰任何渲染器）。四个模版（card_pan /
multi_suite / scroll_gallery / four_grid）都能用，只要给出：
  - 渲染好的成片（可无音轨；有音轨则视为已含 BGM，直接压低）
  - lines  : 口播稿列表（每行对应一个分镜/款式）
  - starts : 每行的起始秒（与 lines 等长；即「时间线 JSON」的最小形态）

做了什么：
  1. edge-tts 逐行合成 + WordBoundary 词级时间戳（7.x 必须显式 boundary=）
  2. 该行音频超窗口时 atempo ≤ 1.12 微加速贴合（红线：再快不自然）
  3. 混音：原声/BGM 压到 duck（默认 0.20）+ 人声 1.0，视频流 copy
  4. 字幕：口播稿拆 ≤5 词短句，时间 = 词级时间戳聚合（说的=显示的）
     ASS 必须逐条加 {\\an2}（本机 libass 不认 Style 行 Align）；只转义 { }
  5. 烧录 _vo.mp4（libx264 一次编码，音频 copy）

captions（屏幕文字，无配音模式）：draft_caption_line 起稿一句产品介绍 +
apply_caption 全程烧在画面**中上方 + 加大字号**——只给没有原生文字的模版
（Product Spotlight / Colour Inspiration）用；Collection/Grid 的文字是渲染器
原生（名牌/中央标题），不走后处理。
字幕位置（用户 2026-10-20 定稿）：口播匹配的字幕**一律中下**（64% 屏高，
四个模版统一）；只有 captions 行用中上方（34% 屏高）。
三段式 beats：多段结构（Collection）自动钉分镜边界；其余模版用 three_act_beats(secs)。

失败语义：网络/合成异常直接抛错——调用方（job）应 catch 后仍交付无口播成片。
"""
import os
import re
import json
import math
import asyncio
import subprocess

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()

DEFAULT_VOICE = "en-US-AriaNeural"
MAX_ATEMPO = 1.12          # 红线：超过听起来不自然
SUB_SIZE = 48              # 口播字幕字号（按 1920 高）
CAP_SIZE = 60              # captions 屏幕文字字号（用户 2026-10-20：要比字幕大一点）
SUB_MARGIN_V = 150         # 底部对齐时距底（pos="bottom" 才用）
SUB_MARGIN_LR = 80         # 字幕左右安全边
SUB_POS_UPPER = 0.34       # 中上方：文字块中心 = 34% 屏高（captions 行专用）
SUB_POS_LOWER = 0.85       # 中下：文字块中心 = 85% 屏高（口播字幕一律用；
                           # 用户 2026-10-21 定稿下移，截图红框 ≈85% 屏高）
MAX_CHUNK_WORDS = 5        # 每条字幕最多几个词（防出画）


# ---------------------------------------------------------------- 基础工具
def probe(path):
    """(fps, 时长, (w,h))。时长一律 ffmpeg 解析 stderr——cv2 读 wav 有 1.00s 假值。"""
    r = subprocess.run([FF, "-i", path], stderr=subprocess.PIPE, text=True)
    err = r.stderr or ""
    dur = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", err)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    fps, w, h = 30.0, 1080, 1920
    m = re.search(r",\s*(\d+(?:\.\d+)?)\s+fps", err)
    if m:
        fps = float(m.group(1))
    m = re.search(r",\s*(\d{2,5})x(\d{2,5})\s", err)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
    return fps, dur, (w, h)


def has_audio(path):
    r = subprocess.run([FF, "-i", path], stderr=subprocess.PIPE)
    return b"Audio" in (r.stderr or b"")


def _run(cmd):
    r = subprocess.run(cmd, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed: %s" % (r.stderr or b"")[-600:])


# ---------------------------------------------------------------- TTS
async def _tts_async(text, out_path, voice):
    """edge-tts 7.x：必须显式 boundary="WordBoundary" 才有词级 offset/duration。"""
    import edge_tts
    comm = edge_tts.Communicate(text, voice, boundary="WordBoundary")
    words, audio = [], b""
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
        elif chunk["type"] == "WordBoundary":
            off = chunk.get("offset") or 0
            dur = chunk.get("duration") or 0
            words.append({
                "text": (chunk.get("text") or "").strip(),
                "start": off / 1e7,
                "end": (off + dur) / 1e7,
            })
    if not audio:
        raise RuntimeError("edge-tts returned no audio for: %r" % text[:40])
    with open(out_path, "wb") as f:
        f.write(audio)
    return words


def tts_line(text, out_audio, voice=DEFAULT_VOICE):
    """合成一行口播 → 落盘 mp3，返回词级时间戳 [{text,start,end}]（秒）。"""
    return asyncio.run(_tts_async(text, out_audio, voice))


def fit_tempo(audio_dur, window_dur):
    """该行音频比分镜窗口长 → atempo 微加速（≤1.12）；短了保持 1.0 不硬拉。"""
    if audio_dur <= 0 or window_dur <= 0:
        return 1.0
    if audio_dur <= window_dur:
        return 1.0
    return min(MAX_ATEMPO, audio_dur / window_dur)


# ---------------------------------------------------------------- 字幕
def chunks_of(text, max_words=MAX_CHUNK_WORDS):
    """按逗号/句号拆；过长从句就近折到每段 ≤max_words；<3 词碎条并回邻条。"""
    parts = [p.strip(" ,;—-") for p in re.split(r"[,;\.]|—", text or "") if p.strip(" ,;—-")]
    chunks = []
    for p in parts:
        queue = [p]
        while queue:
            cur = queue.pop(0)
            words = cur.split()
            if len(words) <= max_words:
                chunks.append(cur)
                continue
            mid = len(words) // 2
            soft = {"for", "and", "with", "to", "of", "in", "on", "your", "the", "a", "an"}
            best = mid
            for i, w in enumerate(words):
                if w.lower().strip(",;") in soft and abs(i - mid) <= 3 and abs(i - mid) < abs(best - mid):
                    best = i
            cut = max(1, min(len(words) - 1, best if best != mid else mid))
            queue.insert(0, " ".join(words[cut:]))
            queue.insert(0, " ".join(words[:cut]))
    merged = []
    for c in chunks:
        if merged and len(c.split()) < 3:
            merged[-1] = merged[-1] + " " + c
        else:
            merged.append(c)
    if len(merged) >= 2 and len(merged[0].split()) < 3:
        merged[1] = merged[0] + " " + merged[1]
        merged = merged[1:]
    return [c for c in merged if c]


def ass_escape(s):
    # ASS 无 XML 实体语法：& 不转义；只转义 { } 覆盖标签
    return (s or "").replace("{", r"\{").replace("}", r"\}")


def ass_time(t):
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    sec = t - h * 3600 - m * 60
    return "%d:%02d:%02d.%02d" % (h, m, int(sec), int(round((sec - int(sec)) * 100)) % 100)


def build_cues(lines, starts, words_by_line, tempos, video_dur):
    """词级时间戳聚合 → [(text, start, end)]。无时间戳时按词数比例兜底。"""
    cues = []
    prev_end = 0.0
    for i, line in enumerate(lines):
        line = (line or "").strip()
        if not line:
            continue
        st = starts[i]
        # 该行不许越过下一行起始（跨镜最多叠到紧邻下一句之前）
        limit = (starts[i + 1] - 0.02) if i + 1 < len(starts) else (video_dur - 0.05)
        words = (words_by_line or {}).get(i) or []
        chunks = chunks_of(line)
        if not chunks:
            chunks = [line]
        tempo = tempos[i] if tempos and i < len(tempos) else 1.0
        if words:
            idx = 0
            n = len(words)
            for c in chunks:
                cnt = len(c.split())
                j = min(n - 1, idx + cnt - 1)
                s = max(prev_end + 0.03, st + words[idx]["start"] / tempo - 0.05)
                e = min(limit, max(s + 0.30, st + words[j]["end"] / tempo + 0.12))
                cues.append((c, s, e))
                prev_end = e
                idx = min(n, idx + cnt)
        else:  # 兜底：窗口内按词数比例均分
            span = max(0.6, limit - st)
            total = sum(len(c.split()) for c in chunks) or 1
            t = st
            for c in chunks:
                d = span * (len(c.split()) / float(total))
                e = min(limit, t + d)
                cues.append((c, t, e))
                prev_end = e
                t = e
    return cues


ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: %(w)d
PlayResY: %(h)d
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Align, MarginL, MarginR, MarginV, Encoding
Style: En,%(fontname)s,%(fsize)d,&H00FFFFFF,&H000019FF,&H00101010,&H96000000,%(fbold)d,%(fitalic)d,0,0,100,100,0,0,1,2.2,1.0,2,%(ml)d,%(mr)d,%(mv)d,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def write_ass(path, cues, w=1080, h=1920, pos="bottom", size=None,
              fontname="Arial", bold=-1, italic=0):
    # ★ PlayResX/PlayResY 必须写：缺省时 libass 按 384x288 解析，48px 字号会被
    #   放大 6 倍多且 MarginL 按比例左偏——字幕巨大、左侧被裁（本次实锤）。
    # 位置（用户 2026-10-20 定稿）：
    #   pos="upper" → 中上方：{\an5\pos()} 钉文字块中心在 34% 屏高，水平居中
    #     ——只给无原生文字的模版（Product Spotlight / Colour Inspiration）的
    #     captions 行用，且字号加大（size=CAP_SIZE）；
    #   pos="lower" → 中下：文字块中心在 64% 屏高——口播匹配的字幕一律用，
    #     四个模版统一（Collection 名牌在画面顶部 ~14%，这里不叠字）；
    #   pos="bottom" → 底部 {\an2}（兼容旧调用，现无调用方）。
    # ★ WrapStyle:0（自动折行）：captions 字号加大后长句必须能按安全边折行，
    #   WrapStyle:2 不折行会横向溢出画布。
    # ★ 覆盖标签不依赖 Style 行 Align：本机 ffmpeg 构建的 libass 对 Style 行
    #   Align 不生效（会左对齐到 MarginL），{\an2}/{\an5\pos()} 才可靠（实测）。
    lines = [ASS_HEAD % {"w": w, "h": h, "fsize": int(size or SUB_SIZE),
                         "ml": SUB_MARGIN_LR, "mr": SUB_MARGIN_LR, "mv": SUB_MARGIN_V,
                         "fontname": fontname, "fbold": int(bold), "fitalic": int(italic)}]
    if pos == "upper":
        tag = "{\\an5\\pos(%d,%d)}" % (w // 2, int(round(h * SUB_POS_UPPER)))
    elif pos == "lower":
        tag = "{\\an5\\pos(%d,%d)}" % (w // 2, int(round(h * SUB_POS_LOWER)))
    else:
        tag = "{\\an2}"
    for c, s, e in cues:
        lines.append("Dialogue: 0,%s,%s,En,,0,0,0,,%s%s\n" % (ass_time(s), ass_time(e), tag, ass_escape(c)))
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    return path


# ---------------------------------------------------------------- 主流程
def apply_voiceover(video_in, lines, starts, out_path, bgm=None, duck=0.20,
                    voice=DEFAULT_VOICE, progress=None, work_dir=None,
                    sub_pos="lower"):
    """成片 + 口播稿 → 带配音/字幕的新成片（不改原文件）。

    video_in : 渲染好的成片（可无音轨；有音轨=已含 BGM，直接压低 duck）
    lines    : 每行一句口播（空串 = 该分镜不配）
    starts   : 每行起始秒（与 lines 等长）
    bgm      : 视频无音轨时用作背景音乐（必需；有音轨时忽略）
    sub_pos  : "upper"|"lower"|"bottom" 字幕位置。默认 "lower"（中下）——
               口播匹配的字幕四个模版一律中下（用户 2026-10-20 定稿）。
    返回 meta：lines/starts/tempos/音频时长/cues/输出路径。
    """
    lines = [str(x or "").strip() for x in (lines or [])]
    starts = [float(s) for s in (starts or [])]
    if len(starts) != len(lines):
        raise ValueError("lines/starts length mismatch")
    if not any(lines):
        raise ValueError("empty narration")

    work_dir = work_dir or os.path.join(os.path.dirname(os.path.abspath(out_path)), "_vo_work")
    os.makedirs(work_dir, exist_ok=True)
    fps, vdur, (w, h) = probe(video_in)

    def prog(pct, msg):
        if progress:
            progress(pct, 100, msg)

    # 1) 逐行 TTS（词级时间戳）+ atempo 适配
    prog(10, "Synthesizing narration")
    metas, words_by_line, tempos = [], {}, []
    for i, line in enumerate(lines):
        if not line:
            metas.append(None)
            tempos.append(1.0)
            continue
        audio = os.path.join(work_dir, "line_%02d.mp3" % i)
        words = tts_line(line, audio, voice)
        adur = max([wd["end"] for wd in words]) if words else probe(audio)[1]
        window = (starts[i + 1] - starts[i]) if i + 1 < len(starts) else (vdur - starts[i])
        tempo = fit_tempo(adur, window)
        if tempo > 1.0:
            print("  line %d: audio %.2fs > window %.2fs -> atempo %.3f" % (i, adur, window, tempo))
        metas.append({"audio": audio, "dur": round(adur, 2)})
        words_by_line[i] = words
        tempos.append(tempo)

    # 2) 混音（视频流 copy）
    prog(55, "Mixing narration over music")
    mixed = os.path.join(work_dir, "mixed.mp4")
    have_audio = has_audio(video_in)
    # 输入装配：[bgm?] , 视频 , 逐行人声音频
    ff_in = []
    if not have_audio and bgm:
        ff_in.append(("loop", bgm))       # 0
    video_idx = len(ff_in)
    ff_in.append(("plain", video_in))
    line_idx = {}
    for i, m in enumerate(metas):
        if m:
            line_idx[i] = len(ff_in)
            ff_in.append(("plain", m["audio"]))

    fc = []
    if have_audio:
        fc.append("[%d:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                  "volume=%.3f[bg]" % (video_idx, duck))
    elif bgm:
        fc.append("[%d:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                  "volume=%.3f,atrim=0:%.3f,afade=t=out:st=%.3f:d=1.5[bg]"
                  % (0, duck, vdur, max(0.0, vdur - 1.5)))
    mix_labels = ["[bg]"] if (have_audio or bgm) else []
    for i, m in enumerate(metas):
        if not m:
            continue
        delay_ms = int(round(starts[i] * 1000))
        chain = "aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo"
        if abs(tempos[i] - 1.0) > 1e-3:
            chain += ",atempo=%.3f" % tempos[i]
        chain += ",adelay=%d:all=1" % delay_ms
        fc.append("[%d:a]%s[vo%d]" % (line_idx[i], chain, i))
        mix_labels.append("[vo%d]" % i)
    if not mix_labels:
        raise RuntimeError("no audio sources to mix")
    fc.append("".join(mix_labels) + "amix=inputs=%d:dropout_transition=0:normalize=0[a]" % len(mix_labels))

    cmd = [FF, "-y"]
    for kind, path in ff_in:
        if kind == "loop":
            cmd += ["-stream_loop", "-1", "-i", path]
        else:
            cmd += ["-i", path]
    cmd += ["-filter_complex", ";".join(fc),
            "-map", "%d:v:0" % video_idx, "-map", "[a]", "-c:v", "copy", "-c:a", "aac",
            "-b:a", "192k", "-t", "%.3f" % vdur, "-movflags", "+faststart", mixed]
    _run(cmd)

    # 3) 字幕 cues + ASS + 烧录（一次编码）
    prog(80, "Burning subtitles")
    cues = build_cues(lines, starts, words_by_line, tempos, vdur)
    ass_path = os.path.join(work_dir, "sub.ass")
    write_ass(ass_path, cues, w, h, pos=sub_pos)
    esc = ass_path.replace("\\", "/").replace(":", "\\:")
    _run([FF, "-y", "-i", mixed, "-vf", "subtitles='%s'" % esc,
          "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
          "-c:a", "copy", "-movflags", "+faststart", out_path])

    prog(100, "Done")
    meta = {
        "source": video_in, "output": out_path, "voice": voice,
        "video_dur": round(vdur, 2), "size": (w, h),
        "lines": [{"text": lines[i], "start": starts[i],
                   "tempo": round(tempos[i], 3),
                   "audio_dur": (metas[i] or {}).get("dur")} for i in range(len(lines))],
        "cues": [[c, round(s, 2), round(e, 2)] for c, s, e in cues],
    }
    with open(os.path.join(work_dir, "vo_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


# ---------------------------------------------------------------- 三段式口播编排
# 规范（用户定稿 2026-10-20）：0-3s 产品亮相+钩子 / 3-10s 核心卖点（只讲一个）/
# 10-15s 行动引导（查看款式 / 咨询定制 / 申请样品 三选一）。
# 口播跨镜连贯叙述，不逐图报名字——款式名牌画面顶部已经有。
CTA_INVITES = {
    "view": "invite viewers to see the full collection of designs",
    "inquire": "invite viewers to inquire about custom orders",
    "samples": "invite viewers to request sample sets",
}


def beats_for_timeline(opener, seg, secs, n):
    """按 0-3 / 3-10 / 10-15 规范生成三段，起点钉在最近的分镜切换处。

    返回 (beats, seg_starts)；beats = [(name, start, end)]。
    注意：识别款名/镜头每次 Ark 返回略有差异 → opener 实测会在 2.2~4.0s 之间飘，
    所以三段起点必须按当次渲染结果现算，绝不硬编码。
    """
    seg_starts = [0.0] + [round(opener + k * seg, 3) for k in range(1, max(2, n))]

    def nearest(t):
        best = seg_starts[0]
        for s in seg_starts:
            if abs(s - t) < abs(best - t):
                best = s
        return best

    mid = nearest(3.0)
    cta = nearest(10.0)
    if mid <= 0.5 or mid >= secs - 2.0:
        mid = min(2.5, max(0.8, secs * 0.27))
    if cta <= mid + 1.0 or cta >= secs - 0.8:
        cta = max(mid + 2.0, secs - 5.0)
    return [("hook", 0.0, mid), ("selling", mid, cta), ("cta", cta, secs)], seg_starts


def three_act_beats(secs):
    """按成片时长给出规范三段（用户定稿规范 2026-10-20）。

    15s → 0-3 / 3-10 / 10-15；12s → 0-3 / 3-8 / 8-12；10s → 0-2 / 2-7 / 7-10。
    没有分镜边界概念的模版（单图运镜 / 滚动 / 四宫格）直接用规范时间，
    只有 Collection Showcase 这种多段结构才会走 beats_for_timeline 钉切换点。
    """
    secs = float(secs or 0.0)
    if secs >= 14:
        mid, cta = 3.0, 10.0
    elif secs >= 11:
        mid, cta = 3.0, 8.0
    elif secs >= 8:
        mid, cta = 2.0, 7.0
    else:
        mid, cta = round(secs * 0.25, 2), round(secs * 0.65, 2)
    return [("hook", 0.0, mid), ("selling", mid, cta), ("cta", cta, secs)]


def build_contact_sheet(srcs, sheet_path, cell_w=360):
    """把 N 张图拼成 4 列网格（左→右、上→下 = 播放顺序），供视觉起稿用。"""
    import cv2
    import numpy as np
    cells = []
    for p in srcs:
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        cells.append(cv2.resize(img, (cell_w, max(1, int(cell_w * h / w)))))
    if not cells:
        raise RuntimeError("no readable images for contact sheet")
    rowh = max(c.shape[0] for c in cells)
    canvas = np.zeros((rowh * 2, cell_w * 4, 3), np.uint8)
    for i, c in enumerate(cells[:8]):
        r, k = divmod(i, 4)
        canvas[r * rowh:r * rowh + c.shape[0], k * cell_w:k * cell_w + c.shape[1]] = c
    cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(sheet_path)
    return sheet_path


def draft_three_act(cfg, srcs, beats, spans, labels, selling, cta, sheet_path, timeout=120):
    """Ark 视觉起稿：给时间线 + 图，产出 hook/selling/cta 三句。失败返回 None。

    beats : beats_for_timeline 的输出
    spans : [(first_idx, last_idx)] 每段画面涵盖的款式下标（仅供起稿参考）
    selling : 用户填写的核心卖点（空 = 让模型自己从图里总结）
    cta      : view / inquire / samples
    """
    import traceback
    from core.showcase import chat_vision
    windows = [b[2] - b[1] for b in beats]
    budgets = [max(3, int(round(w * 2.3))) for w in windows]
    names = [l if l else "the piece on screen" for l in labels]
    while len(names) < len(srcs):
        names.append("the piece on screen")
    (b0, b1), (c0, c1) = spans[1], spans[2]
    tl = (
        "Beat 1 HOOK (0 to %.1fs, opening shot with design 1 %s): product reveal + "
        "attention grab. One sentence that names the product line and its most "
        "attractive quality. Budget ~%d words.\n"
        "Beat 2 SELLING POINT (%.1f to %.1fs, designs %s through %s on screen): "
        "exactly ONE core selling point - the craft, design or styling effect. "
        "%s Budget ~%d words.\n"
        "Beat 3 CALL TO ACTION (%.1f to %.1fs, designs %s through %s on screen): "
        "%s. Budget ~%d words."
        % (beats[0][2], names[0], budgets[0],
           beats[1][1], beats[1][2], names[b0], names[b1],
           ("The user-provided selling point to work in verbatim: \"%s\"." % selling)
           if selling else
           "Infer the single strongest selling point yourself from the photos.",
           budgets[1],
           beats[2][1], beats[2][2], names[c0], names[c1],
           CTA_INVITES.get(cta, CTA_INVITES["view"]).capitalize(),
           budgets[2])
    )
    sys_prompt = (
        "You are a voiceover scriptwriter for a short vertical product "
        "showcase video. FIRST look at the photo grid and identify what the "
        "video actually shows (e.g. wedding invitations, stationery, floral "
        "arrangements, venue decor, cakes, jewelry, apparel - whatever the "
        "photos reveal). The narration MUST match that actual subject matter; "
        "never assume it is wedding invitations unless the photos really show "
        "them. You are given the photos (left to right, top to bottom) in "
        "playback order. Write the narration as THREE continuous sentences "
        "following the beat structure given. Rules: "
        "(1) warm, premium, unhurried; English only; "
        "(2) each sentence MUST respect its word budget - it plays over that "
        "window only; "
        "(3) do NOT list item names inside the narration - the voiceover "
        "carries feeling, the screen carries names; "
        "(4) describe only what is visible (motifs, colors, printed text, "
        "materials); never invent products, prices, or contact channels; "
        "(5) respond with ONLY a JSON object, no code fences: "
        "{\"hook\": \"...\", \"selling\": \"...\", \"cta\": \"...\"}"
    )
    user = ("Playback timeline of the finished video:\n" + tl +
            "\n\nWrite the three narration sentences now.")
    try:
        txt, _u, _e = chat_vision(cfg, sys_prompt, user, sheet_path,
                                  max_tokens=500, timeout=timeout)
        m = re.search(r"\{.*\}", txt or "", re.S)
        j = json.loads(m.group(0)) if m else {}
        cand = [str(j.get(k) or "").strip() for k in ("hook", "selling", "cta")]
        if all(cand):
            return cand
    except Exception:
        traceback.print_exc()
    return None


def voiceover_on_rendered(video_in, out_path, srcs, opener, seg, secs, labels,
                          selling, cta, cfg, progress=None, fallback_bgm=None,
                          duck=0.20, voice=DEFAULT_VOICE, beats=None,
                          sub_pos="lower"):
    """对已渲染成片（通常已含 BGM）追加三段式口播 + 字幕。

    opener/seg/secs 取自渲染结果；labels 是成片上实际显示的款式名。
    beats : 直接给定三段 [(name, start, end)]（None = 按分镜边界自动算，
    见 beats_for_timeline）。单图/滚动/四宫格等无分镜切换概念的模版
    应传 three_act_beats(secs)。
    sub_pos: 字幕位置，默认 "lower"（中下）——口播字幕四个模版统一。
    任何失败都抛错——调用方（job）应 catch 后仍交付无口播成片。
    """
    n = max(2, len(srcs))
    if beats:
        seg_starts = [0.0] + [round(opener + k * seg, 3) for k in range(1, max(2, n))]
    else:
        beats, seg_starts = beats_for_timeline(opener, seg, secs, n)

    def owner(t):
        for i in range(n - 1, -1, -1):
            if seg_starts[i] <= t + 1e-6:
                return i
        return 0

    spans = [(owner(b[1]), owner(max(b[1], b[2] - 0.01))) for b in beats]
    sheet = os.path.join(os.path.dirname(os.path.abspath(out_path)), "_vo_contact.jpg")
    build_contact_sheet(srcs, sheet)
    lines = draft_three_act(cfg, srcs, beats, spans, labels, selling, cta, sheet)
    if not lines:
        raise RuntimeError("three-act narration draft failed")
    starts = [b[1] for b in beats]
    return apply_voiceover(video_in, lines, starts, out_path,
                           bgm=fallback_bgm, duck=duck, voice=voice,
                           progress=progress, sub_pos=sub_pos)


# ---------------------------------------------------------------- captions（屏幕文字）
CAPTION_FALLBACK = "Wedding invitations, crafted with care"


def draft_caption_line(cfg, srcs, sheet_path=None, timeout=60):
    """为无原生屏幕文字的模版（Product Spotlight / Colour Inspiration）
    起稿一句简短产品介绍（≤10 词）。失败返回 CAPTION_FALLBACK，绝不抛错。"""
    import tempfile
    import traceback
    from core.showcase import chat_vision
    srcs = [s for s in (srcs or []) if s][:4]
    if not srcs:
        return CAPTION_FALLBACK
    sheet = sheet_path or os.path.join(tempfile.gettempdir(), "_wb_cap_sheet.jpg")
    try:
        build_contact_sheet(srcs, sheet)
    except Exception:
        traceback.print_exc()
        return CAPTION_FALLBACK
    sys_prompt = (
        "You are a copywriter for short vertical wedding-invitation promo videos. "
        "You are given the product photo(s). Write ONE short on-screen caption "
        "line (maximum 10 words) that introduces the product: warm, premium, "
        "concrete. English only. No hashtags, no quotes, no emoji, no exclamation. "
        "Respond with ONLY a JSON object, no code fences: {\"line\": \"...\"}"
    )
    user = "Write the single on-screen caption line now."
    try:
        txt, _u, _e = chat_vision(cfg, sys_prompt, user, sheet,
                                  max_tokens=120, timeout=timeout)
        m = re.search(r"\{.*\}", txt or "", re.S)
        j = json.loads(m.group(0)) if m else {}
        line = str(j.get("line") or "").strip().strip('"').strip()
        if line and len(line.split()) <= 14:
            return line
    except Exception:
        traceback.print_exc()
    return CAPTION_FALLBACK


def apply_caption(video_in, out_path, line, progress=None, sub_pos="upper",
                  font=None):
    """把一行产品介绍文字全程烧在画面上（captions 开、voiceover 关时用）。

    默认烧在中上方 + 加大字号（CAP_SIZE）——用户 2026-10-20 要求：无原生
    文字的模版（Product Spotlight / Colour Inspiration）captions 放中上方、
    字体大一点；Collection/Grid 顶部/中央有原生文字，不走这里。
    font: 客户自选字体键（2026-10-26 core.fonts 注册表）——ASS Fontname 写
    字体族名、subtitles 滤镜加 fontsdir 指到随包 static/fonts/，libass 按
    族名+Bold/Italic 旗标精确匹配（None = 默认斜体 CG，即各模版现行观感）。
    视频流重编码（烧字幕），音频流 copy；不改原文件。
    """
    from core import fonts as _fonts
    line = str(line or "").strip()
    if not line:
        raise ValueError("empty caption line")
    work_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)), "_cap_work")
    os.makedirs(work_dir, exist_ok=True)
    fps, vdur, (w, h) = probe(video_in)
    if progress:
        progress(50, 100, "Burning on-screen text")
    ass_path = os.path.join(work_dir, "cap.ass")
    fb, fi = _fonts.ass_flags(font)
    write_ass(ass_path, [(line, 0.0, vdur)], w, h, pos=sub_pos, size=CAP_SIZE,
              fontname=_fonts.ass_family(font), bold=fb, italic=fi)
    esc = ass_path.replace("\\", "/").replace(":", "\\:")
    fdir = _fonts.FONT_DIR.replace("\\", "/").replace(":", "\\:")
    _run([FF, "-y", "-i", video_in,
          "-vf", "subtitles='%s':fontsdir='%s'" % (esc, fdir),
          "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
          "-c:a", "copy", "-movflags", "+faststart", out_path])
    if progress:
        progress(100, 100, "Done")
    return {"output": out_path, "line": line, "video_dur": round(vdur, 2)}
