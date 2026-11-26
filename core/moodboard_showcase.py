# -*- coding: utf-8 -*-
"""Moodboard Showcase 模版（Moodboard 品牌板）。

客户上传照片后自动生成「moodboard 品牌板」展示视频：
  * 背景四格：第 2-5 张照片依次缩放淡入，拼成 2×2 全出血拼图；
  * 色块圆点：4 个主色圆点（左侧竖排，白描边）按上传图识别弹出——
    复用 core.collage_palette 的识别口径（上传画像跨图投票 →
    同色系浅→深 4 档），客户手动编辑（palette_edit）最后生效；
  * 主卡片：第 1 张照片做成白边卡片（含投影）上滑淡入居中；
  * 贴纸：可选 —— 对某张上传图做自动抠图（GrabCut + 边缘泛洪清洗，
    带 5-40% 覆盖率与贴边率质量门禁，不达标自动跳过），加白描边后
    从右侧滑入右下角（出血）；sticker 参数可显式指定图或关闭；
  * 定格后进入全屏轮播：主卡原图 + 四格原图依次硬切展示（每张 1.5s，
    上传更多图则追加），最后回到拼合图定格并淡出。

配乐由 server 端混（默认 moodboard_nuanguang.mp3）。

接口
----
render_moodboard(srcs, out_path, ratio="9x16", fps=30, progress=None,
                 palette_edit=None, forced_family=None,
                 sticker="auto", sticker_src=None)
  srcs         : 图片路径列表（≥5 张：第 1 张=主卡片，第 2-5 张=四格，
                 第 6 张起进轮播追加，最多取到 8）
  sticker      : "auto"（默认，自动选图抠图，失败则无贴纸）/ "none" / 路径
  返回 dict：{video, secs, images, ratio, palette, palette_colors,
              sticker_src, sticker_used}
（只出画面，不含配乐）
"""
import os
import json
import time
import traceback

import numpy as np
import cv2

from core.card_pan import FFMPEG, ratio_size, norm_ratio
from core.collage_palette import (
    _imread_bgr, detect_palette, palette_colors_of, complete_palette,
    name_palette, _hex_to_rgb, image_color_profile, pick_family, N_SWATCH,
    palette_from_text,
)

CRF = 18
PRESET = "medium"

# ---------------- 画布与版式常量（与 2026-11-24 定稿样片一致） ----------------
W0, H0 = 1080, 1920
CW, CH = W0 // 2, H0 // 2            # 四格单元 540×960

T_TILE0, T_TILE_STEP, T_TILE_DUR = 0.00, 0.42, 0.78   # 四格依次出现
T_CARD, T_CARD_DUR = 3.30, 1.05                       # 主卡上滑淡入
# 2026-11-25 三次反馈：色块要压在「主卡左上角」上（参考截图），所以必须等
# 主卡落位后再弹出，否则会浮在四格上、主卡滑入时从色块底下穿过。
T_DOT0, T_DOT_STEP, T_DOT_DUR = 4.40, 0.13, 0.40      # 色点依次弹出
T_STK, T_STK_DUR = 5.30, 0.85                         # 贴纸右下滑入
T_HOLD0 = 6.85                                        # 拼合完成定格
T_HOLD1 = 8.70                                        # 轮播开始
G_DUR = 1.50                                          # 每张轮播时长
G_DUR_MANY = 1.20                                     # 轮播超过 10 张时（控总长）
END_HOLD = 0.90                                       # 回拼合后的定格
FADE_OUT = 0.60                                       # 片尾淡出

# 2026-11-25 第 4 版口径：模版更名「Wedding Moodboard」，主卡改竖长形状
# （470×830，经典请柬竖卡比例），仍为照片 _cover 居中裁切，不做合成。
CARD_W, CARD_H = 470, 830                             # 主卡竖长宽×高
STK_RATIO = 0.62                                      # 贴纸高 = 主卡高 × 该比例（客户要求放大）
STROKE_PX = 6                                         # 贴纸白描边
MAX_PHOTOS = 20                                       # 最多参与的张数（防止超长渲染）
# 2026-11-26：原来是 MAX_ROTATION = 8 且直接 paths[:8] —— 客户传 9 张以上时
# 第 9 张起被静默丢弃，成片里根本没有出镜（客户反馈「有的图片没有」）。
MAX_ROTATION = 20                                     # 兼容旧名：现等同于 MAX_PHOTOS

# 色块（2026-11-25 三次反馈）：
#   · 恢复「带描边」样式 —— 实色芯 + 近白描边圈 + 极淡软影（截图里就是这版）；
#   · 位置改为贴主卡左上角（DOT_ANCHOR="right" 时贴右上角）；
#   · 抠图贴纸放到与之对称的另一侧下角，形成对角构图。
DOT_R = 56                                            # 色芯半径
DOT_STROKE = 7                                        # 描边圈宽度
DOT_GAP = 26                                          # 圆与圆的纵向间隙
DOT_ANCHOR = "left"                                   # 左上角（截图样式）
DOT_INSET_X = 6                                       # 圆心相对主卡左/右边缘的内移
DOT_INSET_Y = 70                                      # 首个圆心相对主卡顶边的下移


# ---------------- 缓动 ----------------
def _clamp01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def _ease_out_cubic(t):
    t = _clamp01(t)
    return 1 - (1 - t) ** 3


def _ease_in_out(t):
    t = _clamp01(t)
    return t * t * (3 - 2 * t)


def _ease_out_back(t, s=1.9):
    t = _clamp01(t)
    return 1 + (s + 1) * (t - 1) ** 3 + s * (t - 1) ** 2


# ---------------- 基础图层 ----------------
def _cover(img, tw, th, fx=0.5, fy=0.5):
    h, w = img.shape[:2]
    sc = max(tw / w, th / h)
    r = cv2.resize(img, (int(w * sc + .5), int(h * sc + .5)),
                   interpolation=cv2.INTER_AREA)
    mw, mh = r.shape[1] - tw, r.shape[0] - th
    return r[int(mh * fy):int(mh * fy) + th, int(mw * fx):int(mw * fx) + tw]


def _paste(canvas, rgba, x, y, alpha=1.0):
    if alpha <= 0.003:
        return
    h, w = rgba.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x0 >= x1 or y0 >= y1:
        return
    sub = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    if sub.shape[2] == 3:
        sub = np.dstack([sub, np.full(sub.shape[:2], 255, np.uint8)])
    a = (sub[:, :, 3:4].astype(np.float32) / 255.0) * alpha
    canvas[y0:y1, x0:x1] = (sub[:, :, :3] * a +
                            canvas[y0:y1, x0:x1].astype(np.float32) * (1 - a)).astype(np.uint8)


def _drop_shadow(canvas, rgba, x, y, alpha=1.0, strength=52, blur=20, dy=14):
    if alpha <= 0.003:
        return
    h, w = rgba.shape[:2]
    a = cv2.GaussianBlur(rgba[:, :, 3], (0, 0), blur).astype(
        np.float32) / 255.0 * (strength / 255.0) * alpha
    x0, y0 = max(0, x), max(0, y + dy)
    x1, y1 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h + dy)
    if x0 >= x1 or y0 >= y1:
        return
    sa = a[y0 - dy - y:y1 - dy - y, x0 - x:x1 - x][:, :, None]
    canvas[y0:y1, x0:x1] = (canvas[y0:y1, x0:x1].astype(np.float32) *
                            (1 - sa)).astype(np.uint8)


def _draw_tile(canvas, img, x0, y0, tw, th, scale=1.0, alpha=1.0):
    if alpha <= 0.003:
        return
    if abs(scale - 1.0) > 1e-3:
        # 2026-11-24 踩坑：scale 浮点误差 0.9999999 时裁出 1×1 像素，
        # numpy 广播静默铺满整格 → nw/max 必须不小于目标尺寸
        nw = max(tw, int(round(tw * scale)))
        nh = max(th, int(round(th * scale)))
        tmp = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        cx = max(0, (nw - tw) // 2)
        cy = max(0, (nh - th) // 2)
        r = tmp[cy:cy + th, cx:cx + tw]
        if r.shape[0] != th or r.shape[1] != tw:
            r = img
    else:
        r = img
    if alpha >= 0.997:
        canvas[y0:y0 + th, x0:x0 + tw] = r
    else:
        canvas[y0:y0 + th, x0:x0 + tw] = (
            r.astype(np.float32) * alpha +
            canvas[y0:y0 + th, x0:x0 + tw].astype(np.float32) * (1 - alpha)).astype(np.uint8)


def _draw_dot(canvas, cx, cy, r_out, r_in, bgr, alpha=1.0):
    """带描边的实心圆色块（2026-11-25 第 3 版，对齐客户参考截图）。

    结构由内到外：实色芯 → 近白描边圈 → 极淡软影。
    抗锯齿用「4× 超采样 + INTER_AREA 降采样」而不是 cv2.LINE_AA：上一版
    LINE_AA 的覆盖权重会让边缘像素与底色混出浅色圈（客户两次反馈「一圈有
    描边」）；超采样下采样得到的是 1px 内的干净过渡，既无锯齿也无假边。

    r_out 是描边圈外半径（调用处传 r_in + DOT_STROKE），r_in 是色芯半径。
    """
    if alpha <= 0.003 or r_in <= 0.6:
        return
    r_out = max(float(r_in) + 1.0, float(r_out))
    SS = 4                                   # 超采样倍率
    pad = int(np.ceil(r_out)) + 26           # 留出软影空间
    x0, y0 = max(0, int(cx) - pad), max(0, int(cy) - pad)
    x1, y1 = min(W0, int(cx) + pad), min(H0, int(cy) + pad)
    if x0 >= x1 or y0 >= y1:
        return
    h, w = y1 - y0, x1 - x0
    cc = ((int(cx) - x0) * SS + SS // 2, (int(cy) - y0) * SS + SS // 2)

    core_s = np.zeros((h * SS, w * SS), np.uint8)
    cv2.circle(core_s, cc, max(1, int(round(r_in * SS))), 255, -1, cv2.LINE_8)
    outer_s = np.zeros((h * SS, w * SS), np.uint8)
    cv2.circle(outer_s, cc, max(1, int(round(r_out * SS))), 255, -1, cv2.LINE_8)
    core = cv2.resize(core_s, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    outer = cv2.resize(outer_s, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    ring = np.clip(outer - core, 0.0, 1.0)
    # 软影只落在外圈之外，让圆盘从照片上「托」起来（截图里也有这层淡影）
    halo = cv2.GaussianBlur(outer, (0, 0), max(2.0, r_out * 0.10))
    shadow = np.clip((halo - outer) * 1.6, 0.0, 1.0) * 0.30

    region = canvas[y0:y1, x0:x1].astype(np.float32)
    dark = np.array([58, 52, 48], np.float32)
    ring_c = np.array([250, 248, 244], np.float32)
    a = float(alpha)
    cov = (shadow * a)[:, :, None]
    region = region * (1 - cov) + dark * cov
    cov = (ring * a)[:, :, None]
    region = region * (1 - cov) + ring_c * cov
    cov = (core * a)[:, :, None]
    region = region * (1 - cov) + np.array(bgr, np.float32) * cov
    canvas[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)


# ---------------- 主卡片（白边 + 投影） ----------------
_LIGHT_FAMILIES = ("blush", "rose", "gold", "butter", "sage",
                   "dustyblue", "lilac")


def family_for_moodboard(profs):
    """品牌板主题族：深木/家具色（cocoa 等）不参与投票。

    实测（2026-11-25 样片）：6 张粉调婚礼图里 3 张因深木桌椅被判 cocoa，
    票数压过 rose/blush → 色点整组变深棕，与画面完全不搭。品牌板的
    配色语义是「花材/纸品/布艺的婚礼主题色」，深木家具属于背景道具，
    与叶材（绿）同理。故只要出现任一浅彩族（blush/rose/gold/sage/
    dustyblue/lilac…），就只在这些图里投票；一张浅彩族都没有（真·深色
    主题，如纯酒红/黑白）则退回全量投票，行为与 collage_palette 一致。
    """
    if not profs:
        return None
    light = [p for p in profs if (p.get("family") or "") in _LIGHT_FAMILIES]
    pool = light if light else profs
    fam = pick_family(pool)
    # pick_family 已做旧名映射；这里只做兜底校验
    return fam or None


def _bake_shadow(rgba, strength=52, blur=20, dy=14, pad=64):
    """把投影烘进一张加边 RGBA，供逐帧只做贴图（省掉每帧高斯模糊）。"""
    h, w = rgba.shape[:2]
    out = np.zeros((h + pad * 2 + dy, w + pad * 2, 4), np.uint8)
    sh = cv2.GaussianBlur(rgba[:, :, 3], (0, 0), blur).astype(
        np.float32) / 255.0 * (strength / 255.0)
    lay = out[pad:pad + h, pad:pad + w]
    # 先铺阴影（黑，alpha = 模糊后的主体轮廓），再叠原图
    lay[:, :, 3] = np.clip(sh * 255, 0, 255).astype(np.uint8)
    src = np.zeros((h + pad * 2 + dy, w + pad * 2, 4), np.uint8)
    src[pad + dy:pad + dy + h, pad:pad + w] = rgba
    a1 = out[:, :, 3:4].astype(np.float32) / 255.0
    a2 = src[:, :, 3:4].astype(np.float32) / 255.0
    ra = np.clip(a1 + a2 * (1 - a1), 0, 1)
    rgb = (src[:, :, :3].astype(np.float32) * a2) / np.maximum(ra, 1e-4)
    out[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[:, :, 3] = (ra[:, :, 0] * 255).astype(np.uint8)
    return out, pad


def _make_card(src_bgr, hero_rgba=None):
    """照片 → 竖长主卡 RGBA（CARD_W×CARD_H，无白边）。

    2026-11-25 第 4 版客户口径：中间图片改竖长形状（470×830 请柬竖卡
    比例），仍是照片 _cover 居中裁切、不做「抠图产品 + 虚化底」合成；
    抠出的产品继续放主卡右下角做贴纸（与左上角色块对角对称）。
    hero_rgba 参数保留但已不参与合成，仅为兼容历史调用签名。
    """
    photo = _cover(src_bgr, CARD_W, CARD_H)
    rgba = np.zeros((CARD_H, CARD_W, 4), np.uint8)
    rgba[:, :, :3] = photo
    rgba[:, :, 3] = 255
    return rgba


# ---------------- 版式坐标（主卡 / 色块 / 贴纸共用一套，防动画与定格错位） ----------------
def _card_rect(card):
    chh, cww = card.shape[:2]
    return (W0 - cww) // 2, int(H0 * 0.44 - chh / 2), cww, chh


def _dot_centers(card, n, anchor=None):
    """色块圆心：贴主卡左上角（或右上角）竖排。

    参考截图：第一个圆的圆心略低于主卡顶边、横向压在主卡侧边上，圆有一半
    落在照片上、一半落在四格上——所以这里以主卡边缘为基准而不是画布边缘。
    """
    cx, cy, cww, chh = _card_rect(card)
    anchor = anchor or DOT_ANCHOR
    x = cx + DOT_INSET_X if anchor != "right" else cx + cww - DOT_INSET_X
    step = 2 * DOT_R + DOT_GAP
    # 竖排过高时自动压缩间距（极端比例画幅）
    room = chh - DOT_INSET_Y - DOT_R - 20
    if n > 1 and (n - 1) * step > room:
        step = max(2 * DOT_R + 6, room // (n - 1))
    return [(x, cy + DOT_INSET_Y + i * step) for i in range(n)]


def _fit_sticker(sticker_rgba, card):
    """贴纸按主卡边长等比缩放（约为卡片高度的一半），返回缩放后的 RGBA。"""
    chh = card.shape[0]
    target = int(round(chh * STK_RATIO))
    sh, sw = sticker_rgba.shape[:2]
    s = target / float(sh)
    return cv2.resize(sticker_rgba, (max(1, int(sw * s)), target),
                      interpolation=cv2.INTER_AREA)


def _sticker_xy(st, card, overlap=0.5):
    """贴纸落在与色块对称的右下角：圆心压在主卡右下角，一半外扩。

    overlap=0.5 → 贴纸中心≈主卡右下角。之前的「右下角贴边」写法会把贴纸
    几乎整块塞进卡片里（实测遮住主卡请柬的右半边），这里改为跨在卡角上。
    """
    cx, cy, cww, chh = _card_rect(card)
    x = cx + cww - int(round(st.shape[1] * overlap))
    y = cy + chh - int(round(st.shape[0] * overlap))
    x = max(0, min(x, W0 - st.shape[1]))
    y = max(0, min(y, H0 - st.shape[0]))
    return x, y


# ---------------- 自动抠图（通用：GrabCut 边界初始化 + 边缘泛洪清洗） ----------------
def _bg_color_model(img, thr=24.0):
    """从「四角小块 + 上下边带」估背景色 → 返回 (离背景远的像素掩码, 背景色数组)。

    为什么不用左右边带（2026-11-25 实测）：人物/产品常顶满画幅左右边，
    边带里混进主体本身的颜色，背景模型被污染后会把主体的裙摆、花瓣当成
    背景整片抹掉（粉裙被掏空）。四角 + 上下边的场景信息更干净：
    天花板/墙面在上、地面/桌面在下。
    """
    gh, gw = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    kb = max(6, int(min(gh, gw) * 0.12))          # 角块边长
    tb = max(5, int(gh * 0.06))                    # 上下边带高
    masks = []
    for ys in (slice(0, kb), slice(gh - kb, gh)):
        for xs in (slice(0, kb), slice(gw - kb, gw)):
            m = np.zeros((gh, gw), bool)
            m[ys, xs] = True
            masks.append(m)
    for ys in (slice(0, tb), slice(gh - tb, gh)):
        m = np.zeros((gh, gw), bool)
        m[ys, :] = True
        masks.append(m)
    px = np.concatenate([lab[m] for m in masks], axis=0)
    if len(px) < 64:
        return np.zeros((gh, gw), bool), np.zeros((1, 3), np.float32)
    q = (px // 12).astype(np.int32)
    keys, counts = np.unique(q, axis=0, return_counts=True)
    order = np.argsort(-counts)
    bg, cum, total = [], 0.0, float(len(px))
    for i in order[:8]:
        m = (q == keys[i]).all(axis=1)
        bg.append(px[m].mean(axis=0))
        cum += counts[i] / total
        if cum >= 0.72:
            break
    bg = np.asarray(bg, dtype=np.float32)
    d = np.sqrt(((lab[:, :, None, :] - bg[None, None, :, :]) ** 2).sum(axis=3)).min(axis=2)
    fg = (d > thr).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return fg > 0, bg


def auto_cutout(src_bgr, stroke=STROKE_PX):
    """从照片里抠主体（人物/产品/花束等），输出白描边 RGBA。

    返回 (rgba, score) 或 (None, 0.0)。流程：
      0) 原图自带 alpha（客户上传已抠好的 PNG）→ 直接用；
      1) 四角 + 上下边建模背景色 → 差异像素作候选主体（不依赖中央构图）；
      2) 候选主体作 GrabCut probable-fg 种子，边框作 probable-bg；
      3) 背景色回验清残留 + 保留主连通块 + 填小洞 + 羽化 + 白描边。
    质量门禁：覆盖率 6%-50%、四角不实（<15%）、镂空率 ≤12%、
    边框实心率 ≤55%；任一不过 → None（调用方回退为无贴纸，
    绝不出脏图）。这条门禁偏保守是刻意的：抠坏了比没有贴纸更伤品牌。
    """
    # 0) 自带透明通道：直接用（最可靠）
    if src_bgr.ndim == 3 and src_bgr.shape[2] == 4:
        a0 = src_bgr[:, :, 3]
        if (a0 < 20).mean() > 0.05:
            rgba = src_bgr.copy()
            ys, xs = np.where(a0 > 40)
            if len(ys) == 0:
                return None, 0.0
            rgba = rgba[max(0, ys.min() - 8):ys.max() + 9,
                        max(0, xs.min() - 8):xs.max() + 9]
            return _with_stroke(rgba, stroke), 1.0

    h, w = src_bgr.shape[:2]
    scale = 800.0 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(src_bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    else:
        img = src_bgr.copy()
    gh, gw = img.shape[:2]

    fg0, bg = _bg_color_model(img)
    num0, lab0, st0, _ = cv2.connectedComponentsWithStats(fg0.astype(np.uint8), 8)
    fgc = np.zeros((gh, gw), np.uint8)
    for i in range(1, num0):
        if st0[i, cv2.CC_STAT_AREA] >= gh * gw * 0.012:
            fgc[lab0 == i] = 255
    if fgc.sum() == 0:
        return None, 0.0

    b = max(2, int(min(gh, gw) * 0.02))
    bw = max(6, int(min(gh, gw) * 0.05))
    mask = np.full((gh, gw), cv2.GC_PR_BGD, np.uint8)
    mask[:b, :] = cv2.GC_BGD
    mask[-b:, :] = cv2.GC_BGD
    mask[:, :b] = cv2.GC_BGD
    mask[:, -b:] = cv2.GC_BGD
    core = cv2.erode(fgc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    ring = cv2.dilate(fgc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    mask[core > 0] = cv2.GC_FGD
    mask[(ring > 0) & (core == 0)] = cv2.GC_PR_FGD
    edge = np.zeros((gh, gw), bool)
    edge[:bw, :] = True
    edge[-bw:, :] = True
    edge[:, :bw] = True
    edge[:, -bw:] = True
    mask[edge & (fgc == 0)] = cv2.GC_BGD

    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return None, 0.0
    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    num, labs, st, _ = cv2.connectedComponentsWithStats((alpha > 0).astype(np.uint8), 8)
    if num <= 1:
        return None, 0.0
    keep = np.zeros_like(alpha)
    for i in range(1, num):
        if st[i, cv2.CC_STAT_AREA] < gh * gw * 0.02:
            continue
        keep[labs == i] = 255
    if keep.sum() == 0:
        return None, 0.0
    alpha = keep

    # 2026-11-25 修正：删除「全局背景色回验」。实测四人粉裙合影里粉裙与米白
    # 背景色接近，被整片误删（粉裙挖空成大洞）。GrabCut 已做前后景分离，
    # 不再叠加颜色判据，只在四边极窄带清残留（主体不会触到的地方）。
    edge_band = max(2, int(min(gh, gw) * 0.012))
    alpha[:edge_band, :] = 0
    alpha[-edge_band:, :] = 0
    alpha[:, :edge_band] = 0
    alpha[:, -edge_band:] = 0
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # 保留主连通块（≥3% 面积的块）
    num2, labs2, st2, _ = cv2.connectedComponentsWithStats((alpha > 0).astype(np.uint8), 8)
    if num2 > 1:
        big = 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))
        kk = np.zeros_like(alpha)
        for i in range(1, num2):
            if i == big or st2[i, cv2.CC_STAT_AREA] >= gh * gw * 0.03:
                kk[labs2 == i] = 255
        alpha = kk
    if alpha.sum() == 0:
        return None, 0.0

    # ---- 质量门禁 ----
    cov = float((alpha > 0).mean())
    # 2026-11-25：上限 0.50→0.68。多人合影/顶满画幅的主体覆盖率常到 0.56，
    # 旧上限会把好抠图误拒；0.68 仍能挡住「几乎整张都没抠」的失败情形。
    if not (0.06 <= cov <= 0.68):
        return None, 0.0
    k = max(20, int(min(gh, gw) * 0.09))
    corner = max(float((alpha[ys, xs] > 0).mean())
                 for ys in (slice(0, k), slice(gh - k, gh))
                 for xs in (slice(0, k), slice(gw - k, gw)))
    border = float(((alpha[:3, :] > 0).mean() + (alpha[-3:, :] > 0).mean() +
                    (alpha[:, :3] > 0).mean() + (alpha[:, -3:] > 0).mean()) / 4.0)
    # 封闭孔洞占比（只统计四周都有主体的孔）：轮廓凹处不算
    nh0, hl0, hs0, _ = cv2.connectedComponentsWithStats((alpha == 0).astype(np.uint8), 4)
    enclosed = 0
    for i in range(1, nh0):
        m = hl0 == i
        if (m[:1, :].any() or m[-1:, :].any() or m[:, :1].any() or m[:, -1:].any()):
            continue
        enclosed += int(hs0[i, cv2.CC_STAT_AREA])
    hole_ratio = enclosed / max(float((alpha > 0).sum()), 1.0)
    # 主块占比（碎片过多说明抠碎了）
    nc, lc, sc_, _ = cv2.connectedComponentsWithStats((alpha > 0).astype(np.uint8), 8)
    tot = float((alpha > 0).sum())
    main_share = float(sc_[1:, cv2.CC_STAT_AREA].max()) / max(tot, 1.0) if nc > 1 else 1.0
    if corner > 0.15 or border > 0.55 or hole_ratio > 0.30 or main_share < 0.55:
        return None, 0.0

    # 2026-11-25：细长残条门禁。场景图（长桌/会场/桌布）被 GrabCut 抠出
    # 的多是横跨画幅的细长条 + 碎块，贴出来就是一坨白边残条（客户截图实证）。
    # 真主体（人物/花艺/蛋糕/请柬）裁框后长宽比温和、填充较实；用这两个
    # 判据把「细长 + 稀疏」的场景残片挡在门外，宁可不出贴纸。
    ys_c, xs_c = np.where(alpha > 0)
    if len(xs_c) > 0:
        bh = ys_c.max() - ys_c.min() + 1
        bw = xs_c.max() - xs_c.min() + 1
        ar = bw / float(bh)
        bbox_fill = float((alpha > 0).sum()) / max(bh * bw, 1)
        if (ar > 3.2 or ar < 0.28) or bbox_fill < 0.34:
            return None, 0.0

    # 填小洞 → 羽化 → 裁框 → 白描边
    nh, hole_lab = cv2.connectedComponents((alpha == 0).astype(np.uint8), 4)
    for i in range(1, nh):
        hm = hole_lab == i
        if (hm[:2, :].any() or hm[-2:, :].any() or
                hm[:, :2].any() or hm[:, -2:].any()):
            continue
        if hm.sum() < gh * gw * 0.05:
            alpha[hm] = 255

    soft = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 0.9)
    soft = np.clip(np.maximum(soft, alpha.astype(np.float32) * 0.85), 0, 255)
    rgba = np.dstack([img, soft.astype(np.uint8)])
    ys, xs = np.where(soft > 40)
    rgba = rgba[max(0, ys.min() - 8):ys.max() + 9, max(0, xs.min() - 8):xs.max() + 9]
    out = _with_stroke(rgba, stroke)
    score = cov * (1.0 - corner) * (1.0 - min(hole_ratio, 0.99) * 0.5)
    return out, round(float(score), 3)

def cutout_for_card(src_bgr, box=None):
    """主卡居中产品抠图（2026-11-25 客户要求「居中产品做抠图」）。

    两条路径：
      1) box 给了（视觉模型返回的归一化产品框 [x0,y0,x1,y1]）→ 框内 GrabCut。
         这是主路径：实测对「餐盘上的请柬卡」能精确抠出卡片本体；纯靠颜色
         建模会把整片餐桌（桌布+餐盘+蜡烛）当前景，贴上去就是一块矩形照片。
      2) box 为 None → 全局背景建模兜底（粗，但不会因模型没给框而放弃）。

    与 auto_cutout（右下角贴纸）的区别：不要白描边、覆盖率放宽到 3%-85%、
    允许竖长/横长卡片，只挡真正失败的碎块。
    返回 (rgba, score)，失败 (None, 0.0)。调用方把结果贴到原图虚化底上。
    """
    # 自带 alpha：直接用
    if src_bgr.ndim == 3 and src_bgr.shape[2] == 4:
        a0 = src_bgr[:, :, 3]
        if (a0 < 20).mean() > 0.05:
            rgba = src_bgr.copy()
            ys, xs = np.where(a0 > 40)
            if len(ys) == 0:
                return None, 0.0
            rgba = rgba[max(0, ys.min() - 6):ys.max() + 7,
                        max(0, xs.min() - 6):xs.max() + 7]
            return rgba, 1.0

    h, w = src_bgr.shape[:2]
    scale = 800.0 / max(h, w)
    img = (cv2.resize(src_bgr, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA) if scale < 1.0
           else src_bgr.copy())
    gh, gw = img.shape[:2]

    if box is not None:
        # ---- 主路径：AI 产品框内 GrabCut（框外全背景，框心强前景）----
        x0, y0, x1, y1 = [int(v * d) for v, d in zip(box, (gw, gh, gw, gh))]
        x0 = max(0, min(gw - 4, x0))
        x1 = max(x0 + 4, min(gw, x1))
        y0 = max(0, min(gh - 4, y0))
        y1 = max(y0 + 4, min(gh, y1))
        mask = np.full((gh, gw), cv2.GC_BGD, np.uint8)
        mx = max(2, int((x1 - x0) * 0.06))
        my = max(2, int((y1 - y0) * 0.06))
        mask[y0 + my:y1 - my, x0 + mx:x1 - mx] = cv2.GC_FGD
        mask[max(0, y0 - my):y1 + my, max(0, x0 - mx):x1 + mx] = cv2.GC_PR_FGD
        try:
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            cv2.grabCut(img, mask, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
        except Exception:
            return None, 0.0
        alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                         255, 0).astype(np.uint8)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    else:
        # ---- 兜底：全局背景建模 + GrabCut ----
        fg0, _bg = _bg_color_model(img)
        num0, lab0, st0, _ = cv2.connectedComponentsWithStats(fg0.astype(np.uint8), 8)
        fgc = np.zeros((gh, gw), np.uint8)
        for i in range(1, num0):
            if st0[i, cv2.CC_STAT_AREA] >= gh * gw * 0.006:   # 比贴纸的 0.012 更宽
                fgc[lab0 == i] = 255
        if fgc.sum() == 0:
            return None, 0.0

        b = max(2, int(min(gh, gw) * 0.02))
        bw = max(6, int(min(gh, gw) * 0.05))
        mask = np.full((gh, gw), cv2.GC_PR_BGD, np.uint8)
        mask[:b, :] = cv2.GC_BGD
        mask[-b:, :] = cv2.GC_BGD
        mask[:, :b] = cv2.GC_BGD
        mask[:, -b:] = cv2.GC_BGD
        core = cv2.erode(fgc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        ring = cv2.dilate(fgc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        mask[core > 0] = cv2.GC_FGD
        mask[(ring > 0) & (core == 0)] = cv2.GC_PR_FGD
        edge = np.zeros((gh, gw), bool)
        edge[:bw, :] = True
        edge[-bw:, :] = True
        edge[:, :bw] = True
        edge[:, -bw:] = True
        mask[edge & (fgc == 0)] = cv2.GC_BGD
        try:
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            cv2.grabCut(img, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        except Exception:
            return None, 0.0
        alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                         255, 0).astype(np.uint8)

    # 只留最大连通块（产品是单一主体，直接掐掉碎块）
    n, lab, st, _ = cv2.connectedComponentsWithStats((alpha > 0).astype(np.uint8), 8)
    if n <= 1:
        return None, 0.0
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    alpha = np.where(lab == big, 255, 0).astype(np.uint8)
    cov = float((alpha > 0).mean())
    if cov < 0.03 or cov > 0.85:
        return None, 0.0
    ys_c, xs_c = np.where(alpha > 0)
    bh = ys_c.max() - ys_c.min() + 1
    bw2 = xs_c.max() - xs_c.min() + 1
    ar = bw2 / float(bh)
    fill = float((alpha > 0).sum()) / max(bh * bw2, 1)
    # 允许竖长请柬(ar~0.4)与横长菜单(ar~2.5)，但挡细长碎条与稀疏碎片。
    # 实测迎宾牌/布幔主体 fill=0.25（贯穿画面的立柱，形状正常只是不"实"），
    # 0.28 会误拒，放宽到 0.20；更低的碎块仍由「只留最大连通块」挡掉。
    if (ar > 4.5 or ar < 0.18) or fill < 0.20:
        return None, 0.0

    # 填封闭小洞（放宽到 15%）
    nh, hl = cv2.connectedComponents((alpha == 0).astype(np.uint8), 4)
    for i in range(1, nh):
        m = hl == i
        if (m[:2, :].any() or m[-2:, :].any() or
                m[:, :2].any() or m[:, -2:].any()):
            continue
        if m.sum() < gh * gw * 0.15:
            alpha[m] = 255
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    soft = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 0.8)
    soft = np.clip(np.maximum(soft, alpha.astype(np.float32) * 0.9), 0, 255)
    rgba = np.dstack([img, soft.astype(np.uint8)])
    ys, xs = np.where(soft > 40)
    if len(xs) == 0:
        return None, 0.0
    rgba = rgba[max(0, ys.min() - 6):ys.max() + 7,
                max(0, xs.min() - 6):xs.max() + 7]
    return rgba, round(float(cov), 3)


def _with_stroke(rgba, stroke=STROKE_PX):
    """给 RGBA 加一圈白描边（贴纸质感），返回加边后的 RGBA。

    2026-11-25 修正：旧版用 cv2.dilate 硬膨胀，边缘出现台阶锯齿（用户实测截图）。
    改为「羽化外扩」：膨胀后在浮点域做小高斯模糊，白边外轮廓平滑；再用
    3 层 alpha 合成（主体 / 白边 / 透明），避免硬阈值导致的白边断裂。
    """
    rh, rw = rgba.shape[:2]
    pad = stroke + 2
    a = rgba[:, :, 3].astype(np.float32) / 255.0
    # 主体 alpha 先轻微羽化，贴合轮廓
    a_soft = cv2.GaussianBlur(a, (0, 0), 0.8)
    a_soft = np.maximum(a_soft, a * 0.9)
    # 白边 = 膨胀后的 alpha，再做高斯羽化让外缘平滑（关键：不产生台阶）
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke * 2 + 1, stroke * 2 + 1))
    a_dil = cv2.dilate((a > 0.5).astype(np.uint8) * 255, k).astype(np.float32) / 255.0
    a_ring = cv2.GaussianBlur(a_dil, (0, 0), max(1.0, stroke * 0.45))
    a_ring = np.maximum(a_ring, a_dil * 0.85)
    # 合成：白边在下，主体在上
    H, W = rh + pad * 2, rw + pad * 2
    out_a = np.zeros((H, W), np.float32)
    out_rgb = np.zeros((H, W, 3), np.float32)
    ys, xs = slice(pad, pad + rh), slice(pad, pad + rw)
    ring_a = a_ring
    subj_a = a_soft
    # 白边层
    out_a[ys, xs] = ring_a
    out_rgb[ys, xs] = 255.0
    # 主体层叠上去
    ra = np.clip(subj_a + ring_a * (1 - subj_a), 0, 1)
    src_rgb = rgba[:, :, :3].astype(np.float32)
    blend = np.where(subj_a[:, :, None] > 1e-4,
                     (src_rgb * subj_a[:, :, None] + 255.0 * ring_a[:, :, None] * (1 - subj_a[:, :, None])) /
                     np.maximum(ra[:, :, None], 1e-4),
                     255.0)
    out_rgb[ys, xs] = blend
    out_a[ys, xs] = ra
    return np.dstack([np.clip(out_rgb, 0, 255).astype(np.uint8),
                      (np.clip(out_a, 0, 1) * 255).astype(np.uint8)])


def _read_maybe_alpha(p):
    """读图：PNG 保留 alpha（客户可直接上传已抠好的透明底图）。"""
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return img
    return img[:, :, :3]


# ---------------- AI 图片内容分类（自动选图：请柬居中 / 抠人物·蛋糕 / 四格不重复） ----------------
_CLASSIFY_SYS = """Classify wedding photos. Reply ONLY compact JSON, no markdown:
{"kind": "<invitation|people|cake|floral|scene>", "has_person": 0|1, "has_cake": 0|1, "paper_count": <int 0..9>, "subject_prominence": <0..1>, "isolatable": <0..1>, "product_box": [x0,y0,x1,y1] or null}
Definitions:
- invitation: the focal subject is PRINTED PAPER STATIONERY that you could pick up - an invitation card, envelope, RSVP, menu, place card, or a flat-lay of several such paper pieces (a suite). It may lie on a table.
- IMPORTANT: an acrylic / glass / foam-board / wooden SIGN, table number, welcome board, seating chart, or neon sign is NOT stationery even if it carries printed text -> use kind "scene" with paper_count 0. A draped fabric backdrop, arch, curtain, chair row, table set-up or large floral installation is also NOT invitation -> "scene" (even if a small card appears in it).
- people: a bride, groom, bridesmaid, couple or any person is the main subject.
- cake: a wedding cake / dessert is the main subject.
- floral: flowers / bouquet / arrangement is the main subject.
- scene: venue, aisle, long table, decor with no single pickable product.
paper_count = how many separate PRINTED PAPER pieces are clearly visible and in focus (0 = none, 1 = one card, 2+ = a suite / multiple pieces). This is the main signal for "is this real stationery product".
subject_prominence = how much one main subject fills and stands out in the frame (0 = background only, 1 = fills frame).
isolatable = how cleanly that main subject could be cut out as a standalone sticker: 1 = plain / contrasting background, single object; 0 = same-tone cluttered background, edges blended into the scene.
product_box = normalized 0..1 [x0,y0,x1,y1] TIGHTLY around that ONE subject (person, cake, bouquet, or the paper suite), with almost no surrounding background. EXCLUDE plates, napkins, tablecloth, candles, chairs, fabric and whole venue scenes. If there is no single isolatable subject, use null."""


def classify_photos(srcs, vision_cfg, progress=None):
    """并行调用视觉模型给每张图打内容标签。

    返回 list[dict]（与 srcs 等长）：{kind, has_person, has_cake,
    subject_prominence}。任一失败回退 kind=scene/prominence=0.5，绝不让
    分类失败阻断出片。配置无 key 时也直接回退（纯启发式排布）。
    """
    n = len(srcs)
    fallback = [{"kind": "scene", "has_person": False, "has_cake": False,
                 "subject_prominence": 0.5, "product_box": None,
                 "paper_count": 0, "isolatable": 0.3}
                for _ in range(n)]
    cfg = vision_cfg or {}
    if not str(cfg.get("api_key") or "").strip() or n == 0:
        return fallback
    try:
        from concurrent.futures import ThreadPoolExecutor
        from core.showcase import chat_vision
        import re as _re

        def _one(idx):
            # 2026-11-25 第 7 版：8 路并发实测会被 API 瞬时限流整批打挂
            # （8 个全进 fallback → 全判 scene、框全空）。加两次退避重试。
            for attempt in range(3):
                try:
                    if attempt:
                        time.sleep(2.5 * attempt)
                    txt, _u, _el = chat_vision(cfg, _CLASSIFY_SYS,
                                               "Classify this photo.", srcs[idx],
                                               max_tokens=220, timeout=90)
                    m = _re.search(r"\{.*?\}", txt, _re.S)
                    d = json.loads(m.group(0)) if m else {}
                    kind = str(d.get("kind") or "scene").strip().lower()
                    if kind not in ("invitation", "people", "cake", "floral", "scene"):
                        kind = "scene"
                    try:
                        prom = float(d.get("subject_prominence", 0.5))
                    except Exception:
                        prom = 0.5
                    prom = max(0.0, min(1.0, prom))
                    try:
                        iso = float(d.get("isolatable", 0.5))
                    except Exception:
                        iso = 0.5
                    iso = max(0.0, min(1.0, iso))
                    try:
                        paper = int(float(d.get("paper_count", 0)))
                    except Exception:
                        paper = 0
                    paper = max(0, min(9, paper))
                    # 产品框：严格校验归一化区间，格式不对就当没有（走兜底抠图）
                    pb = None
                    raw_box = d.get("product_box")
                    if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
                        try:
                            v = [float(t) for t in raw_box]
                            if (0.0 <= v[0] < v[2] <= 1.0 and
                                    0.0 <= v[1] < v[3] <= 1.0 and
                                    (v[2] - v[0]) >= 0.05 and (v[3] - v[1]) >= 0.05):
                                pb = v
                        except Exception:
                            pb = None
                    return idx, {"kind": kind,
                                 "has_person": bool(d.get("has_person")),
                                 "has_cake": bool(d.get("has_cake")),
                                 "subject_prominence": prom,
                                 "isolatable": iso,
                                 "paper_count": paper,
                                 "product_box": pb}
                except Exception:
                    continue
            return idx, None

        with ThreadPoolExecutor(max_workers=min(8, n)) as ex:
            for idx, d in ex.map(_one, range(n)):
                if d is not None:
                    fallback[idx] = d
        if progress:
            progress(15, 100, "Photos classified")
    except Exception:
        traceback.print_exc()
    return fallback


def pick_hero_by_cutout(paths, infos, bgrs):
    """在请柬候选中选「最容易抠出干净产品」的那张做居中主卡。

    2026-11-25 客户要求居中产品抠图后发现：AI 的 invitation 标签会把
    「场景里的布幔装置」和「餐盘上的请柬卡」都判成请柬，但前者抠出来是
    一坨布幔+木架+餐桌残片，后者才是干净产品。product_suitability 两者
    都是 0.7、区分度不足；改为直接跑一次宽松抠图，用「覆盖率 × bbox 填充率
    + 质量分」排序，选真正能抠出独立产品的那张。失败回退到 prominence 最高。
    """
    n = len(infos)
    if n == 0:
        return 0
    inv = [i for i, x in enumerate(infos) if x.get("kind") == "invitation"]
    if not inv:
        return max(range(n), key=lambda i: infos[i].get("subject_prominence", 0.5))
    if len(inv) == 1:
        return inv[0]
    best_i, best_q = inv[0], -1.0
    for i in inv:
        try:
            rgba, sc = cutout_for_card(bgrs[paths[i]],
                                       box=infos[i].get("product_box"))
            if rgba is None:
                q = 0.0
            else:
                a = rgba[:, :, 3] > 128
                ys, xs = np.where(a)
                bh = ys.max() - ys.min() + 1
                bw = xs.max() - xs.min() + 1
                fill = float(a.sum()) / max(bh * bw, 1)
                # 卡片/纸品是纯色平面：亮度方差小；布幔/装饰有强明暗渐变。
                # 实测布幔 L_std=74 vs 菜单卡 L_std=27，是区分两者的硬指标。
                lab = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2LAB)
                lstd = float(lab[:, :, 0][a].std()) if a.any() else 255.0
                # 轮廓规整度（凸包填充率）：卡片接近矩形 ≈0.97，碎布 ≈0.92
                sol = 0.0
                cnts, _ = cv2.findContours(a.astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    ca = cv2.contourArea(c)
                    ha = cv2.contourArea(cv2.convexHull(c))
                    sol = ca / max(ha, 1.0)
                q = sol * (1.0 - min(lstd / 120.0, 1.0)) * fill * 2.0
        except Exception:
            q = 0.0
        if q > best_q:
            best_i, best_q = i, q
    return best_i


# ---------------- 选图（2026-11-25 第 8 版：按内容决策，不看上传顺序） ----------------
def _subject_pri(info):
    """可抠主体优先级：人物 > 蛋糕 > 花艺 > 请柬纸品；纯场景 = 0。"""
    k = info.get("kind")
    if k == "people" or info.get("has_person"):
        return 5
    if k == "cake" or info.get("has_cake"):
        return 4
    if k == "floral":
        return 3
    if k == "invitation":
        return 2
    return 0


def _hero_score(info):
    """居中主卡评分：真纸品套系 > 单张纸卡。

    客户口径「居中优先请柬产品」。丙烯 / 泡沫板指示牌（桌号、迎宾牌）会被
    分类器判成 scene，不再抢占主卡；paper_count 是区分「套系纸品」与
    「场景里恰好有张卡」的硬信号。
    """
    if info.get("kind") != "invitation":
        return -1.0
    paper = int(info.get("paper_count") or 0)
    suite = 1.0 if paper >= 3 else (0.78 if paper == 2 else (0.45 if paper == 1 else 0.0))
    return (suite * 1.6
            + float(info.get("subject_prominence") or 0.5) * 0.7
            + float(info.get("isolatable") or 0.3) * 0.4)


def _img_hash(bgr, n=12):
    """平均哈希：用于剔除近似重复的图（同一场景连拍/同素材多份）。"""
    if bgr is None:
        return None
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (n, n), interpolation=cv2.INTER_AREA)
    return g > g.mean()


def _is_dup(h1, h2, thr=14):
    return h1 is not None and h2 is not None and int((h1 != h2).sum()) <= thr


def plan_shot_order(infos, count=None, force_hero=None, imgs=None):
    """内容驱动选图：返回 (hero_idx, tile_idxs[4], sticker_idx|None)。

    与上传顺序无关，全部按 AI 语义标签 + 质量分决策：
      1. hero（居中主卡）：必须是真纸品请柬（paper_count 越多越优先）。
         一张请柬都没有时，退到 人物 > 蛋糕 > 花艺，最后才用任意图。
      2. sticker（右下贴纸）：有人物图优先抠人物（客户口径），其次蛋糕、
         花艺、非居中的请柬纸品；纯场景一律不抠——GrabCut 在长桌/布幔上
         只会留下白边残条，正是客户说的「抠图不行」。
      3. tiles（四格）：按「种类多样性 + 主体突出 + 可抠性」打分挑选，
         同种类第二次出现扣分，近似重复图（平均哈希 ≤14）直接扣穿；
         绝不复用 hero / sticker 的图，从根上杜绝「同一张图出现两次」。
      4. 版式微调：把最安静的一张场景图放到右下格——贴纸压在这一格上，
         避免贴纸压在自己的原图或请柬细节上，产生视觉重复。
    """
    n = len(infos)
    if n == 0:
        return 0, [], None
    proms = [float(x.get("subject_prominence") or 0.5) for x in infos]
    iso = [float(x.get("isolatable") or 0.3) for x in infos]
    kinds = [x.get("kind") for x in infos]

    # ---- 1) 居中主卡：请柬产品优先 ----
    inv = [i for i in range(n) if kinds[i] == "invitation"]
    if force_hero is not None and 0 <= force_hero < n and kinds[force_hero] == "invitation":
        hero = force_hero
    elif inv:
        hero = max(inv, key=lambda i: _hero_score(infos[i]))
    else:
        pool = [i for i in range(n) if _subject_pri(infos[i]) > 0] or list(range(n))
        hero = max(pool, key=lambda i: (_subject_pri(infos[i]), proms[i], iso[i]))
    used = {hero}

    # ---- 2) 贴纸源：人物 > 蛋糕 > 花艺 > 请柬；排除 hero 与近似重复 ----
    sticker = None
    if n >= 6:      # 5 张时四格要占满，强出贴纸必然重复（客户硬性口径）
        h_hero = _img_hash(imgs[hero]) if imgs else None
        cands = sorted((i for i in range(n) if i not in used),
                       key=lambda i: (_subject_pri(infos[i]), iso[i], proms[i]),
                       reverse=True)
        for i in cands:
            if _subject_pri(infos[i]) <= 0:
                break
            if imgs and _is_dup(_img_hash(imgs[i]), h_hero):
                continue
            sticker = i
            break
    if sticker is not None:
        used.add(sticker)

    # ---- 3) 四格：按内容打分挑选，种类不重复优先 ----
    rest = [i for i in range(n) if i not in used]
    hashes = [_img_hash(b) for b in imgs] if imgs else None
    kind_cnt = {}
    picked = []
    st_kind = kinds[sticker] if sticker is not None else None
    pool = list(rest)
    while len(picked) < 4 and pool:
        best, bs = None, -1e9
        for i in pool:
            k = kinds[i]
            s = proms[i] + 0.35 * iso[i]
            s -= 0.5 * kind_cnt.get(k, 0)                 # 同质化扣分
            if st_kind and k == st_kind and _subject_pri(infos[i]) >= 2:
                s -= 0.9                                  # 与贴纸同主体（另一张蛋糕）扣分
            if hashes and any(_is_dup(hashes[i], hashes[j]) for j in picked):
                s -= 3.0                                  # 近似重复直接排除
            if s > bs:
                best, bs = i, s
        if best is None:
            break
        picked.append(best)
        pool.remove(best)
        kind_cnt[kinds[best]] = kind_cnt.get(kinds[best], 0) + 1
    for i in rest:                                        # 图少时补齐
        if len(picked) >= 4:
            break
        if i not in picked:
            picked.append(i)

    # ---- 4) 版式：右下格放最安静的场景图（贴纸压在这里） ----
    if imgs and len(picked) >= 2:
        def _busy(i):
            try:
                g = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
                g = cv2.resize(g, (160, 160), interpolation=cv2.INTER_AREA)
                return float(cv2.Laplacian(g, cv2.CV_32F).var())
            except Exception:
                return 1e9
        quiet_pool = [i for i in picked if kinds[i] == "scene"] or []
        if quiet_pool:
            quiet = min(quiet_pool, key=_busy)
            picked = [i for i in picked if i != quiet] + [quiet]
    return hero, picked, sticker


def _sticker_cutout(img, box=None, stroke=STROKE_PX, person=False):
    """贴纸抠图（2026-11-25 第 8 版）：AI 分割优先，GrabCut 兜底。

    第 7 版用「框内 GrabCut」，但实测在浅色产品 + 同色调背景上无法收敛：
    蛋糕图覆盖面 0.69~0.80（等于把背景整块留下），并在框边留下直切线。
    现在改为：
      1) core.matting（U^2-Net portable，按 product_box 裁剪后推理）——
         实测蛋糕 / 婚纱 / 请柬纸品都能完整抠净；模型缺失或异常时自动跳过；
      2) GrabCut 框内路径（旧口径）作为兜底；
      3) 全局背景建模（auto_cutout）作为最后兜底。
    三条路径最后都统一加白描边，保证贴纸质感一致。

    person=True（图里有人）时**不用产品框**：视觉模型常把框只框在手捧花上，
    框内分割就只抠出一束花（客户要的是人）。整图分割时 u2netp 会把人物
    作为最大显著主体抠出来，实测能连人带花完整抠净。
    """
    seg_box = None if person else box
    try:
        from core import matting
        rgba, sc = matting.cutout(img, box=seg_box, stroke=stroke)
        if rgba is not None:
            return rgba, sc
    except Exception:
        traceback.print_exc()
    if box:
        try:
            rgba, sc = cutout_for_card(img, box=box)
            if rgba is not None:
                return _with_stroke(rgba, stroke), sc
        except Exception:
            pass
    return auto_cutout(img, stroke=stroke)


def _alpha_hits_top(rgba, flat=0.35):
    """抠图顶部是否是「被原图裁断的平切边」（人物头部被裁掉）。

    贴纸画布四周有描边留白，所以不能直接看前几行；取最上面一行有内容的
    位置，再看该行填充宽度占主体宽度的比例：自然头顶/头发很窄（实测
    0.01），被裁的躯干是一条宽平边（实测 0.51）。
    """
    if rgba is None or rgba.ndim != 3:
        return False
    a = rgba[:, :, 3]
    ys, xs = np.where(a > 128)
    if len(ys) == 0:
        return False
    row = a[int(ys.min())]
    span = int((row > 128).sum())
    wide = int(xs.max() - xs.min() + 1)
    return bool(span >= max(8, wide * flat))


def _pick_sticker(srcs, bgrs, sticker, sticker_src, prefer_path=None,
                  prefer_box=None, boxes=None, exclude=None, order=None,
                  persons=None, kinds=None):
    """选贴纸来源并返回 (rgba, path_used)。

    2026-11-25 第 8 版重写：上一版 auto 兜底会在「优先图抠不出来」时扫全部
    上传图取分数最高者，可能扫回四格/主卡已经用过的照片 —— 客户截图里
    「蛋糕出现两次」就是这么来的（贴纸抠自蛋糕图，而蛋糕图同时进了四格）。
    现在：
      * exclude 里的图（主卡 + 四格）一律不参与，绝不重复用图；
      * 候选按语义优先级（人物 > 蛋糕 > 花艺 > 请柬）依次尝试，先成先得；
      * 返回真实使用的路径，调用方据此从轮播里剔除，避免片上再出现一次原图。
    """
    exclude = set(exclude or [])
    boxes = boxes or {}
    persons = set(persons or [])

    cands = []
    if sticker_src and os.path.isfile(sticker_src) and sticker_src not in exclude:
        cands = [sticker_src]
    elif sticker == "auto":
        if prefer_path and prefer_path in srcs and prefer_path not in exclude:
            cands.append(prefer_path)
        for p in (order or []):
            if p in srcs and p not in cands and p not in exclude:
                cands.append(p)
        for p in srcs:
            if p not in cands and p not in exclude:
                cands.append(p)
    if not cands:
        return None, None

    # 人物优先的三级取法（第 8 版定稿）：
    #   1) 完整人物（头部没被原图裁掉）——直接返回；
    #   2) 干净产品（蛋糕/花艺/请柬，按语义排序）——比「平切掉头」的躯干好看；
    #   3) 被裁断的人物；
    #   4) 场景图（只有实在没有主体时才用，抠出来多是杂块）。
    rank_of = {"people": 0, "cake": 1, "floral": 2, "invitation": 3}
    kinds = kinds or {}
    orphan_person = None                     # 头顶被裁的人物
    best_product = None                      # (rank, rgba, path)
    best_scene = None                        # 场景兜底
    for p in cands:
        try:
            img = _read_maybe_alpha(p)
            if img is None:
                continue
            box = (prefer_box if p == prefer_path else boxes.get(p))
            rgba, _sc = _sticker_cutout(img, box=box, person=(p in persons))
            if rgba is None:
                continue
            if p in persons:
                if _alpha_hits_top(rgba):
                    if orphan_person is None:
                        orphan_person = (rgba, p)
                    continue
                return rgba, p
            rk = rank_of.get(str(kinds.get(p) or ""), 9)
            if rk >= 9:
                if best_scene is None:
                    best_scene = (rgba, p)
            elif best_product is None or rk < best_product[0]:
                best_product = (rk, rgba, p)
        except Exception:
            continue
    if best_product is not None:
        return best_product[1], best_product[2]
    return (orphan_person or best_scene or (None, None))


# ---------------- 静态拼合完成态 ----------------
def _render_base(tiles, card, sticker_rgba, dot_bgrs=None):
    c = np.full((H0, W0, 3), 250, np.uint8)
    for i in range(4):
        r, q = divmod(i, 2)
        c[r * CH:r * CH + CH, q * CW:q * CW + CW] = tiles[i]
    cx, cy = _card_rect(card)[:2]
    _drop_shadow(c, card, cx, cy, 1.0)
    _paste(c, card, cx, cy, 1.0)
    # 色块压在卡角上 → 必须在主卡之后画，否则被照片盖住
    if dot_bgrs:
        for (dx_, dy_), bgr in zip(_dot_centers(card, len(dot_bgrs)), dot_bgrs):
            _draw_dot(c, dx_, dy_, DOT_R + DOT_STROKE, DOT_R, bgr, 1.0)
    if sticker_rgba is not None:
        st = _fit_sticker(sticker_rgba, card)
        x, y = _sticker_xy(st, card)
        _drop_shadow(c, st, x, y, 1.0, strength=38, blur=14, dy=8)
        _paste(c, st, x, y, 1.0)
    return c


# ---------------- 主渲染 ----------------
def render_moodboard(srcs, out_path, ratio="9x16", fps=30, progress=None,
                     palette_edit=None, forced_family=None,
                     sticker="auto", sticker_src=None, vision_cfg=None,
                     no_classify=False, palette_text=None):
    paths = [p for p in (srcs or []) if p and os.path.isfile(p)]
    if len(paths) < 5:
        raise ValueError("Moodboard Showcase needs at least 5 photos")
    paths = paths[:MAX_PHOTOS]

    # ---- AI 自动选图（2026-11-25）：请柬居中 / 抠人物·蛋糕·主体最突出 / 四格不重复 ----
    # 仅在自动贴纸（未显式指定 sticker_src）且未禁用分类时启用，避免覆盖客户手选。
    shot_info = {"classified": False, "hero_kind": None, "tile_kinds": [],
                 "sticker_kind": None, "kinds": []}
    if not no_classify and sticker == "auto" and not sticker_src and len(paths) >= 5:
        if progress:
            progress(9, 100, "Recognizing photo content")
        infos = classify_photos(paths, vision_cfg, progress)
        # 选图完全由内容决定（第 8 版）：hero=请柬纸品，sticker=人物>蛋糕>花艺，
        # 四格按多样性挑选并剔除近似重复 —— 不再受上传顺序影响。
        try:
            _probe = [_read_maybe_alpha(p) for p in paths]
        except Exception:
            traceback.print_exc()
            _probe = None
        hero_i, tile_is, st_i = plan_shot_order(infos, len(paths), imgs=_probe)
        sticker_path = paths[st_i] if (st_i is not None and 0 <= st_i < len(paths)) else None
        # product_box 映射按「重排前」的原始顺序建立（infos 与原 paths 对齐），
        # 之后 paths 被重排，框仍要能按路径找到对应图。
        product_boxes = {paths[i]: infos[i].get("product_box")
                         for i in range(len(paths))}
        # 有人的图：贴纸走整图分割（视觉模型常把框只框在手捧花上）
        person_paths = [paths[i] for i in range(len(paths))
                        if infos[i].get("has_person")]
        kinds_by_path = {paths[i]: infos[i].get("kind")
                         for i in range(len(paths))}
        # 用选出的索引重排：hero -> tiles -> 其余（其余进轮播追加）
        ordered = [paths[hero_i]] + [paths[i] for i in tile_is]
        used = set([hero_i] + tile_is)
        for i, p in enumerate(paths):
            if i not in used:
                ordered.append(p)
        paths = ordered
        shot_info = {"classified": True,
                     "hero_kind": infos[hero_i].get("kind"),
                     "tile_kinds": [infos[i].get("kind") for i in tile_is],
                     "sticker_kind": infos[st_i].get("kind") if st_i is not None else None,
                     "kinds": [x.get("kind") for x in infos],
                     "sticker_path": sticker_path,
                     "hero_box": infos[hero_i].get("product_box"),
                     "sticker_box": (product_boxes.get(sticker_path)
                                     if sticker_path else None),
                     "person_paths": person_paths,
                     "kinds_by_path": kinds_by_path,
                     "sticker_has_person": (sticker_path in set(person_paths)
                                            if sticker_path else False),
                     "product_boxes": product_boxes}
    else:
        shot_info["sticker_path"] = None

    OW, OH = ratio_size(norm_ratio(ratio))
    global W0, H0, CW, CH
    W0, H0 = int(OW), int(OH)
    CW, CH = W0 // 2, H0 // 2

    if progress:
        progress(6, 100, "Reading photos")
    bgrs = {p: _imread_bgr(p) for p in paths}

    if progress:
        progress(12, 100, "Recognizing colors")
    fam = forced_family
    if fam is None:
        try:
            fam = family_for_moodboard([image_color_profile(b) for b in bgrs.values()])
        except Exception:
            traceback.print_exc()
            fam = None
    palette = detect_palette(list(bgrs.values()), N_SWATCH, forced_family=fam)
    palette_colors = palette_colors_of(palette)
    if palette_edit:
        new_cols = []
        for c in (palette_edit.get("colors") or [])[:N_SWATCH]:
            if not isinstance(c, dict):
                continue
            rgb = c.get("rgb") or _hex_to_rgb(c.get("hex"))
            if not rgb:
                continue
            rgb = tuple(int(round(float(v))) for v in rgb)
            new_cols.append({"name": str(c.get("name") or "").strip() or "",
                             "rgb": list(rgb),
                             "hex": "#%02X%02X%02X" % rgb})
        if new_cols:
            palette_colors = complete_palette(new_cols)
    elif palette_text:
        # 2026-11-25 客户口径：文字输入匹配合适色块（与 Color Palette
        # Showcase 的 Subtitle 驱动同款）——"Navy Blue & Gold" 这类写法
        # 直接决定色块；认不出任何色名时忽略，回退照片识别。
        try:
            _pt = palette_from_text(palette_text)
        except Exception:
            _pt = None
            traceback.print_exc()
        if _pt and _pt.get("colors"):
            palette_colors = _pt["colors"][:N_SWATCH]

    # 色点（BGR）。2026-11-25 第 3 版：色块恢复「带描边」样式后，米白色芯
    # 靠描边圈也能从照片上读出来（客户参考截图的第一个圆就是米白），
    # 所以取消上一版「跳过亮度>236」的过滤，保持色板 4 档与实际识别一致。
    dot_bgrs = []
    for c in palette_colors:
        r, g, b = (int(v) for v in c["rgb"][:3])
        dot_bgrs.append((b, g, r))
        if len(dot_bgrs) >= 4:
            break

    if progress:
        progress(18, 100, "Preparing cutout")
    _prefer = shot_info.get("sticker_path") if shot_info.get("classified") else None
    # 主卡 + 四格占用的图不允许再做贴纸源（客户口径：同一张图不能出现两次）
    _used_imgs = set(paths[:5]) if shot_info.get("classified") else set()
    sticker_rgba, sticker_path_used = _pick_sticker(
        paths, bgrs, sticker, sticker_src,
        prefer_path=_prefer,
        prefer_box=shot_info.get("sticker_box"),
        boxes=shot_info.get("product_boxes") or {},
        exclude=_used_imgs,
        order=[paths[i] for i in range(len(paths))],
        persons=shot_info.get("person_paths") or [],
        kinds=shot_info.get("kinds_by_path") or {})
    shot_info["sticker_path_used"] = sticker_path_used
    sticker_kind = shot_info.get("sticker_kind")

    hero = bgrs[paths[0]]
    tiles = [_cover(bgrs[p], CW, CH) for p in paths[1:5]]
    # 轮播不再回放主卡原图与贴纸源图：主卡已用它做卡面、贴纸已用它做抠图，
    # 再全屏放一次就是客户截图里「同一张图出现两次」的另一半原因。
    _skip = {paths[0]}
    if sticker_path_used:
        _skip.add(sticker_path_used)
    rotation = [_cover(bgrs[p], W0, H0) for p in paths[1:] if p not in _skip]

    # 主卡 = 照片方形裁切（2026-11-25 第 3 版：不做抠图合成）
    card = _make_card(hero)
    # 无人像/蛋糕/花艺可抠时，退而把主卡产品抠出来放到右下角（与色块对角），
    # 保证客户「抠图产品放对称位置」在任何素材下都成立。
    hero_cut_score = 0.0
    if sticker_rgba is None:
        try:
            _hrgba, hero_cut_score = cutout_for_card(
                hero, box=shot_info.get("hero_box"))
            if _hrgba is not None:
                sticker_rgba = _with_stroke(_hrgba)
                sticker_kind = "hero_product"
        except Exception:
            traceback.print_exc()
    sticker_used = sticker_rgba is not None
    shot_info["corner_sticker_kind"] = sticker_kind
    shot_info["hero_cut_score"] = hero_cut_score
    BASE_IMG = _render_base(tiles, card, sticker_rgba, dot_bgrs)

    CARD_X, CARD_Y = _card_rect(card)[:2]
    # 投影预烘焙：逐帧只贴图，不再每帧高斯模糊
    CARD_SH, CARD_PAD = _bake_shadow(card)
    DOT_XY = _dot_centers(card, len(dot_bgrs))
    st_resized = _fit_sticker(sticker_rgba, card) if sticker_rgba is not None else None
    STK_XY = _sticker_xy(st_resized, card) if st_resized is not None else (0, 0)
    STK_SH, STK_PAD = None, 0
    if st_resized is not None:
        STK_SH, STK_PAD = _bake_shadow(st_resized, strength=38, blur=14, dy=8)

    # 每张全屏时长：张数多时略缩短，避免成片过长（20 张 ≈ 28s）
    g_dur = G_DUR if len(rotation) <= 10 else G_DUR_MANY
    T_END_SHOW = T_HOLD1 + len(rotation) * g_dur
    T_OUT_START = T_END_SHOW + END_HOLD
    T_TOTAL = T_OUT_START + FADE_OUT
    NFRAMES = int(round(T_TOTAL * fps))

    if progress:
        progress(24, 100, "Rendering")

    import subprocess
    import tempfile
    cmd = [FFMPEG, "-y", "-hide_banner",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (W0, H0), "-r", str(fps), "-i", "-",
           "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", str(CRF), "-preset", PRESET, "-movflags", "+faststart",
           out_path]
    err_f = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=err_f, stderr=err_f)

    def _frame(t):
        # ---- 全屏轮播（硬切） ----
        if T_HOLD1 <= t < T_END_SHOW:
            k = int((t - T_HOLD1) // g_dur)
            k = min(k, len(rotation) - 1)
            return rotation[k]
        # ---- 收尾：回拼合图定格 → 淡出到白 ----
        if t >= T_END_SHOW:
            c = BASE_IMG.copy()
            if t >= T_OUT_START:
                a = _clamp01((t - T_OUT_START) / FADE_OUT)
                white = np.full_like(c, 252)
                c = (c.astype(np.float32) * (1 - a) +
                     white.astype(np.float32) * a).astype(np.uint8)
            return c
        # ---- 拼合阶段 ----
        c = np.full((H0, W0, 3), 250, np.uint8)
        for i in range(4):
            st_ = T_TILE0 + i * T_TILE_STEP
            p = _clamp01((t - st_) / T_TILE_DUR)
            if p <= 0:
                continue
            e = _ease_out_cubic(p)
            r, q = divmod(i, 2)
            _draw_tile(c, tiles[i], q * CW, r * CH, CW, CH,
                       scale=1.14 - 0.14 * e, alpha=e)
        # 主卡（上滑淡入，投影已烘焙）
        p = _clamp01((t - T_CARD) / T_CARD_DUR)
        if p > 0:
            e = _ease_out_cubic(p)
            dy = int(150 * (1 - e))
            _paste(c, CARD_SH, CARD_X - CARD_PAD, CARD_Y + dy - CARD_PAD, e)
        # 色点：压在卡角上 → 必须画在主卡之后（2026-11-25 第 3 版）
        for i, bgr in enumerate(dot_bgrs):
            st_ = T_DOT0 + i * T_DOT_STEP
            p = _clamp01((t - st_) / T_DOT_DUR)
            if p <= 0:
                continue
            e = _ease_out_back(p, 1.7)
            a = _clamp01(p / 0.45)
            dx_, dy_ = DOT_XY[i]
            _draw_dot(c, dx_, dy_, (DOT_R + DOT_STROKE) * e, DOT_R * e, bgr, a)
        # 贴纸（与色块对角：从右下滑入）
        if st_resized is not None:
            p = _clamp01((t - T_STK) / T_STK_DUR)
            if p > 0:
                e = _ease_out_cubic(p)
                dx = int(280 * (1 - e))
                if STK_SH is not None:
                    _paste(c, STK_SH, STK_XY[0] + dx - STK_PAD,
                           STK_XY[1] - STK_PAD, e)
                else:
                    _paste(c, st_resized, STK_XY[0] + dx, STK_XY[1], e)
        return c

    try:
        for i in range(NFRAMES):
            t = i / float(fps)
            fr = _frame(t)
            if fr.shape[2] == 3:
                fr = fr[:, :, ::-1]            # BGR→RGB（红线 13）
            proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
            if progress and (i % 60 == 0):
                progress(int(24 + 70 * i / max(NFRAMES, 1)), 100, "Rendering")
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        err_f.seek(0)
        raise RuntimeError("ffmpeg failed rc=%s: %s" %
                           (proc.returncode, err_f.read().decode("utf-8", "ignore")[-800:]))

    if progress:
        progress(100, 100, "Done")
    return {"video": out_path, "secs": round(T_TOTAL, 2),
            "images": len(paths), "ratio": ratio,
            "palette": [list(c) for c in palette],
            "palette_colors": palette_colors,
            "palette_name": name_palette(palette) if not palette_edit else
            (palette_edit.get("name") or name_palette(palette)),
            "sticker_used": sticker_used,
            "sticker_src": sticker_src or ("auto" if sticker_used else None),
            "shot_info": shot_info}
