"""轻量 AI 调用计量 + 额度闸门（客户化前必装，防止成本失控）。

设计：
- 按「自然日 + 会话(sid)」计数 AI 调用；支持每会话上限与全局每日上限。
- 计数器落盘到 store_dir/_ai_meter.json（只保留最近 7 天，避免无限增长）。
- 闸门在「请求入口」拦截（同步），超限直接返回 402，不进渲染/生成任务。
- 额度数值走环境变量，方便运营调：
    AI_DAILY_CAP   全局每日 AI 请求上限（默认 200）
    AI_SESSION_CAP 每会话 AI 请求上限（默认 20）
  admin（X-Studio-Key / Basic）不受限。
"""
import os
import time
import json
import threading

_meter = None
_lock = threading.Lock()


class Meter:
    def __init__(self, store_dir, daily_cap=None, session_cap=None):
        self.store_dir = store_dir
        self.daily_cap = int(os.environ.get("AI_DAILY_CAP", daily_cap or 200))
        self.session_cap = int(os.environ.get("AI_SESSION_CAP", session_cap or 20))
        self.path = os.path.join(store_dir, "_ai_meter.json")
        self._cache = None
        self._cache_day = None

    # ---- 持久化 ----
    def _load(self):
        day = time.strftime("%Y-%m-%d")
        if self._cache is not None and self._cache_day == day:
            return self._cache
        data = {}
        try:
            if os.path.isfile(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
        except Exception:
            data = {}
        # 只保留最近 7 天
        keep = {}
        for d in data:
            if (time.time() - time.mktime(time.strptime(d, "%Y-%m-%d"))) <= 7 * 86400:
                keep[d] = data[d]
        self._cache, self._cache_day = keep, day
        return keep

    def _save(self, data):
        try:
            os.makedirs(self.store_dir, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
        except Exception:
            pass

    # ---- 计数 + 闸门 ----
    def count(self, sid, kind="ai"):
        """计数一次 AI 调用；返回 (ok, reason, used, cap)。
        ok=False 时调用方应拒绝请求（返回 402）。"""
        with _lock:
            data = self._load()
            day = time.strftime("%Y-%m-%d")
            data.setdefault(day, {})
            sk = sid or "anon"
            rec = data[day].setdefault(sk, {"ai": 0, "by_kind": {}})
            rec["ai"] += 1
            rec["by_kind"][kind] = rec["by_kind"].get(kind, 0) + 1
            # 全局当日总量
            total = sum(v.get("ai", 0) for v in data[day].values())
            self._save(data)
            if sid:  # 浏览器会话才限每会话；anon/admin 走全局
                if rec["ai"] > self.session_cap:
                    return (False, "session", rec["ai"], self.session_cap)
            if total > self.daily_cap:
                return (False, "daily", total, self.daily_cap)
            return (True, "", total, self.daily_cap)

    def snapshot(self):
        data = self._load()
        day = time.strftime("%Y-%m-%d")
        total = sum(v.get("ai", 0) for v in data.get(day, {}).values())
        return {"day": day, "daily_used": total, "daily_cap": self.daily_cap,
                "session_cap": self.session_cap}


def init(store_dir, daily_cap=None, session_cap=None):
    global _meter
    _meter = Meter(store_dir, daily_cap, session_cap)
    return _meter


def get_meter():
    return _meter
