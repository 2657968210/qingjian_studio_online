# -*- coding: utf-8 -*-
"""
额度账本（sqlite3，标准库零新依赖）。

业务口径（对应设计讨论）：
  - 权威在 Magento，台子只做「预占 -> 成功扣减 / 失败释放」
  - 每次消耗带唯一 flow_id 保证幂等，重复提交不重复扣
  - 按 customer_id + 账期(月) 记账，支持定期对账
  - 初期统一额度：每客户 200 点，每段扣 20 点（units 字段单位为「点数」，非段数）
"""
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime

_LOCK = threading.RLock()


def period_now():
    return datetime.now().strftime("%Y-%m")


class QuotaLedger:
    """持久化账本。绝不放在会被发布重置的 data/ 里——本副本用独立 _ledger/。"""

    def __init__(self, db_path):
        self.db_path = db_path
        self._report_sink = None
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS holds (
                    hold_id      TEXT PRIMARY KEY,
                    customer_id  TEXT NOT NULL,
                    period       TEXT NOT NULL,
                    units        INTEGER NOT NULL,
                    status       TEXT NOT NULL,
                    flow_id      TEXT NOT NULL UNIQUE,
                    created_at   TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ledger (
                    flow_id      TEXT PRIMARY KEY,
                    customer_id  TEXT NOT NULL,
                    period       TEXT NOT NULL,
                    units        INTEGER NOT NULL,
                    created_at   TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_holds_cust
                    ON holds(customer_id, period, status);
                CREATE INDEX IF NOT EXISTS idx_ledger_cust
                    ON ledger(customer_id, period);
                CREATE TABLE IF NOT EXISTS generations (
                    flow_id      TEXT PRIMARY KEY,
                    customer_id  TEXT NOT NULL,
                    template     TEXT NOT NULL,
                    points       INTEGER NOT NULL,
                    created_ts   REAL NOT NULL,
                    created_at   TEXT NOT NULL,
                    note         TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_gen_cust
                    ON generations(customer_id, created_ts);
                -- 充值流水：IT 在客户下单后调用加积分接口写入；order_no 唯一键保证幂等
                CREATE TABLE IF NOT EXISTS topups (
                    order_no     TEXT PRIMARY KEY,
                    customer_id  TEXT NOT NULL,
                    points       INTEGER NOT NULL,
                    plan         TEXT,
                    note         TEXT,
                    created_at   TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_topups_cust
                    ON topups(customer_id);
            """)

    def _conn(self):
        c = sqlite3.connect(self.db_path, timeout=20, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    @staticmethod
    def _balance_on(c, customer_id, cap, period):
        """在同一连接/事务内查余额，避免嵌套连接造成 SQLite 锁竞争。"""
        held = c.execute(
            "SELECT COALESCE(SUM(units),0) FROM holds "
            "WHERE customer_id=? AND period=? AND status='held'",
            (customer_id, period),
        ).fetchone()[0]
        used = c.execute(
            "SELECT COALESCE(SUM(units),0) FROM ledger "
            "WHERE customer_id=? AND period=?",
            (customer_id, period),
        ).fetchone()[0]
        return {
            "cap": cap,
            "held": held,
            "used": used,
            "available": max(0, cap - held - used),
            "period": period,
        }

    def balance(self, customer_id, cap, period=None):
        """返回 {cap, held, used, available}。"""
        with _LOCK, self._conn() as c:
            return self._balance_on(c, customer_id, cap, period or period_now())

    def hold(self, customer_id, cap, units=1, flow_id=None):
        """
        预占额度。返回 (ok, detail, balance, idempotent)。
        flow_id 重复时直接返回原结果，不重复扣（并发幂等）。
        """
        flow_id = flow_id or uuid.uuid4().hex
        period = period_now()
        with _LOCK, self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            old = c.execute(
                "SELECT hold_id, status FROM holds WHERE flow_id=?", (flow_id,)
            ).fetchone()
            if old:
                c.commit()
                bal = self._balance_on(c, customer_id, cap, period)
                return True, {"hold_id": old["hold_id"], "status": old["status"],
                              "flow_id": flow_id}, bal, True

            bal = self._balance_on(c, customer_id, cap, period)
            if bal["available"] < units:
                c.commit()
                return False, {"reason": "quota_exhausted"}, bal, False

            hid = uuid.uuid4().hex
            c.execute(
                "INSERT INTO holds VALUES (?,?,?,?,?,?,?)",
                (hid, customer_id, period, units, "held", flow_id,
                 datetime.now().isoformat(timespec="seconds")),
            )
            c.commit()
            bal = self._balance_on(c, customer_id, cap, period)
        return True, {"hold_id": hid, "status": "held", "flow_id": flow_id}, bal, False

    def commit(self, hold_id):
        """出片成功：把预占转成正式消耗流水。返回 (ok, reason)。"""
        with _LOCK, self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            r = c.execute(
                "SELECT * FROM holds WHERE hold_id=?", (hold_id,)
            ).fetchone()
            if not r or r["status"] != "held":
                c.rollback()
                return False, "hold_not_active"
            c.execute(
                "UPDATE holds SET status='committed' WHERE hold_id=?", (hold_id,)
            )
            c.execute(
                "INSERT OR IGNORE INTO ledger VALUES (?,?,?,?,?)",
                (r["flow_id"], r["customer_id"], r["period"], r["units"],
                 datetime.now().isoformat(timespec="seconds")),
            )
            c.commit()
        sink = getattr(self, "_report_sink", None)
        if sink:
            try:
                sink({"flow_id": r["flow_id"], "customer_id": r["customer_id"],
                      "period": r["period"], "units": r["units"]})
            except Exception:
                pass
        return True, "committed"

    def release(self, hold_id):
        """出片失败：释放预占，额度回退。返回 (ok, reason)。"""
        with _LOCK, self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            r = c.execute(
                "SELECT status FROM holds WHERE hold_id=?", (hold_id,)
            ).fetchone()
            if not r or r["status"] != "held":
                c.rollback()
                return False, "hold_not_active"
            c.execute(
                "UPDATE holds SET status='released' WHERE hold_id=?", (hold_id,)
            )
            c.commit()
        return True, "released"

    def add_generation(self, flow_id, customer_id, template, points,
                       created_ts=None, note=None):
        """出片成功记一条生成记录；flow_id 与扣点流水同源，重复调用不覆盖。"""
        created_ts = created_ts or datetime.now().timestamp()
        created_at = datetime.fromtimestamp(created_ts).isoformat(timespec="seconds")
        with _LOCK, self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO generations "
                "(flow_id, customer_id, template, points, created_ts, created_at, note) "
                "VALUES (?,?,?,?,?,?,?)",
                (flow_id, customer_id, template, points, created_ts, created_at,
                 note if note is None or isinstance(note, str) else json.dumps(note, ensure_ascii=False)),
            )
            c.commit()
        return flow_id

    def topup(self, customer_id, order_no, points, plan=None, note=None):
        """客户下单后加积分：以 order_no 为唯一键幂等。
        返回 (ok, detail, idempotent)。同一订单重复通知不得重复加分。
        若同 order_no 换了 customer_id，返回冲突（防止串户）。"""
        order_no = str(order_no or "").strip()
        if not order_no or not customer_id:
            return False, {"reason": "missing_order_or_customer"}, False
        try:
            points = int(points)
        except (TypeError, ValueError):
            return False, {"reason": "bad_points"}, False
        if points <= 0:
            return False, {"reason": "nonpositive_points"}, False
        with _LOCK, self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            old = c.execute(
                "SELECT customer_id, points, plan, created_at FROM topups WHERE order_no=?",
                (order_no,)).fetchone()
            if old:
                c.commit()
                if old["customer_id"] != customer_id:
                    return False, {"reason": "order_customer_mismatch",
                                   "order_no": order_no}, True
                return True, {
                    "order_no": order_no, "customer_id": customer_id,
                    "points": old["points"], "plan": old["plan"],
                    "created_at": old["created_at"],
                }, True
            c.execute(
                "INSERT INTO topups (order_no, customer_id, points, plan, note, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (order_no, customer_id, points, str(plan or ""),
                 note if note is None or isinstance(note, str)
                 else json.dumps(note, ensure_ascii=False),
                 datetime.now().isoformat(timespec="seconds")),
            )
            c.commit()
        return True, {"order_no": order_no, "customer_id": customer_id,
                      "points": points, "plan": plan or ""}, False

    def topup_total(self, customer_id):
        """该客户累计充值积分（所有订单加积分之和）。"""
        with _LOCK, self._conn() as c:
            return int(c.execute(
                "SELECT COALESCE(SUM(points),0) FROM topups WHERE customer_id=?",
                (customer_id,)).fetchone()[0])

    def list_topups(self, customer_id=None, limit=300):
        with _LOCK, self._conn() as c:
            if customer_id:
                rows = c.execute(
                    "SELECT * FROM topups WHERE customer_id=? "
                    "ORDER BY created_at DESC LIMIT ?", (customer_id, limit)).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM topups ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def list_generations(self, customer_id=None, limit=300):
        """生成记录列表（新→旧）。customer_id=None 表示全量（运营方视角）。"""
        with _LOCK, self._conn() as c:
            if customer_id:
                rows = c.execute(
                    "SELECT * FROM generations WHERE customer_id=? "
                    "ORDER BY created_ts DESC LIMIT ?", (customer_id, limit)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM generations ORDER BY created_ts DESC LIMIT ?", (limit,)
                ).fetchall()
        return [dict(r) for r in rows]

    def set_report_sink(self, fn):
        """预留对接点：出片成功后把流水回写 Magento（公网阶段实现）。
        fn(flow_row) -> None；异常被吞，不影响本地记账。默认 None（无网络调用）。"""
        self._report_sink = fn

    def flows(self, customer_id):
        """消耗流水，供与 Magento 定期对账。"""
        with _LOCK, self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT flow_id, period, units, created_at FROM ledger "
                "WHERE customer_id=? ORDER BY created_at", (customer_id,)
            )]

    def wipe(self):
        """仅本地自测用：清空账本重来。"""
        with _LOCK, self._conn() as c:
            c.execute("DELETE FROM holds")
            c.execute("DELETE FROM ledger")
            c.execute("DELETE FROM topups")
            c.commit()
