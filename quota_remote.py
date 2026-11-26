# -*- coding: utf-8 -*-
"""
额度路由适配器：把 server.py 现有的 QUOTA（本地 QuotaLedger）包一层。

- magento_it.enabled=false（默认）：所有方法原样透传本地账本，行为与改造前完全一致。
- magento_it.enabled=true：balance/hold/commit/release/add_generation 走 IT 真实接口
  （itclient），本地账本仅留生成记录供对账；IT 不可用时 fail-closed，不降级放行。

设计目的：业务层（server.py 的 QUOTA.balance/hold/commit/release/add_generation）零改动，
IT 给出地址后只需 config.json 置 enabled=true + 填 base_url/service_token 即可切换。
"""
import uuid
from datetime import datetime
import itclient


class RemoteQuotaAdapter:
    def __init__(self, local, client):
        self.local = local
        self.client = client
        self._hid2flow = {}          # IT hold_id -> flow_id（commit/release 按 hold_id 调用时用）

    def _on(self):
        return bool(self.client and self.client.available())

    # ---- 与 QuotaLedger 同签名的方法 ----
    def balance(self, customer_id, cap, period=None):
        if not self._on():
            return self.local.balance(customer_id, cap, period)
        d = self.client.get_quota(customer_id)      # fail-closed: 异常直接抛出
        q = d.get("quota") or {}
        return {
            "cap": int(q.get("cap", cap)),
            "held": int(q.get("held", 0)),
            "used": int(q.get("used", 0)),
            "available": int(q.get("available", 0)),
            "period": d.get("period") or q.get("period") or period or "",
        }

    def hold(self, customer_id, cap, units=1, flow_id=None):
        if not self._on():
            return self.local.hold(customer_id, cap, units, flow_id)
        flow_id = flow_id or uuid.uuid4().hex
        try:
            d = self.client.hold(customer_id, flow_id, units)
        except itclient.QuotaNotEnough as e:
            bal = e.balance or {"cap": cap, "held": 0, "used": 0, "available": 0}
            return False, {"reason": "quota_exhausted"}, bal, False
        hid = d.get("hold_id") or ""
        if hid:
            self._hid2flow[hid] = flow_id
        detail = {"hold_id": hid, "status": "held", "flow_id": flow_id}
        bal = d.get("balance") or {}
        return True, detail, bal, bool(d.get("idempotent"))

    def commit(self, hold_id):
        if not self._on():
            return self.local.commit(hold_id)
        flow = self._hid2flow.get(hold_id) or hold_id
        self.client.commit(flow, hold_id)
        return True, "committed"

    def release(self, hold_id):
        if not self._on():
            return self.local.release(hold_id)
        flow = self._hid2flow.get(hold_id) or hold_id
        self.client.release(flow, hold_id)
        return True, "released"

    def add_generation(self, flow_id, customer_id, template, points,
                       created_ts=None, note=None, order_no=None, video_url=None):
        # 生成记录一律本地留存（对账）；IT 模式下额外回写 Magento §4.5
        out = self.local.add_generation(flow_id, customer_id, template, points,
                                        created_ts=created_ts, note=note)
        if self._on():
            created_at = ""
            if created_ts:
                created_at = datetime.fromtimestamp(created_ts).isoformat(timespec="seconds")
            self.client.add_record(
                cid=customer_id, flow_id=flow_id,
                order_no=order_no or customer_id,
                template=template, points=points,
                video_url=video_url or (note if isinstance(note, str) else "") or "",
                created_at=created_at,
            )
        return out

    # ---- 其余方法透传本地（生成记录列表 / 对账 / wipe / sink 等）----
    def __getattr__(self, name):
        return getattr(self.local, name)
