# -*- coding: utf-8 -*-
"""
Magento IT 接口实装层（对应《账户系统对接接口契约 v1》§4 的 5 个接口）。

设计原则：
  1. 纯标准库（urllib），零新依赖；
  2. 地址 / 服务令牌 / 头名全部配置化，IT 给出真实地址即可切换，不改业务代码；
  3. fail-closed：任何网络/解析/HTTP 非预期错误都抛 ITClientError，调用方据此拒绝生成；
  4. 幂等：所有写接口透传 flow_id，由服务端按 flow_id 去重；
  5. 本机代理劫持治理：对 127.0.0.1/localhost 自动 bypass 系统代理（本机有 v2rayN 等全局代理时，
     否则本地假端点会被劫持出假 502）；生产地址仍走系统代理。
  6. 密钥不硬编码：service token 从 config.json 的 magento_it.service_token 或环境变量
     IT_SERVICE_TOKEN 读；绝不写死在源码。

配置示例（config.json 增加一段；本文件不含任何真实值）：
  "magento_it": {
    "enabled": false,
    "base_url": "https://<magento-host>/rest/V1/janice-studio",
    "service_token": "",
    "token_header": "X-Service-Token",
    "cid_header": "X-Customer-Id",
    "timeout": 8
  }
  enabled=false 时不发起任何网络请求（保持本地假账本模式，向后兼容）。
"""
import json
import os
import urllib.request
import urllib.error
from urllib.parse import urlencode, urlparse

BASE = os.path.dirname(os.path.abspath(__file__))


class ITClientError(Exception):
    """IT 接口不可用/返回异常。调用方必须据此 fail-closed，不得降级放行。"""
    def __init__(self, message, status=None, payload=None):
        super().__init__(message)
        self.status = status
        self.payload = payload


class QuotaNotEnough(Exception):
    """402 积分不足，单独成类，便于业务层给出友好提示（不是系统故障）。"""
    def __init__(self, balance=None, need=None):
        super().__init__("not enough quota")
        self.balance = balance
        self.need = need


def _is_loopback(url):
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return host in ("127.0.0.1", "localhost", "::1")


def _opener_for(url):
    """loopback 显式空代理；其它地址走系统默认（继承环境代理）。"""
    if _is_loopback(url):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


class MagentoITClient:
    def __init__(self, base_url, service_token="", token_header="X-Service-Token",
                 cid_header="X-Customer-Id", timeout=8, enabled=True):
        self.base_url = (base_url or "").rstrip("/")
        self.service_token = service_token or ""
        self.token_header = token_header or "X-Service-Token"
        self.cid_header = cid_header or "X-Customer-Id"
        self.timeout = float(timeout or 8)
        self.enabled = bool(enabled)

    def available(self):
        return bool(self.enabled and self.base_url)

    # ---- 底层请求 ----
    def _request(self, method, path, body=None, cid=None, expect=(200, 201)):
        if not self.available():
            raise ITClientError("IT client disabled or base_url empty")
        url = self.base_url + path
        data = None
        headers = {"Accept": "application/json"}
        if self.service_token:
            headers[self.token_header] = self.service_token
        if cid:
            headers[self.cid_header] = cid
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            resp = _opener_for(url).open(req, timeout=self.timeout)
            raw = resp.read()
            status = resp.status
        except urllib.error.HTTPError as e:
            raw = e.read()
            status = e.code
        except Exception as e:
            raise ITClientError("IT request failed: %s" % e)
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            raise ITClientError("IT response not JSON", status=status)
        if status == 402:
            raise QuotaNotEnough(payload.get("balance"), payload.get("need"))
        if status not in expect:
            raise ITClientError("IT HTTP %s: %s" % (status, payload.get("error", "")),
                                status=status, payload=payload)
        if isinstance(payload, dict) and payload.get("ok") is False \
                and payload.get("error") != "not_enough_quota":
            raise ITClientError("IT ok=false: %s" % payload.get("error"),
                                status=status, payload=payload)
        return payload

    # ---- §4 五接口 ----
    def get_quota(self, cid):
        """§4.1 查询额度 -> {cid, plan, quota:{cap,used,held,available}, period}"""
        if not cid:
            raise ITClientError("missing cid")
        return self._request("GET", "/account/quota?" + urlencode({"cid": cid}), cid=cid)

    def hold(self, cid, flow_id, units, period=None):
        """§4.2 预占 -> {ok, hold_id, flow_id, balance, idempotent?}；402 抛 QuotaNotEnough。"""
        body = {"cid": cid, "flow_id": flow_id, "units": int(units)}
        if period:
            body["period"] = period
        return self._request("POST", "/account/quota/hold", body=body, cid=cid)

    def commit(self, flow_id, hold_id=None):
        """§4.3 提交扣点 -> {ok, flow_id, balance}"""
        body = {"flow_id": flow_id}
        if hold_id:
            body["hold_id"] = hold_id
        return self._request("POST", "/account/quota/commit", body=body)

    def release(self, flow_id, hold_id=None):
        """§4.4 释放预占 -> {ok, flow_id, balance}"""
        body = {"flow_id": flow_id}
        if hold_id:
            body["hold_id"] = hold_id
        return self._request("POST", "/account/quota/release", body=body)

    def add_record(self, cid, flow_id, order_no, template, points, video_url, created_at):
        """§4.5 作品回写 -> {ok, record_id, idempotent?}"""
        body = {
            "cid": cid, "flow_id": flow_id, "order_no": order_no,
            "template": template, "points": int(points),
            "video_url": video_url, "created_at": created_at,
        }
        return self._request("POST", "/account/records", body=body, cid=cid)


def from_config(config_path=None):
    """从 config.json 的 magento_it 段构造客户端；环境变量可覆盖令牌/地址。"""
    path = config_path or os.path.join(BASE, "config.json")
    cfg = {}
    try:
        with open(path, encoding="utf-8") as f:
            cfg = (json.load(f).get("magento_it") or {})
    except Exception:
        cfg = {}
    base_url = os.environ.get("IT_BASE_URL") or cfg.get("base_url") or ""
    token = os.environ.get("IT_SERVICE_TOKEN") or cfg.get("service_token") or ""
    return MagentoITClient(
        base_url=base_url,
        service_token=token,
        token_header=cfg.get("token_header") or "X-Service-Token",
        cid_header=cfg.get("cid_header") or "X-Customer-Id",
        timeout=cfg.get("timeout") or 8,
        enabled=bool(cfg.get("enabled")),
    )


def report_sink_from(client, order_no_of=None, url_of=None):
    """构造 set_report_sink(fn) 用的回写钩子：把内部流水字段适配成 §4.5 契约字段。

    order_no_of(flow_row) -> str   订单号（默认取 customer_id）
    url_of(flow_row) -> str        成片地址（默认空；生产应由调用方传真实媒体地址）
    """
    if not client or not client.available():
        return None

    def sink(row):
        fid = row.get("flow_id")
        cid = row.get("customer_id")
        order = (order_no_of(row) if order_no_of else None) or cid
        url = (url_of(row) if url_of else None) or ""
        client.add_record(
            cid=cid, flow_id=fid, order_no=order,
            template=row.get("template") or "",
            points=row.get("units") or 0,
            video_url=url,
            created_at=row.get("created_at") or "",
        )

    return sink


if __name__ == "__main__":
    # 自检：打印当前配置解析结果（不回显令牌明文）
    c = from_config()
    print("enabled=%s base_url=%s token_set=%s timeout=%s"
          % (c.enabled, c.base_url or "(empty)", bool(c.service_token), c.timeout))
