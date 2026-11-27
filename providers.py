# -*- coding: utf-8 -*-
"""
AccountProvider —— 可插拔账号来源。

业务层（登录 / 额度 / 隔离）只依赖这个接口，不关心账号来自哪里。
本地测试用 LocalFixtureProvider（读脱敏夹具）；未来公网切 Magento 只需
换一个 MagentoRestProvider 实现，业务层一行不改。
"""
import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

import fixtures

BASE = os.path.dirname(os.path.abspath(__file__))


class AccountLocked(Exception):
    """账号已停用/封禁（登录接口据此返回 403）。"""


class AuthUnavailable(Exception):
    """验证服务不可达/超时/异常 —— 必须按 fail-closed 处理（登录失败而非放行）。"""


def load_config():
    with open(os.path.join(BASE, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _configured_plan(plan_key, fallback_key):
    plans = load_config().get("plans") or {}
    plan = dict(plans.get(plan_key) or plans.get(fallback_key) or {})
    plan.setdefault("label", plan_key or fallback_key)
    plan["key"] = plan_key
    return plan


class AccountProvider(ABC):
    """统一身份契约。返回值只有 customer_id / email / group / plan / account_lock。"""

    @abstractmethod
    def authenticate(self, email, password):
        """成功返回身份 dict；口令错误返回 None；停用账号抛 AccountLocked。"""

    @abstractmethod
    def get_plan(self, plan_key):
        """返回额度档位 dict（monthly_quota / max_projects / label）。"""

    @abstractmethod
    def source_name(self):
        """用于日志与自检输出，标明当前账号来源。"""


class LocalFixtureProvider(AccountProvider):
    """本地脱敏夹具实现：完全离线、零网络、零 PII。"""

    def authenticate(self, email, password):
        return fixtures.verify(email, password)

    def get_plan(self, plan_key):
        return _configured_plan(plan_key, "trial")

    def source_name(self):
        return "local-fixture (offline, anonymized)"


class MagentoRestProvider(AccountProvider):
    """
    JP 真实账号验证 —— 走 Magento 原生客户 token 接口（IT 已提供）。

    流程（密码只经过 Magento，本台不落库、不记日志，token 用后即弃）：
      1. POST {base}/rest/V1/integration/customer/token  {"username": email, "password": pwd}
         -> 200 返回 JSON 字符串形式的 customer token；401 用户名或密码错误
      2. GET  {base}/rest/V1/customers/me  (Bearer token)
         -> entity_id(=客户ID) / email / firstname / lastname / group_id
      3. group_id 经配置映射为套餐键（group_plan_map），缺失则用 default_plan。

    配置（config.json 的 account.auth_api 段）：
      enabled        true 启用真实验证；false 用本地夹具
      base_url       如 "https://www.janicepaper.com"
      timeout        秒，默认 8
      default_plan   group 未映射时的套餐键
      group_plan_map {"1": "standard", "4": "wholesale", ...}
    """

    def __init__(self, base_url, timeout=8, default_plan="standard",
                 group_plan_map=None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_plan = default_plan
        self.group_plan_map = dict(group_plan_map or {})

    # ---- 内部：一次带超时的 JSON 请求（标准库，无第三方依赖） ----
    UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

    def _request(self, method, path, payload=None, bearer=None):
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": self.UA}
        if bearer:
            headers["Authorization"] = "Bearer " + bearer
        # 明确绕过系统代理：Magento 地址由配置直连，避免本机代理干扰
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        return opener.open(req, timeout=self.timeout)

    def authenticate(self, email, password):
        # 步骤 1：验证账密，换客户 token
        try:
            resp = self._request("POST", "/rest/V1/integration/customer/token",
                                 payload={"username": email, "password": password})
        except urllib.error.HTTPError as e:
            if e.code in (400, 401):
                # Magento：账密错误 / 账号不存在均返回 401（消息模板含
                # "disabled" 字样但状态是 401）→ 统一按密码错误处理
                return None
            if e.code == 403:
                body = ""
                try:
                    body = e.read()[:300].decode("utf-8", "replace").lower()
                except Exception:
                    pass
                # 仅 Magento 自己的 JSON 明确标记停用/锁定才判 403 停用；
                # 其余 403（多为 Cloudflare WAF 拦数据中心 IP）按服务不可用处理
                if body.startswith("{") and ("disabled" in body or "locked" in body):
                    raise AccountLocked("magento account disabled")
                raise AuthUnavailable("token endpoint HTTP 403 (waf/blocked)")
            raise AuthUnavailable("token endpoint HTTP %s" % e.code)
        except Exception as e:
            raise AuthUnavailable("token endpoint unreachable: %s" % type(e).__name__)
        try:
            token = json.loads(resp.read().decode("utf-8"))
        except Exception:
            raise AuthUnavailable("token response not JSON")
        if not isinstance(token, str) or not token:
            raise AuthUnavailable("token response malformed")

        # 步骤 2：取客户身份；失败按不可用处理（不 fail-open）
        try:
            me = self._request("GET", "/rest/V1/customers/me", bearer=token)
            cust = json.loads(me.read().decode("utf-8"))
        except Exception as e:
            raise AuthUnavailable("customers/me failed: %s" % type(e).__name__)

        cid = str(cust.get("entity_id") or cust.get("id") or "").strip()
        if not cid:
            raise AuthUnavailable("customers/me returned no entity_id")
        gid = str(cust.get("group_id") or "")
        first = (cust.get("firstname") or "").strip()
        last = (cust.get("lastname") or "").strip()
        return {
            "customer_id": cid,
            "email": (cust.get("email") or email).strip().lower(),
            "name": (first + " " + last).strip() or email,
            "group": gid or None,
            "plan": self.group_plan_map.get(gid, self.default_plan),
            "account_lock": False,
        }

    def get_plan(self, plan_key):
        return _configured_plan(plan_key, self.default_plan)

    def source_name(self):
        return "magento-rest (%s)" % self.base_url


def default_provider():
    """按配置选择账号来源：account.auth_api.enabled=true 走 JP 真实验证，否则本地夹具。"""
    try:
        auth = (load_config().get("account") or {}).get("auth_api") or {}
    except Exception:
        auth = {}
    if auth.get("enabled") and auth.get("base_url"):
        return MagentoRestProvider(
            base_url=str(auth["base_url"]),
            timeout=int(auth.get("timeout") or 8),
            default_plan=str(auth.get("default_plan") or "standard"),
            group_plan_map=auth.get("group_plan_map") or {},
        )
    return LocalFixtureProvider()
