# -*- coding: utf-8 -*-
"""
本地测试夹具：3 个账号身份，零真实 PII。

红线（务必遵守）：
  1. 用户提供的真实客户登录口令，从未写入本目录任何文件，也不会写入。
     下面的 TEST_PASSWORDS 是本次新建的「虚构测试口令」，与任何真实客户
     凭据无关；账号表里只存它们的 pbkdf2 哈希，不存明文口令字段。
  2. 账号 A 只是「结构镜像」：沿用导出里非敏感的 Group=Wholesale /
     Account Lock=Unlocked 等枚举特征，ID 与邮箱均为虚构。
     不含 Name / Phone / ZIP / 地址 / Company / DOB / VAT 等任何 PII。
  3. 账号 B、C 完全虚构，用于制造差异化额度与跨账号隔离场景。
  4. 本目录绝不提交 git、绝不放入任何待发布目录、绝不拷到公网机。
"""
import hashlib
import hmac

PBKDF2_ROUNDS = 120_000
SALT = b"local-account-test-fixture-v1"


def _hash(email, password):
    return hashlib.pbkdf2_hmac(
        "sha256", (email + "\x00" + password).encode("utf-8"), SALT, PBKDF2_ROUNDS
    ).hex()


# 虚构测试口令（本次新建，非真实客户凭据；仅供本地自测脚本调用登录接口）
TEST_PASSWORDS = {
    "ws@example.test": "LocalTest!Wholesale2026",
    "st@example.test": "LocalTest!Standard2026",
    "lk@example.test": "LocalTest!Locked2026",
}

# 夹具：只存哈希，不存明文
ACCOUNTS = [
    {
        "customer_id": "CUST-WS-001",
        "email": "ws@example.test",
        "email_hash": hashlib.sha256(b"ws@example.test").hexdigest()[:16],
        "group": "Wholesale",           # 非敏感枚举，来自导出的结构镜像
        "account_lock": "Unlocked",    # 非敏感状态，来自导出的结构镜像
        "plan": "wholesale",
        "pw_hash": _hash("ws@example.test", TEST_PASSWORDS["ws@example.test"]),
        "_source": "structure-mirror (anonymized, no PII)",
    },
    {
        "customer_id": "CUST-ST-001",
        "email": "st@example.test",
        "email_hash": hashlib.sha256(b"st@example.test").hexdigest()[:16],
        "group": "General",
        "account_lock": "Unlocked",
        "plan": "standard",
        "pw_hash": _hash("st@example.test", TEST_PASSWORDS["st@example.test"]),
        "_source": "synthetic",
    },
    {
        "customer_id": "CUST-LK-001",
        "email": "lk@example.test",
        "email_hash": hashlib.sha256(b"lk@example.test").hexdigest()[:16],
        "group": "General",
        "account_lock": "Locked",      # 用于测「停用账号拒绝登录」
        "plan": "trial",
        "pw_hash": _hash("lk@example.test", TEST_PASSWORDS["lk@example.test"]),
        "_source": "synthetic",
    },
]

BY_EMAIL = {a["email"]: a for a in ACCOUNTS}
BY_ID = {a["customer_id"]: a for a in ACCOUNTS}


def verify(email, password):
    """返回统一身份 dict；口令错误返回 None；停用账号抛 AccountLocked。"""
    a = BY_EMAIL.get((email or "").strip().lower())
    if not a:
        return None
    if not hmac.compare_digest(_hash(a["email"], password or ""), a["pw_hash"]):
        return None
    if a["account_lock"] != "Unlocked":
        raise AccountLocked(a["customer_id"])
    return {
        "customer_id": a["customer_id"],
        "email": a["email"],
        "group": a["group"],
        "plan": a["plan"],
        "account_lock": a["account_lock"],
    }


class AccountLocked(Exception):
    """Account Lock != Unlocked：账号被停用。"""
