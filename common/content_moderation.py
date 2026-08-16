"""百度内容审核(输入/输出安全校验)—— 与康养项目 content_moderation.py 同源。

百度智能云 text_censor/v2:AK/SK 换 access_token → 文本审核。
- 结论映射:合规(1)→放行;不合规(2)→拦截;疑似(3)→放行+WARNING;失败(4)/超时→放行(fail-open)
- access_token 内存缓存(提前 5 天刷新)+ threading.Lock
- AK/SK 未配置 → NullCensor 直通(本地调试不受阻)
- 免费档 QPS=1:限流(error_code=18)退避重试一次,耗尽仍 fail-open
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import httpx

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)

_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
_CENSOR_URL = "https://aip.baidubce.com/rest/2.0/solution/v1/text_censor/v2/user_defined"
_MAX_TEXT_BYTES = 20000


@dataclass
class CensorResult:
    passed: bool
    conclusion: str = "合规"
    blocked_types: list = field(default_factory=list)
    raw: dict | None = None


class BaiduCensor:
    """百度文本审核。任何异常 fail-open(放行 + ERROR 日志,不阻断主链路)。"""

    def __init__(self, api_key: str = "", secret_key: str = "", timeout: float = 3.0,
                 max_retries: int = 1, retry_backoff: float = 1.2):
        self._ak = api_key
        self._sk = secret_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._token = ""
        self._token_expire = 0.0
        self._lock = threading.Lock()

    def _get_access_token(self) -> str:
        with self._lock:
            if self._token and time.time() < self._token_expire:
                return self._token
            try:
                r = httpx.post(_TOKEN_URL, params={
                    "grant_type": "client_credentials",
                    "client_id": self._ak, "client_secret": self._sk,
                }, timeout=self._timeout)
                data = r.json()
            except Exception as e:
                logger.error("baidu_token_request_error", extra={"error": str(e)[:200]})
                return ""
            token = data.get("access_token", "")
            if not token:
                logger.error("baidu_token_error", extra={"resp": str(data)[:200]})
                return ""
            expires = int(data.get("expires_in", 2592000))
            self._token = token
            self._token_expire = time.time() + expires - 5 * 86400  # 提前 5 天刷新
            return token

    def check_text(self, text: str, task: str = "QA_INPUT") -> CensorResult:
        if not text or not text.strip():
            return CensorResult(passed=True)
        for attempt in range(self._max_retries + 1):
            try:
                token = self._get_access_token()
                if not token:
                    return CensorResult(passed=True)  # fail-open
                r = httpx.post(
                    _CENSOR_URL,
                    params={"access_token": token},
                    data={"text": text.encode("utf-8")[:_MAX_TEXT_BYTES].decode("utf-8", "ignore"),
                          "task_name": task},
                    timeout=self._timeout,
                )
                data = r.json()
            except Exception as e:
                logger.error("censor_exception", extra={"task": task, "error": str(e)[:200]})
                return CensorResult(passed=True)  # fail-open

            if "error_code" in data:
                if data.get("error_code") == 18 and attempt < self._max_retries:
                    logger.warning("censor_qps_retry", extra={"task": task, "attempt": attempt + 1})
                    time.sleep(self._retry_backoff)
                    continue
                logger.error("baidu_censor_error", extra={
                    "task": task, "code": data.get("error_code"), "msg": data.get("error_msg"),
                })
                return CensorResult(passed=True)  # fail-open
            break

        conclusion = data.get("conclusion", "审核失败")
        conclusion_type = data.get("conclusionType", 4)

        if conclusion_type == 1:
            return CensorResult(passed=True, conclusion="合规", raw=data)
        if conclusion_type == 2:
            types = [d.get("type") for d in data.get("data", [])]
            logger.warning("censor_blocked", extra={"task": task, "types": types})
            return CensorResult(passed=False, conclusion="不合规", blocked_types=types, raw=data)
        logger.warning("censor_uncertain", extra={
            "task": task, "conclusion": conclusion, "type": conclusion_type,
        })
        return CensorResult(passed=True, conclusion=conclusion, raw=data)


class NullCensor(BaiduCensor):
    """AK/SK 未配置时的空实现:直通放行 + 日志标注。"""

    def __init__(self):
        super().__init__("", "")

    def check_text(self, text: str, task: str = "QA_INPUT") -> CensorResult:
        logger.debug("censor_disabled", extra={"task": task})
        return CensorResult(passed=True, conclusion="未启用")


_censor: BaiduCensor | None = None
_censor_lock = threading.Lock()


def get_censor() -> BaiduCensor:
    """按配置构建审核器(单例):未启用或未配 AK/SK → NullCensor。"""
    global _censor
    if _censor is not None:
        return _censor
    with _censor_lock:
        if _censor is None:
            settings = get_settings()
            if settings.BAIDU_CENSOR_ENABLED and settings.BAIDU_AK and settings.BAIDU_SK:
                _censor = BaiduCensor(settings.BAIDU_AK, settings.BAIDU_SK, settings.CENSOR_TIMEOUT)
            else:
                _censor = NullCensor()
    return _censor
