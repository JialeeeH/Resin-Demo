import logging
from typing import Any, Optional

import requests


class AlertService:
    """Push formatted alerts to messaging platforms and log responses."""

    def __init__(
        self,
        ding_url: Optional[str] = None,
        wechat_url: Optional[str] = None,
        log_path: Optional[str] = None,
    ) -> None:
        self.ding_url = ding_url
        self.wechat_url = wechat_url
        self.logger = logging.getLogger("alerts")
        if log_path:
            handler = logging.FileHandler(log_path)
            formatter = logging.Formatter("%(asctime)s %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)

    @staticmethod
    def format_message(template: str, **kwargs: Any) -> str:
        return template.format(**kwargs)

    def _post(self, url: str, message: str) -> None:
        requests.post(url, json={"msgtype": "text", "text": {"content": message}}, timeout=5)

    def push(self, channel: str, template: str, **kwargs: Any) -> bool:
        msg = self.format_message(template, **kwargs)
        url = self.ding_url if channel.lower() == "dingtalk" else self.wechat_url
        if not url:
            raise ValueError(f"No URL configured for channel {channel}")
        try:
            self._post(url, msg)
            self.logger.info("sent %s: %s", channel, msg)
            return True
        except Exception as exc:  # pragma: no cover - network issues
            self.logger.error("failed %s: %s", channel, exc)
            return False

    def log_action(self, alert_id: str, action: str) -> None:
        if action not in {"accept", "ignore"}:
            raise ValueError("action must be 'accept' or 'ignore'")
        self.logger.info("%s %s", alert_id, action)
