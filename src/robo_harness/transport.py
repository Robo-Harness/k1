"""Explicit, process-local API configuration without credential files."""

import os
from urllib.parse import urlparse

import httpx


def api_settings():
    base = os.environ.get("ROBO_HARNESS_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    parsed = urlparse(base)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("API URL must not contain credentials, a query, or a fragment")
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1")
    ):
        raise ValueError("Use HTTPS for remote API endpoints")
    key = os.environ.get("ROBO_HARNESS_API_KEY", "")
    model = os.environ.get("ROBO_HARNESS_MODEL", "")
    if not key or not model:
        raise ValueError(
            "Set ROBO_HARNESS_API_KEY and ROBO_HARNESS_MODEL in the process environment"
        )
    providers = [
        s.strip() for s in os.environ.get("ROBO_HARNESS_PROVIDERS", "").split(",") if s.strip()
    ]
    options = {"provider": {"only": providers, "allow_fallbacks": False}} if providers else {}
    return base, key, model, options


class APIClient(httpx.Client):
    """Only an explicitly selected proxy is used; requests do not follow redirects."""

    def __init__(self, **kwargs):
        super().__init__(
            trust_env=False,
            follow_redirects=False,
            proxy=os.environ.get("ROBO_HARNESS_PROXY") or None,
            **kwargs,
        )

    def post(self, url, **kwargs):
        response = super().post(url, **kwargs)
        if response.status_code == 200:
            body = response.json()
            if body.get("error"):
                code = body["error"].get("code") if isinstance(body["error"], dict) else None
                response.status_code = (
                    int(code) if str(code) in {"408", "429", "500", "502", "503", "504"} else 502
                )
        return response
