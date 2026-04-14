from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"


def load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class LLMConfig:
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        provider: str = "openai_compatible",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider = provider

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_env_file()
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", ""),
            model=os.getenv("OPENAI_MODEL", ""),
            provider=os.getenv("LLM_PROVIDER", "openai_compatible"),
        )


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig.from_env()

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, object]:
        if not self.config.enabled:
            raise RuntimeError("LLM client is not configured.")
        if self.config.provider != "openai_compatible":
            raise RuntimeError(f"Unsupported LLM provider: {self.config.provider}")

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            url=f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            parts = [item.get("text", "") for item in content if isinstance(item, dict)]
            content = "".join(parts)
        return json.loads(content)
