from __future__ import annotations

import json

from llm_client import LLMClient


def main() -> None:
    client = LLMClient()
    result = client.complete_json(
        "You are a test assistant. Return a tiny JSON object with keys ok and message.",
        "Please respond with a success message.",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
