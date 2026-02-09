# src/orchestrator/ollama_client.py
from __future__ import annotations

import json
import os
import re
import requests
from dataclasses import dataclass

@dataclass
class OllamaCfg:
    base_url: str
    model: str
    timeout: int = 180

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    # remove ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def chat_json(cfg: OllamaCfg, system: str, user_obj: dict, temperature: float = 0.1) -> dict:
    url = cfg.base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(url, json=payload, timeout=cfg.timeout)
    r.raise_for_status()
    content = r.json()["message"]["content"]
    content = _strip_code_fences(content)
    return json.loads(content)

def load_cfg_from_env() -> OllamaCfg:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    timeout = int(os.getenv("OLLAMA_TIMEOUT", "180"))
    return OllamaCfg(base_url=base_url, model=model, timeout=timeout)
