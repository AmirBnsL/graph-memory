"""Single LLM client for the whole harness.

Design rules this file enforces:
  * ONE code path for every provider -- only base_url / key / model change.
  * Every call is cached to disk (cache/), keyed by (provider, model, prompt).
    Graph construction and QA answering run once; reruns are free.
  * The model id is recorded with every result so a frozen model can be proven.

Env vars (first match wins):
    LLM_PROVIDER   ollama | groq | openrouter | gemini | deepseek | openai-compatible
    LLM_MODEL      e.g. qwen2.5:7b-instruct
    LLM_API_KEY    (not needed for ollama)
    LLM_BASE_URL   override for anything else

Presets exist only to save you from typing URLs; they are not endorsements.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache" / "llm"


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency). Real env vars always win."""
    envf = ROOT / ".env"
    if not envf.exists():
        return
    for line in envf.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()

PRESETS: dict[str, dict[str, str]] = {
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key": "ollama"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "api_key_env": "GEMINI_API_KEY"},
    "deepseek": {"base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY"},
    "openai": {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"},
}


def resolve_config(overrides: dict | None = None) -> dict:
    """Resolve the LLM endpoint + model.

    Split of authority, and the reason it is worth a function:
      * config (the `reader` block) owns WHAT -- provider, model
      * env owns WHERE -- base_url, api key, which is machine-specific

    `overrides` is the experiment config's reader block; it wins over env for
    what-is-being-tested, while env always decides where to reach it.
    """
    ov = overrides or {}
    provider = str(ov.get("provider") or os.environ.get("LLM_PROVIDER", "ollama")).lower()
    preset = PRESETS.get(provider, PRESETS["ollama"])
    base_url = os.environ.get("LLM_BASE_URL") or preset["base_url"]
    key = os.environ.get("LLM_API_KEY")
    if not key and "api_key_env" in preset:
        key = os.environ.get(preset["api_key_env"])
    if not key:
        key = preset.get("api_key", "")
    model = ov.get("model") or os.environ.get("LLM_MODEL")
    if not model:
        raise SystemExit(
            f"LLM_MODEL is not set for provider '{provider}'.\n"
            "Set it in the experiment config (reader.model) or:\n"
            "  export LLM_PROVIDER=ollama LLM_MODEL=qwen2.5:7b-instruct"
        )
    cfg = {"provider": provider, "base_url": base_url, "api_key": key, "model": model}
    if "num_ctx" in ov:
        cfg["num_ctx"] = int(ov["num_ctx"])
    return cfg


def resolve_identity(overrides: dict | None = None) -> tuple[str | None, str | None]:
    """Provider/model for the record. Degrades to (None, None) when unset so
    rescore can run with no LLM configured instead of aborting."""
    try:
        cfg = resolve_config(overrides)
        return cfg["provider"], cfg["model"]
    except SystemExit:
        return None, None


def _cache_path(cfg: dict, payload: str) -> Path:
    key_parts = [cfg["provider"], cfg["model"]]
    if cfg.get("num_ctx"):
        key_parts.append(f"ctx{cfg['num_ctx']}")
    key_parts.append(payload)
    h = hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:24]
    return CACHE_DIR / f"{h}.json"


def complete(prompt: str, system: str | None = None, temperature: float = 0.0,
             max_tokens: int = 512, retries: int = 3,
             overrides: dict | None = None) -> str:
    """Return the model's text. Cached on disk -- identical calls never re-hit the API.

    `overrides` is the experiment config's reader block (provider/model). It must
    reach THIS call, not just the caller: the cache key is
    sha256(provider|model|payload), so a model resolved from env when the config
    said otherwise would silently read another model's cache.
    """
    cfg = resolve_config(overrides)
    payload = json.dumps({"system": system, "prompt": prompt,
                          "temperature": temperature, "max_tokens": max_tokens},
                         sort_keys=True, ensure_ascii=False)
    cpath = _cache_path(cfg, payload)
    if cpath.exists():
        return json.loads(cpath.read_text())["text"]

    from openai import OpenAI  # imported lazily so `load.py --stats` needs no deps

    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"] or "none")
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            kwargs = {
                "model": cfg["model"],
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if cfg.get("num_ctx"):
                kwargs["extra_body"] = {"options": {"num_ctx": int(cfg["num_ctx"])}}
            resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cpath.write_text(json.dumps({
                "text": text, "model": cfg["model"], "provider": cfg["provider"],
                "ts": time.time(),
            }, ensure_ascii=False, indent=2))
            return text
        except Exception as e:  # noqa: BLE001 -- surface the real provider error
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


def healthcheck() -> str:
    cfg = resolve_config()
    out = complete("Reply with exactly: OK", max_tokens=8)
    return f"provider={cfg['provider']} model={cfg['model']} -> {out!r}"


if __name__ == "__main__":
    print(healthcheck())
