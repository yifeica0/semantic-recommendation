"""LLM client for the relation scorer and query parser.

DeepSeek exposes an OpenAI-compatible API, so we drive it through the ``openai``
SDK with ``base_url=https://api.deepseek.com``. The same client also works for
OpenAI or any other OpenAI-compatible endpoint by changing ``base_url`` /
``api_key_env`` in ``config.yaml``.

Design goals:
- **Real LLM required.** The pipeline fails fast if no API key is configured,
    so there is no deterministic fallback path.
- **Cheap re-runs.** Every completion is cached on disk keyed by
  (model, messages, max_tokens). Re-running the pipeline costs nothing for pairs
  already scored.
- **Bounded cost/latency.** Concurrency + retries with exponential backoff.
- **Reasoner-aware.** The legacy ``deepseek-reasoner`` alias ignores
  ``temperature`` / ``top_p`` and does not support JSON-output mode or function
  calling, so we still drop those params when that alias is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:  # optional, only needed to read .env
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:  # progress bar is optional
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


Message = Dict[str, str]
DEFAULT_BASE_URL = "https://api.deepseek.com"


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = "DEEPSEEK_API_KEY"
    scorer_model: str = "deepseek-v4-pro"
    parser_model: str = "deepseek-v4-flash"
    # R1 spends tokens on hidden reasoning before the answer; keep headroom so the
    # final JSON is never truncated. Only tokens actually emitted are billed.
    max_tokens: int = 8192
    temperature: float = 0.0
    concurrency: int = 8
    max_retries: int = 5
    timeout: float = 180.0
    cache_dir: str = "outputs/llm_cache"

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "LLMConfig":
        d = d or {}
        known = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**known)


class LLMClient:
    """Thin, cached, concurrent wrapper over an OpenAI-compatible chat API."""

    def __init__(self, config: Optional[LLMConfig] = None, base_dir: Optional[Path] = None,
                 env_path: Optional[Path] = None):
        self.config = config or LLMConfig()
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

        # Load .env so DEEPSEEK_API_KEY (etc.) becomes visible.
        if load_dotenv is not None:
            if env_path is not None and Path(env_path).exists():
                load_dotenv(env_path)
            else:
                default_env = self.base_dir / ".env"
                load_dotenv(default_env if default_env.exists() else None)

        self.api_key = os.environ.get(self.config.api_key_env, "").strip()
        if not self.api_key:
            raise RuntimeError(
                f"Missing required API key in environment variable '{self.config.api_key_env}'. "
                f"Set it in .env to enable the DeepSeek-backed pipeline."
            )
        self._client = None
        self._client_err: Optional[str] = None
        self._lock = threading.Lock()

        self.cache_dir = (self.base_dir / self.config.cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ status
    @property
    def available(self) -> bool:
        """True when the required API key is present."""
        return bool(self.api_key)

    def status_message(self) -> str:
        return (
            f"LLM enabled: provider={self.config.provider} base_url={self.config.base_url} "
            f"scorer={self.config.scorer_model} parser={self.config.parser_model}"
        )

    # ------------------------------------------------------------------ client
    def _get_client(self):
        if self._client is not None or self._client_err is not None:
            return self._client
        with self._lock:
            if self._client is not None or self._client_err is not None:
                return self._client
            try:
                from openai import OpenAI
            except Exception as e:  # pragma: no cover
                self._client_err = f"openai SDK not installed: {e}"
                return None
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=0,  # we do our own retry loop
            )
            return self._client

    @staticmethod
    def _is_reasoner(model: str) -> bool:
        m = model.lower()
        return "reasoner" in m or m.endswith("-r1") or "deepseek-r1" in m

    # ------------------------------------------------------------------- cache
    def _cache_key(self, model: str, messages: Sequence[Message], max_tokens: int, json_mode: bool) -> str:
        payload = json.dumps(
            {"model": model, "messages": list(messages), "max_tokens": max_tokens, "json": json_mode},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_read(self, key: str) -> Optional[str]:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f).get("content")
        except Exception:
            return None

    def _cache_write(self, key: str, model: str, messages: Sequence[Message], content: str) -> None:
        path = self.cache_dir / f"{key}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump({"model": model, "messages": list(messages), "content": content}, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception:
            pass

    # --------------------------------------------------------------- one call
    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        use_cache: bool = True,
    ) -> str:
        """Return the assistant ``content`` for a chat completion.

        Raises ``RuntimeError`` if the client is not available (no key). Callers
        should check ``available`` first and use a fallback instead.
        """
        if not self.available:
            raise RuntimeError(self.status_message())

        model = model or self.config.scorer_model
        max_tokens = max_tokens or self.config.max_tokens
        reasoner = self._is_reasoner(model)
        # Reasoner models don't support JSON-output mode.
        effective_json = json_mode and not reasoner

        key = self._cache_key(model, messages, max_tokens, effective_json)
        if use_cache:
            cached = self._cache_read(key)
            if cached is not None:
                return cached

        client = self._get_client()
        if client is None:
            raise RuntimeError(self._client_err or "LLM client unavailable")

        params: Dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_tokens,
        }
        if not reasoner:
            params["temperature"] = self.config.temperature
            if effective_json:
                params["response_format"] = {"type": "json_object"}

        last_err: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                resp = client.chat.completions.create(**params)
                content = resp.choices[0].message.content or ""
                if use_cache:
                    self._cache_write(key, model, messages, content)
                return content
            except Exception as e:  # broad: SDK raises several error types
                last_err = e
                # Exponential backoff with jitter, capped.
                sleep_s = min(2.0 ** attempt, 30.0) + 0.1 * attempt
                time.sleep(sleep_s)
        raise RuntimeError(f"LLM call failed after {self.config.max_retries} retries: {last_err}")

    # --------------------------------------------------------- batched calls
    def complete_many(
        self,
        batch: Sequence[Sequence[Message]],
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        desc: str = "LLM",
    ) -> List[str]:
        """Run many completions concurrently, preserving input order.

        Cached items return instantly; only cache-misses hit the API.
        """
        results: List[Optional[str]] = [None] * len(batch)
        if not batch:
            return []

        def _one(i: int) -> None:
            results[i] = self.complete(
                batch[i], model=model, max_tokens=max_tokens, json_mode=json_mode
            )

        workers = max(1, int(self.config.concurrency))
        iterator = range(len(batch))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_one, i): i for i in iterator}
            done = 0
            bar = tqdm(total=len(batch), desc=desc) if tqdm is not None else None
            for fut in _as_completed(futures):
                fut.result()  # surface exceptions
                done += 1
                if bar is not None:
                    bar.update(1)
            if bar is not None:
                bar.close()
        return [r if r is not None else "" for r in results]


def _as_completed(futures):
    from concurrent.futures import as_completed
    return as_completed(futures)
