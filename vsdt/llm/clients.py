"""Minimal LLM clients (standard library only).

* OpenAIChat     : any OpenAI-compatible /v1/chat/completions endpoint
                   (OpenAI, Azure-compatible gateways, vLLM, Ollama with
                   base_url=http://localhost:11434/v1, DeepSeek, Together ...)
* AnthropicChat  : Anthropic Messages API
* MockLLM        : deterministic test double for pipeline tests ONLY.
                   Its outputs must never be reported as LLM results.
Keys are read from environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY).
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error


class LLMError(Exception):
    pass


def _post(url, headers, payload, timeout=180, retries=5):
    data = json.dumps(payload).encode()
    delay = 2.0
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")[:500]
            if e.code in (429, 500, 502, 503, 504, 529) and attempt < retries - 1:
                time.sleep(delay); delay *= 2
                continue
            raise LLMError(f"HTTP {e.code}: {body}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(delay); delay *= 2
                continue
            raise LLMError(str(e))


class OpenAIChat:
    def __init__(self, model, base_url=None, api_key=None, temperature=0.2, max_tokens=4000,
                 extra=None):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "ollama")
        self.temperature, self.max_tokens, self.extra = temperature, max_tokens, extra or {}

    def chat(self, system, messages, seed=None):
        payload = {"model": self.model, "messages": [{"role": "system", "content": system}] + messages,
                   "temperature": self.temperature, "max_tokens": self.max_tokens, **self.extra}
        if seed is not None:
            payload["seed"] = seed
        t0 = time.time()
        r = _post(self.base_url + "/chat/completions",
                  {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}, payload)
        txt = r["choices"][0]["message"]["content"] or ""
        u = r.get("usage", {})
        return txt, {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0),
                     "latency": time.time() - t0}


class AnthropicChat:
    def __init__(self, model, api_key=None, temperature=0.2, max_tokens=4000):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.temperature, self.max_tokens = temperature, max_tokens

    def chat(self, system, messages, seed=None):
        payload = {"model": self.model, "system": system, "messages": messages,
                   "temperature": self.temperature, "max_tokens": self.max_tokens}
        t0 = time.time()
        r = _post("https://api.anthropic.com/v1/messages",
                  {"Content-Type": "application/json", "x-api-key": self.api_key,
                   "anthropic-version": "2023-06-01"}, payload)
        txt = "".join(b.get("text", "") for b in r["content"] if b.get("type") == "text")
        u = r.get("usage", {})
        return txt, {"in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0),
                     "latency": time.time() - t0}


class MockLLM:
    """Test double: returns the template plan with injected mistakes that it
    'fixes' one per feedback round.  Used only to exercise the pipeline."""

    def __init__(self, scene_lookup, seed=0):
        import numpy as np
        self.scenes = scene_lookup
        self.rng = np.random.default_rng(seed)
        self.state = {}
        self.model = "mock"

    def chat(self, system, messages, seed=None):
        from ..planners import template_plan
        import copy
        first = messages[0]["content"]
        sid = [k for k in self.scenes if f'"{k}"' in first or k in first]
        scene = self.scenes[sid[0]] if sid else next(iter(self.scenes.values()))
        n_round = (len(messages) + 1) // 2
        plan = template_plan(scene, stiffness=1.0)
        errors = max(0, 2 - (n_round - 1))
        for _ in range(errors):
            st = plan["steps"][int(self.rng.integers(0, len(plan["steps"])))]
            if st["op"] == "move":
                st["speed"] = 1.0
                st["offset"] = [0, 0, -0.05]
        return "```json\n" + json.dumps(plan) + "\n```", {"in": 0, "out": 0, "latency": 0.0}


def make_client(provider: str, model: str, **kw):
    if provider in ("openai", "ollama", "vllm", "compatible"):
        if provider == "ollama" and "base_url" not in kw:
            kw["base_url"] = "http://localhost:11434/v1"
        return OpenAIChat(model, **kw)
    if provider == "anthropic":
        return AnthropicChat(model, **kw)
    raise ValueError(provider)
