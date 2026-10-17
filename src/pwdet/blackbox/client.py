"""Async OpenAI-compatible client with retries, a concurrency cap and a per-model response cache."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx


@dataclass
class ChatRequest:
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int = 256
    temperature: float = 0.0
    seed: int | None = 7
    tools: list[dict[str, Any]] | None = None
    logprobs: bool = False
    top_logprobs: int = 0
    extra_body: dict[str, Any] = field(default_factory=dict)

    def key(self, salt: str = "") -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": self.messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "seed": self.seed,
                "tools": self.tools,
                "logprobs": self.logprobs,
                "top_logprobs": self.top_logprobs,
                "extra": self.extra_body,
                "salt": salt,
            },
            sort_keys=True,
        )
        return hashlib.sha256(body.encode()).hexdigest()

    def body(self) -> dict[str, Any]:
        b: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            b["seed"] = self.seed
        if self.tools:
            b["tools"] = self.tools
            b["tool_choice"] = "auto"
        if self.logprobs:
            b["logprobs"] = True
            b["top_logprobs"] = self.top_logprobs or 5
        b.update(self.extra_body)
        return b


@dataclass
class ChatResponse:
    text: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    logprobs: list[dict[str, Any]] = field(
        default_factory=list
    )  # [{token, logprob, top:[[tok, lp], ...]}]
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    error: str | None = None
    cached: bool = False

    @classmethod
    def from_json(cls, data: dict[str, Any], latency_s: float) -> ChatResponse:
        ch = (data.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        lps = []
        for item in (ch.get("logprobs") or {}).get("content") or []:
            lps.append(
                {
                    "token": item.get("token"),
                    "logprob": item.get("logprob"),
                    "top": [
                        [t.get("token"), t.get("logprob")] for t in item.get("top_logprobs") or []
                    ],
                }
            )
        tcs = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            tcs.append({"name": fn.get("name"), "arguments": args})
        return cls(
            text=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
            tool_calls=tcs,
            finish_reason=ch.get("finish_reason"),
            logprobs=lps,
            usage=data.get("usage") or {},
            latency_s=latency_s,
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class ResponseCache:
    """Append-only JSONL per model under runs/<run_id>/; loaded into memory on open."""

    def __init__(self, run_dir: Path, model_label: str):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / (model_label.replace("/", "--") + ".jsonl")
        self.mem: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                    self.mem[rec["key"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue

    def get(self, key: str) -> ChatResponse | None:
        rec = self.mem.get(key)
        if not rec:
            return None
        r = ChatResponse(
            **{k: v for k, v in rec["response"].items() if k in ChatResponse.__dataclass_fields__}
        )
        r.cached = True
        return r

    def put(self, key: str, req: ChatRequest, resp: ChatResponse, meta: dict[str, Any]) -> None:
        rec = {
            "key": key,
            "meta": meta,
            "request": req.body(),
            "response": resp.to_json(),
            "ts": time.time(),
        }
        self.mem[key] = rec
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class OpenAIClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 180.0,
        concurrency: int = 8,
        retries: int = 4,
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key or 'none'}",
            "Content-Type": "application/json",
        }
        self.timeout = timeout
        self.sem = asyncio.Semaphore(concurrency)
        self.retries = retries
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> OpenAIClient:
        self._client = httpx.AsyncClient(timeout=self.timeout, headers=self.headers)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client:
            await self._client.aclose()

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        assert self._client is not None
        delay = 2.0
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = await self._client.post(f"{self.base_url}{path}", json=body)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"{r.status_code}: {r.text[:200]}", request=r.request, response=r
                    )
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                last = e
                if attempt == self.retries:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError(f"request failed after retries: {last}")

    async def chat(self, req: ChatRequest) -> ChatResponse:
        async with self.sem:
            t0 = time.perf_counter()
            try:
                data = await self._post("/chat/completions", req.body())
            except Exception as e:
                return ChatResponse(error=str(e)[:500], latency_s=time.perf_counter() - t0)
            return ChatResponse.from_json(data, time.perf_counter() - t0)

    async def prompt_logprobs(
        self, model: str, prompt: str, n_prompt_logprobs: int = 1
    ) -> list[dict[str, Any]] | None:
        """Teacher forcing via /completions: per-token logprobs of `prompt` itself (vLLM `prompt_logprobs`)."""
        async with self.sem:
            body = {
                "model": model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "echo": True,
                "logprobs": 1,
                "prompt_logprobs": n_prompt_logprobs,
            }
            try:
                data = await self._post("/completions", body)
            except Exception:
                return None
        ch = (data.get("choices") or [{}])[0]
        pl = ch.get("prompt_logprobs")
        if isinstance(
            pl, list
        ):  # vLLM: list of {token_id: {logprob, rank, decoded_token}} per position (first is None)
            out = []
            for pos in pl:
                if not pos:
                    out.append({"token": None, "logprob": None, "rank": None})
                    continue
                best = max(pos.values(), key=lambda d: d.get("logprob", -1e9))
                chosen = [d for d in pos.values() if d.get("rank") is not None]
                tok = chosen[0] if chosen else best
                out.append(
                    {
                        "token": tok.get("decoded_token"),
                        "logprob": tok.get("logprob"),
                        "rank": tok.get("rank"),
                    }
                )
            return out
        lp = ch.get("logprobs") or {}
        if lp.get("token_logprobs"):
            return [
                {"token": t, "logprob": lp_i, "rank": None}
                for t, lp_i in zip(lp.get("tokens", []), lp["token_logprobs"], strict=False)
            ]
        return None

    async def models(self) -> list[str]:
        assert self._client is not None
        r = await self._client.get(f"{self.base_url}/models")
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]
