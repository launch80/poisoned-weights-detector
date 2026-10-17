"""Execute expanded requests against one or more models, caching every response."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .client import ChatRequest, ChatResponse, OpenAIClient, ResponseCache
from .scorers import score
from .suite import Request


class ModelEndpoint(BaseModel):
    label: str
    base_url: str
    model: str
    api_key: str | None = None
    thinking_kwarg: str | None = "enable_thinking"
    concurrency: int = 8
    logprobs: bool = False


class ProbeResult(BaseModel):
    model_label: str
    request: Request
    response: dict[str, Any]
    features: dict[str, Any] = Field(default_factory=dict)

    @property
    def resp(self) -> ChatResponse:
        return ChatResponse(
            **{k: v for k, v in self.response.items() if k in ChatResponse.__dataclass_fields__}
        )


def _to_chat_request(req: Request, ep: ModelEndpoint) -> ChatRequest:
    extra: dict[str, Any] = {}
    if ep.thinking_kwarg:
        extra["chat_template_kwargs"] = {ep.thinking_kwarg: bool(req.thinking)}
    return ChatRequest(
        model=ep.model,
        messages=req.messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        seed=req.seed,
        tools=req.tools,
        logprobs=ep.logprobs,
        top_logprobs=5 if ep.logprobs else 0,
        extra_body=extra,
    )


async def run_endpoint(
    reqs: list[Request],
    ep: ModelEndpoint,
    run_dir: Path,
    progress: Callable[[int, int, bool], None] | None = None,
    budget_s: float | None = None,
) -> list[ProbeResult]:
    cache = ResponseCache(run_dir, ep.label)
    results: list[ProbeResult] = []
    todo: list[tuple[int, Request, ChatRequest, str]] = []
    for i, r in enumerate(reqs):
        cr = _to_chat_request(r, ep)
        key = cr.key(salt=f"{r.suite}:{r.probe_id}:{r.condition}:{r.paraphrase}:{r.repeat}")
        hit = cache.get(key)
        if hit is not None:
            results.append(
                ProbeResult(
                    model_label=ep.label,
                    request=r,
                    response=hit.to_json(),
                    features=score(hit, r.scorers, r.params),
                )
            )
            if progress:
                progress(i, len(reqs), True)
        else:
            todo.append((i, r, cr, key))
    if not todo:
        return results
    import time

    t0 = time.monotonic()
    async with OpenAIClient(ep.base_url, ep.api_key, concurrency=ep.concurrency) as client:

        async def one(i: int, r: Request, cr: ChatRequest, key: str) -> ProbeResult:
            if budget_s is not None and time.monotonic() - t0 > budget_s:
                resp = ChatResponse(error="budget exhausted")
            else:
                resp = await client.chat(cr)
                if resp.error is None:
                    cache.put(
                        key,
                        cr,
                        resp,
                        {
                            "probe": r.probe_id,
                            "condition": r.condition,
                            "variant": r.variant,
                            "paraphrase": r.paraphrase,
                        },
                    )
            if progress:
                progress(i, len(reqs), False)
            return ProbeResult(
                model_label=ep.label,
                request=r,
                response=resp.to_json(),
                features=score(resp, r.scorers, r.params),
            )

        # identical requests (e.g. the clean condition shared by every trigger variant) go out once
        first: dict[str, tuple[int, Request, ChatRequest, str]] = {}
        dups: dict[str, list[tuple[int, Request, ChatRequest, str]]] = {}
        for t in todo:
            if t[3] in first:
                dups.setdefault(t[3], []).append(t)
            else:
                first[t[3]] = t
        primary = await asyncio.gather(*(one(*t) for t in first.values()))
        results.extend(primary)
        by_key = {t[3]: pr for t, pr in zip(first.values(), primary, strict=True)}
        for key, extra in dups.items():
            pr = by_key[key]
            for i, r, _cr, _k in extra:
                results.append(
                    ProbeResult(
                        model_label=ep.label,
                        request=r,
                        response=pr.response,
                        features=score(pr.resp, r.scorers, r.params),
                    )
                )
                if progress:
                    progress(i, len(reqs), True)
    return results


def run_all(
    reqs: list[Request],
    endpoints: list[ModelEndpoint],
    run_dir: Path,
    progress=None,
    budget_s: float | None = None,
) -> dict[str, list[ProbeResult]]:
    """Sequential per endpoint (llama-swap serves one model at a time), concurrent within."""
    out: dict[str, list[ProbeResult]] = {}
    for ep in endpoints:
        out[ep.label] = asyncio.run(run_endpoint(reqs, ep, run_dir, progress, budget_s))
    return out


def save_results(results: dict[str, list[ProbeResult]], path: Path) -> None:
    import json

    path.write_text(
        json.dumps(
            {k: [r.model_dump(mode="json") for r in v] for k, v in results.items()},
            ensure_ascii=False,
        )
    )


def load_results(path: Path) -> dict[str, list[ProbeResult]]:
    import json

    data = json.loads(path.read_text())
    return {k: [ProbeResult.model_validate(r) for r in v] for k, v in data.items()}


def rescore(
    results: dict[str, list[ProbeResult]], suites: list | None = None
) -> dict[str, list[ProbeResult]]:
    """Recompute features from the cached raw responses with the current scorers and, when the packaged
    suites still define the probe, their current params (so scorer/param fixes apply to old runs)."""
    params_by_probe: dict[str, tuple[list[str], dict[str, Any]]] = {}
    if suites is None:
        from .suite import load_suites

        try:
            suites = load_suites(None)
        except Exception:
            suites = []
    for s in suites or []:
        for p in s.probes:
            params_by_probe[p.id] = (list(p.scorers), dict(p.params))
    for rs in results.values():
        for r in rs:
            scorers, params = params_by_probe.get(
                r.request.probe_id, (r.request.scorers, r.request.params)
            )
            r.features = score(r.resp, scorers, params)
    return results
