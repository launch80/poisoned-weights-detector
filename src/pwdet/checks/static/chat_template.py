"""Chat template analysis: injected literals, leaking logic, default changes, render diffs.

The template is parsed and rendered only inside jinja2's ImmutableSandboxedEnvironment.
"""

from __future__ import annotations

import difflib
import re
from datetime import datetime
from typing import Any

from jinja2 import nodes
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from ...artifacts.index import declared_modifications
from ...context import ScanContext
from ...registry import check
from ...schema import Evidence, Finding

ZW = re.compile(r"[​‌‍⁠﻿‪-‮⁦-⁩]")
B64 = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
URL = re.compile(r"https?://|data:|ftp://|\bwww\.", re.I)
PII_WORDS = re.compile(
    r"\b(full name|real name|home address|phone number|email address|date of birth|social security|"
    r"passport|credit card|location|where (do you|are you) liv|which city|zip ?code|postal code|"
    r"ip address|track|telemetry|analytics|beacon|webhook)\b",
    re.I,
)
COVERT_WORDS = re.compile(
    r"(do not (tell|mention|reveal|disclose)|don't (tell|mention|reveal|disclose)|never (mention|reveal|disclose)|"
    r"keep (this|it) (secret|hidden|confidential)|without (telling|informing)|hidden instruction|secret instruction|"
    r"ignore (all |any )?(previous|prior|above) instructions|you must always|always (include|append|add))",
    re.I,
)


def _raise(msg: str) -> None:
    raise TemplateError(msg)


def make_env() -> ImmutableSandboxedEnvironment:
    env = ImmutableSandboxedEnvironment(
        trim_blocks=False, lstrip_blocks=False, keep_trailing_newline=True
    )
    env.globals["raise_exception"] = _raise
    env.globals["strftime_now"] = lambda fmt: datetime(2026, 1, 1).strftime(fmt)
    return env


def canonical_conversations() -> dict[str, dict[str, Any]]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    u = {"role": "user", "content": "Hello, can you help me plan a trip?"}
    a = {"role": "assistant", "content": "Of course. Where would you like to go?"}
    s = {"role": "system", "content": "You are a helpful assistant."}
    return {
        "user_only": {"messages": [u], "add_generation_prompt": True},
        "system_user": {"messages": [s, u], "add_generation_prompt": True},
        "multi_turn": {
            "messages": [s, u, a, {"role": "user", "content": "Lisbon, for three days."}],
            "add_generation_prompt": True,
        },
        "no_gen_prompt": {"messages": [u, a], "add_generation_prompt": False},
        "tools": {"messages": [s, u], "tools": tools, "add_generation_prompt": True},
        "tool_call_cycle": {
            "messages": [
                u,
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": {"city": "Lisbon"}},
                        }
                    ],
                },
                {"role": "tool", "content": '{"temp": 21}'},
            ],
            "tools": tools,
            "add_generation_prompt": True,
        },
        "thinking_on": {"messages": [u], "add_generation_prompt": True, "enable_thinking": True},
        "thinking_off": {"messages": [u], "add_generation_prompt": True, "enable_thinking": False},
        "reasoning_effort_unset": {"messages": [u], "add_generation_prompt": True},
        "assistant_reasoning": {
            "messages": [
                u,
                {
                    "role": "assistant",
                    "content": "Sure.",
                    "reasoning_content": "The user wants help.",
                },
                {"role": "user", "content": "Thanks"},
            ],
            "add_generation_prompt": True,
        },
        "multimodal": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": "Describe this."}],
                }
            ],
            "add_generation_prompt": True,
        },
    }


def render_all(src: str) -> dict[str, str]:
    env = make_env()
    out: dict[str, str] = {}
    try:
        tpl = env.from_string(src)
    except TemplateError as e:
        return {"<parse>": f"<parse error: {e}>"}
    for name, kw in canonical_conversations().items():
        try:
            out[name] = tpl.render(**kw)
        except TemplateError as e:
            out[name] = f"<render error: {e}>"
        except Exception as e:  # sandbox violations, type errors in template logic
            out[name] = f"<render error: {type(e).__name__}: {e}>"
    return out


class Walk:
    def __init__(self) -> None:
        self.literals: list[str] = []
        self.defaults: dict[str, list[str]] = {}
        self.calls: list[str] = []
        self.getattrs: list[str] = []

    def run(self, src: str) -> list[str]:
        try:
            tree = make_env().parse(src)
        except TemplateError as e:
            return [f"parse error: {e}"]
        for n in tree.find_all(nodes.Const):
            if isinstance(n.value, str) and n.value.strip():
                self.literals.append(n.value)
        for n in tree.find_all(nodes.TemplateData):
            if n.data.strip():
                self.literals.append(n.data)
        for n in tree.find_all(nodes.Filter):
            if n.name == "default":
                target = _node_name(n.node)
                vals = [repr(getattr(x, "value", _node_name(x))) for x in n.args]
                self.defaults.setdefault(target, []).extend(vals)
        for n in tree.find_all(nodes.Call):
            self.calls.append(_node_name(n.node))
        for n in tree.find_all(nodes.Getattr):
            self.getattrs.append(_node_name(n))
        return []


def _node_name(n: Any) -> str:
    if isinstance(n, nodes.Name):
        return n.name
    if isinstance(n, nodes.Getattr):
        return f"{_node_name(n.node)}.{n.attr}"
    if isinstance(n, nodes.Getitem):
        return f"{_node_name(n.node)}[{getattr(n.arg, 'value', _node_name(n.arg))}]"
    if isinstance(n, nodes.Const):
        return repr(n.value)
    return type(n).__name__


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def literal_flags(lit: str) -> list[str]:
    why = []
    if URL.search(lit):
        why.append("URL")
    if ZW.search(lit):
        why.append("zero-width/bidi chars")
    if B64.search(lit) and not re.search(r"<\|?[\w_]+\|?>", lit):
        why.append("base64-like run")
    if PII_WORDS.search(lit):
        why.append("PII/tracking vocabulary")
    if COVERT_WORDS.search(lit):
        why.append("covert-instruction phrasing")
    return why


@check(
    "static.chat_template",
    "static",
    requires=["artifact:template"],
    description="Chat template injection, defaults and render diff",
    tags=["template"],
)
def chat_template(ctx: ScanContext) -> list[Finding]:
    a = ctx.artifact
    assert a is not None
    out: list[Finding] = []
    declared = bool(declared_modifications(a))
    tpls = a.chat_templates()

    # all copies must agree ------------------------------------------------------
    distinct = {}
    for k, v in tpls.items():
        distinct.setdefault(_norm(v), []).append(k)
    if len(distinct) > 1:
        groups = list(distinct.values())
        out.append(
            Finding(
                check_id="static.chat_template",
                title=f"{len(distinct)} different chat templates ship in this repo",
                status="warn",
                severity="medium",
                confidence=0.7,
                evidence=[
                    Evidence(kind="template_snippet", locator=", ".join(g), summary="one variant")
                    for g in groups
                ],
                remediation=(
                    "Which copy the runtime uses depends on the loader (vLLM prefers "
                    "chat_template.jinja, transformers the tokenizer_config copy). They must match."
                ),
                tags=["template"],
            )
        )
    primary_key = "chat_template.jinja" if "chat_template.jinja" in tpls else next(iter(tpls))
    src = tpls[primary_key]

    # absolute analysis ----------------------------------------------------------
    w = Walk()
    errs = w.run(src)
    if errs:
        out.append(
            Finding(
                check_id="static.chat_template",
                title="chat template does not parse",
                status="fail",
                severity="medium",
                confidence=0.8,
                evidence=[
                    Evidence(kind="template_snippet", locator=primary_key, summary=e) for e in errs
                ],
                tags=["template"],
            )
        )
        return out
    flagged = [(lit, literal_flags(lit)) for lit in w.literals]
    flagged = [(lit, f) for lit, f in flagged if f]
    dangerous_calls = [
        c
        for c in w.calls
        if c.split(".")[-1]
        in ("eval", "exec", "import", "__import__", "open", "system", "getattr", "attr")
    ]
    if dangerous_calls:
        out.append(
            Finding(
                check_id="static.chat_template",
                title="chat template calls unusual functions",
                status="fail",
                severity="high",
                confidence=0.8,
                evidence=[
                    Evidence(kind="template_snippet", locator=primary_key, summary=c)
                    for c in dangerous_calls[:20]
                ],
                tags=["template", "code-exec"],
            )
        )

    # vs reference -----------------------------------------------------------------
    near = ctx.nearest_reference()
    ref_art = ctx.artifact_for(near[1]) if near else None
    rsrc = None
    if ref_art is not None:
        rt = ref_art.chat_templates()
        rsrc = rt.get(primary_key) or (next(iter(rt.values())) if rt else None)
    if rsrc is None:
        if flagged:
            hard = any(
                set(f) & {"URL", "zero-width/bidi chars", "base64-like run"} for _, f in flagged
            )
            out.append(
                Finding(
                    check_id="static.chat_template",
                    title=f"{len(flagged)} template literal(s) with suspicious content (no reference to compare)",
                    status="warn",
                    severity="medium" if hard else "low",
                    confidence=0.5,
                    evidence=[
                        Evidence(
                            kind="template_snippet",
                            locator=primary_key,
                            summary=f"{', '.join(f)}: {lit[:200]!r}",
                        )
                        for lit, f in flagged[:30]
                    ],
                    tags=["template", "injection"],
                )
            )
        else:
            out.append(
                Finding.passed(
                    "static.chat_template",
                    "no suspicious literals in the template",
                    evidence=[
                        Evidence(
                            kind="metric",
                            locator=primary_key,
                            summary=f"{len(w.literals)} literals, defaults {w.defaults}",
                        )
                    ],
                )
            )
        out.append(
            ctx.no_reference(
                "static.chat_template", "rendered prompts not diffed against a reference template"
            )
        )
        return out

    hop = near[0]
    rw = Walk()
    rw.run(rsrc)
    ref_lits = {_norm(x) for x in rw.literals}
    new_lits = [x for x in w.literals if _norm(x) not in ref_lits]
    removed = [x for x in rw.literals if _norm(x) not in {_norm(y) for y in w.literals}]
    new_flagged = [(x, literal_flags(x)) for x in new_lits]
    bad_new = [(x, f) for x, f in new_flagged if f]
    comment_only = [x for x in new_lits if x.strip().startswith("{#")]
    if bad_new:
        out.append(
            Finding(
                check_id="static.chat_template",
                title=f"{len(bad_new)} new template literal(s) with URL/PII/covert phrasing vs {hop}",
                status="fail",
                severity="high",
                confidence=0.85,
                hop="suspect",
                declared=False,
                evidence=[
                    Evidence(
                        kind="template_snippet",
                        locator=primary_key,
                        summary=f"{', '.join(f)}: {x[:300]!r}",
                        reference_locator=f"{near[1].label()}:{primary_key}",
                    )
                    for x, f in bad_new[:30]
                ],
                remediation=(
                    "Text the template injects into every prompt is invisible to users; "
                    "treat as prompt injection unless documented."
                ),
                tags=["template", "injection"],
            )
        )
    plain_new = [x for x, f in new_flagged if not f and x not in comment_only]
    if plain_new:
        out.append(
            Finding(
                check_id="static.chat_template",
                title=f"{len(plain_new)} new template literal(s) vs {hop}",
                status="warn",
                severity="medium",
                confidence=0.7,
                declared=declared,
                hop="suspect",
                evidence=[
                    Evidence(kind="template_snippet", locator=primary_key, summary=repr(x[:300]))
                    for x in plain_new[:40]
                ],
                tags=["template"],
            )
        )
    if removed:
        out.append(
            Finding(
                check_id="static.chat_template",
                title=f"{len(removed)} template literal(s) removed vs {hop}",
                status="warn",
                severity="low",
                confidence=0.6,
                declared=declared,
                hop="suspect",
                evidence=[
                    Evidence(kind="template_snippet", locator=primary_key, summary=repr(x[:300]))
                    for x in removed[:40]
                ],
                tags=["template"],
            )
        )
    dchanges = [
        (k, rw.defaults.get(k), v) for k, v in w.defaults.items() if rw.defaults.get(k) != v
    ] + [(k, v, None) for k, v in rw.defaults.items() if k not in w.defaults]
    if dchanges:
        out.append(
            Finding(
                check_id="static.chat_template",
                title=f"{len(dchanges)} default() value(s) changed vs {hop}",
                status="warn",
                severity="low",
                confidence=0.8,
                declared=declared,
                hop="suspect",
                evidence=[
                    Evidence(
                        kind="template_snippet",
                        locator=f"{primary_key}#default({k})",
                        summary=f"{old} -> {new}",
                    )
                    for k, old, new in dchanges
                ],
                tags=["template", "behaviour"],
            )
        )

    # render diff -------------------------------------------------------------------
    r_s, r_r = render_all(src), render_all(rsrc)
    diffs: list[Evidence] = []
    for name in r_s:
        if r_s[name] != r_r.get(name):
            ws_only = _norm(r_s[name]) == _norm(r_r.get(name, ""))
            d = "\n".join(
                difflib.unified_diff(
                    r_r.get(name, "").splitlines(),
                    r_s[name].splitlines(),
                    "reference",
                    "suspect",
                    lineterm="",
                    n=1,
                )
            )
            diffs.append(
                Evidence(
                    kind="diff",
                    locator=f"render:{name}",
                    summary="whitespace only" if ws_only else d[:1500],
                    data={"whitespace_only": ws_only},
                )
            )
    substantive = [d for d in diffs if not d.data.get("whitespace_only")]
    if substantive:
        out.append(
            Finding(
                check_id="static.chat_template",
                title=f"rendered prompt differs from {hop} in {len(substantive)}/{len(r_s)} canonical conversations",
                status="warn",
                severity="low" if (plain_new or bad_new or dchanges) else "medium",
                confidence=0.8,
                declared=declared,
                hop="suspect",
                evidence=substantive[:12],
                tags=["template"],
            )
        )
    elif diffs:
        out.append(
            Finding.passed(
                "static.chat_template",
                f"rendered prompts differ from {hop} only in whitespace",
                evidence=diffs[:12],
            )
        )
    else:
        out.append(
            Finding.passed(
                "static.chat_template",
                f"rendered prompts identical to {hop} across {len(r_s)} canonical conversations",
            )
        )
    if comment_only:
        out.append(
            Finding.passed(
                "static.chat_template",
                "template comments added",
                evidence=[
                    Evidence(kind="template_snippet", locator=primary_key, summary=x[:300])
                    for x in comment_only
                ],
            )
        )
    return out
