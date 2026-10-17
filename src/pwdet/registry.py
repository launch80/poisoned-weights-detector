"""Check registry: a check is a function (ScanContext) -> list[Finding] with declared needs.

`requires` capability strings (see ScanContext.capabilities): artifact:dir, artifact:safetensors,
artifact:gguf, artifact:pickle, artifact:config, artifact:template, reference:parent,
reference:base, reference:any, endpoint:suspect, endpoint:reference, torch, network.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Literal

from .schema import Finding, Layer

if TYPE_CHECKING:
    from .context import ScanContext

Cost = Literal["cheap", "medium", "expensive"]
CheckFn = Callable[["ScanContext"], list[Finding]]


@dataclass(frozen=True)
class CheckSpec:
    id: str
    layer: Layer
    fn: CheckFn
    requires: frozenset[str] = field(default_factory=frozenset)
    cost: Cost = "cheap"
    description: str = ""
    tags: tuple[str, ...] = ()


_REGISTRY: dict[str, CheckSpec] = {}
_DISCOVERED = False


def check(
    id: str,
    layer: Layer,
    requires: Iterable[str] = (),
    cost: Cost = "cheap",
    description: str = "",
    tags: Iterable[str] = (),
) -> Callable[[CheckFn], CheckFn]:
    def deco(fn: CheckFn) -> CheckFn:
        if id in _REGISTRY:
            raise ValueError(f"duplicate check id {id}")
        _REGISTRY[id] = CheckSpec(
            id=id,
            layer=layer,
            fn=fn,
            requires=frozenset(requires),
            cost=cost,
            description=description or (fn.__doc__ or "").strip().splitlines()[0:1][0]
            if (description or fn.__doc__)
            else "",
            tags=tuple(tags),
        )
        return fn

    return deco


def discover() -> dict[str, CheckSpec]:
    """Import every pwdet.checks.* module plus third-party entry points, once."""
    global _DISCOVERED
    if _DISCOVERED:
        return _REGISTRY
    from . import checks as checks_pkg

    for mod in pkgutil.walk_packages(checks_pkg.__path__, checks_pkg.__name__ + "."):
        importlib.import_module(mod.name)
    for ep in entry_points(group="pwdet.checks"):
        try:
            ep.load()
        except Exception:  # third-party plugin failures must not break the scan
            pass
    _DISCOVERED = True
    return _REGISTRY


def all_specs() -> list[CheckSpec]:
    return sorted(discover().values(), key=lambda s: (s.layer, s.id))


def select(
    layers: Iterable[str] | None = None,
    ids: Iterable[str] | None = None,
    skip: Iterable[str] = (),
    max_cost: Cost | None = None,
) -> list[CheckSpec]:
    order = {"cheap": 0, "medium": 1, "expensive": 2}
    specs = all_specs()
    if ids:
        wanted = set(ids)
        specs = [s for s in specs if s.id in wanted]
    if layers:
        ls = set(layers)
        specs = [s for s in specs if s.layer in ls]
    if max_cost:
        specs = [s for s in specs if order[s.cost] <= order[max_cost]]
    sk = set(skip)
    return [s for s in specs if s.id not in sk]
