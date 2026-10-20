"""ScanContext: everything a check may need, created lazily and shared across checks."""

from __future__ import annotations

import importlib.util
import shutil
import socket
from functools import cached_property
from typing import Any

from .artifacts.index import ArtifactIndex
from .config import Settings, resolve_model
from .schema import Finding, Hop, LineageHop, ModelRef

FIRST_PARTY_NA = (
    "not applicable: a first-party release has nothing upstream to compare with; "
    "its files are checked byte for byte against the Hub instead"
)


class ScanContext:
    def __init__(
        self,
        subject: ModelRef,
        settings: Settings,
        references: dict[str, ModelRef] | None = None,
        options: dict[str, Any] | None = None,
    ):
        self.subject = subject
        self.settings = settings
        self.references: dict[str, ModelRef] = references or {}
        self.options: dict[str, Any] = options or {}
        self.lineage: list[LineageHop] = []
        self.notes: list[str] = []
        self.shared: dict[str, Any] = {}  # cross-check scratch (e.g. trigger candidates)
        self._artifacts: dict[str, ArtifactIndex] = {}

    # artifacts --------------------------------------------------------------
    def artifact_for(self, ref: ModelRef) -> ArtifactIndex | None:
        if not ref.path:
            return None
        if ref.key not in self._artifacts:
            self._artifacts[ref.key] = ArtifactIndex(ref.path)
        return self._artifacts[ref.key]

    @cached_property
    def artifact(self) -> ArtifactIndex | None:
        return self.artifact_for(self.subject)

    def reference(self, hop: Hop) -> ModelRef | None:
        return self.references.get(hop)

    def reference_artifact(self, hop: Hop) -> ArtifactIndex | None:
        ref = self.reference(hop)
        return self.artifact_for(ref) if ref else None

    def nearest_reference(self, local: bool = True) -> tuple[Hop, ModelRef] | None:
        """Closest lineage hop we can compare against; local=True requires files on disk."""
        for hop in ("parent", "base"):
            ref = self.references.get(hop)
            if ref is not None and (ref.path or not local):
                return hop, ref  # type: ignore[return-value]
        return None

    def no_reference(self, check_id: str, what: str) -> Finding:
        """The skip a comparison reports when it has nothing on disk to compare with."""
        if self.shared.get("first_party_subject"):
            return Finding.skipped(check_id, FIRST_PARTY_NA)
        return Finding.skipped(check_id, f"missing capability: reference:any:local ({what})")

    def expects(self, profile: str) -> bool:
        return profile in self.subject.expect or profile in self.options.get("expect", [])

    # capabilities -----------------------------------------------------------
    def capabilities(self) -> set[str]:
        caps: set[str] = set()
        a = self.artifact
        if a is not None:
            caps.add("artifact:dir")
            if a.safetensors_files:
                caps.add("artifact:safetensors")
                caps.add("artifact:weights")
            if a.gguf_files:
                caps.add("artifact:gguf")
                if a.gguf:  # readable through gguf-py: tensors are comparable like safetensors
                    caps.add("artifact:weights")
            if a.pickle_like_files:
                caps.add("artifact:pickle")
            if a.config:
                caps.add("artifact:config")
            if a.chat_templates():
                caps.add("artifact:template")
        for hop in ("parent", "base"):
            if hop in self.references:
                caps.add(f"reference:{hop}")
                caps.add("reference:any")
                if self.references[hop].path:
                    caps.add(f"reference:{hop}:local")
        if self.subject.endpoint:
            caps.add("endpoint:suspect")
        if any(r.endpoint for r in self.references.values()):
            caps.add("endpoint:reference")
        if importlib.util.find_spec("torch") is not None:
            caps.add("torch")
        if shutil.which("docker") is not None and not self.options.get("no_docker"):
            caps.add("docker")
        for hop in ("parent", "base"):
            if hop in self.references and self.references[hop].path:
                caps.add("reference:any:local")
        if not self.settings.offline and _online():
            caps.add("network")
        if self.subject.kind in ("hf-cache", "hf-remote") and self.subject.repo:
            caps.add("hub:repo")
        return caps

    @classmethod
    def build(
        cls,
        subject_spec: str,
        settings: Settings,
        parent: str | None = None,
        base: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ScanContext:
        subject = resolve_model(subject_spec, settings)
        refs: dict[str, ModelRef] = {}
        lin = settings.lineage.get(subject.key)
        parent = parent or (lin.parent if lin else None)
        base = base or (lin.base if lin else None)
        if parent:
            refs["parent"] = resolve_model(parent, settings, key=parent)
        if base:
            refs["base"] = resolve_model(base, settings, key=base)
        ctx = cls(subject, settings, refs, options)
        ctx.options["user_refs"] = bool(parent or base)
        ctx.detect_lineage(user_parent=parent, user_base=base)
        return ctx

    def detect_lineage(self, user_parent: str | None = None, user_base: str | None = None) -> None:
        """Fill self.lineage from the artifact's own claims; attach cached parents as references."""
        a = self.artifact
        if a is None:
            return
        from .artifacts.lineage import build_hops

        hops, auto = build_hops(a, self.settings)
        self.lineage = hops
        for hop, ref in auto.items():
            if hop not in self.references:
                self.references[hop] = ref
                self.notes.append(
                    f"{hop} reference auto-attached from lineage claim: {ref.label()}"
                )
        for hop, spec in (("parent", user_parent), ("base", user_base)):
            if spec and hop in self.references and not any(h.hop == hop for h in hops):
                self.lineage.append(
                    LineageHop(
                        hop=hop,
                        ref=self.references[hop],
                        relation="unknown",
                        source="user",
                        verified=False,
                    )  # type: ignore[arg-type]
                )


def _online(host: str = "huggingface.co", timeout: float = 1.5) -> bool:
    try:
        socket.create_connection((host, 443), timeout=timeout).close()
        return True
    except OSError:
        return False
