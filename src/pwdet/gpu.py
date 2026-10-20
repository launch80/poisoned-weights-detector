"""Which GPU the docker-based layers use.

`auto` (the default) picks the first discrete Intel GPU on the PCI bus, so no machine's
address is baked in. `PWDET_GPU` or `--gpu <bdf>` choose one explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

PCI = Path("/sys/bus/pci/devices")
INTEL = "0x8086"


def discrete_intel_gpus(root: Path = PCI) -> list[str]:
    """PCI addresses of display-class Intel devices, integrated ones (bus 00) excluded."""
    out = []
    for dev in sorted(root.glob("*")):
        try:
            vendor = (dev / "vendor").read_text().strip()
            klass = (dev / "class").read_text().strip()
        except OSError:
            continue
        if vendor != INTEL or not klass.startswith("0x03"):
            continue
        bus = dev.name.split(":")[1] if dev.name.count(":") == 2 else ""
        if bus == "00":
            continue
        out.append(dev.name)
    return out


def resolve_gpu(spec: str | None = "auto", root: Path = PCI) -> str:
    """A PCI address for `spec`: an explicit address is kept; `auto` picks one."""
    spec = (spec or "auto").strip()
    if spec == "auto":
        spec = os.environ.get("PWDET_GPU", "auto").strip() or "auto"
    if spec != "auto":
        return spec
    found = discrete_intel_gpus(root)
    if not found:
        raise RuntimeError(
            "no discrete Intel GPU found for the docker-based layers; pass --gpu <pci address>"
        )
    return found[0]
