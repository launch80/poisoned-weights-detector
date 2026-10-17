from __future__ import annotations

import json
from pathlib import Path

from ..schema import Report


def write_json(report: Report, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return p


def export_schema(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(Report.model_json_schema(), indent=2), encoding="utf-8")
    return p
