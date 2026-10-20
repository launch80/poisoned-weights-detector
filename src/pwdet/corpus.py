"""The evaluation corpus: labelled models, verdict-level accuracy, calibration and a regression gate.

eval/corpus.yaml lists every item: a model family's base, then attacks and benign changes built
from it (fixture/builders.py), trained fixtures, and real community re-uploads pinned to a Hub
commit. `pwdet corpus build` makes what is missing, `pwdet corpus eval` scans each item at each
assurance level exactly as a user's scan would run, and `pwdet corpus calibrate` turns the results
into calibration.json: measured noise floors, per-finding-kind confidence and the accuracy each
report quotes. `pwdet corpus gate` fails when a change makes detection or false alarms worse.

Labels: `tampered` (built attacks, known ground truth), `benign` (built benign changes) and
`presumed_benign` (community re-uploads: nobody can prove a third-party upload clean, but a
popular re-upload of a first-party model is overwhelmingly likely to be; they count towards false
alarms only).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from . import __version__, calibration, registry
from .calibration import FLOOR_MARGIN, MIN_EXAMPLES, wilson95
from .config import Settings
from .schema import Finding, LineageHop, ModelRef

FLAGGED = {"suspicious", "likely_tampered", "malicious"}
CONFIDENT = {"likely_tampered", "malicious"}
BENIGN_LABELS = {"benign", "presumed_benign"}
LEVELS = ("basic", "standard", "thorough")
# what each level tests, so a miss outside it is reported as out of scope, not hidden
OUT_OF_SCOPE = {"basic": {"trained_behaviour"}, "standard": {"trained_behaviour"}}
HUB_PATTERNS = ["*.json", "*.txt", "*.jinja", "*.safetensors", "README.md", "*.model"]


@dataclass
class Item:
    id: str
    family: str
    label: str = ""
    threat: str | None = None
    kind: str = ""
    fixture: str | None = None
    build: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    source: str | None = None  # another item's id: the builder's input
    hub: str | None = None  # org/repo@sha
    file: str | None = None  # one file of the hub repo (a GGUF)
    extra: list[str] = field(default_factory=list)  # more hub files to fetch with the repo
    probe_run: str | None = None
    floor: bool = False  # a benign format conversion: measures that format's noise floor
    expected: dict[str, str] = field(default_factory=dict)
    note: str = ""


@dataclass
class Corpus:
    name: str
    root: Path
    fixtures: Path
    probe_runs: Path
    families: dict[str, dict[str, Any]]
    items: list[Item]
    path: Path

    def item(self, iid: str) -> Item:
        return next(i for i in self.items if i.id == iid)

    def digest(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]


def load(path: Path) -> Corpus:
    raw = yaml.safe_load(Path(path).read_text())
    items = []
    for d in raw["items"]:
        d = dict(d)
        d.setdefault("family", d["id"].split("/")[0])
        items.append(Item(**d))
    here = Path(path).resolve().parent.parent

    def p(s: str) -> Path:
        q = Path(os.path.expanduser(s))
        return q if q.is_absolute() else here / q

    return Corpus(
        name=raw.get("name", "pwdet corpus"),
        root=p(raw.get("root", "~/.cache/pwdet/corpus")),
        fixtures=p(raw.get("fixtures", "~/.cache/pwdet/fixtures")),
        probe_runs=p(raw.get("probe_runs", "runs/fixtures")),
        families=raw["families"],
        items=items,
        path=Path(path),
    )


# ------------------------------------------------------------------ paths --------------------
def _split(spec: str) -> tuple[str, str | None]:
    repo, _, rev = spec.partition("@")
    return repo, rev or None


def hub_snapshot(spec: str, settings: Settings, patterns: list[str] | None = None) -> Path:
    from huggingface_hub import snapshot_download

    from .fetch import _download_dir

    repo, rev = _split(spec)
    kw = dict(revision=rev, allow_patterns=patterns or HUB_PATTERNS, token=settings.token())
    if rev and len(rev) == 40:  # a pinned snapshot already on disk needs no Hub call
        import fnmatch

        for cache in settings.hf_cache_dirs():
            snap = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
            names = [q.name for q in snap.iterdir()] if snap.is_dir() else []
            explicit = [x for x in (patterns or []) if "*" not in x and x != "README.md"]
            if (
                names
                and all(x in names for x in explicit)
                and any(fnmatch.fnmatch(n, x) for n in names for x in (patterns or HUB_PATTERNS))
            ):
                return snap
    return Path(snapshot_download(repo, cache_dir=_download_dir(settings), **kw))


def item_path(c: Corpus, it: Item, settings: Settings) -> Path:
    if it.fixture:
        return c.fixtures / it.fixture
    if it.hub and not it.file:
        return hub_snapshot(it.hub, settings, HUB_PATTERNS + it.extra)
    return c.root / it.id


def base_ref(c: Corpus, family: str, settings: Settings) -> ModelRef:
    spec = c.families[family]["base"]
    local = Path(os.path.expanduser(spec))
    if local.is_dir():
        return ModelRef(key=str(local), kind="local", path=str(local))
    repo, rev = _split(spec)
    try:
        return ModelRef(
            key=repo, kind="local", repo=repo, revision=rev, path=str(hub_snapshot(spec, settings))
        )
    except Exception:
        from .fetch import mirror_reference

        return mirror_reference(
            ModelRef(key=repo, kind="hf-remote", repo=repo, revision=rev), settings
        )


# ------------------------------------------------------------------ build --------------------
def build(c: Corpus, settings: Settings, only: list[str] | None = None, log=print) -> list[str]:
    """Make every built or single-file item that is not on disk yet; returns the ids made."""
    from .fixture.builders import BUILDERS
    from .fixture.common import read_truth

    made = []
    for it in c.items:
        if only and it.id not in only:
            continue
        dst = c.root / it.id
        if it.build:
            if (dst / "FIXTURE.json").exists():
                continue
            src = (
                item_path(c, c.item(it.source), settings)
                if it.source
                else Path(base_ref(c, it.family, settings).path or "")
            )
            if it.source and not read_truth(src) and not (src / "config.json").exists():
                raise RuntimeError(f"{it.id}: build {it.source} first")
            log(f"build {it.id}: {it.build} {it.args or ''}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".partial")
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
            try:
                BUILDERS[it.build](src, tmp, **it.args)
            except Exception:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
            shutil.rmtree(dst, ignore_errors=True)
            tmp.rename(dst)
            made.append(it.id)
        elif it.hub and it.file:
            if (dst / it.file).exists():
                continue
            log(f"fetch {it.id}: {it.hub} {it.file}")
            snap = hub_snapshot(it.hub, settings, [it.file, *it.extra])
            dst.mkdir(parents=True, exist_ok=True)
            for name in [it.file, *it.extra]:
                s = (snap / name).resolve()
                if not s.exists():
                    continue
                try:
                    os.link(s, dst / name)
                except OSError:
                    import shutil

                    shutil.copyfile(s, dst / name)
            made.append(it.id)
    return made


def truth_of(c: Corpus, it: Item, path: Path) -> dict[str, Any]:
    from .fixture.common import read_truth

    t = read_truth(path) or {}
    label = it.label or t.get("label") or ("tampered" if t.get("expected") else "benign")
    if not it.label and not t.get("label") and str(t.get("kind", "")).startswith("benign"):
        label = "benign"
    return {
        "label": label,
        "threat": it.threat if it.threat is not None else t.get("threat"),
        "kind": it.kind or t.get("kind") or ("community" if it.hub else ""),
        "expected": it.expected or t.get("expected") or {},
    }


# ------------------------------------------------------------------ scan ---------------------
def _floors_of(findings: list[Finding]) -> list[dict[str, Any]]:
    out = []
    for f in findings:
        if f.check_id not in ("static.embedding_compare", "static.unembedding_compare"):
            continue
        for e in f.evidence:
            d = e.data or {}
            if e.locator == "row_rel_l2" and d.get("floor_key") and d.get("rel_l2"):
                out.append(
                    {
                        "check": f.check_id,
                        "key": d["floor_key"],
                        "floor": d.get("floor"),
                        "median": d["rel_l2"].get("median"),
                        "p99": d["rel_l2"].get("p99"),
                        "max": d["rel_l2"].get("max"),
                        "rows_deviant": d.get("rows_deviant"),
                    }
                )
    return out


def scan_item(
    c: Corpus,
    it: Item,
    level: str,
    settings: Settings,
    checks: list[str] | None = None,
    serve: bool = False,
) -> dict[str, Any]:
    """One scan of one item at one level, the way `pwdet scan` runs it."""
    from .blackbox.differential import load_thresholds
    from .context import ScanContext
    from .fixture.evaluate import behavioural_findings
    from .runner import build_report, run_checks
    from .schema import Report

    t0 = time.perf_counter()
    path = item_path(c, it, settings)
    truth = truth_of(c, it, path)
    rec: dict[str, Any] = {"id": it.id, "family": it.family, "level_asked": level, **truth}
    try:
        subj = it.hub if (it.hub and not it.file) else str(path)
        ctx = ScanContext.build(subj, settings)
        ctx.references.clear()
        ctx.lineage = [h for h in ctx.lineage if h.hop not in ("parent", "base")]
        if level != "basic":
            ref = base_ref(c, it.family, settings)
            ctx.references["base"] = ref
            ctx.options["user_refs"] = True
            ctx.lineage.append(LineageHop(hop="base", ref=ref, source="corpus manifest"))
        specs = registry.select(layers=["static", "provenance"], ids=checks, max_cost=None)
        findings = run_checks(ctx, specs)
        caps = ctx.capabilities()
        if level == "thorough" and serve:
            # live: serve the item and its base on this machine's GPU and probe both, as
            # `pwdet scan --profile thorough` does
            ctx.options.update(serve=True, gpu=os.environ.get("PWDET_GPU", "auto"))
            bspecs = registry.select(ids=["blackbox.probes"])
            findings += run_checks(ctx, bspecs)
            specs = specs + bspecs
            caps = ctx.capabilities()
        elif level == "thorough":
            fam = c.families[it.family]
            run, base_run = (
                c.probe_runs / (it.probe_run or "-"),
                c.probe_runs / fam.get("probe_base", "BASE"),
            )
            if not (run / "results.json").exists() or not (base_run / "results.json").exists():
                return {**rec, "skipped": "no probe run"}
            th = load_thresholds(settings.thresholds)
            bf = behavioural_findings(run, base_run, th) or [
                Finding.passed("blackbox.probes", "no behavioural difference")
            ]
            for f in bf:
                f.check_id = f.check_id or "blackbox.probes"
                calibration.apply(f)
            findings += bf
            specs = specs + registry.select(ids=["blackbox.probes"])
            caps = caps | {"endpoint:suspect", "endpoint:reference"}
            ctx.capabilities = lambda: caps  # type: ignore[method-assign]
        rep = build_report(ctx, specs, findings, Report.now(), f"corpus:{level}")
        rec.update(
            level=rep.assurance.level if rep.assurance else None,
            verdict=rep.verdict.label if rep.verdict else None,
            rationale=rep.verdict.rationale if rep.verdict else [],
            layer_of={s.id: s.layer for s in specs},
            caps=sorted(caps),
            findings=[f.model_dump(mode="json") for f in findings],
            floors=_floors_of(findings),
        )
    except Exception as e:  # one broken item must not stop the run
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    return rec


def _worker(args: tuple) -> dict[str, Any]:
    cpath, iid, level, offline, checks, serve = args
    s = Settings(offline=offline)
    from .config import load_settings

    try:
        s = load_settings(None)
        s.offline = offline
        if s.thresholds and not Path(s.thresholds).expanduser().exists():
            s.thresholds = None  # the packaged thresholds, not a path relative to somewhere else
    except Exception:
        pass
    c = load(Path(cpath))
    return scan_item(c, c.item(iid), level, s, checks, serve)


def evaluate(
    c: Corpus,
    levels: list[str],
    only: list[str] | None = None,
    jobs: int = 4,
    offline: bool = False,
    checks: list[str] | None = None,
    log=print,
    serve: bool = False,
) -> list[dict[str, Any]]:
    """`serve`: thorough scans probe live, one at a time on this machine's GPU; without it they
    use recorded probe runs where the manifest names one."""
    todo, live = [], []
    for it in c.items:
        if only and it.id not in only and it.family not in only:
            continue
        for lv in levels:
            a = (str(c.path.resolve()), it.id, lv, offline, checks, serve and lv == "thorough")
            if lv == "thorough" and serve:
                live.append(a)
            elif lv != "thorough" or it.probe_run:
                todo.append(a)
    out: list[dict[str, Any]] = []
    total = len(todo) + len(live)

    def done(r: dict[str, Any]) -> None:
        out.append(r)
        what = r.get("error") or r.get("skipped") or f"{r.get('verdict')} · {r.get('level')}"
        log(
            f"[{len(out)}/{total}] {r['id']} ({r['level_asked']}, {r['label']}): {what} ({r['seconds']}s)"
        )

    if todo:
        with ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
            for fut in as_completed([ex.submit(_worker, a) for a in todo]):
                done(fut.result())
    for a in live:  # one GPU: in turn
        done(_worker(a))
    out.sort(key=lambda r: (r["id"], LEVELS.index(r["level_asked"])))
    return out


# ------------------------------------------------------------------ metrics ------------------
def _findings(r: dict[str, Any]) -> list[Finding]:
    return [Finding.model_validate(f) for f in r.get("findings", [])]


def redecide(r: dict[str, Any]) -> tuple[str, str]:
    """Verdict and level for a recorded scan under the calibration now loaded."""
    from .assurance import assess
    from .verdict import decide

    fs = _findings(r)
    for f in fs:
        calibration.apply(f)
    v = decide(fs, r.get("layer_of", {}))
    a = assess(fs, set(r.get("caps", [])), verified_reference=True)
    return v.label, a.level


def summarize(results: list[dict[str, Any]], by: str = "level") -> dict[str, Any]:
    """Per assurance level reached: tampered detected and benign flagged, with 95% intervals."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        if r.get("verdict") and not r.get("error"):
            groups[r[by]].append(r)
    out: dict[str, Any] = {}
    for lv in LEVELS:
        rs = groups.get(lv, [])
        if not rs:
            continue
        t = [r for r in rs if r["label"] == "tampered"]
        b = [r for r in rs if r["label"] in BENIGN_LABELS]
        det = [r for r in t if r["verdict"] in FLAGGED]
        conf = [r for r in t if r["verdict"] in CONFIDENT]
        flg = [r for r in b if r["verdict"] in FLAGGED]
        threats: dict[str, Counter] = defaultdict(Counter)
        for r in t:
            threats[r.get("threat") or "?"]["n"] += 1
            threats[r.get("threat") or "?"]["detected"] += r["verdict"] in FLAGGED
        scope_out = OUT_OF_SCOPE.get(lv, set())
        ts = [r for r in t if r.get("threat") not in scope_out]
        ds = [r for r in ts if r["verdict"] in FLAGGED]
        entry = {
            "tampered": {
                "n": len(t),
                "detected": len(det),
                "confident": len(conf),
                "ci95": list(wilson95(len(det), len(t))),
                "missed": [r["id"] for r in t if r["verdict"] not in FLAGGED],
            },
            "benign": {
                "n": len(b),
                "flagged": len(flg),
                "clean": sum(r["verdict"] == "clean" for r in b),
                "inconclusive": sum(r["verdict"] == "inconclusive" for r in b),
                "ci95": list(wilson95(len(flg), len(b))),
                "false_alarms": [r["id"] for r in flg],
            },
            "by_threat": {k: dict(v) for k, v in sorted(threats.items())},
        }
        if scope_out and len(ts) != len(t):
            lo, hi = wilson95(len(ds), len(ts))
            entry["in_scope"] = (
                f"Of the tampering this level tests (all but trained-in behaviour, which needs "
                f"behavioural probes), it flagged {len(ds)}/{len(ts)} (95% CI {lo:.0%}-{hi:.0%})."
            )
        out[lv] = entry
    return out


def confidence_table(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Per (check, severity): how often a finding of that kind was right. One scan per item (its
    highest level), so the same finding is not counted twice."""
    best: dict[str, dict[str, Any]] = {}
    for r in results:
        if r.get("error") or not r.get("verdict"):
            continue
        cur = best.get(r["id"])
        if cur is None or LEVELS.index(r["level_asked"]) > LEVELS.index(cur["level_asked"]):
            best[r["id"]] = r
    tp: Counter = Counter()
    fp: Counter = Counter()
    for r in best.values():
        expected = r.get("expected") or {}
        seen = set()
        for f in _findings(r):
            if f.status not in ("warn", "fail") or f.effective_severity == "info":
                continue
            k = calibration.kind_key(f)
            if k in seen:
                continue
            seen.add(k)
            if r["label"] in BENIGN_LABELS:
                fp[k] += 1
            elif r["label"] == "tampered" and f.check_id in expected:
                tp[k] += 1
    table = {}
    for k in sorted(set(tp) | set(fp)):
        n = tp[k] + fp[k]
        table[k] = {"tp": tp[k], "fp": fp[k], "n": n, "lb95": round(wilson95(tp[k], n)[0], 4)}
    return table


def noise_floors(results: list[dict[str, Any]], c: Corpus) -> dict[str, Any]:
    floor_items = {i.id for i in c.items if i.floor}
    obs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        if r["id"] in floor_items and r["label"] in BENIGN_LABELS and r.get("floors"):
            for o in r["floors"]:
                if o.get("p99") is not None:
                    obs[o["key"]].append({**o, "item": r["id"]})
    out = {}
    for key, os_ in sorted(obs.items()):
        if key == "dense" or key.endswith((":F32", ":F16", ":BF16")):
            continue  # an exact format copies rows: nothing to measure but zero
        worst = max(o["p99"] for o in os_)
        out[key] = {
            "floor": float(f"{worst * FLOOR_MARGIN:.3g}"),
            "n": len({o["item"] for o in os_}),
            "observed_p99_max": worst,
            "observed_max": max(o["max"] or 0 for o in os_),
            "observed_median_max": max(o["median"] or 0 for o in os_),
            "items": sorted({o["item"] for o in os_}),
        }
    return out


def calibrate(c: Corpus, results: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """calibration.json content from a corpus run, and warnings (stale floors, thin kinds)."""
    floors = noise_floors(results, c)
    warnings = []
    used = {(o["key"], o["floor"]) for r in results for o in r.get("floors", [])}
    for key, m in floors.items():
        old = {f for k, f in used if k == key}
        if old and any(abs(m["floor"] - (f or 0)) > 0.01 * m["floor"] for f in old):
            warnings.append(
                f"noise floor {key} moved to {m['floor']} (the run used {sorted(old)}): "
                "run `pwdet corpus calibrate` again so the findings use it"
            )
    conf = confidence_table(results)
    cal = {
        "corpus": {
            "name": c.name,
            "digest": c.digest(),
            "items": len(c.items),
            "tampered": sum(
                1 for r in results if r["label"] == "tampered" and r["level_asked"] == "standard"
            ),
            "benign": sum(
                1 for r in results if r["label"] in BENIGN_LABELS and r["level_asked"] == "standard"
            ),
            "families": sorted(c.families),
            "generated": datetime.now(UTC).isoformat(timespec="seconds"),
            "tool_version": __version__,
        },
        "min_examples": MIN_EXAMPLES,
        "floor_margin": FLOOR_MARGIN,
        "noise_floors": floors,
        "confidence": conf,
    }
    # accuracy is re-decided under this calibration (confidence changes verdicts, not findings)
    tmp = c.root / ".calibration-preview.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(cal))
    prev = os.environ.get("PWDET_CALIBRATION")
    os.environ["PWDET_CALIBRATION"] = str(tmp)
    calibration.reset()
    try:
        re = []
        for r in results:
            if r.get("verdict") and not r.get("error"):
                v, lv = redecide(r)
                re.append({**r, "verdict": v, "level": lv})
    finally:
        if prev is None:
            os.environ.pop("PWDET_CALIBRATION", None)
        else:
            os.environ["PWDET_CALIBRATION"] = prev
        calibration.reset()
        tmp.unlink(missing_ok=True)
    cal["accuracy"] = summarize(re)
    if warnings:
        cal["provisional"] = warnings
    return cal, warnings


# ------------------------------------------------------------------ gate ---------------------
def gate(results: list[dict[str, Any]], baseline: dict[str, Any]) -> list[str]:
    """Regressions against a baseline run: a detected attack now missed, a clean benign model now
    flagged, or a level's counts getting worse."""
    old = {(r["id"], r["level_asked"]): r for r in baseline.get("results", [])}
    bad = []
    for r in results:
        o = old.get((r["id"], r["level_asked"]))
        if not o or not r.get("verdict") or not o.get("verdict"):
            continue
        if r["label"] == "tampered" and o["verdict"] in FLAGGED and r["verdict"] not in FLAGGED:
            bad.append(f"{r['id']} ({r['level_asked']}): was {o['verdict']}, now {r['verdict']}")
        if r["label"] in BENIGN_LABELS and o["verdict"] not in FLAGGED and r["verdict"] in FLAGGED:
            bad.append(f"{r['id']} ({r['level_asked']}): was {o['verdict']}, now {r['verdict']}")
    for r in results:
        if r.get("error") and (r["id"], r["level_asked"]) in old:
            bad.append(f"{r['id']} ({r['level_asked']}): {r['error']}")
    return bad


def baseline_of(results: list[dict[str, Any]]) -> dict[str, Any]:
    keep = ("id", "level_asked", "label", "threat", "verdict", "level")
    return {
        "tool_version": __version__,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "results": [{k: r.get(k) for k in keep} for r in results if r.get("verdict")],
    }


# ------------------------------------------------------------------ report -------------------
def render_md(c: Corpus, results: list[dict[str, Any]], cal: dict[str, Any] | None) -> str:
    acc = (cal or {}).get("accuracy") or summarize(results)
    n_t = len({r["id"] for r in results if r["label"] == "tampered"})
    n_b = len({r["id"] for r in results if r["label"] in BENIGN_LABELS})
    L = [
        f"# {c.name}: pwdet {__version__}",
        "",
        f"{len(c.items)} items over {len(c.families)} families ({', '.join(sorted(c.families))}): "
        f"{n_t} tampered, {n_b} benign or presumed benign. Corpus digest `{c.digest()}`.",
        "",
        "A model counts as **flagged** when its verdict is suspicious, likely_tampered or "
        "malicious. Intervals are 95% Wilson intervals.",
        "",
        "## Verdicts by assurance level",
        "",
        "| level | tampered flagged | of which confident | benign flagged | benign clean | benign inconclusive |",
        "|---|---|---|---|---|---|",
    ]
    for lv, a in acc.items():
        t, b = a["tampered"], a["benign"]
        L.append(
            f"| {lv} | {t['detected']}/{t['n']} ({t['ci95'][0]:.0%}-{t['ci95'][1]:.0%}) | "
            f"{t['confident']}/{t['n']} | {b['flagged']}/{b['n']} ({b['ci95'][0]:.0%}-{b['ci95'][1]:.0%}) | "
            f"{b['clean']}/{b['n']} | {b['inconclusive']}/{b['n']} |"
        )
    L += [
        "",
        "## Tampering flagged, by threat",
        "",
        "| level | threat | flagged |",
        "|---|---|---|",
    ]
    for lv, a in acc.items():
        for th, v in a["by_threat"].items():
            L.append(f"| {lv} | {th} | {v.get('detected', 0)}/{v['n']} |")
    for lv, a in acc.items():
        if a.get("in_scope"):
            L.append(f"\n{lv.capitalize()}: {a['in_scope']}")
    L += ["", "## Misses and false alarms", ""]
    for lv, a in acc.items():
        if a["tampered"]["missed"]:
            L.append(f"- {lv}, missed: {', '.join(a['tampered']['missed'])}")
        if a["benign"]["false_alarms"]:
            L.append(f"- {lv}, false alarms: {', '.join(a['benign']['false_alarms'])}")
    if cal:
        L += [
            "",
            "## Measured noise floors",
            "",
            "| format | floor | benign conversions | worst p99 row error |",
            "|---|---|---|---|",
        ]
        for k, m in cal["noise_floors"].items():
            L.append(f"| {k} | {m['floor']} | {m['n']} | {m['observed_p99_max']:.3g} |")
        L += [
            "",
            "## Finding confidence",
            "",
            f"Measured where the corpus has at least {cal['min_examples']} findings of a kind; "
            "the rest keep the check's own estimate.",
            "",
            "| check · severity | right | wrong | confidence (95% lower bound) | used |",
            "|---|---|---|---|---|",
        ]
        for k, m in cal["confidence"].items():
            used = "measured" if m["n"] >= cal["min_examples"] else "check's estimate"
            L.append(
                f"| {k.replace('|', ' · ')} | {m['tp']} | {m['fp']} | {m['lb95']:.2f} | {used} |"
            )
        for w in cal.get("provisional", []):
            L.append(f"\n**Provisional:** {w}")
    L += [
        "",
        "## Every scan",
        "",
        "| item | label | asked | reached | verdict | seconds |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        v = r.get("verdict") or r.get("error") or r.get("skipped")
        L.append(
            f"| {r['id']} | {r['label']} | {r['level_asked']} | {r.get('level') or '-'} | {v} | {r.get('seconds', '')} |"
        )
    L += [
        "",
        "## Limits of this evaluation",
        "",
        "- Built attacks of one kind share a builder, so they are correlated: an interval over "
        "five seeds of one builder is narrower than five independent attacks would earn.",
        "- Calibration and accuracy are measured on the same corpus (in-sample). Families added "
        "later are the out-of-sample check.",
        "- Community re-uploads are presumed benign, not proven benign.",
        "- Trained-in behaviour is evaluated only where probe runs exist (the Qwen3-0.6B "
        "fixtures); the other families are static and provenance only.",
        "",
    ]
    return "\n".join(L)
