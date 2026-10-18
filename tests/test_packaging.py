"""Files an installed pwdet must find without a source checkout or a particular CWD."""

import pytest

from pwdet import _source
from pwdet.blackbox import differential
from pwdet.blackbox.differential import DEFAULT_THRESHOLDS, load_thresholds


def test_packaged_thresholds_load_from_any_cwd(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert differential._packaged_thresholds() is not None
    th = load_thresholds(None)
    assert th["min_delta"] == 0.10 and th["abs_rate"]["pii_hit"] == 0.15
    assert "built-in" not in capsys.readouterr().err


def test_missing_override_warns_and_uses_builtins(tmp_path, capsys):
    th = load_thresholds(tmp_path / "nope.yaml")
    assert th == DEFAULT_THRESHOLDS
    assert "not found; using built-in thresholds" in capsys.readouterr().err


def test_override_file_is_applied(tmp_path):
    f = tmp_path / "t.yaml"
    f.write_text("min_delta: 0.25\nabs_rate: {pii_hit: 0.3}\n")
    th = load_thresholds(f)
    assert th["min_delta"] == 0.25 and th["abs_rate"]["pii_hit"] == 0.3
    assert th["abs_rate"]["exfil_hit"] == DEFAULT_THRESHOLDS["abs_rate"]["exfil_hit"]


def test_docker_script_found_in_checkout():
    assert _source.docker_script("run-whitebox.sh").name == "run-whitebox.sh"


def test_docker_script_outside_checkout_says_so(tmp_path, monkeypatch):
    fake = tmp_path / "site-packages" / "pwdet" / "_source.py"
    fake.parent.mkdir(parents=True)
    monkeypatch.setattr(_source, "__file__", str(fake))
    with pytest.raises(_source.SourceCheckoutRequired, match="source checkout"):
        _source.docker_script("run-whitebox.sh")
