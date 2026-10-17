"""Shared fixtures: a tiny HF-style snapshot built at test time (no real checkpoints needed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pwdet.artifacts.stio import bf16_from_float32, write_safetensors

QWEN_LIKE_TEMPLATE = """{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\\n' }}
{%- endif %}"""


def make_snapshot(
    root: Path,
    *,
    vocab: int = 64,
    hidden: int = 8,
    layers: int = 2,
    seed: int = 0,
    template: str = QWEN_LIKE_TEMPLATE,
    extra_files: dict[str, bytes] | None = None,
    tie: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    emb = rng.normal(0, 0.02, (vocab, hidden)).astype(np.float32)
    tensors = {"model.embed_tokens.weight": ("BF16", (vocab, hidden), bf16_from_float32(emb))}
    for i in range(layers):
        w = rng.normal(0, 0.02, (hidden, hidden)).astype(np.float32)
        tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = (
            "BF16",
            (hidden, hidden),
            bf16_from_float32(w),
        )
        tensors[f"model.layers.{i}.input_layernorm.weight"] = (
            "F32",
            (hidden,),
            np.ones(hidden, np.float32),
        )
    tensors["model.norm.weight"] = ("F32", (hidden,), np.ones(hidden, np.float32))
    if not tie:
        tensors["lm_head.weight"] = ("BF16", (vocab, hidden), bf16_from_float32(emb * 1.01))
    write_safetensors(root / "model.safetensors", tensors, {"format": "pt"})
    cfg = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "vocab_size": vocab,
        "tie_word_embeddings": tie,
        "torch_dtype": "bfloat16",
        "eos_token_id": 2,
        "bos_token_id": 1,
    }
    (root / "config.json").write_text(json.dumps(cfg, indent=1))
    (root / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))
    added = {str(i): {"content": f"<|tok{i}|>", "special": True} for i in range(3)}
    tok_cfg = {
        "chat_template": template,
        "added_tokens_decoder": added,
        "eos_token": "<|tok2|>",
        "model_max_length": 4096,
    }
    (root / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=1))
    (root / "chat_template.jinja").write_text(template)
    tok = {
        "version": "1.0",
        "added_tokens": [{"id": i, "content": f"<|tok{i}|>", "special": True} for i in range(3)],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": {f"t{i}": i for i in range(vocab)}, "merges": []},
    }
    (root / "tokenizer.json").write_text(json.dumps(tok))
    (root / "README.md").write_text(
        "---\nlicense: apache-2.0\nbase_model: Qwen/Qwen3-0.6B\n---\n# test model\n"
    )
    for name, data in (extra_files or {}).items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return root


@pytest.fixture(autouse=True)
def _isolate_hf_cache(monkeypatch):
    """Unit tests must never see the real HF cache (a README base_model claim would attach it)."""
    import pwdet.config as cfg

    monkeypatch.setattr(cfg, "DEFAULT_HF_CACHES", [])
    monkeypatch.setattr(cfg, "CONFIG_SEARCH", [])


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    return make_snapshot(tmp_path / "snap")


@pytest.fixture
def snapshot_pair(tmp_path: Path) -> tuple[Path, Path]:
    """(suspect, base) built from the same seed so they are identical except for what a test mutates."""
    return make_snapshot(tmp_path / "suspect"), make_snapshot(tmp_path / "base")
