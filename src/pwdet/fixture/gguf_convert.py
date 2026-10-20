"""Minimal HF -> GGUF converter for dense llama-style checkpoints (qwen3 / qwen2 / llama), numpy only.

llama.cpp's `convert_hf_to_gguf.py` is not vendored here; this writes the same tensor names (via
gguf-py's architecture map), the KVs llama.cpp needs for these architectures, a GPT-2-style BPE
tokenizer block and the chat template, with 2-D weights block-quantised to Q8_0 (or stored F16/BF16)
and 1-D tensors kept F32 (gguf-py quantises Q8_0/Q4_0/Q5_0/Q4_1/Q5_1 in numpy; the K-quants are
dequantise-only there, so Q4_K_M files must come from llama.cpp). It exists so the GGUF code path can be exercised on controlled fixtures:
B2G (benign Q8_0 of the base) and F3G (the surgery fixture re-serialised as GGUF).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts.index import ArtifactIndex
from ..artifacts.quant import QuantIndex
from .common import copy_snapshot, read_truth, write_truth

PRE_TOKENIZER = {"qwen3": "qwen2", "qwen2": "qwen2", "llama": "llama-bpe"}


def pre_tokenizer(arch: str, tok: dict[str, Any]) -> str:
    """llama.cpp's name for how the tokenizer splits text. Its own converter tells them apart by
    hashing the tokenizer; here the HF pre-tokenizer decides: SmolLM splits every digit and then
    goes byte-level, with no Llama-3 split regex, and llama.cpp calls that `smollm`."""
    pre = json.dumps(tok.get("pre_tokenizer") or {})
    if arch == "llama" and '"individual_digits": true' in pre and '"Split"' not in pre:
        return "smollm"
    return PRE_TOKENIZER.get(arch, "default")


def permute_qk(w: np.ndarray, n_head: int) -> np.ndarray:
    """HF stores llama's q/k rows for rotate-half RoPE; llama.cpp's llama arch applies RoPE to
    adjacent pairs, so its converter interleaves each head's two halves. Without this the GGUF loads
    and talks, but every position is rotated wrongly (qwen archs use NeoX RoPE and need none)."""
    return np.ascontiguousarray(
        w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:])
        .swapaxes(1, 2)
        .reshape(w.shape)
    )


SAFETENSORS_ONLY_CHECKS = {"static.weight_stats", "static.st_header"}


def convert_to_gguf(
    src: Path,
    out_file: Path,
    qtype: str = "Q8_0",
    base_repo: str | None = None,
    quantized_by: str = "pwdet-fixture",
) -> dict[str, Any]:
    import gguf

    a = ArtifactIndex(src)
    cfg = a.text_config
    arch = str(cfg.get("model_type") or "")
    by_name = {v: k for k, v in gguf.MODEL_ARCH_NAMES.items()}
    if arch not in by_name:
        raise ValueError(f"{arch!r}: not an architecture gguf-py knows")
    n_layers = int(cfg.get("num_hidden_layers") or 0)
    q = gguf.GGMLQuantizationType[qtype]
    w = gguf.GGUFWriter(str(out_file), arch)
    w.add_name(src.name)
    w.add_block_count(n_layers)
    w.add_context_length(int(cfg.get("max_position_embeddings") or 4096))
    w.add_embedding_length(int(cfg["hidden_size"]))
    if cfg.get("intermediate_size"):
        w.add_feed_forward_length(int(cfg["intermediate_size"]))
    heads = int(cfg.get("num_attention_heads") or 1)
    w.add_head_count(heads)
    w.add_head_count_kv(int(cfg.get("num_key_value_heads") or heads))
    head_dim = int(cfg.get("head_dim") or int(cfg["hidden_size"]) // heads)
    w.add_key_length(head_dim)
    w.add_value_length(head_dim)
    w.add_rope_freq_base(float(cfg.get("rope_theta") or 10000.0))
    w.add_layer_norm_rms_eps(float(cfg.get("rms_norm_eps") or 1e-6))
    ft = {
        "Q8_0": "MOSTLY_Q8_0",
        "Q4_0": "MOSTLY_Q4_0",
        "Q5_0": "MOSTLY_Q5_0",
        "Q4_K": "MOSTLY_Q4_K_M",
        "Q5_K": "MOSTLY_Q5_K_M",
        "Q6_K": "MOSTLY_Q6_K",
        "F16": "MOSTLY_F16",
        "BF16": "MOSTLY_BF16",
        "F32": "ALL_F32",
    }
    w.add_file_type(gguf.LlamaFileType[ft.get(qtype, "MOSTLY_Q8_0")])
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)
    w.add_quantized_by(quantized_by)
    base_repo = base_repo or str(a.readme_frontmatter.get("base_model") or "") or None
    if base_repo:
        w.add_base_model_count(1)
        w.add_base_model_name(0, base_repo.split("/")[-1])
        w.add_base_model_repo_url(0, f"https://huggingface.co/{base_repo}")

    # tokenizer --------------------------------------------------------------------------
    tok = a.json("tokenizer.json") or {}
    model = tok.get("model") or {}
    vocab: dict[str, int] = dict(model.get("vocab") or {})
    added = {int(t["id"]): t for t in tok.get("added_tokens", []) or [] if "id" in t}
    for i, t in added.items():
        vocab.setdefault(str(t.get("content", "")), i)
    n_vocab = max(int(cfg.get("vocab_size") or 0), (max(vocab.values()) + 1) if vocab else 0)
    id2tok = {i: t for t, i in vocab.items()}
    tokens, types = [], []
    for i in range(n_vocab):
        t = id2tok.get(i)
        if t is None:
            tokens.append(f"[PAD{i}]")
            types.append(gguf.TokenType.UNUSED)
        elif i in added:
            tokens.append(t)
            types.append(
                gguf.TokenType.CONTROL if added[i].get("special") else gguf.TokenType.USER_DEFINED
            )
        else:
            tokens.append(t)
            types.append(gguf.TokenType.NORMAL)
    merges = [m if isinstance(m, str) else " ".join(m) for m in (model.get("merges") or [])]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre(pre_tokenizer(arch, tok))
    w.add_token_list(tokens)
    w.add_token_types(types)
    if merges:
        w.add_token_merges(merges)
    tc = a.tokenizer_config

    def tok_id(spec: Any) -> int | None:
        if isinstance(spec, dict):
            spec = spec.get("content")
        return vocab.get(spec) if isinstance(spec, str) else None

    bos = cfg.get("bos_token_id")
    eos = cfg.get("eos_token_id")
    eos = eos[0] if isinstance(eos, list) and eos else eos
    if isinstance(bos, int):
        w.add_bos_token_id(bos)
    if isinstance(eos, int):
        w.add_eos_token_id(eos)
    pad = tok_id(tc.get("pad_token"))
    if pad is not None:
        w.add_pad_token_id(pad)
    w.add_add_bos_token(bool(tc.get("add_bos_token", False)))
    templates = a.chat_templates()
    if templates:
        w.add_chat_template(next(iter(templates.values())))

    # tensors ----------------------------------------------------------------------------
    tm = gguf.get_tensor_name_map(by_name[arch], n_layers)
    qi = QuantIndex(a)
    written: dict[str, str] = {}
    skipped: list[str] = []
    for name, lt in sorted(qi.logical.items()):
        gname = tm.get_name(name, try_suffixes=(".weight", ".bias"))
        if gname is None:
            skipped.append(name)
            continue
        x = np.ascontiguousarray(qi.dequant(lt), dtype=np.float32)
        if arch == "llama" and name.endswith(("q_proj.weight", "k_proj.weight")):
            n_head = int(cfg["num_attention_heads"])
            if name.endswith("k_proj.weight"):
                n_head = int(cfg.get("num_key_value_heads") or n_head)
            x = permute_qk(x, n_head)
        if x.ndim == 2 and qtype != "F32" and x.shape[-1] % 32 == 0:
            if qtype == "F16":
                w.add_tensor(gname, x.astype(np.float16))
                written[gname] = "F16"
            elif qtype == "BF16":
                raw = (x.view(np.uint32) >> 16).astype(np.uint16)
                w.add_tensor(gname, raw, raw_dtype=gguf.GGMLQuantizationType.BF16)
                written[gname] = "BF16"
            else:
                w.add_tensor(gname, gguf.quants.quantize(x, q), raw_dtype=q)
                written[gname] = qtype
        else:
            w.add_tensor(gname, x)
            written[gname] = "F32"
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {
        "arch": arch,
        "qtype": qtype,
        "tensors": written,
        "skipped": skipped,
        "n_vocab": n_vocab,
    }


def make_gguf_variant(
    src: Path, out: Path, variant: str, qtype: str = "Q8_0", base_repo: str | None = None
) -> dict:
    """GGUF re-serialisation of a snapshot or fixture dir: same JSON side files, weights only as GGUF.

    A fixture's ground truth carries over (minus checks that only exist for safetensors); a plain
    snapshot becomes a benign control.
    """
    a = ArtifactIndex(src)
    skip = {f.rel for f in a.safetensors_files} | {"model.safetensors.index.json"}
    copy_snapshot(src, out, skip=skip)
    info = convert_to_gguf(src, out / f"model-{qtype}.gguf", qtype=qtype, base_repo=base_repo)
    parent = read_truth(src)
    if parent:
        truth = dict(parent)
        truth["expected"] = {
            k: v for k, v in parent.get("expected", {}).items() if k not in SAFETENSORS_ONLY_CHECKS
        }
        truth["kind"] = f"{parent.get('kind')}+gguf"
        truth["source_variant"] = parent.get("variant")
    else:
        truth = {"kind": "benign-gguf", "modified": [], "expected": {}}
    truth.update({"variant": variant, "gguf": {"qtype": qtype, "skipped_tensors": info["skipped"]}})
    write_truth(out, truth)
    return truth
