"""The fixture GGUF converter names the pre-tokenizer as llama.cpp's own converter does."""

import numpy as np

from pwdet.fixture.gguf_convert import permute_qk, pre_tokenizer

SMOLLM = {
    "pre_tokenizer": {
        "type": "Sequence",
        "pretokenizers": [
            {"type": "Digits", "individual_digits": True},
            {"type": "ByteLevel", "add_prefix_space": False},
        ],
    }
}
LLAMA3 = {
    "pre_tokenizer": {
        "type": "Sequence",
        "pretokenizers": [{"type": "Split", "pattern": {"Regex": "x"}}, {"type": "ByteLevel"}],
    }
}


def test_pre_tokenizer_names():
    assert pre_tokenizer("llama", SMOLLM) == "smollm"
    assert pre_tokenizer("llama", LLAMA3) == "llama-bpe"
    assert pre_tokenizer("qwen3", SMOLLM) == "qwen2"
    assert pre_tokenizer("olmo2", {}) == "default"


def test_llama_qk_rows_interleave_each_heads_halves():
    # one head of 4 rows: rotate-half pairs (0, 2), (1, 3) become adjacent, as llama.cpp's converter
    w = np.arange(8, dtype=np.float32).reshape(4, 2)
    assert permute_qk(w, 1)[:, 0].tolist() == [0, 4, 2, 6]
    assert permute_qk(w, 2)[:, 0].tolist() == [0, 2, 4, 6]  # two heads of 2 rows: unchanged
