"""Token id -> human readable string, from tokenizer.json alone (no tokenizers library)."""

from __future__ import annotations

from functools import lru_cache

from .index import ArtifactIndex


@lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


class TokenMap:
    def __init__(self, a: ArtifactIndex):
        tok = a.json("tokenizer.json") if a.has("tokenizer.json") else None
        self.id2tok: dict[int, str] = {}
        self.special: set[int] = set()
        if isinstance(tok, dict):
            vocab = (tok.get("model") or {}).get("vocab")
            if isinstance(vocab, dict):
                self.id2tok = {int(i): t for t, i in vocab.items()}
            for t in tok.get("added_tokens", []) or []:
                if isinstance(t, dict) and "id" in t:
                    self.id2tok[int(t["id"])] = str(t.get("content", ""))
                    if t.get("special"):
                        self.special.add(int(t["id"]))
        if not self.id2tok and getattr(a, "gguf", None):  # GGUF-only repo: vocab lives in the KVs
            g = next(iter(a.gguf.values())).g
            toks = g.token_strings()
            types = g.kv.get("tokenizer.ggml.token_type") or []
            self.id2tok = {i: t for i, t in enumerate(toks)}
            self.special = {i for i, ty in enumerate(types) if ty == 3}  # CONTROL
        self.size = (max(self.id2tok) + 1) if self.id2tok else 0

    def decode(self, i: int) -> str:
        t = self.id2tok.get(i)
        if t is None:
            return f"<id {i}: unused/padding>"
        dec = _byte_decoder()
        if all(ch in dec for ch in t):
            try:
                s = bytes(dec[ch] for ch in t).decode("utf-8")
                return repr(s)
            except UnicodeDecodeError:
                return repr(t) + " (partial utf-8)"
        return repr(t)
