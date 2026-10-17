"""Signed provenance manifest: what we scanned, byte for byte, and what we concluded.

Format: in-toto Statement v1 wrapped in a minimal DSSE-like envelope signed with Ed25519.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

KEY_DIR = Path(os.path.expanduser("~/.config/pwdet/keys"))
PREDICATE_TYPE = "https://pwdet.local/provenance/v1"


def load_or_create_key(key_dir: Path = KEY_DIR) -> tuple[Ed25519PrivateKey, str]:
    key_dir.mkdir(parents=True, exist_ok=True)
    priv_p, pub_p = key_dir / "ed25519.key", key_dir / "ed25519.pub"
    if priv_p.exists():
        priv = serialization.load_pem_private_key(priv_p.read_bytes(), password=None)
        assert isinstance(priv, Ed25519PrivateKey)
    else:
        priv = Ed25519PrivateKey.generate()
        priv_p.write_bytes(
            priv.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        os.chmod(priv_p, 0o600)
        pub_p.write_bytes(
            priv.public_key().public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
            )
        )
    pub = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return priv, hashlib.sha256(pub).hexdigest()[:16]


def build_statement(
    subject_name: str, file_hashes: dict[str, str], sizes: dict[str, int], predicate: dict[str, Any]
) -> dict[str, Any]:
    combined = hashlib.sha256(
        "\n".join(f"{k} {v}" for k, v in sorted(file_hashes.items())).encode()
    ).hexdigest()
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": subject_name, "digest": {"sha256": combined}}],
        "predicateType": PREDICATE_TYPE,
        "predicate": {
            "files": {
                k: {"sha256": v, "size": sizes.get(k)} for k, v in sorted(file_hashes.items())
            },
            "generated": datetime.now(UTC).isoformat(),
            **predicate,
        },
    }


def sign(statement: dict[str, Any], key_dir: Path = KEY_DIR) -> dict[str, Any]:
    priv, keyid = load_or_create_key(key_dir)
    payload = json.dumps(statement, sort_keys=True, separators=(",", ":")).encode()
    sig = priv.sign(_pae(payload))
    pub = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return {
        "payloadType": "application/vnd.in-toto+json",
        "payload": base64.b64encode(payload).decode(),
        "signatures": [
            {
                "keyid": keyid,
                "sig": base64.b64encode(sig).decode(),
                "pubkey": base64.b64encode(pub).decode(),
            }
        ],
    }


def _pae(payload: bytes, ptype: bytes = b"application/vnd.in-toto+json") -> bytes:
    return b"DSSEv1 %d %s %d " % (len(ptype), ptype, len(payload)) + payload


def verify(
    envelope: dict[str, Any],
    rehash_root: Path | None = None,
    trusted_pubkeys: list[bytes] | None = None,
) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    payload = base64.b64decode(envelope["payload"])
    ok_sig = False
    for s in envelope.get("signatures", []):
        pub = base64.b64decode(s["pubkey"])
        if trusted_pubkeys is not None and pub not in trusted_pubkeys:
            msgs.append(f"signature by untrusted key {s['keyid']}")
            continue
        try:
            Ed25519PublicKey.from_public_bytes(pub).verify(
                base64.b64decode(s["sig"]), _pae(payload)
            )
            ok_sig = True
            msgs.append(f"signature valid (key {s['keyid']})")
        except Exception:
            msgs.append(f"signature INVALID (key {s['keyid']})")
    if not ok_sig:
        return False, msgs
    st = json.loads(payload)
    if rehash_root is not None:
        files = st["predicate"]["files"]
        bad = 0
        for rel, rec in files.items():
            p = rehash_root / rel
            if not p.exists():
                msgs.append(f"missing: {rel}")
                bad += 1
                continue
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            if h.hexdigest() != rec["sha256"]:
                msgs.append(f"CHANGED: {rel}")
                bad += 1
        msgs.append(f"{len(files) - bad}/{len(files)} files match the manifest")
        return bad == 0, msgs
    return True, msgs
