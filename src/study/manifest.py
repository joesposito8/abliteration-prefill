"""Shared manifest primitives: content hashing and canonical serialization.

Both freeze scripts (``build_datasets.py``, ``freeze_portfolio.py``) record artifacts
as SHA-256 content hashes in a committed JSON manifest. Centralizing the operations
that must stay byte-stable keeps the two builders from drifting.

The field-tested pattern is *canonicalize for hashing, pretty-print for storage*:
- ``sha256_file`` / ``sha256_bytes`` — per-artifact content digests.
- ``canonical_bytes`` — the RFC 8785 (JSON Canonicalization Scheme) form used for
  hashing, delegated to the ``rfc8785`` library (sorted keys, RFC-8785 number
  normalization, no insignificant whitespace). It is the number normalization that
  earns the dependency: the frozen sampling params are floats (1.0 / 0.95 / 0.0), which
  ``json.dumps`` does not canonicalize. Imported lazily, so plain hashing/writing works
  without it.
- ``rollup_sha256`` — one digest over the canonical serialization of a spec (the
  in-toto / OCI "digest of the manifest" pattern).
- ``write_manifest`` — the pretty, key-sorted, diff-friendly form that gets committed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_bytes(obj: Any) -> bytes:
    """RFC 8785 (JCS) canonical UTF-8 bytes for deterministic hashing."""
    import rfc8785

    return rfc8785.dumps(obj)


def rollup_sha256(obj: Any) -> str:
    """One SHA-256 over the canonical serialization of ``obj`` — a manifest digest."""
    return sha256_bytes(canonical_bytes(obj))


def write_manifest(path: str | Path, obj: Any) -> None:
    """Write ``obj`` as a pretty, key-sorted JSON manifest (the committed form)."""
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    Path(path).write_text(text, encoding="utf-8")
