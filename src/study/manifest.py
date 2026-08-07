"""Shared manifest primitives for the freeze scripts: content hashing and serialization.

The pattern is canonicalize-for-hashing, pretty-print-for-storage. ``canonical_bytes``
uses RFC 8785 (JSON Canonicalization Scheme) via ``rfc8785`` — its number normalization
matters because some manifest values are floats, which ``json.dumps`` does not
canonicalize. ``rfc8785`` is imported lazily, so hashing and writing work without it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


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


def write_csv(path: str | Path, frame: pd.DataFrame, float_format: str | None = None) -> str:
    """Write ``frame`` as a CSV — no index, LF, UTF-8 — and return its SHA-256."""
    frame.to_csv(
        path, index=False, lineterminator="\n", encoding="utf-8", float_format=float_format
    )
    return sha256_file(path)


def assert_frozen_unchanged(
    manifest_path: str | Path,
    frozen: Mapping[str, str] | None,
    built: Mapping[str, str],
) -> None:
    """Abort if a rebuild changed an artifact the committed manifest already pins.

    Both mappings are ``{filename: sha256}``; ``frozen`` is ``None`` before the first
    freeze. A build rewrites its artifacts and its manifest together, so a drifted
    derivation would re-freeze itself with every downstream check still passing —
    comparing against what is committed is what makes a rebuild a reproduction test.
    """
    if frozen is None:
        return  # first freeze; there is nothing frozen yet

    drifted = [
        f"  {name}\n    frozen {frozen[name]}\n    built  {sha256}"
        for name, sha256 in sorted(built.items())
        if name in frozen and frozen[name] != sha256
    ]
    if drifted:
        raise SystemExit(
            "REBUILD CHANGED A FROZEN ARTIFACT — refusing to re-freeze:\n"
            + "\n".join(drifted)
            + "\n\nThe committed file on disk has been overwritten; restore it with"
            "\n  git checkout -- data/"
            "\nIf this change is intentional, drop the recorded hash from"
            f"\n  {manifest_path}\nand rebuild, so re-freezing is a deliberate act."
        )
