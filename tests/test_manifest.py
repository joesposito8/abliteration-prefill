"""Offline invariants for the shared manifest primitives (``study.manifest``).

Every freeze script routes its hashing and serialization through these, so what is locked
here is what a manifest's recorded digests mean: the canonical form they are taken over,
and the byte-exact file the writer produces.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from study.datasets import FREEZE_MANIFEST_JSON, VALIDATION_HARMFUL_CSV
from study.manifest import (
    canonical_bytes,
    render_csv,
    rollup_sha256,
    sha256_bytes,
    sha256_file,
    write_bytes,
    write_manifest,
)


def test_sha256_bytes_matches_hashlib():
    assert sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_sha256_file(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"payload")
    assert sha256_file(p) == hashlib.sha256(b"payload").hexdigest()


def test_canonical_bytes_is_sorted_and_compact():
    assert canonical_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_canonical_bytes_normalizes_floats():
    # The reason for the rfc8785 dependency: JCS number canonicalization renders 1.0
    # as "1" and keeps 0.95 minimal — json.dumps would emit "1.0". The frozen sampling
    # params are floats, so this normalization is load-bearing for a stable roll-up.
    assert canonical_bytes({"temperature": 1.0, "top_p": 0.95}) == b'{"temperature":1,"top_p":0.95}'


def test_rollup_deterministic_and_change_sensitive():
    spec = {"a": [1, 2, 3], "b": {"x": 1.0}}
    assert rollup_sha256(spec) == rollup_sha256(spec)
    assert rollup_sha256(spec) == sha256_bytes(canonical_bytes(spec))
    assert rollup_sha256(spec) != rollup_sha256({"a": [1, 2, 3], "b": {"x": 2.0}})
    assert len(rollup_sha256(spec)) == 64


def test_write_manifest_pretty_sorted_trailing_newline(tmp_path):
    out = tmp_path / "m.json"
    write_manifest(out, {"b": 1, "a": 2})
    text = out.read_text(encoding="utf-8")
    assert text == '{\n  "a": 2,\n  "b": 1\n}\n'


def test_write_manifest_reproduces_frozen_dataset_manifest(tmp_path):
    # Proof that routing build_datasets' write through write_manifest is output-neutral:
    # re-serializing the committed dataset manifest yields byte-identical content.
    original = FREEZE_MANIFEST_JSON.read_bytes()
    out = tmp_path / "freeze_manifest.json"
    write_manifest(out, json.loads(original))
    assert out.read_bytes() == original


def test_render_csv_hashes_what_write_bytes_puts_on_disk(tmp_path):
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    payload = render_csv(frame)
    out = tmp_path / "t.csv"
    write_bytes(out, payload)
    assert out.read_bytes() == payload
    assert sha256_file(out) == sha256_bytes(payload)


def test_render_csv_reproduces_a_committed_table():
    """The split is output-neutral: rendering re-reads to the frozen file's own bytes."""
    original = VALIDATION_HARMFUL_CSV.read_bytes()
    frame = pd.read_csv(VALIDATION_HARMFUL_CSV)
    assert render_csv(frame) == original
