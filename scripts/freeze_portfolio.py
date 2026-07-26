#!/usr/bin/env python3
"""Freeze the 13-prefill portfolio: hash every authored file and the frozen rules into
``data/portfolio_manifest.json`` with one roll-up ``portfolio_sha256``.

``build_manifest()`` is pure and deterministic (no clock, no randomness), so re-running
produces a byte-identical manifest and the tests can verify hashing without a committed
file. The single ``portfolio_sha256`` is what gets recorded in the preregistration
tracker; after that the portfolio is never revised.

Run:  python scripts/freeze_portfolio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from study.manifest import (  # noqa: E402
    rollup_sha256,
    sha256_bytes,
    sha256_file,
    write_manifest,
)
from prefills import (  # noqa: E402
    FAMILIES,
    FIXED_FALLBACK,
    HELPER_MODEL,
    HELPER_REVISION,
    HELPER_SAMPLING,
    MAX_ATTEMPTS,
    MAX_NEW_TOKENS,
    PLACEHOLDER,
    PORTFOLIO,
    STATIC_BASELINE,
    VARIANTS_PER_FAMILY,
    prompt_path,
)

MANIFEST_PATH = REPO / "data" / "portfolio_manifest.json"

# Frozen source files whose bytes the portfolio hash pins, beyond the six prompts.
CODE_FILES = ("rules.py", "families.py", "__init__.py")


def build_manifest() -> dict:
    """The full manifest dict, including per-file hashes and the roll-up hash."""
    prompt_dir = prompt_path(FAMILIES[0].id).parent
    code_dir = REPO / "src" / "prefills"

    families = [
        {
            "id": f.id,
            "prompt_file": f.prompt_file,
            "sha256": sha256_file(prompt_path(f.id)),
            "struppek_definition": f.struppek_definition,
            "review_note": f.review_note,
        }
        for f in FAMILIES
    ]
    code = {name: sha256_file(code_dir / name) for name in CODE_FILES}

    rules = {
        "helper_model": HELPER_MODEL,
        "helper_revision": HELPER_REVISION,
        "helper_sampling": dict(HELPER_SAMPLING),
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_attempts": MAX_ATTEMPTS,
        "variants_per_family": VARIANTS_PER_FAMILY,
        "fixed_fallback": FIXED_FALLBACK,
        "static_baseline": STATIC_BASELINE,
        "static_baseline_sha256": sha256_bytes(STATIC_BASELINE.encode("utf-8")),
        "prompt_placeholder": PLACEHOLDER,
    }

    # The spec that the roll-up hash covers: file hashes + frozen rule values + slots.
    spec = {
        "families": families,
        "code": code,
        "rules": rules,
        "slots": [s.slot_id for s in PORTFOLIO],
    }
    portfolio_sha256 = rollup_sha256(spec)

    manifest = dict(spec)
    manifest["portfolio_sha256"] = portfolio_sha256
    manifest["counts"] = {"families": len(FAMILIES), "slots": len(PORTFOLIO)}
    manifest["_note"] = (
        "Prompts + code authored from Struppek (arXiv:2602.14689); baseline from Kuo "
        "(arXiv:2605.26526). portfolio_sha256 is the frozen preregistration hash."
    )
    return manifest


def main() -> None:
    manifest = build_manifest()
    write_manifest(MANIFEST_PATH, manifest)
    print(f"froze portfolio -> {MANIFEST_PATH}")
    print(f"  portfolio_sha256: {manifest['portfolio_sha256']}")
    for fam in manifest["families"]:
        print(f"  {fam['id']:22s} {fam['sha256'][:16]}…")


if __name__ == "__main__":
    main()
