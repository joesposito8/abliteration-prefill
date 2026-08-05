"""The one branch of the width rule that would be wrong without looking wrong."""

from __future__ import annotations

import batch_sweep as sweep


def test_a_flat_curve_takes_the_smaller_width():
    """Diminishing returns is the shape the arithmetic predicts, so this is the branch
    the real numbers will take — and taking the fastest instead reads as a valid answer."""
    curve = [
        {"width": w, "prompts_per_second": r, "peak_vram_gb": 10.0}
        for w, r in ((16, 1.0), (32, 1.05), (64, 1.08))
    ]

    assert sweep.choose_width(curve)[0] == 16
