"""The preflight around direction extraction.

The tensor path is gated on torch, which is in the ``gpu`` extra rather than ``dev``. The
matrix-inventory guard the script also runs belongs to ``abliteration.edit`` and is tested
there.
"""

from __future__ import annotations

import extract_directions as extract
import pytest

from conftest import needs_torch

try:
    import torch
except ImportError:  # every test that uses it is skipped
    pass


# --- the tensor path -------------------------------------------------------


@needs_torch
def test_nan_directions_are_refused():
    directions = torch.ones(4, 8)
    directions[2, 0] = float("nan")

    with pytest.raises(SystemExit, match="NaN"):
        extract.check_directions(directions)


@needs_torch
def test_two_layers_with_an_identical_direction_are_refused():
    """Adjacent duplicates mean the extraction collapsed, not that the layers agree."""
    directions = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    directions[2] = directions[1]

    with pytest.raises(SystemExit, match="identical direction"):
        extract.check_directions(directions)


@needs_torch
def test_distinct_directions_pass():
    extract.check_directions(torch.nn.functional.normalize(torch.randn(4, 8), dim=-1))
