"""The rule that picks the batch width, which has to be settled before the numbers are."""

from __future__ import annotations

import batch_sweep as sweep


def curve(*points: tuple[int, float], vram: float = 10.0) -> list[dict]:
    return [
        {"width": width, "prompts_per_second": rate, "peak_vram_gb": vram}
        for width, rate in points
    ]


def test_a_flat_curve_takes_the_smaller_width():
    """Diminishing returns is the expected shape, so this is the live branch."""
    width, rule = sweep.choose_width(curve((16, 1.0), (32, 1.05), (64, 1.08)))

    assert width == 16
    assert "within" in rule


def test_a_steep_curve_takes_the_fastest():
    width, rule = sweep.choose_width(curve((16, 1.0), (32, 2.0), (64, 4.0)))

    assert width == 64
    assert rule == "fastest within the memory ceiling"


def test_stepping_down_stops_at_the_first_real_loss():
    """It does not skip past a width that costs too much to reach a smaller one."""
    width, _ = sweep.choose_width(curve((16, 1.0), (32, 0.5), (64, 1.05), (128, 1.1)))

    assert width == 64


def test_a_width_over_the_memory_ceiling_is_not_eligible_however_fast():
    measurements = curve((32, 1.0)) + curve((128, 9.0), vram=sweep.VRAM_CEILING_GB + 1)

    width, _ = sweep.choose_width(measurements)

    assert width == 32
