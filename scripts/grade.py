#!/usr/bin/env python3
"""Grade a sweep's generation logs against the StrongREJECT rubric.

Wants an API key and no GPU: generation recorded its model arguments as null, so the
provider rebuilds weightless and nothing here reaches it.

Grades are written to a separate tree, mirroring the generation one, so what the GPU
produced stays as it was and a condition can be regraded without losing the first run.

Run:  python scripts/grade.py <generation-root> <scored-root>
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from grading.scorers import GRADER, strongreject  # noqa: E402
from inspect_ai import score  # noqa: E402
from inspect_ai.log import (  # noqa: E402
    EvalLog,
    list_eval_logs,
    read_eval_log,
    write_eval_log,
)


def finished_logs(root: Path) -> list[EvalLog]:
    """The successful log header of every condition under ``root``.

    A failed or killed attempt stays in its directory, so the tree cannot be scanned
    without asking each log how it ended — and the header that answers is returned rather
    than dropped, so a caller reading the results need not open the file again.
    """
    headers = (read_eval_log(info, header_only=True) for info in list_eval_logs(str(root)))
    return [log for log in headers if log.status == "success"]


def grade_log(source: Path, output: Path) -> EvalLog:
    """Score one condition, whole. The largest is 8,138 samples and ~250 MB read in.

    ``score`` reads the judge from the header and never writes it back, so declaring it
    there is also what records it.
    """
    log = read_eval_log(str(source))
    log.eval.model_roles = GRADER
    scored = score(log, strongreject(), action="append", copy=False)
    write_eval_log(scored, str(output))
    scored.location = str(output)
    return scored


def grade_sweep(source_root: Path, output_root: Path) -> list[EvalLog]:
    """Grade every finished condition into a mirror of its tree."""
    scored = []
    for log in finished_logs(source_root):
        source = _local(log.location)
        output = output_root / source.relative_to(source_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        scored.append(grade_log(source, output))
    return scored


def _local(uri: str) -> Path:
    """``list_eval_logs`` reports a URI; these logs are files on a disk."""
    return Path(urlparse(uri).path)


def main(argv: list[str]) -> None:
    source_root, output_root = (Path(arg).resolve() for arg in argv)
    for log in grade_sweep(source_root, output_root):
        print(f"{log.eval.model}: {log.results.total_samples} samples, {log.location}")


if __name__ == "__main__":
    main(sys.argv[1:])
