#!/usr/bin/env python3
"""Grade a sweep's generation logs against the StrongREJECT rubric.

Wants an API key and no GPU: generation recorded its model arguments as null, so the
provider rebuilds weightless and nothing here reaches it.

Grades are written to a separate tree, mirroring the generation one, so what the GPU
produced stays as it was and a condition can be regraded without losing the first run.

Run:  python scripts/grade.py <generation-root> <scored-root>
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from urllib.parse import urlparse

import anyio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from grading.scorers import JUDGE_CONNECTIONS, strongreject  # noqa: E402
from grading.strongreject_grader import JUDGE_MODEL  # noqa: E402
from inspect_ai import score_async  # noqa: E402
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log  # noqa: E402
from inspect_ai.log._recorders import create_recorder_for_location  # noqa: E402
from inspect_ai.model import ModelConfig  # noqa: E402

# Written onto the header rather than passed to score_async, which reads roles from the
# header and never writes them back. Set here or nothing records what graded the run.
GRADER = {"grader": ModelConfig(model=f"openai/{JUDGE_MODEL}")}

# Samples held in memory at once. No point in more than the judge can be asked about.
STREAM = JUDGE_CONNECTIONS


def finished_logs(root: Path) -> list:
    """The successful log of every condition under ``root``.

    A failed or killed attempt stays in its directory, so the tree cannot be scanned
    without asking each log how it ended.
    """
    return [
        info
        for info in list_eval_logs(str(root))
        if read_eval_log(info, header_only=True).status == "success"
    ]


async def grade_log(log_file: str, output_file: str) -> EvalLog:
    """Score one condition, reading and writing ``STREAM`` samples at a time.

    A whole condition is hundreds of megabytes of transcript, and the judge needs one
    sample at a time.
    """
    reader = create_recorder_for_location(log_file, str(Path(log_file).parent))
    writer = create_recorder_for_location(output_file, str(Path(output_file).parent))

    log = await reader.read_log(log_file, header_only=True)
    log.eval.model_roles = GRADER

    sample_ids = await reader.read_log_sample_ids(log_file)
    resident = anyio.Semaphore(STREAM)
    written = 0

    @contextlib.asynccontextmanager
    async def read_sample(index: int):
        async with resident:
            sample = await reader.read_log_sample(log_file, *sample_ids[index])
            yield sample
            await writer.log_sample(log.eval, sample)
            nonlocal written
            written += 1
            if written % STREAM == 0:
                await writer.flush(log.eval)

    await writer.log_init(log.eval, location=output_file)
    await writer.log_start(log.eval, log.plan)
    scored = await score_async(
        log, strongreject(), action="append", copy=False, samples=read_sample
    )
    # Returned rather than ``scored``, whose location is still the log we read.
    return await writer.log_finish(
        scored.eval,
        scored.status,
        scored.stats,
        scored.results,
        scored.reductions,
        scored.error,
        header_only=True,
    )


async def grade_sweep(source_root: Path, output_root: Path) -> list[EvalLog]:
    """Grade every finished condition, one at a time, into a mirror of its tree."""
    scored = []
    for info in finished_logs(source_root):
        source = _local(info.name)
        output = output_root / source.relative_to(source_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        scored.append(await grade_log(str(source), str(output)))
    return scored


def _local(uri: str) -> Path:
    """``list_eval_logs`` reports a URI; these logs are files on a disk."""
    return Path(urlparse(uri).path)


def main(argv: list[str]) -> None:
    source_root, output_root = (Path(arg).resolve() for arg in argv)
    for log in anyio.run(grade_sweep, source_root, output_root):
        print(f"{log.eval.model}: {log.results.total_samples} samples, {log.location}")


if __name__ == "__main__":
    main(sys.argv[1:])
