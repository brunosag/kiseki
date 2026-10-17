"""Worker entry point: `python -m kiseki.worker <command> <path>`.

  train <run_dir>      run or resume a training run (see kiseki.worker.train)
  report <report_dir>  build a report (see kiseki.worker.report)

The core launches it; it is the only process that imports JAX.
"""

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from kiseki.worker import runtime

    runtime.ensure_driver_path()
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[0] not in {"train", "report"}:
        print(__doc__, file=sys.stderr)
        return 2
    command, target = args[0], Path(args[1])
    if command == "train":
        from kiseki.store import Run

        run = Run(target)
        runtime.configure(deterministic=run.spec().deterministic)
        from kiseki.worker import train

        return train.main(run)
    if command == "report":
        from kiseki.report import Report

        runtime.configure()
        from kiseki.worker import report

        return report.main(Report(target))
    return 2


if __name__ == "__main__":
    sys.exit(main())
