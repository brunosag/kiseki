"""SQL over every run on this machine (DuckDB, reading the run files in place).

Views:
  runs   one row per run: id, every RunSpec field (params as JSON) and every status field
  train  one row per step:        run, step, loss, accuracy, optimizer extras
  val    one row per validation:  run, step, elapsed, val_loss, val_accuracy, extras

Example: kiseki query "select optimizer, max(best_accuracy) from runs group by 1"
"""

from pathlib import Path

import duckdb

from kiseki import paths

_RUN = r"regexp_extract(filename, '([^/]+)/[^/]+$', 1)"


def connect(root: Path | None = None) -> duckdb.DuckDBPyConnection:
    runs = root or paths.runs_dir()
    db = duckdb.connect()
    sources = {  # view: (file, read options, extra select)
        "spec": ("spec.json", "maximum_depth=2", " replace (to_json(params) as params)"),
        "status": ("status.json", "maximum_depth=2", ""),
        "train": ("train.ndjson", "format='newline_delimited', ignore_errors=true", ""),
        "val": ("val.ndjson", "format='newline_delimited', ignore_errors=true", ""),
    }
    for view, (name, options, extra) in sources.items():
        pattern = str(runs / "*" / name).replace("'", "''")
        if any(runs.glob(f"*/{name}")):
            db.execute(
                f"create view {view} as select {_RUN} as run, * exclude (filename){extra} from "
                f"read_json('{pattern}', filename=true, union_by_name=true, {options})"
            )
        else:
            db.execute(f"create view {view} as select null::varchar as run where false")
    db.execute("create view runs as select * from spec join status using (run)")
    return db


def query(sql: str, root: Path | None = None) -> tuple[list[str], list[tuple]]:
    result = connect(root).execute(sql)
    columns = [column[0] for column in result.description or []]
    return columns, result.fetchall()
