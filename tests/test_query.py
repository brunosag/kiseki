from kiseki.query import query
from kiseki.spec import RunSpec
from kiseki.store import Run


def test_views_join_specs_statuses_and_rows(home):
    for name, optimizer, accuracies in (("aaaa", "sgd", [0.5, 0.9]), ("bbbb", "leea", [0.3, 0.6])):
        run = Run.create(
            f"20261017-100000-mnist-{optimizer}-{name}",
            RunSpec(optimizer=optimizer),
            machine="local",
        )
        run.append(
            "val",
            [
                {"step": 10 * (i + 1), "elapsed": 1.0, "val_loss": 1.0, "val_accuracy": a}
                for i, a in enumerate(accuracies)
            ],
        )
    columns, rows = query(
        "select optimizer, max(val_accuracy) best from val join runs using (run) "
        "group by 1 order by 1"
    )
    assert columns == ["optimizer", "best"]
    assert rows == [("leea", 0.6), ("sgd", 0.9)]
