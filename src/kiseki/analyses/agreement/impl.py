"""Agreement between pairs of models on individual images."""

import math

from kiseki.analyses.agreement import TITLE, Params
from kiseki.worker.report import Context, Section, md_table


def mcnemar_exact(only_first: int, only_second: int) -> float:
    """Two-sided exact McNemar p-value: a binomial test on the discordant pairs."""
    n = only_first + only_second
    if n == 0:
        return 1.0
    k = min(only_first, only_second)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def run(params: Params, context: Context) -> Section:
    labels = context.labels
    summary, data, blocks = {}, {}, []
    for first, second in context.pairs():
        a, b = context.evaluate(first), context.evaluate(second)
        both = a.correct & b.correct
        only_a = a.correct & ~b.correct
        only_b = ~a.correct & b.correct
        neither = ~a.correct & ~b.correct
        same_wrong = neither & (a.predictions == b.predictions)
        agree = a.predictions == b.predictions
        counts = {
            "both_correct": int(both.sum()),
            "only_first": int(only_a.sum()),
            "only_second": int(only_b.sum()),
            "both_wrong_same": int(same_wrong.sum()),
            "both_wrong_different": int((neither & ~same_wrong).sum()),
        }
        errors = int((~a.correct).sum()) + int((~b.correct).sum())
        key = f"{first.label} vs {second.label}"
        # Error overlap: shared errors over the smaller error set (1: one's errors ⊂ other's).
        smaller = min(int((~a.correct).sum()), int((~b.correct).sum()))
        overlap = float(neither.sum() / smaller) if smaller else 0.0
        p_value = mcnemar_exact(counts["only_first"], counts["only_second"])
        summary[key] = {
            "agreement": float(agree.mean()),
            "error_overlap": overlap,
            "mcnemar_p": p_value,
            **counts,
        }
        per_class = [
            {
                "class": name,
                "agreement": float(agree[labels == i].mean()),
                "only_first": int(only_a[labels == i].sum()),
                "only_second": int(only_b[labels == i].sum()),
            }
            for i, name in enumerate(context.class_names)
        ]
        data[key] = {"per_class": per_class, "errors": errors}
        blocks.append(
            f"**{key}**: predictions agree on {100 * agree.mean():.2f}% of images; "
            f"{100 * overlap:.1f}% of the smaller error set is shared; McNemar "
            f"p = {p_value:.3g}.\n\n"
            + md_table(
                [
                    "both right",
                    f"only {first.label}",
                    f"only {second.label}",
                    "both wrong (same)",
                    "both wrong (different)",
                ],
                [
                    [
                        counts["both_correct"],
                        counts["only_first"],
                        counts["only_second"],
                        counts["both_wrong_same"],
                        counts["both_wrong_different"],
                    ]
                ],
            )
        )
    if not blocks:
        blocks.append("One model: nothing to compare.")
    return Section(
        name="agreement", title=TITLE, summary=summary, data=data, markdown="\n\n".join(blocks)
    )
