"""Layer-wise relevance propagation for `kiseki.models.layers` networks, in JAX.

Relevance starts as the target logit and flows back layer by layer (Montavon et al. 2019,
"Layer-Wise Relevance Propagation: An Overview"), with the composite rules recommended
there for convolutional networks:
- first layer (on pixels): z^B, bounded by the pixel range [0, 1] in normalized units;
- other convolutions: LRP-γ (positive weights boosted by γ);
- hidden dense layers: LRP-ε, ε relative to each sample's RMS pre-activation;
- output layer: LRP-0;
- max pooling: winner takes all; average pooling: proportional (z-rule); ReLU: identity.

Every rule is "gradient × input" of a modified layer, computed with `jax.vjp`, so it works
for any layer list the model DSL can declare. Rules other than ε conserve relevance up to
what biases absorb; `kept` reports Σ input relevance / target logit per image.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import lax

from kiseki.analyses.lrp import TITLE, Params
from kiseki.models.layers import AvgPool, Conv, Dense, Flatten, MaxPool, Network, ReLU, apply_layer
from kiseki.worker.report import Context, Section, md_table

TINY = 1e-9


def _stabilize(z: jax.Array, epsilon: jax.Array | float) -> jax.Array:
    return z + jnp.where(z >= 0, 1.0, -1.0) * (epsilon + TINY)


def _linear(layer) -> Callable:
    def apply(a, w, b):
        return apply_layer(layer, a, (w, b))

    return apply


def _rule(apply, a, relevance, w, b, epsilon: jax.Array | float = 0.0):
    """Generic rule: R_in = a ⊙ ∂/∂a [ s · z(a) ], s = R / stabilized z."""
    z, vjp = jax.vjp(lambda t: apply(t, w, b), a)
    (c,) = vjp(relevance / _stabilize(z, epsilon))
    return a * c


def _z_bounded(apply, a, relevance, w, low, high):
    zero = jnp.zeros(w.shape[-1], w.dtype)
    positive, negative = jnp.maximum(w, 0), jnp.minimum(w, 0)
    z, vjp_a = jax.vjp(lambda t: apply(t, w, zero), a)
    lz, vjp_l = jax.vjp(lambda t: apply(t, positive, zero), jnp.broadcast_to(low, a.shape))
    hz, vjp_h = jax.vjp(lambda t: apply(t, negative, zero), jnp.broadcast_to(high, a.shape))
    s = relevance / _stabilize(z - lz - hz, 0.0)
    return a * vjp_a(s)[0] - low * vjp_l(s)[0] - high * vjp_h(s)[0]


def relevance(
    net: Network,
    flat: jax.Array,
    x: jax.Array,
    targets: jax.Array,
    *,
    gamma: float,
    epsilon: float,
    low: jax.Array,
    high: jax.Array,
):
    """Input relevance (B, H, W, C) of `targets` logits, and the target logits."""
    tensors = net.unflatten(flat)
    inputs, h = [], x
    for index, layer in enumerate(net.layers):
        inputs.append(h)
        slots = net._param_index.get(index)
        h = apply_layer(layer, h, (tensors[slots[0]], tensors[slots[1]]) if slots else None)
    logits = h
    r = logits * jax.nn.one_hot(targets, net.classes)
    parametric = [i for i, layer in enumerate(net.layers) if isinstance(layer, Conv | Dense)]
    first, last = parametric[0], parametric[-1]
    for index in reversed(range(len(net.layers))):
        layer, a = net.layers[index], inputs[index]
        match layer:
            case Conv() | Dense():
                w_slot, b_slot = net._param_index[index]
                w, b = tensors[w_slot], tensors[b_slot]
                apply = _linear(layer)
                if index == first:
                    r = _z_bounded(apply, a, r, w, low, high)
                elif index == last:
                    r = _rule(apply, a, r, w, b)
                elif isinstance(layer, Conv):
                    r = _rule(
                        apply, a, r, w + gamma * jnp.maximum(w, 0), b + gamma * jnp.maximum(b, 0)
                    )
                else:
                    z = apply(a, w, b)
                    rms = jnp.sqrt((z**2).mean(axis=tuple(range(1, z.ndim)), keepdims=True))
                    r = _rule(apply, a, r, w, b, epsilon * rms)
            case ReLU():
                pass
            case MaxPool():
                _, vjp = jax.vjp(lambda t, layer=layer: apply_layer(layer, t, None), a)
                (r,) = vjp(r)
            case AvgPool():
                z, vjp = jax.vjp(lambda t, layer=layer: apply_layer(layer, t, None), a)
                (c,) = vjp(r / _stabilize(z, 0.0))
                r = a * c
            case Flatten():
                r = r.reshape(a.shape)
    target_logits = jnp.take_along_axis(logits, targets[:, None], 1)[:, 0]
    return r, target_logits


def maps_for(net, flat, x, targets, params: Params, low, high, batch: int = 250):
    """Heatmaps (N, H, W) summed over channels, and kept relevance per image."""

    def one(args: tuple[jax.Array, jax.Array]):
        images, labels = args
        r, logit = relevance(
            net,
            flat,
            images,
            labels,
            gamma=params.gamma,
            epsilon=params.epsilon,
            low=low,
            high=high,
        )
        total = r.sum(axis=(1, 2, 3))
        return r.sum(-1), total / jnp.where(jnp.abs(logit) > TINY, logit, 1.0)

    count = x.shape[0]
    pad = -(-count // batch) * batch - count
    xs = jnp.pad(x, ((0, pad), (0, 0), (0, 0), (0, 0))).reshape(-1, batch, *x.shape[1:])
    ys = jnp.pad(targets, (0, pad)).reshape(-1, batch)
    heat, kept = lax.map(one, (xs, ys))
    return heat.reshape(-1, *x.shape[1:3])[:count], kept.reshape(-1)[:count]


def normalized(maps: np.ndarray) -> np.ndarray:
    scale = np.abs(maps).reshape(maps.shape[0], -1).max(1)
    return maps / np.where(scale > 0, scale, 1.0)[:, None, None]


GROUPS = ("only_first", "only_second", "both_wrong_different", "both_wrong_same", "both_right")


def groups(context: Context) -> np.ndarray:
    first = context.evaluate(context.subjects[0])
    if len(context.subjects) == 1:
        return np.where(first.correct, "both_right", "both_wrong_same")
    second = context.evaluate(context.subjects[1])
    agree = first.predictions == second.predictions
    return np.select(
        [
            first.correct & ~second.correct,
            ~first.correct & second.correct,
            ~first.correct & ~second.correct & ~agree,
            ~first.correct & ~second.correct,
        ],
        ["only_first", "only_second", "both_wrong_different", "both_wrong_same"],
        "both_right",
    )


def gallery_indices(context: Context, count: int) -> list[int]:
    """Round-robin over groups, most informative first: where the models differ."""
    rng = np.random.default_rng(context.seed)
    labels = groups(context)
    queues = {g: list(rng.permutation(np.flatnonzero(labels == g))) for g in GROUPS}
    chosen: list[int] = []
    while len(chosen) < count and any(queues.values()):
        for group in GROUPS:
            if queues[group] and len(chosen) < count:
                chosen.append(int(queues[group].pop(0)))
    return chosen


def focus(maps: np.ndarray) -> dict[str, float]:
    """How concentrated relevance is: share of |R| in the top 10% pixels; positive share."""
    flat = np.abs(maps.reshape(maps.shape[0], -1))
    top = max(1, flat.shape[1] // 10)
    concentrated = np.sort(flat, axis=1)[:, -top:].sum(1) / np.maximum(flat.sum(1), 1e-12)
    positive = np.clip(maps, 0, None).reshape(maps.shape[0], -1).sum(1) / np.maximum(
        flat.sum(1), 1e-12
    )
    return {"top10_share": float(concentrated.mean()), "positive_share": float(positive.mean())}


def run(params: Params, context: Context) -> Section:
    low = (0.0 - context.data.mean) / context.data.std
    high = (1.0 - context.data.mean) / context.data.std
    gallery = gallery_indices(context, params.gallery)
    rng = np.random.default_rng(context.seed + 1)
    per_class = (
        {
            c: rng.permutation(np.flatnonzero(context.labels == c))[: params.per_class]
            for c in range(len(context.class_names))
        }
        if params.per_class
        else {}
    )
    wanted = sorted(set(gallery).union(*[set(v.tolist()) for v in per_class.values()]))
    if not wanted:
        return Section(
            name="lrp", title=TITLE, summary={}, data={}, markdown="No images requested."
        )
    position = {index: i for i, index in enumerate(wanted)}
    x = context.x[jnp.asarray(wanted)]
    group_names = groups(context)
    summary, data = {}, {"gallery": [], "arrays": {}}
    arrays = []
    gallery_maps, class_maps = {}, {}
    for number, subject in enumerate(context.subjects):
        evaluation = context.evaluate(subject)
        source = context.labels if params.target == "label" else evaluation.predictions
        targets = jnp.asarray(source[wanted], jnp.int32)
        compute = jax.jit(
            lambda flat, x, t, net=subject.net: maps_for(net, flat, x, t, params, low, high)
        )
        heat, kept = compute(subject.weights, x, targets)
        heat = np.asarray(heat)
        kept = np.asarray(kept)
        gallery_maps[number] = (
            normalized(heat[[position[i] for i in gallery]]) if gallery else heat[:0]
        )
        means = (
            np.stack([heat[[position[i] for i in per_class[c]]].mean(0) for c in per_class])
            if per_class
            else heat[:0]
        )
        class_maps[number] = normalized(means) if len(means) else means
        summary[subject.label] = {"kept_relevance": float(np.median(kept)), **focus(heat)}
        data["arrays"][subject.label] = {
            "gallery": context.save_array(
                f"lrp-{number}-gallery", gallery_maps[number].astype(np.float16)
            ),
            "classes": context.save_array(
                f"lrp-{number}-classes", class_maps[number].astype(np.float16)
            ),
        }
        arrays += list(data["arrays"][subject.label].values())
    images = (
        np.asarray(context.pixels(context.x[jnp.asarray(gallery)])) if gallery else np.zeros((0,))
    )
    data["arrays"]["images"] = context.save_array(
        "lrp-images", np.clip(images, 0, 1).astype(np.float16)
    )
    arrays.append(data["arrays"]["images"])
    for index in gallery:
        entry = {
            "index": index,
            "label": int(context.labels[index]),
            "group": str(group_names[index]),
        }
        for subject in context.subjects:
            evaluation = context.evaluate(subject)
            entry[subject.label] = {
                "prediction": int(evaluation.predictions[index]),
                "confidence": float(evaluation.probabilities[index].max()),
            }
        data["gallery"].append(entry)

    figures = []
    if gallery:
        shown = gallery[:12]
        rows = 1 + len(context.subjects)
        figure, axes = plt.subplots(
            rows, len(shown), figsize=(1.3 * len(shown), 1.4 * rows), squeeze=False
        )
        for column, index in enumerate(shown):
            picture = images[column]
            axes[0][column].imshow(
                picture[..., 0] if picture.shape[-1] == 1 else picture,
                cmap="gray" if picture.shape[-1] == 1 else None,
            )
            axes[0][column].set_title(
                f"{context.class_names[context.labels[index]][:8]}", fontsize=7
            )
            for row, subject in enumerate(context.subjects, start=1):
                prediction = context.evaluate(subject).predictions[index]
                axes[row][column].imshow(gallery_maps[row - 1][column], cmap="bwr", vmin=-1, vmax=1)
                axes[row][column].set_title(f"→{context.class_names[prediction][:8]}", fontsize=7)
        for axis in axes.flat:
            axis.set_xticks([])
            axis.set_yticks([])
        for row, subject in enumerate(context.subjects, start=1):
            axes[row][0].set_ylabel(subject.label, fontsize=7)
        figures.append(context.save_figure(figure, "lrp-gallery"))
    if per_class:
        classes = len(per_class)
        figure, axes = plt.subplots(
            len(context.subjects),
            classes,
            figsize=(1.2 * classes, 1.4 * len(context.subjects)),
            squeeze=False,
        )
        for row, subject in enumerate(context.subjects):
            for column in range(classes):
                axes[row][column].imshow(class_maps[row][column], cmap="bwr", vmin=-1, vmax=1)
                axes[row][column].set_xticks([])
                axes[row][column].set_yticks([])
                if row == 0:
                    axes[row][column].set_title(context.class_names[column][:8], fontsize=7)
            axes[row][0].set_ylabel(subject.label, fontsize=7)
        figures.append(context.save_figure(figure, "lrp-classes"))
    rows_md = [
        [
            s.label,
            summary[s.label]["kept_relevance"],
            100 * summary[s.label]["top10_share"],
            100 * summary[s.label]["positive_share"],
        ]
        for s in context.subjects
    ]
    markdown = (
        f"Relevance of the {params.target} class. Gallery: {len(gallery)} images, chosen "
        "round-robin from where the models differ first. Red supports the class, blue "
        "speaks against it. Kept: median Σ relevance / logit (1: conserved); top-10%: "
        "share of |relevance| on the most relevant tenth of pixels (higher: more focused)."
        "\n\n" + md_table(["model", "kept", "top-10% share %", "positive share %"], rows_md)
    )
    return Section(
        name="lrp",
        title=TITLE,
        summary=summary,
        data=data,
        markdown=markdown,
        figures=figures,
        arrays=arrays,
    )
