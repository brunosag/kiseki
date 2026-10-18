# Analyses

Reports compare saved models: `kiseki report RUN[:NAME]...` (or the app) runs a worker
(`kiseki/worker/report.py`) that evaluates each model once over the chosen split and runs
the analyses in `kiseki/analyses/`, writing `report.json` (for the app), `report.md` (for
agents and people), `figures/` and `arrays/`.

## One evaluation per model, shared

Logits, probabilities, predictions and penultimate features over the split are computed
once per model and cached in the report's `Context`; analyses add only their own work.
A full report of nine analyses on three MNIST models took 38 s on the RTX 3050
(2026-10-17), 19 s of it t-SNE.

## Embeddings: one exact t-SNE map per model, on the GPU

Each model's penultimate features (5000 class-balanced images) get PCA and exact t-SNE:
kNN affinities at perplexity 30, every pairwise repulsion each iteration, row-blocked;
reference optimisation (exaggeration 12 for 250 iterations, momentum 0.5→0.8, gains,
learning rate max(n/48, 50), PCA init). Scores: kNN accuracy in feature space and on
the map, and trustworthiness.

- **Why:** penultimate spaces of different networks are not aligned, so one joint map
  (the old report) mostly separated the models from each other. Exact t-SNE on the GPU
  at n ≤ 10k is faster than Barnes-Hut on the CPU and has no approximation error.
- **Rejected:** a joint map of all models (the former answer, for the reason above).

## LRP with the composite rules for CNNs

z^B on the first layer (bounded by the pixel range), LRP-γ on other convolutions, LRP-ε
(relative to each sample's RMS pre-activation) on hidden dense layers, LRP-0 on the
output, winner-takes-all for max pooling, proportional for average pooling. Every rule
is gradient × input of a modified layer via `jax.vjp`, so it works for any model the
layer DSL declares. The gallery picks images round-robin from where the models differ
first; class averages use up to 500 images per class.

- **Why:** the rules Montavon et al. (2019) recommend for CNNs. With zero biases and
  ε → 0 relevance is conserved to 0.15% on all three models (`tests/test_lrp.py`).

## Loss landscape: filter-normalized planes and linear paths

For each model, the loss on a 31 × 31 grid of θ + α·d1 + β·d2 with Gaussian directions
filter-normalized to the weights (biases fixed), α, β ∈ [-1, 1], on 2000 training
images; directions come from the report's seed, so models of one architecture share
them. For pairs of the same architecture, the loss along the straight line between them
(25% past each end) and its barrier. Grid points are scored as a population.

- **Why:** the analysis added to test the plugin path, chosen because it asks the
  project's question directly: whether evolution and SGD end in minima of different
  sharpness, and whether they share a basin. Agent's choice.
- **Note:** the loss rise at r/2 and at r can rank models differently. On MNIST
  (2026-10-17) SGD, OpenAI-ES and LEEA rose 0.81 / 0.78 / 1.13 at r/2 but 13.3 / 7.4 /
  7.5 at r, so the report gives both and the planes rather than one sharpness number.

## Other analyses

Metrics (confusion, per-class F1), calibration (ECE, Brier, NLL), agreement (exact
McNemar), weights (per-tensor statistics and distances), representations (activation
statistics, linear CKA), robustness (noise, brightness, contrast, occlusion, in pixel
space). Each module's docstring states its method.

## Sources

- van der Maaten & Hinton 2008, "Visualizing Data using t-SNE"; Kobak & Berens 2019,
  "The art of using t-SNE for single-cell transcriptomics" (PCA init, learning rate).
- Montavon, Binder, Lapuschkin, Samek & Müller 2019, "Layer-Wise Relevance Propagation:
  An Overview".
- Li, Xu, Taylor, Studer & Goldstein 2018, "Visualizing the Loss Landscape of Neural Nets".
- Frankle, Dziugaite, Roy & Carbin 2020, "Linear Mode Connectivity and the Lottery
  Ticket Hypothesis".
- Kornblith, Norouzi, Lee & Hinton 2019, "Similarity of Neural Network Representations
  Revisited" (CKA).
