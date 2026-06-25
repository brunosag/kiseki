# `kiseki`

A PyTorch-based machine learning framework designed to investigate whether gradient-based and evolutionary optimization algorithms produce distinct learning patterns and internal representations in structurally identical artificial neural networks. By facilitating the training and comparison of models across both paradigms, the framework aims to expand deep learning theory and clarify how the choice of optimizer influences network behavior, decision rules, and interpretability. The repository features a command-line interface for headless execution and a reactive web application for real-time telemetry.

> **/kiseki/** `noun`
> 
> 1. 軌跡: locus, trajectory, or path; traces left by a process.
> 2. 奇跡: miracle, wonder, or marvel.

## Architecture and core modules

* **Models:** Neural network architectures are implemented in PyTorch. The backend supports MNIST and CIFAR-10 models.
* **Optimizers:** The framework exports `SGD` for standard gradient descent and `LEEA`, a population-based evolutionary optimization algorithm with mutation, crossover, retention, and fitness decay controls.
* **Data handling:** The backend loads MNIST and CIFAR-10 through torchvision and creates deterministic streams when exact reproducibility is requested.
* **Experiment management:** `ExperimentManager` coordinates model initialization, CPU/GPU device mapping, batch sizing, checkpointing, pause/resume, and early stopping based on target accuracy.
* **Checkpoints:** Training checkpoints are stored under `backend/checkpoints/` with model state, optimizer state, status, config, runtime metadata, and JSON summaries.

## Interfaces

### Local development

Run the backend and frontend together from the repository root:

```bash
npm run dev
```

This starts the FastAPI backend at `http://127.0.0.1:8000` and the Vite frontend
at `http://127.0.0.1:5173`. The frontend dev server proxies `/api` requests to
the backend. Install frontend dependencies first with
`npm --prefix frontend install` if `frontend/node_modules` is not present.

On NixOS, run `direnv allow` once from the repository root before `npm run dev`
so CUDA driver libraries and the system CA bundle are exported before the
backend Python process starts.

### Web dashboard

The repository features a reactive web user interface built with GenieFramework and StippleLatex. The application (`scripts/app.jl`) allows users to:

* Configure experiment hyperparameters, including dataset parameters, hardware device, and optimizer-specific variables (e.g., learning rate for SGD, or population size and initial mutation step size for LEEA).
* Visualize real-time training telemetry, plotting loss and accuracy metrics onto Plotly-based graphs.
* Start, interrupt, and monitor experiments asynchronously.

### Command-line interface

For SSH-safe headless training, start the tmux helper from the repository root:

```bash
npm run train -- --device gpu --optimizer LEEA --iterations 100000
```

This creates and attaches to a `kiseki-train` tmux session, runs
`uv run kiseki train` from `backend/`, and writes a matching log under `logs/`.
Detach with `Ctrl-b d`; reattach with `tmux attach -t kiseki-train`.

Resume from a checkpoint:

```bash
npm run train -- --resume <run_id>
```

Run the Python CLI directly from `backend/` when tmux is not needed:

```bash
uv run kiseki train --device cpu --optimizer SGD --iterations 100
```

## Development Roadmap

* [x] Implement scalar metric telemetry for continuous loss and validation accuracy tracking.
* [x] Develop binary state serialization and JSON-based metadata logging for experiment persistence.
* [x] Build a reactive web dashboard for real-time hyperparameter configuration.
* [x] Integrate live graphical rendering of training trajectories.
* [ ] Create specialized hooks for high-dimensional data extraction, including intermediate neuronal activations and weight configurations.
* [ ] Develop automated visualization modules for comparative trajectory overlays between gradient-based and evolutionary runs.
* [ ] Implement dimensionality reduction views, such as t-SNE or PCA, for analyzing internal representation spaces.
* [ ] Script set-theoretic classification divergence modules to compare prediction sets across optimizers.
* [ ] Integrate post-hoc interpretability algorithms, including LRP, DeepLIFT, and Shapley values, for relevance attribution.
* [ ] Implement shortcut learning evaluation protocols to test generalization robustness.
* [ ] Engineer an automated report compiler to aggregate metrics and visual renders into static exportable formats.
* [ ] Draft comprehensive technical documentation covering the API, interpretability tools, and training replication.

## Dependencies

* **PyTorch & torchvision:** Neural network training, GPU acceleration, and dataset loading.
* **FastAPI:** Backend API and server-sent experiment events.
* **React & Vite:** Dashboard frontend.
