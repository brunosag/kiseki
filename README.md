# `kiseki`

A Julia-based machine learning framework designed to investigate whether gradient-based and evolutionary optimization algorithms produce distinct learning patterns and internal representations in structurally identical artificial neural networks. By facilitating the training and comparison of models across both paradigms, the framework aims to expand deep learning theory and clarify how the choice of optimizer influences network behavior, decision rules, and interpretability. The repository features a command-line interface for headless execution and a reactive web application for real-time telemetry.  

> **/kiseki/** `noun`
> 
> 1. 軌跡: locus, trajectory, or path; traces left by a process.
> 2. 奇跡: miracle, wonder, or marvel.

## Architecture and core modules

* **Models:** Neural network architectures are implemented utilizing the Lux.jl ecosystem. The default reference model is a ~50K-parameter Convolutional Neural Network (`CNN_2C2D_MNIST`) based on LeNet-5.
* **Optimizers:** The framework exports `SGD` for standard gradient descent via Zygote.jl and `LEEA`, a population-based evolutionary optimization algorithm. `LEEA` maintains a parameter matrix, applying asynchronous reproduction mechanisms such as mutation, crossover, and fitness decay.
* **Data handling:** The system implements a custom `BalancedDataLoader` to ensure proportional class representation during training batches. The data ingestion pipeline defaults to the MNIST dataset via MLDatasets.jl.
* **Experiment management:** The `Experiment` struct coordinates model initialization, hardware device mapping (CPU or GPU via LuxCUDA), batch sizing, and early stopping based on defined target accuracy thresholds.
* **Callbacks:** Training loops support extensible callbacks, including `ConsoleLogger` for standard output telemetry, `Tracker` for in-memory metric history, and `CheckpointSaver` for serialized state persistence using JSON metadata and JLS files.

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

The `scripts/train.jl` script provides a robust CLI via ArgParse for batch or remote execution. Supported arguments include device selection (`--device`), optimizer designation (`--optimizer`), execution length (`--iterations`), validation 
frequency (`--val-freq`), and state resumption from checkpoint files (`--resume`).

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
* [ ] Draft comprehensive technical documentation covering the API, interpretability tools, and benchmark replication.

## Dependencies

* **Lux.jl & LuxCUDA.jl:** Neural network parameterization and GPU acceleration.
* **Zygote.jl:** Algorithmic differentiation for the gradient-based SGD optimizer.
* **GenieFramework:** Web server execution and reactive frontend.
* **PlotlyBase:** Graphical metric visualization.
