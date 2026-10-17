# Training

How a run trains: the compiled loop in `kiseki/worker/train.py`, the optimizers in
`kiseki/optimizers/`, and the datasets and models they use.

## Many steps per host round trip

One compiled block is a `lax.while_loop` over steps with a runtime step count (at most
1024), so a single compilation serves every block. Inside it: minibatches drawn from the
on-device training set, the optimizer step, validation every `validation_interval` steps
(and at the last), best-network tracking, and the target-accuracy stop. Per-step metrics
go to fixed device buffers the host reads once per block. Blocks aim at about 2 s,
growing by at most 8× per block, and end at checkpoint steps. The carry is donated.

- **Why:** the old loop waited on the GPU every step; here the host touches the device
  about every 2 s whatever the step rate ("Everything is" optimized for performance,
  PREFERENCES.md). Agent's design.
- **Rejected:** one compiled step per call (host sync per step); fixed-size blocks
  (recompile per size, or blocks too long for responsive pause and progress).

## Data lives on the device; a run is a function of (seed, step)

Each split is uploaded once. Each epoch's order is a permutation keyed by the run's seed
and the epoch; the validation split comes from `split_seed`, shared by all runs on a
dataset so their curves compare. Step keys are `fold_in(step_key, step)`.

- **Why:** no input pipeline; resuming at any step reproduces the same batches, so a
  paused-and-resumed run equals an uninterrupted one bit for bit
  (`tests/test_train.py`).

## Populations are scored in chunks sized to GPU memory

Evolutionary optimizers score networks `chunk` at a time, vmapped over one minibatch. The
first chunk targets 30% of device memory for activations (1 to 256 networks); compiling
checks the plan against 85% of the device, and an out-of-memory first block halves the
chunk and retries. OpenAI-ES regenerates each perturbation from its key in both passes
instead of storing it, so memory is O(chunk · weights) for any population.

- **Why:** the largest chunk that fits is the fastest; failures fall back without the
  user picking numbers.

## Checkpoints are the flattened carry

`state/` holds the whole loop carry as safetensors plus an atomic manifest; files are
written on a background thread. On resume, metric rows past the checkpoint (and a torn
last line) are truncated. Saved models are separate files: `best`, `final`, `step-N`.

- **Why:** a crash or kill at any moment leaves either the old or the new checkpoint,
  never half of one; the GPU never waits for the disk.

## Optimizers

SGD (heavy-ball momentum, L2), LEEA (Morse & Stanley 2016), CoSyNE (Gomez et al. 2008)
and OpenAI-ES (Salimans et al. 2017), each a package with `Params` and a pure `build()`
(`kiseki/worker/optimizer.py` is the interface).

- **OpenAI-ES** is the algorithm added to test the plugin path: the standard scalable
  evolution strategy, and the closest evolutionary counterpart to SGD (it follows a
  gradient estimate), so it separates "gradient-free" from "gradient-following" in the
  comparison. Mirrored sampling, centred ranks, Adam. Agent's choice.
- **Bet:** OpenAI-ES's defaults (σ 0.02, Adam 0.01, population 1000) learn MNIST to
  ≥ 96% in 300 steps. Result 2026-10-17: 96.55% test (best validation 95.64%); LEEA 84.9%,
  CoSyNE 63.0% in the same 300 steps at their defaults. Tuning LEEA and CoSyNE is open.

## Precision: fp32 by default, bf16 per run

`precision: bf16` runs population scoring in bfloat16 (1.4–2.3× faster in the stack
benchmark).

- **Bet:** bf16 selects networks as fp32 does. Test: the same LEEA run in both, comparing
  validation curves. Not yet run; until then fp32 stays the default.

## Datasets and models

MNIST, Fashion-MNIST, KMNIST and CIFAR-10, downloaded from their original hosts with
pinned SHA-256 hashes and parsed without pickle or archive extraction to disk. Models
small_cnn, wide_cnn and vgg_mini are layer lists (`kiseki/models/layers.py`) from which
init, forward pass, activation taps and LRP all derive.

- **KMNIST** is the dataset added to test the plugin path: same shape and cost as MNIST
  but much harder, so optimizers that tie near 99% on MNIST separate. Agent's choice.

## Sources

- Morse & Stanley 2016, "Simple Evolutionary Optimization Can Rival Stochastic Gradient
  Descent in Neural Networks" (LEEA).
- Gomez, Schmidhuber & Miikkulainen 2008, "Accelerated Neural Evolution through
  Cooperatively Coevolved Synapses" (CoSyNE).
- Salimans, Ho, Chen, Sidor & Sutskever 2017, "Evolution Strategies as a Scalable
  Alternative to Reinforcement Learning" (OpenAI-ES).
- Clanuwat et al. 2018, "Deep Learning for Classical Japanese Literature" (KMNIST).
