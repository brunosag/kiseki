# kiseki

Training small image classifiers with gradient descent or with evolution, and comparing
what each produces.

## Runs

**Run**:
One training of one model by one optimizer on one dataset, from its spec to its saved
models. Lives on one machine; other machines hold mirrors.
_Avoid_: experiment (for a single run), job

**Spec**:
Everything that determines a run: dataset, model, optimizer and its params, seed, batch
size, step count and intervals. Frozen when the run is queued.
_Avoid_: config

**Step**:
One update: one minibatch update for SGD, one generation for an evolutionary optimizer.
A spec's `iterations` counts steps.
_Avoid_: iteration, generation, epoch (an epoch is a pass over the training images)

**Validation**:
Measuring the current network on the held-out validation split, every
`validation_interval` steps. Best accuracy is the best validation.

**Checkpoint**:
The training state a paused or interrupted run resumes from.
_Avoid_: using it for saved models

**Saved model**:
A run's network weights kept for analysis: `best` (best validation), `final`, and
`step-N` at each checkpoint step. Referred to as `run:name`.

**Preset**:
Named spec overrides; `smoke` trains for seconds to check that a configuration runs.

## Training

**Model**:
A network architecture from the catalog (small_cnn, wide_cnn, vgg_mini). Built for a
dataset's image shape and classes, it is a `Network` in code, in variables named `net`.
_Avoid_: architecture; net (in prose and the UI)

**Optimizer**:
The training algorithm: SGD, LEEA, CoSyNE or OpenAI-ES.
_Avoid_: algorithm, trainer

**Population**:
The networks an evolutionary optimizer scores each step (OpenAI-ES: the perturbations of
its centre).

**Chunk**:
How many networks are scored together on the device; chosen to fit GPU memory.

**Block**:
A stretch of steps compiled and run on the device between two reads by the host.

## Machines

**Machine**:
Where runs execute: `local` (this laptop) or a remote reached over SSH (lugia, vps).

**Queue runner**:
The per-machine process that starts that machine's queued runs one at a time.

**Notice**:
The block in a shared server's message file (lugia's /etc/motd.tail) naming who is using
its GPU and until when.

## Reports

**Report**:
A set of analyses over saved models on one dataset split, with its numbers, prose and
figures.
_Avoid_: comparison

**Analysis**:
One method in a report (calibration, LRP, loss landscape, ...), producing one section.
