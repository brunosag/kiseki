# Terms

Words kiseki uses in code, the CLI, the UI and docs where the obvious one is wrong.

- **Run**: one training of one model by one optimizer on one dataset, from its spec to its
  saved models; not an experiment or a job. It lives on one machine; other machines hold
  mirrors.
- **Spec**: everything that determines a run (dataset, model, optimizer and its params,
  seed, batch size, step count and intervals), frozen when the run is queued; not config.
- **Step**: one update: a minibatch update for SGD, a generation for an evolutionary
  optimizer. A spec's `iterations` counts steps. Not iteration, generation or epoch (an
  epoch is a pass over the training images).
- **Best accuracy**: the best validation, measured on the held-out validation split every
  `validation_interval` steps.
- **Saved model**: weights kept for analysis: `best` (best validation), `final`, and
  `step-N` at each checkpoint step, referred to as `run:name`. A **checkpoint** is only
  the training state a paused or interrupted run resumes from.
- **Model**: a network architecture from the catalog; built for a dataset's image shape
  and classes, it is a `Network` in code, in variables named `net`. Not architecture, and
  not net in prose or the UI.
- **Optimizer**: the training algorithm; not algorithm or trainer.
- **Home**: a directory of runs, reports and their queue (`KISEKI_HOME`), not `$HOME`. A
  machine has its own, one for runs sent from other machines, and any scratch homes.
- **Report**: a set of analyses over saved models on one dataset split; not comparison.
