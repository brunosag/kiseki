# TODO

- Tune LEEA's and CoSyNE's defaults: in 300 MNIST steps they reach 84.9% and 63.0% test
  accuracy, OpenAI-ES 96.55% (`docs/design/training.md`, Optimizers).
- Test bf16 on CIFAR-10 in the performance profile: time scoring in fp32 and bf16, then
  the same LEEA run in both, comparing validation curves; remove `precision` if it gains
  nothing (`docs/design/training.md`, Precision).
- Revisit the CUDA 13.0 pin: both of its triggers have fired, since lugia's GPU failed
  again with 13.0 and its driver is now 615 (`docs/design/architecture.md`, CUDA
  user-space libraries; the lugia gotcha in `AGENTS.md`).
- Publish the first AppImage (no `v<version>` tag exists yet) and watch its first real
  install on the Ubuntu laptop (`docs/design/architecture.md`, Other machines run an
  AppImage).
