# TODO

- Tune LEEA's and CoSyNE's defaults: in 300 MNIST steps they reach 84.9% and 63.0% test accuracy, OpenAI-ES 96.55% (`docs/design/training.md`, Optimizers).
- Test bf16 on CIFAR-10 in the performance profile, and remove `precision` if it gains nothing (`docs/design/training.md`, Precision).
- Revisit the CUDA 13.0 pin: both its triggers have fired (lugia's GPU failed again with 13.0; its driver is now 615) (`docs/design/architecture.md`, CUDA user-space libraries).
- Publish the first AppImage (no `v<version>` tag yet) and watch its first install on the Ubuntu laptop.
