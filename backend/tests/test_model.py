import torch

from kiseki.models import CIFARResNet20, CNN2C2DMNIST, count_parameters
from kiseki.optimizers import LEEAConfig, LEEARunner


def test_model_output_shape_and_parameter_count() -> None:
    model = CNN2C2DMNIST()
    output = model(torch.randn(4, 1, 28, 28))

    assert output.shape == (4, 10)
    assert count_parameters(model) == 50578


def test_model_named_final_hidden_activation_shape() -> None:
    model = CNN2C2DMNIST()
    activations = model.named_activations(torch.randn(4, 1, 28, 28), ("fc1_relu",))

    assert activations["fc1_relu"].shape == (4, 120)


def test_cifar_resnet20_output_shape_hidden_activation_and_parameter_count() -> None:
    model = CIFARResNet20()
    inputs = torch.randn(4, 3, 32, 32)
    output = model(inputs)

    assert output.shape == (4, 10)
    assert model.final_hidden(inputs).shape == (4, 64)
    assert model.named_activations(inputs, ("final_hidden",))["final_hidden"].shape == (4, 64)
    assert count_parameters(model) == 269722


def test_leea_can_step_cifar_resnet20_population() -> None:
    model = CIFARResNet20()
    runner = LEEARunner(
        model,
        LEEAConfig(population_size=2, evaluation_chunk_size=1),
        device=torch.device("cpu"),
        seed=3,
    )

    loss = runner.step(torch.randn(2, 3, 32, 32), torch.tensor([1, 2]))

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))
