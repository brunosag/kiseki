import torch

from kiseki.models import CIFAR10CNN, CNN2C2DMNIST, count_parameters
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


def test_cifar10_cnn_output_shape_hidden_activation_and_parameter_count() -> None:
    model = CIFAR10CNN()
    inputs = torch.randn(4, 3, 32, 32)
    output = model(inputs)

    assert output.shape == (4, 10)
    assert model.final_hidden(inputs).shape == (4, 32)
    assert model.named_activations(inputs, ("final_hidden",))["final_hidden"].shape == (4, 32)
    probabilities = model.predict_proba(inputs)
    assert probabilities.shape == (4, 10)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1e-6)
    assert count_parameters(model) == 18514


def test_leea_can_step_cifar10_cnn_population() -> None:
    model = CIFAR10CNN()
    runner = LEEARunner(
        model,
        LEEAConfig(population_size=2, evaluation_chunk_size=1),
        device=torch.device("cpu"),
        seed=3,
    )

    loss = runner.step(torch.randn(2, 3, 32, 32), torch.tensor([1, 2]))

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))
