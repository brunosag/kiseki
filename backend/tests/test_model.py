import torch

from kiseki.models import CNN2C2DMNIST, count_parameters


def test_model_output_shape_and_parameter_count() -> None:
    model = CNN2C2DMNIST()
    output = model(torch.randn(4, 1, 28, 28))

    assert output.shape == (4, 10)
    assert count_parameters(model) == 50578
