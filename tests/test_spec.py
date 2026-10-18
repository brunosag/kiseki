import pytest

from kiseki import actions
from kiseki.spec import ModelRef, RunSpec, from_overrides, normalize


def test_overrides_route_to_fields_and_params():
    spec = from_overrides(RunSpec(), {"optimizer": "leea", "population_size": 100, "seed": 3})
    assert spec.optimizer == "leea"
    assert spec.seed == 3
    assert spec.params["population_size"] == 100


def test_unknown_setting_is_rejected():
    with pytest.raises(KeyError):
        from_overrides(RunSpec(), {"optimizer": "sgd", "population_size": 10})


def test_normalize_fills_dataset_defaults():
    spec = normalize(RunSpec(dataset="mnist", optimizer="sgd"))
    assert spec.model == "small_cnn"
    assert spec.validation_size == 10000
    assert spec.params["learning_rate"] > 0


def test_grid_with_preset_drops_population_for_sgd():
    specs = actions.build_specs({"dataset": "mnist"}, {"optimizer": ["sgd", "leea"]}, "smoke")
    assert [s.optimizer for s in specs] == ["sgd", "leea"]
    assert all(s.iterations == 20 for s in specs)
    assert "population_size" not in specs[0].params
    assert specs[1].params["population_size"] == 32


def test_grid_is_a_product():
    specs = actions.build_specs({}, {"seed": [1, 2, 3], "batch_size": [64, 128]})
    assert sorted((s.seed, s.batch_size) for s in specs) == [
        (1, 64),
        (1, 128),
        (2, 64),
        (2, 128),
        (3, 64),
        (3, 128),
    ]


def test_model_refs():
    assert ModelRef.parse("abcd") == ModelRef("abcd", "best")
    assert str(ModelRef.parse("abcd:step-100")) == "abcd:step-100"
