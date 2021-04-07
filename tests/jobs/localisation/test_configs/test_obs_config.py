from semeio.workflows.localisation.localisation_config import ObsConfig


def test_obs_config():
    config_dict = {"add": "*", "context": ["a", "b", "c"]}

    config = ObsConfig(**config_dict)
    assert config.result_items == ["a", "b", "c"]
