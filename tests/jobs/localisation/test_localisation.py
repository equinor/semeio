from semeio.workflows.localisation.local_config_scalar import create_ministep
from unittest.mock import Mock


def test_create_ministep():
    local_mock = Mock()
    mock_ministep = Mock()
    mock_model_group = Mock()
    mock_obs_group = Mock()
    mock_update_step = Mock()

    local_mock.createMinistep.return_value = mock_ministep
    mock_ministep.attachDataset.return_value = None
    mock_ministep.attachObsset.return_value = None
    mock_model_group.name.return_value = "OBS"
    mock_obs_group.name.return_value = "PARA"

    obs = "OBS"
    para = "PARA"
    create_ministep(
        local_config=local_mock,
        mini_step_name="TEST",
        model_group=mock_model_group,
        obs_group=mock_obs_group,
        updatestep=mock_update_step,
    )
    assert local_mock.createMinistep.called_once()
    assert mock_ministep.attachDataset.called_once_with(para)
    assert mock_ministep.attachObsset.called_once_with(obs)
    assert mock_update_step.attachMinistep.called_once_with(mock_ministep)
