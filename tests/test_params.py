from unittest.mock import patch

import pytest

from biolm_utils.params import _validate_config
from biolm_utils.structured_config import BioLMConfig, DataSourceConfig


class TestParams:
    def test_validate_config_tokenize_mode_no_validation(self):
        cfg = BioLMConfig(mode="tokenize")
        # Should not raise
        _validate_config(cfg)

    def test_validate_config_fine_tune_requires_task(self):
        cfg = BioLMConfig(mode="fine-tune", task=None)
        with pytest.raises(ValueError, match="task is required"):
            _validate_config(cfg)

    def test_validate_config_predict_requires_task(self):
        cfg = BioLMConfig(mode="predict", task=None)
        with pytest.raises(ValueError, match="task is required"):
            _validate_config(cfg)

    def test_validate_config_interpret_requires_task(self):
        cfg = BioLMConfig(mode="interpret", task=None)
        with pytest.raises(ValueError, match="task is required"):
            _validate_config(cfg)

    def test_validate_config_fine_tune_with_task_valid(self):
        cfg = BioLMConfig(mode="fine-tune", task="regression")
        # Should not raise
        _validate_config(cfg)

    def test_validate_config_splitratio_invalid_length(self):
        data_source = DataSourceConfig(splitratio=[50])
        cfg = BioLMConfig(mode="fine-tune", task="regression", data_source=data_source)
        with pytest.raises(ValueError, match="must be a list of at least 2 integers"):
            _validate_config(cfg)

    def test_validate_config_splitratio_not_sum_100(self):
        data_source = DataSourceConfig(splitratio=[50, 30, 30])
        cfg = BioLMConfig(mode="fine-tune", task="regression", data_source=data_source)
        with pytest.raises(ValueError, match="must sum to 100"):
            _validate_config(cfg)

    def test_validate_config_splitratio_valid(self):
        data_source = DataSourceConfig(splitratio=[80, 10, 10])
        cfg = BioLMConfig(mode="fine-tune", task="regression", data_source=data_source)
        # Should not raise
        _validate_config(cfg)

    def test_validate_config_splitpos_without_devsplits(self):
        data_source = DataSourceConfig(splitpos=3, devsplits=None)
        cfg = BioLMConfig(mode="fine-tune", task="regression", data_source=data_source)
        with pytest.raises(ValueError, match="devsplits is required"):
            _validate_config(cfg)

    def test_validate_config_splitpos_valid(self):
        data_source = DataSourceConfig(splitpos=3, devsplits=[1, 2])
        cfg = BioLMConfig(mode="fine-tune", task="regression", data_source=data_source)
        # Should not raise
        _validate_config(cfg)
