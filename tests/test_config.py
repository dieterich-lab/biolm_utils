from unittest.mock import MagicMock

import pytest

from biolm_utils.config import Config, get_config, set_config


class TestConfig:
    def test_set_and_get_config(self):
        # Create a mock config
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_dataset = MagicMock()
        mock_config_cls = MagicMock()
        mock_data_collator = MagicMock()

        config = Config(
            MODEL_CLS_FOR_PRETRAINING=mock_model,
            MODEL_CLS_FOR_FINETUNING=mock_model,
            TOKENIZER_CLS=mock_tokenizer,
            LEARNINGRATE=1e-3,
            MAX_GRAD_NORM=0.4,
            WEIGHT_DECAY=0.001,
            SPECIAL_TOKENIZER_FOR_TRAINER_CLS=MagicMock(),
            DATACOLLATOR_CLS_FOR_PRETRAINING=mock_data_collator,
            DATACOLLATOR_CLS_FOR_FINETUNING=mock_data_collator,
            ADD_SPECIAL_TOKENS=False,
            CONFIG_CLS=mock_config_cls,
            PRETRAINING_REQUIRED=False,
            DATASET_CLS=mock_dataset,
        )

        set_config(config)
        retrieved = get_config()
        assert retrieved == config

    def test_get_config_without_set_raises_exception(self):
        # Reset config
        import biolm_utils.config

        biolm_utils.config._config = None

        with pytest.raises(Exception, match="Config not initialized"):
            get_config()
