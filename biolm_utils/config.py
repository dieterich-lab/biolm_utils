from dataclasses import dataclass

from transformers import (
    DefaultDataCollator,
    PretrainedConfig,
    PreTrainedModel,
    XLNetTokenizerFast,
)
from transformers.image_processing_utils import ImageProcessingMixin

# from biolm_utils.rna_datasets import RNABaseDataset


@dataclass
class Config:
    MODEL_CLS_FOR_PRETRAINING: PreTrainedModel  # 0
    MODEL_CLS_FOR_FINETUNING: PreTrainedModel  # 1
    TOKENIZER_CLS: XLNetTokenizerFast  # 2
    # DATASET_CLS: RNABaseDataset  # 3
    LEARNINGRATE: float  # 4
    MAX_GRAD_NORM: float  # 5
    WEIGHT_DECAY: float  # 6
    SPECIAL_TOKENIZER_FOR_TRAINER_CLS: ImageProcessingMixin  # 7
    DATACOLLATOR_CLS_FOR_PRETRAINING: DefaultDataCollator  # 8
    DATACOLLATOR_CLS_FOR_FINETUNING: DefaultDataCollator  # 9
    ADD_SPECIAL_TOKENS: bool  # 10
    CONFIG_CLS: PretrainedConfig  # 11
    PRETRAINING_REQUIRED: bool  # 12


_config: Config | None = None


def get_config():
    global _config

    if _config is None:
        raise Exception("Config not initialized")
    return _config


def set_config(config: Config):
    global _config
    _config = config
