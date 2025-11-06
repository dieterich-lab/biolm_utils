import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from .structured_config import SalukiConfig


def _validate_task_requirement(cfg: SalukiConfig):
    """Check that --task is provided for relevant modes."""
    if cfg.mode in ["fine-tune", "predict", "interpret"] and not cfg.task:
        raise ValueError(f"Argument 'task' is required when mode is '{cfg.mode}'.")


def _validate_splitratio(cfg: SalukiConfig):
    """Check that --splitratio is a list of 2 or 3 integers summing to 100."""
    if not cfg.data_source or not cfg.data_source.splitratio:
        return
    splitratio = cfg.data_source.splitratio
    if not (
        isinstance(splitratio, list)
        and all(isinstance(item, int) for item in splitratio)
    ):
        raise ValueError("'splitratio' must be a list of integers.")

    if len(splitratio) not in [2, 3]:
        raise ValueError(
            f"'splitratio' must contain 2 (train, val) or 3 (train, val, test) values, but got {len(splitratio)}."
        )

    if sum(splitratio) != 100:
        raise ValueError(
            f"Values in 'splitratio' must sum to 100, but got {sum(splitratio)} for {splitratio}."
        )


def _validate_split_definitions(cfg: SalukiConfig):
    """Check that dev/test splits have the correct structure based on cross-validation."""
    if not cfg.data_source:
        return

    def is_list_of_ints(value: Any) -> bool:
        return isinstance(value, list) and all(isinstance(item, int) for item in value)

    def is_list_of_lists_of_ints(value: Any) -> bool:
        return (
            isinstance(value, list)
            and all(isinstance(sublist, list) for sublist in value)
            and all(isinstance(item, int) for sublist in value for item in sublist)
        )

    for split_name in ["devsplits", "testsplits"]:
        split_value = getattr(cfg.data_source, split_name, None)
        if not split_value:
            continue

        crossvalidation = (
            cfg.data_source.crossvalidation
            if cfg.data_source.crossvalidation
            else False
        )
        if crossvalidation:
            if not is_list_of_lists_of_ints(split_value):
                raise ValueError(
                    f"With 'crossvalidation', '{split_name}' must be a list of lists of integers (e.g., '[[1, 2], [3]]')."
                )
        else:
            if not is_list_of_ints(split_value):
                raise ValueError(
                    f"Without 'crossvalidation', '{split_name}' must be a flat list of integers (e.g., '[1, 2]')."
                )

    if cfg.data_source.testsplits and cfg.data_source.devsplits:
        if not len(cfg.data_source.devsplits) == len(cfg.data_source.testsplits):
            raise ValueError(
                f"With 'testsplits', 'devsplits' and 'testsplits' must have the same length."
            )


def _validate_split_exclusivity(cfg: SalukiConfig):
    """
    Check that exactly one of --splitratio or --splitpos is provided.
    Skip for tokenize mode as it may not need splits.
    """
    # Skip validation for tokenize mode
    if cfg.mode == "tokenize":
        return

    if not cfg.data_source:
        return

    ratio_is_set = cfg.data_source.splitratio is not None
    pos_is_set = cfg.data_source.splitpos is not None

    # Condition 1: Not both can be set.
    if ratio_is_set and pos_is_set:
        raise ValueError(
            f"In mode '{cfg.mode}', 'splitratio' and 'splitpos' are mutually exclusive. Please provide only one.",
        )

    # Condition 2: At least one must be set.
    if not ratio_is_set and not pos_is_set:
        raise ValueError(
            f"Either 'splitratio' or 'splitpos' must be provided to define data splits.",
        )


def _validate_splitpos_dependencies(cfg: SalukiConfig):
    """Check that if --splitpos is given, --devsplits is also provided."""
    if not cfg.data_source:
        return
    if cfg.data_source.splitpos is not None and cfg.data_source.devsplits is None:
        raise ValueError(
            "Argument 'devsplits' is required when 'splitpos' is provided.",
        )


def _validate_all_args(cfg: SalukiConfig):
    """Runs all validation checks on the parsed arguments."""
    _validate_task_requirement(cfg)
    _validate_splitratio(cfg)
    _validate_split_definitions(cfg)
    _validate_split_exclusivity(cfg)
    _validate_splitpos_dependencies(cfg)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def parse_args(cfg: DictConfig) -> argparse.Namespace:
    """
    Parses configuration using Hydra and returns argparse.Namespace for backward compatibility.
    """
    from .structured_config import (
        DataSourceConfig,
        DebuggingConfig,
        InferenceConfig,
        SettingsConfig,
        TokenizationConfig,
        TrainingConfig,
    )

    # Convert DictConfig to container
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Manually instantiate nested dataclasses
    data_source = (
        DataSourceConfig(**config_dict["data_source"])
        if "data_source" in config_dict
        else None
    )
    tokenization = (
        TokenizationConfig(**config_dict["tokenization"])
        if "tokenization" in config_dict
        else None
    )
    training = (
        TrainingConfig(**config_dict["training"]) if "training" in config_dict else None
    )
    inference = (
        InferenceConfig(**config_dict["inference"])
        if "inference" in config_dict
        else None
    )
    settings = (
        SettingsConfig(**config_dict["settings"])
        if "settings" in config_dict and config_dict["settings"]
        else None
    )
    debugging = (
        DebuggingConfig(**config_dict["debugging"])
        if "debugging" in config_dict
        else DebuggingConfig()
    )

    # Create SalukiConfig instance
    saluki_cfg = SalukiConfig(
        mode=config_dict.get("mode", "tokenize"),
        outputpath=config_dict.get("outputpath"),
        task=config_dict.get("task"),
        data_source=data_source,
        tokenization=tokenization,
        training=training,
        inference=inference,
        settings=settings,
        debugging=debugging,
    )

    # Validate the configuration
    _validate_all_args(saluki_cfg)

    # Convert to namespace for backward compatibility
    return saluki_cfg.to_namespace()
