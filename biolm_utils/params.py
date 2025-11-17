import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf, ValidationError

from .structured_config import BioLMConfig

logger = logging.getLogger(__name__)


def _validate_config(cfg: BioLMConfig) -> None:
    """
    Validate the configuration based on the selected mode.
    Uses comprehensive validation with clear error messages.
    """
    # Task validation - required for training and inference modes
    if cfg.mode in ["fine-tune", "predict", "interpret"]:
        if not cfg.task:
            raise ValueError(
                f"task is required when mode='{cfg.mode}'. "
                f"Valid tasks depend on the specific model plugin."
            )

    # Skip data split validation for modes that don't need it
    if cfg.mode in ["tokenize", "predict", "interpret"]:
        return

    # Data split validation for training modes
    if not cfg.data_source:
        logger.warning(
            f"No data_source specified for mode '{cfg.mode}'. "
            "This may cause issues during training."
        )
        return

    # Validate splitratio if provided
    if cfg.data_source.splitratio is not None:
        splitratio = cfg.data_source.splitratio
        if not (isinstance(splitratio, list) and len(splitratio) >= 2):
            raise ValueError(
                f"data_source.splitratio must be a list of at least 2 integers, got {splitratio}"
            )
        if not all(isinstance(item, int) and item > 0 for item in splitratio):
            raise ValueError(
                f"All values in data_source.splitratio must be positive integers, got {splitratio}"
            )
        if len(splitratio) not in [2, 3]:
            raise ValueError(
                f"data_source.splitratio must contain 2 or 3 values, got {len(splitratio)}"
            )
        if sum(splitratio) != 100:
            raise ValueError(
                f"Values in data_source.splitratio must sum to 100, got {sum(splitratio)}"
            )

    # Validate split exclusivity and requirements
    ratio_is_set = cfg.data_source.splitratio is not None
    pos_is_set = cfg.data_source.splitpos is not None

    if ratio_is_set and pos_is_set:
        raise ValueError(
            f"In mode '{cfg.mode}', data_source.splitratio and data_source.splitpos are mutually exclusive"
        )

    if not ratio_is_set and not pos_is_set:
        raise ValueError(
            f"Either data_source.splitratio or data_source.splitpos must be provided for mode '{cfg.mode}'"
        )

    # Validate splitpos dependencies
    if cfg.data_source.splitpos is not None:
        if cfg.data_source.devsplits is None:
            raise ValueError(
                "data_source.devsplits is required when data_source.splitpos is provided"
            )
        if (
            not isinstance(cfg.data_source.splitpos, int)
            or cfg.data_source.splitpos < 0
        ):
            raise ValueError(
                f"data_source.splitpos must be a non-negative integer, got {cfg.data_source.splitpos}"
            )

    # Validate split definitions for cross-validation
    if cfg.data_source.crossvalidation:
        for split_name in ["devsplits", "testsplits"]:
            split_value = getattr(cfg.data_source, split_name, None)
            if split_value is not None:
                if not (
                    isinstance(split_value, list)
                    and all(isinstance(sublist, list) for sublist in split_value)
                ):
                    raise ValueError(
                        f"With data_source.crossvalidation=True, {split_name} must be a list of lists"
                    )
                if not all(
                    isinstance(item, int) for sublist in split_value for item in sublist
                ):
                    raise ValueError(
                        f"With data_source.crossvalidation=True, {split_name} must contain only integers"
                    )


def load_config(overrides: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Load configuration using Hydra programmatically with best practices.

    Args:
        overrides: List of configuration overrides in the format 'key=value'

    Returns:
        argparse.Namespace for backward compatibility with existing code

    Raises:
        ValueError: If configuration is invalid or cannot be loaded
        RuntimeError: If Hydra initialization fails
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf, open_dict

    config_path = Path(__file__).parent / "conf"

    try:
        # Initialize Hydra with proper version base
        with initialize_config_dir(config_dir=str(config_path), version_base="1.1"):
            # Separate mode override from other overrides
            mode = "tokenize"
            other_overrides = []

            if overrides:
                for override in overrides:
                    if override.startswith("mode="):
                        mode = override.split("=", 1)[1]
                    else:
                        other_overrides.append(override)

            # Load base config
            cfg = compose(config_name="config", overrides=[])

            # Load and merge mode-specific config
            try:
                mode_cfg = compose(config_name=f"mode/{mode}", overrides=[])
                cfg = OmegaConf.merge(cfg, mode_cfg)
            except Exception as e:
                logger.warning(f"Could not load mode config for '{mode}': {e}")

            # Apply other overrides after mode merge
            for override in other_overrides:
                key, value = override.split("=", 1)
                try:
                    with open_dict(cfg):
                        OmegaConf.update(cfg, key, value, merge=True)
                except Exception as e:
                    logger.warning(f"Failed to apply override '{override}': {e}")

            # Ensure mode is set correctly
            with open_dict(cfg):
                cfg.mode = mode

            # Process and validate
            return _process_hydra_config(cfg)

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise RuntimeError(f"Configuration loading failed: {e}") from e
    finally:
        # Clean up Hydra instance
        GlobalHydra.instance().clear()


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def parse_args(cfg: DictConfig) -> argparse.Namespace:
    """
    Parses configuration using Hydra CLI and returns argparse.Namespace for backward compatibility.

    This is the entry point when called via Hydra CLI (e.g., python script.py mode=fine-tune task=regression).

    Args:
        cfg: Hydra DictConfig object

    Returns:
        argparse.Namespace for backward compatibility
    """
    try:
        return _process_hydra_config(cfg)
    except Exception as e:
        logger.error(f"Configuration parsing failed: {e}")
        raise


def _process_hydra_config(cfg: DictConfig) -> argparse.Namespace:
    """
    Process a Hydra DictConfig into our structured config and return argparse.Namespace.

    Args:
        cfg: Hydra configuration object

    Returns:
        argparse.Namespace for backward compatibility

    Raises:
        ValueError: If configuration validation fails
        RuntimeError: If config processing fails
    """
    from .structured_config import (
        DataSourceConfig,
        DebuggingConfig,
        InferenceConfig,
        SettingsConfig,
        TokenizationConfig,
        TrainingConfig,
    )

    try:
        # Convert DictConfig to container with resolution
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(config_dict, dict):
            raise RuntimeError("Configuration must resolve to a dictionary")

        # Safely instantiate nested dataclasses with error handling
        def safe_instantiate(cls, data, name):
            if data is None:
                return None
            try:
                return cls(**data)
            except TypeError as e:
                raise ValueError(f"Invalid {name} configuration: {e}") from e

        data_source = safe_instantiate(
            DataSourceConfig, config_dict.get("data_source"), "data_source"
        )
        tokenization = safe_instantiate(
            TokenizationConfig, config_dict.get("tokenization"), "tokenization"
        )
        training = safe_instantiate(
            TrainingConfig, config_dict.get("training"), "training"
        )
        inference = safe_instantiate(
            InferenceConfig, config_dict.get("inference"), "inference"
        )
        settings = safe_instantiate(
            SettingsConfig,
            config_dict.get("settings") if config_dict.get("settings") else None,
            "settings",
        )
        debugging = (
            safe_instantiate(DebuggingConfig, config_dict.get("debugging"), "debugging")
            or DebuggingConfig()
        )

        # Create BioLMConfig instance
        biolm_cfg = BioLMConfig(
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

        # Auto-detect number of GPUs if using GPU accelerator and ngpus is default
        if biolm_cfg.debugging.accelerator == "gpu" and biolm_cfg.debugging.ngpus == 1:
            # Check if ngpus was explicitly set in config (either in debugging or settings.environment for backward compatibility)
            ngpus_explicitly_set = "ngpus" in config_dict.get("debugging", {}) or (
                config_dict.get("settings")
                and "environment" in config_dict["settings"]
                and "ngpus" in config_dict["settings"]["environment"]
            )

            if not ngpus_explicitly_set:
                try:
                    import torch

                    detected_gpus = torch.cuda.device_count()
                    if detected_gpus > 0:
                        biolm_cfg.debugging.ngpus = detected_gpus
                        logger.info(f"Auto-detected {detected_gpus} GPU(s) available")
                    else:
                        logger.warning(
                            "GPU accelerator selected but no GPUs detected, falling back to CPU"
                        )
                        biolm_cfg.debugging.accelerator = "cpu"
                except ImportError:
                    logger.warning("PyTorch not available for GPU detection, using CPU")
                    biolm_cfg.debugging.accelerator = "cpu"

        # Validate the configuration
        _validate_config(biolm_cfg)

        # Convert to namespace for backward compatibility
        return biolm_cfg.to_namespace()

    except ValidationError as e:
        raise ValueError(f"Configuration validation failed: {e}") from e
    except Exception as e:
        logger.error(f"Configuration processing failed: {e}")
        raise RuntimeError(f"Failed to process configuration: {e}") from e
