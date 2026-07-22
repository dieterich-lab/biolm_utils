"""Hydra-backed configuration loader moved out of params.py.

This module exposes load_config() and parse_args() (the latter as a hydra
entrypoint). The implementation is intentionally small and delegates
validation/autodetection to the structured dataclass.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from .plugin_config import merge_plugin_defaults
from .structured_config import (
    BioLMConfig,
    DataSourceConfig,
    DebuggingConfig,
    InferenceConfig,
    SettingsConfig,
    TokenizationConfig,
    TrainingConfig,
)


def _process_hydra_config(cfg: DictConfig) -> BioLMConfig:
    """Turn a resolved DictConfig into our structured BioLMConfig.

    This mirrors the previous logic from params._process_hydra_config but is
    extracted here so `params.py` can become a thin re-exporting wrapper.
    """
    from omegaconf import OmegaConf

    # Backward compatibility: allow legacy `model=` CLI overrides to select plugins.
    if cfg.get("model") and isinstance(cfg.get("model"), str) and not cfg.get("plugin"):
        cfg.plugin = cfg.get("model")

    cfg = merge_plugin_defaults(cfg)

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config_dict, dict):
        raise RuntimeError("Configuration must resolve to a dictionary")

    # learning_rate is fixed by plugins; block user overrides in Hydra configs
    def _has_lr(conf) -> bool:
        if not conf:
            return False
        return isinstance(conf, dict) and "learning_rate" in conf

    if _has_lr(config_dict) or _has_lr(config_dict.get("training")):
        raise ValueError(
            "'learning_rate' is fixed by the selected plugin and cannot be set via Hydra config."
        )

    # Validate that task is provided for modes that require it
    if "mode" in config_dict:
        mode = config_dict["mode"]
        if mode in ["fine-tune", "predict", "interpret"]:
            if "task" not in config_dict:
                raise ValueError(f"Task must be defined for mode '{mode}'")

    def safe_instantiate(cls, data, name):
        if data is None:
            return None
        try:
            return cls(**data)
        except TypeError as e:
            raise ValueError(f"Invalid {name} configuration: {e}") from e

    # Disallow legacy GPU-count keys from overrides early
    if config_dict.get("settings") and "environment" in config_dict["settings"]:
        if "ngpus" in config_dict["settings"]["environment"]:
            raise ValueError(
                "The 'settings.environment.detected_ngpus' option has been removed. "
                "GPU count is auto-detected; see debugging.detected_ngpus."
            )

    if "debugging" in config_dict and "ngpus" in config_dict["debugging"]:
        raise ValueError(
            "The 'debugging.detected_ngpus' option has been removed. "
            "GPU count is auto-detected; use debugging.detected_ngpus if needed."
        )

    data_source = safe_instantiate(
        DataSourceConfig, config_dict.get("data_source"), "data_source"
    )
    tokenization = safe_instantiate(
        TokenizationConfig, config_dict.get("tokenization"), "tokenization"
    )
    training = safe_instantiate(TrainingConfig, config_dict.get("training"), "training")
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

    # Build kwargs for BioLMConfig, only including fields that are present
    biolm_kwargs = {}
    for field_name in ["mode", "outputpath", "task", "plugin", "model"]:
        if field_name in config_dict:
            biolm_kwargs[field_name] = config_dict[field_name]

    biolm_cfg = BioLMConfig(
        **biolm_kwargs,
        data_source=data_source,
        tokenization=tokenization,
        training=training,
        inference=inference,
        settings=settings,
        debugging=debugging,
    )

    # Let the dataclass perform validation and runtime GPU auto-detection
    biolm_cfg.validate()
    biolm_cfg.autodetect_gpus()

    # Apply plugin if specified
    if biolm_cfg.plugin:
        # Plugin system removed - plugins should be handled differently
        pass

    return biolm_cfg


def load_config(overrides: Optional[List[str]] = None) -> BioLMConfig:
    """Public loader: programmatic Hydra composition into BioLMConfig.

    The loader accepts an explicit list of Hydra 'key=value' overrides.
    We deliberately do not attempt to auto-parse sys.argv; callers should pass
    overrides explicitly when invoking programmatically.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_path = Path(__file__).parent / "conf"
    if overrides is None:
        overrides = []

    # Provide an early consistent error when callers attempt to use legacy GPU
    # options via overrides (these are not supported anymore).
    for ov in list(overrides):
        if isinstance(ov, str) and (
            ov.startswith("settings.environment.ngpus=")
            or ov.startswith("debugging.ngpus=")
        ):
            raise ValueError(
                "The 'settings.environment.detected_ngpus' option has been removed. "
                "GPU count is auto-detected; see debugging.detected_ngpus."
            )

    try:
        with initialize_config_dir(config_dir=str(config_path), version_base="1.1"):
            cfg = compose(config_name="config", overrides=overrides)

            # merge mode-specific config if present (best-effort)
            try:
                mode = "tokenize"
                for ov in overrides:
                    if isinstance(ov, str) and ov.startswith("mode="):
                        mode = ov.split("=", 1)[1]
                        break
                mode_cfg = compose(config_name=f"mode/{mode}", overrides=[])
                cfg = OmegaConf.merge(cfg, mode_cfg)
            except Exception:
                # mode file optional — don't fail hard if not present
                pass

            if not cfg.data_source.splitratio:
                cfg.data_source.splitratio = [80, 20]

            # Explicitly handle filepath override
            for ov in overrides:
                if isinstance(ov, str) and ov.startswith("data_source.filepath="):
                    cfg.data_source.filepath = ov.split("=", 1)[1]
                    break

            return _process_hydra_config(cfg)
    finally:
        GlobalHydra.instance().clear()


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def parse_args(cfg: DictConfig) -> BioLMConfig:
    """Hydra CLI entrypoint — returns processed BioLMConfig.

    Keeping this here mirrors previous behavior while still delegating work
    to the dataclass and the loader internals.
    """
    return _process_hydra_config(cfg)
