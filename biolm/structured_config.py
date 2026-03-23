from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast


@dataclass
class DataSourceConfig:
    filepath: Optional[Path] = None
    stripheader: bool = False
    columnsep: str = "\t"
    tokensep: Optional[str] = ","
    specifiersep: Optional[str] = None
    inferenceonsplits: Optional[List[int]] = None
    idpos: Optional[int] = None
    seqpos: Optional[int] = None
    labelpos: Optional[int] = None
    crossvalidation: Optional[bool] = None
    splitratio: Optional[List[int]] = None
    splitpos: Optional[int] = None
    devsplits: Optional[Any] = None
    testsplits: Optional[Any] = None

    def __post_init__(self):
        if self.filepath is not None and not isinstance(self.filepath, Path):
            self.filepath = Path(self.filepath)


@dataclass
class TokenizationConfig:
    samplesize: Optional[int] = None
    encoding: str = "bpe"
    lefttailing: bool = False
    vocabsize: int = 20000
    minfreq: int = 2
    maxtokenlength: int = 10
    atomicreplacements: Optional[Dict[str, str]] = None


@dataclass
class TrainingConfig:
    # canonical snake_case name for training settings
    seed: int = 42
    batchsize: int = 8
    gradacc: int = 4
    blocksize: Optional[int] = None
    nepochs: int = 100
    num_epochs: Optional[int] = None
    patience: int = 10
    resume: bool = False
    scaling: str = "log"
    weightedregression: bool = False

    def __post_init__(self):
        if self.num_epochs is not None:
            self.nepochs = self.num_epochs


@dataclass
class InferenceConfig:
    looscores: Dict[str, Any] = field(default_factory=dict)
    pretrainedmodel: Optional[Path] = None


@dataclass
class SettingsConfig:
    data_pre_processing: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    # MLflow integration settings (opt-in)
    mlflow: Optional[Dict[str, Any]] = None
    training: Optional[TrainingConfig] = None


@dataclass
class DebuggingConfig:
    silent: bool = False
    dev: bool = False
    getdata: bool = False
    forcenewdata: bool = False
    accelerator: str = "auto"
    # Computed/auto-detected GPU count (do not set directly in configs) — kept for runtime
    # use and backward compatibility; this replaces the former GPU-count top-level arg.
    detected_ngpus: Optional[int] = None


@dataclass
class BioLMConfig:
    # Core settings
    mode: str
    task: Optional[str] = None
    plugin: Optional[str] = None
    outputpath: Optional[Path] = None
    model: Optional[Dict[str, Any]] = None
    data_source: Optional[DataSourceConfig] = None
    tokenization: Optional[TokenizationConfig] = None
    training: Optional[TrainingConfig] = None
    inference: Optional[InferenceConfig] = None
    settings: Optional[SettingsConfig] = None
    debugging: DebuggingConfig = field(default_factory=DebuggingConfig)

    def validate(self) -> None:
        """Validate the configuration.

        This performs the same sanity checks previously done in params._validate_config
        but attached to the structured dataclass so validation is colocated with the
        data model. Raises ValueError on invalid configuration.
        """
        # Task requirement for modes that perform a task
        if self.mode in ["fine-tune", "predict", "interpret"] and not self.task:
            raise ValueError(
                f"task is required when mode='{self.mode}'. "
                "Valid tasks depend on the specific model plugin."
            )

        # Skip split validation for modes that don't use training splits
        if (
            self.mode in ["tokenize", "predict", "interpret", "pre-train"]
        ):
            return

        # data_source must exist for training modes.
        if not self.data_source:
            raise ValueError(
                f"No data_source specified for mode '{self.mode}'. This may cause issues during training."
            )

        ds = self.data_source

        # Validate splitratio
        if ds.splitratio is not None:
            if not (isinstance(ds.splitratio, list) and len(ds.splitratio) >= 2):
                raise ValueError(
                    f"data_source.splitratio must be a list of at least 2 integers, got {ds.splitratio}"
                )
            if not all(isinstance(item, int) and item > 0 for item in ds.splitratio):
                raise ValueError(
                    f"All values in data_source.splitratio must be positive integers, got {ds.splitratio}"
                )
            if len(ds.splitratio) not in [2, 3]:
                raise ValueError(
                    f"data_source.splitratio must contain 2 or 3 values, got {len(ds.splitratio)}"
                )
            if sum(ds.splitratio) != 100:
                raise ValueError(
                    f"Values in data_source.splitratio must sum to 100, got {sum(ds.splitratio)}"
                )

        # Ensure splitratio and splitpos are exclusive
        ratio_is_set = ds.splitratio is not None
        pos_is_set = ds.splitpos is not None

        if ratio_is_set and pos_is_set:
            raise ValueError(
                f"In mode '{self.mode}', data_source.splitratio and data_source.splitpos are mutually exclusive"
            )

        if not ratio_is_set and not pos_is_set:
            raise ValueError(
                f"Either data_source.splitratio or data_source.splitpos must be provided for mode '{self.mode}'"
            )

        # Validate splitpos details
        if ds.splitpos is not None:
            if ds.devsplits is None:
                raise ValueError(
                    "data_source.devsplits is required when data_source.splitpos is provided"
                )
            if not (isinstance(ds.splitpos, int) and ds.splitpos >= 0):
                raise ValueError(
                    f"data_source.splitpos must be a non-negative integer, got {ds.splitpos}"
                )

        # Cross-validation split shapes
        if ds.crossvalidation:
            for split_name in ["devsplits", "testsplits"]:
                split_value = getattr(ds, split_name, None)
                if split_value is not None:
                    if not (
                        isinstance(split_value, list)
                        and all(isinstance(sublist, list) for sublist in split_value)
                    ):
                        raise ValueError(
                            f"With data_source.crossvalidation=True, {split_name} must be a list of lists"
                        )
                    if not all(
                        isinstance(item, int)
                        for sublist in split_value
                        for item in sublist
                    ):
                        raise ValueError(
                            f"With data_source.crossvalidation=True, {split_name} must contain only integers"
                        )

    def autodetect_gpus(self) -> None:
        """Detect or normalise GPU settings and write result to debugging.detected_ngpus.

        Logs a warning if GPU is requested but not available or detection fails.
        """
        import logging

        final_ngpus = None
        if getattr(self.debugging, "accelerator", "auto") in ["gpu", "auto"]:
            try:
                import torch

                detected_gpus = torch.cuda.device_count()
                if detected_gpus > 0:
                    # choose highest power-of-two <= detected_gpus
                    def _is_power_of_two(x):
                        return x > 0 and (x & (x - 1)) == 0

                    def _highest_power_of_two_leq(x):
                        return 1 << (x.bit_length() - 1)

                    if _is_power_of_two(detected_gpus):
                        final_ngpus = detected_gpus
                    else:
                        final_ngpus = _highest_power_of_two_leq(detected_gpus)
                else:
                    logging.warning(
                        "GPU requested but no GPUs detected. Falling back to CPU."
                    )
                    self.debugging.accelerator = "cpu"
            except Exception as e:
                logging.warning(
                    f"GPU requested but detection failed ({e}). Falling back to CPU."
                )
                self.debugging.accelerator = "cpu"
        # attach the computed value
        self.debugging.detected_ngpus = final_ngpus if final_ngpus is not None else 1
