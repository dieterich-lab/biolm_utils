import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DataSourceConfig:
    filepath: Optional[Path] = None
    stripheader: bool = False
    columnsep: str = "\t"
    tokensep: Optional[str] = ","
    specifiersep: Optional[str] = None
    idpos: Optional[int] = None
    seqpos: Optional[int] = None
    labelpos: Optional[int] = None
    crossvalidation: Optional[bool] = None
    splitratio: Optional[List[int]] = None
    splitpos: Optional[int] = None
    devsplits: Optional[Any] = None
    testsplits: Optional[Any] = None


@dataclass
class TokenizationConfig:
    samplesize: Optional[int] = None
    encoding: str = "atomic"
    lefttailing: bool = False


@dataclass
class TrainingConfig:
    learningrate: float = 1e-4
    seed: int = 42
    batchsize: int = 8
    gradacc: int = 4
    blocksize: int = 12288
    nepochs: int = 100
    patience: int = 10
    resume: bool = False
    scaling: str = "log"
    weightedregression: bool = False


@dataclass
class InferenceConfig:
    looscores: Dict[str, Any] = field(default_factory=dict)
    pretrainedmodel: Optional[Path] = None


@dataclass
class SettingsConfig:
    data_pre_processing: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    training: Optional[TrainingConfig] = None


@dataclass
class DebuggingConfig:
    silent: bool = False
    dev: bool = False
    getdata: bool = False
    forcenewdata: bool = False
    accelerator: str = "gpu"


@dataclass
class SalukiConfig:
    mode: str = "tokenize"
    outputpath: Optional[Path] = None
    task: Optional[str] = None
    data_source: Optional[DataSourceConfig] = None
    tokenization: Optional[TokenizationConfig] = None
    training: Optional[TrainingConfig] = None
    inference: Optional[InferenceConfig] = None
    settings: Optional[SettingsConfig] = None
    debugging: DebuggingConfig = field(default_factory=DebuggingConfig)

    def to_namespace(self) -> argparse.Namespace:
        """Convert the config to an argparse.Namespace for backward compatibility."""
        # Flatten the config to match the old structure
        flat = {}

        # Add top-level fields
        flat["mode"] = self.mode
        flat["outputpath"] = cast(
            Any, str(self.outputpath) if self.outputpath else None
        )
        flat["task"] = cast(Any, self.task)

        # Handle nested structures
        if self.data_source:
            for k, v in self.data_source.__dict__.items():
                flat[k] = v
        if self.tokenization:
            for k, v in self.tokenization.__dict__.items():
                flat[k] = v
        if self.training:
            for k, v in self.training.__dict__.items():
                flat[k] = v
        if self.inference:
            for k, v in self.inference.__dict__.items():
                flat[k] = v
        if self.debugging:
            for k, v in self.debugging.__dict__.items():
                flat[k] = v

        return argparse.Namespace(**flat)


# ConfigStore registration removed for simplicity
