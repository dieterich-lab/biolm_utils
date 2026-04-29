import pytest
from omegaconf import OmegaConf

from biolm import loader
from biolm.plugin_config import PluginConfig
from biolm.structured_config import BioLMConfig


def test_process_hydra_config_from_dictconfig():
    cfg = OmegaConf.create({"mode": "tokenize", "debugging": {"accelerator": "cpu"}})
    out = loader._process_hydra_config(cfg)
    assert isinstance(out, BioLMConfig)
    assert out.mode == "tokenize"
    assert out.debugging.accelerator == "cpu"


def test_load_config_overrides_accepts_list():
    # Ensure programmatic overrides are accepted and applied
    out = loader.load_config(
        [
            "mode=tokenize",
            "debugging.accelerator=cpu",
            "data_source.splitratio=[80, 20]",
            "data_source.filepath=/path/to/data",
        ]
    )
    assert isinstance(out, BioLMConfig)
    assert out.mode == "tokenize"
    assert out.debugging.accelerator == "cpu"
    assert out.data_source.splitratio == [80, 20]
    # Convert PosixPath to string for comparison
    assert str(out.data_source.filepath) == "/path/to/data"


def test_load_config_rejects_legacy_ngpus_override():
    # Legacy overrides like settings.environment.ngpus are disallowed and should
    # raise a ValueError with a helpful message.
    with pytest.raises(ValueError):
        loader.load_config(
            [
                "mode=tokenize",
                "settings.environment.ngpus=3",
                "data_source.splitratio=[80,20]",
            ]
        )


def test_plugin_defaults_are_merged(monkeypatch):
    """Plugin defaults should merge beneath user overrides."""

    class DummyEntryPoint:
        name = "dummy"

        def load(self):
            return lambda: (
                PluginConfig(),
                {
                    "tokenization": {"vocabsize": 123},
                    "training": {"batchsize": 64},
                },
            )

    def fake_entry_points(group=None):
        if group == "biolm.plugins":
            return [DummyEntryPoint()]
        return []

    monkeypatch.setattr("importlib.metadata.entry_points", fake_entry_points)

    cfg = OmegaConf.create(
        {
            "mode": "tokenize",
            "plugin": "dummy",
            "debugging": {"accelerator": "cpu"},
            # User override should win over plugin default
            "training": {"batchsize": 8},
        }
    )

    out = loader._process_hydra_config(cfg)

    assert out.training.batchsize == 8
    assert out.tokenization.vocabsize == 123
