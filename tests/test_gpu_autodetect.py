import sys
import types

import pytest

from biolm.params import get_detected_ngpus, load_config


def _make_dummy_torch(gpu_count: int):
    # Create a minimal stub of torch.cuda.device_count using simple classes
    class Cuda:
        def __init__(self, n):
            self._n = n

        def device_count(self):
            return self._n

    class Torch:
        def __init__(self, n):
            self.cuda = Cuda(n)

    return Torch(gpu_count)


class TestGPUAutodetect:
    def test_no_torch_fallbacks_to_cpu(self, monkeypatch):
        # Ensure no torch module available
        monkeypatch.delitem(sys.modules, "torch", raising=False)

        ns = load_config(
            [
                "mode=tokenize",
                "debugging.accelerator=gpu",
                "data_source.splitratio=[80, 20]",
                "data_source.filepath=/path/to/data",
            ]
        )
        # No torch -> fallback to CPU and detected_ngpus should be 1
        assert ns.debugging.detected_ngpus == 1
        assert get_detected_ngpus(ns) == 1

    def test_non_power_of_two_reduced(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _make_dummy_torch(3))
        ns = load_config(
            [
                "mode=tokenize",
                "debugging.accelerator=gpu",
                "data_source.splitratio=[80, 20]",
                "data_source.filepath=/path/to/data",
            ]
        )
        # 3 GPUs -> reduced to 2 (highest power of two <= 3)
        assert ns.debugging.detected_ngpus == 2
        assert get_detected_ngpus(ns) == 2

    def test_power_of_two_respected(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _make_dummy_torch(4))
        ns = load_config(
            [
                "mode=tokenize",
                "debugging.accelerator=gpu",
                "data_source.splitratio=[80, 20]",
                "data_source.filepath=/path/to/data",
            ]
        )
        assert ns.debugging.detected_ngpus == 4
        assert get_detected_ngpus(ns) == 4

    def test_explicit_invalid_raises(self, monkeypatch):
        # Explicitly set detected_ngpus to a non-power-of-two value -> should raise
        monkeypatch.setitem(sys.modules, "torch", _make_dummy_torch(4))
        with pytest.raises(ValueError):
            # settings.environment.ngpus is a legacy key and should be disallowed
            load_config(
                [
                    "mode=tokenize",
                    "debugging.accelerator=gpu",
                    "settings.environment.ngpus=3",
                ]
            )

    def test_explicit_settings_invalid_raises(self, monkeypatch):
        # Also test if settings.environment.detected_ngpus is explicit and invalid -> should raise
        monkeypatch.setitem(sys.modules, "torch", _make_dummy_torch(4))
        with pytest.raises(ValueError):
            load_config(
                [
                    "mode=tokenize",
                    "debugging.accelerator=gpu",
                    "settings.environment.ngpus=3",
                ]
            )

    def test_explicit_valid_is_removed(self, monkeypatch):
        # Explicit settings.environment.detected_ngpus usage should be disallowed
        monkeypatch.setitem(sys.modules, "torch", _make_dummy_torch(4))
        with pytest.raises(ValueError):
            load_config(
                [
                    "mode=tokenize",
                    "debugging.accelerator=gpu",
                    # using the legacy 'ngpus' option should raise; 'detected_ngpus' is auto set
                    "settings.environment.ngpus=4",
                ]
            )
