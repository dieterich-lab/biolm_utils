"""Smoke tests for Hydra configuration system.

These tests verify that the Hydra configuration system works correctly
with plugins, modes, and configuration merging.
"""

import subprocess
import sys
import tempfile
from pathlib import Path
import importlib.util

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_hydra_config_composition_basic():
    """Test that basic Hydra config composition works."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "biolm.cli",
            "mode=fine-tune",
            "task=classification",
            "--cfg",
            "job",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, f"Config composition failed: {result.stderr}"
    assert "mode: fine-tune" in result.stdout


def test_hydra_config_with_plugin():
    """Test that Hydra config works with plugin selection."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "biolm.cli",
            "mode=fine-tune",
            "plugin=saluki",
            "task=classification",
            "--cfg",
            "job",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, f"Plugin config failed: {result.stderr}"
    assert "plugin: saluki" in result.stdout
    assert "mode: fine-tune" in result.stdout


def test_hydra_config_fine_tune_without_task_fails():
    """Fine-tune mode should fail early when task is not selected."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "biolm.cli",
            "mode=fine-tune",
            "--cfg",
            "job",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode != 0
    assert "You must specify 'task'" in (result.stderr + result.stdout)


def test_hydra_config_pre_train_without_task():
    """Test that pre-train mode composes without task override."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "biolm.cli",
            "mode=pre-train",
            "plugin=xlnet",
            "--cfg",
            "job",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, f"Pre-train config failed: {result.stderr}"
    assert "mode: pre-train" in result.stdout


def test_hydra_config_all_modes():
    """Test that all modes can compose their configurations."""
    modes = ["tokenize", "pre-train", "fine-tune", "predict", "interpret"]

    for mode in modes:
        result = subprocess.run(
            (
                [
                    sys.executable,
                    "-m",
                    "biolm.cli",
                    f"mode={mode}",
                    "plugin=saluki",
                    "task=classification",
                    "--cfg",
                    "job",
                ]
                if mode in ["fine-tune", "predict", "interpret"]
                else [
                sys.executable,
                "-m",
                "biolm.cli",
                f"mode={mode}",
                "plugin=saluki",
                "--cfg",
                "job",
                ]
            ),
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )

        assert result.returncode == 0, f"Mode {mode} config failed: {result.stderr}"
        assert f"mode: {mode}" in result.stdout


def test_hydra_config_with_custom_values():
    """Test that custom configuration values are properly merged."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "biolm.cli",
            "mode=fine-tune",
            "plugin=saluki",
            "task=classification",
            "training.nepochs=5",
            "data_source.filepath=/tmp/test.tsv",
            "--cfg",
            "job",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, f"Custom config failed: {result.stderr}"
    assert "nepochs: 5" in result.stdout
    assert "filepath: /tmp/test.tsv" in result.stdout


def test_hydra_config_file_override(tmp_path):
    """Test that configuration files work with Hydra."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
mode: fine-tune
plugin: saluki
task: classification
outputpath: /tmp/test_output
training:
  nepochs: 3
  batchsize: 4
"""
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "biolm.cli",
            "--config-path",
            str(tmp_path),
            "--config-name",
            "test_config",
            "--cfg",
            "job",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, f"Config file failed: {result.stderr}"
    assert "mode: fine-tune" in result.stdout
    assert "plugin: saluki" in result.stdout
    assert "nepochs: 3" in result.stdout
    assert "batchsize: 4" in result.stdout


def test_hydra_config_plugin_invariants_enforced():
    """Test that plugin invariants are properly enforced."""
    if importlib.util.find_spec("saluki_plugin") is None:
        pytest.skip("saluki_plugin is not installed in this environment")

    # Test that saluki enforces atomic encoding
    script = """
import sys
sys.path.insert(0, '__REPO_ROOT__')
from saluki_plugin.dataset import RNACNNDataset
try:
    # This should fail because encoding is not atomic
    dataset = RNACNNDataset(
        args=type('Args', (), {
            'tokenization': type('Tok', (), {'encoding': 'bpe'})(),
            'training': type('Train', (), {'blocksize': 12288})(),
        })()
    )
    print("ERROR: Should have failed")
    sys.exit(1)
except ValueError as e:
    if "atomic" in str(e):
        print("SUCCESS: Atomic encoding enforced")
    else:
        print(f"ERROR: Wrong error: {e}")
        sys.exit(1)
""".replace("__REPO_ROOT__", str(REPO_ROOT))

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, f"Invariant test failed: {result.stderr}"
    assert "SUCCESS: Atomic encoding enforced" in result.stdout


def test_hydra_config_xlnet_plugin():
    """Test that XLNet plugin configuration works."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "biolm.cli",
            "mode=fine-tune",
            "plugin=xlnet",
            "task=classification",
            "--cfg",
            "job",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, f"XLNet config failed: {result.stderr}"
    assert "plugin: xlnet" in result.stdout
    assert "mode: fine-tune" in result.stdout


def test_hydra_config_help_works():
    """Test that help command works without config composition errors."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "biolm.cli",
            "--help",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, f"Help failed: {result.stderr}"
    assert "BioLM Framework CLI" in result.stdout
    assert "Management Commands:" in result.stdout
