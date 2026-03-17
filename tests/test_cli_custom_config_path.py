import subprocess
import sys


def test_cli_accepts_custom_config_path(tmp_path):
    """CLI should accept --config-path with a standalone custom config file."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "\n".join(
            [
                "defaults:",
                "  - _self_",
                "mode: tokenize",
                "outputpath: /tmp/biolm_custom_cfg_test",
                "debugging:",
                "  accelerator: cpu",
                "data_source:",
                "  filepath: /tmp/nonexistent.tsv",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "biolm.cli",
            "--config-path",
            str(tmp_path),
            "--config-name",
            "config",
            "--cfg",
            "job",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "mode: tokenize" in result.stdout
