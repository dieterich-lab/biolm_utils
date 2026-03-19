"""CLI entry point and argument parsing."""

import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .loader import _process_hydra_config
from .logging_config import setup_logging

warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.",
)

HYDRA_CONFIG_PATH = str(Path(__file__).resolve().parent / "conf")


@hydra.main(
    config_path=HYDRA_CONFIG_PATH,
    config_name="config",
    version_base="1.1",
)
def _hydra_main(cfg: DictConfig):
    """Hydra CLI entrypoint — returns processed BioLMConfig."""
    # Set up logging early
    setup_logging()

    # Allow legacy overrides that address keys not present in the base config
    OmegaConf.set_struct(cfg, False)

    processed_config = _process_hydra_config(cfg)

    # Set the config for the framework to use
    from .biolm import initialize_runtime

    # Reset cached module-level state to use the new config
    initialize_runtime(processed_config)

    # Run the main function
    from .biolm import main

    main()


def parse_args():
    """
    Main entry point dispatcher.

    Routes commands to either:
    1. The Plugin Manager (for 'install-plugin', 'list-plugins', etc.)
    2. The Hydra Application (for 'train', 'tokenize', etc.)
    """
    import sys

    from .plugin_manager import handle_plugin_command

    # Define management commands and their mapping to plugin_manager actions
    # This acts as a router before Hydra takes over
    MANAGEMENT_COMMANDS = {
        "plugin": lambda args: handle_plugin_command(args),
        "install-plugin": lambda args: handle_plugin_command(["install"] + args),
        "develop-plugin": lambda args: handle_plugin_command(["develop"] + args),
        "list-plugins": lambda args: handle_plugin_command(["list"] + args),
        "remove-plugin": lambda args: handle_plugin_command(["remove"] + args),
    }

    # Intercept help to show management commands
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print("BioLM Framework CLI")
        print("===================")
        print("\nManagement Commands:")
        print("  install-plugin <url>   Install a plugin from a Git URL")
        print("  develop-plugin <path>  Install a local plugin in editable mode")
        print("  list-plugins           List installed plugins")
        print("  remove-plugin <name>   Uninstall a plugin")
        print("  plugin <command>       Access advanced plugin management")
        print("\n" + "-" * 20 + "\n")
        print("For mode-specific help, run: biolm mode=<mode> --help")
        print("Available modes: tokenize, pre-train, fine-tune, predict, interpret")
        return

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command in MANAGEMENT_COMMANDS:
            # Execute the mapped management command with remaining arguments
            MANAGEMENT_COMMANDS[command](sys.argv[2:])
            return

    # If not a management command, pass control to Hydra
    _hydra_main()


if __name__ == "__main__":
    parse_args()
