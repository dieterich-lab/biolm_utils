import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse, urlunparse


def _parse_plugin_source(source: str) -> tuple[str, str | None]:
    """Parse a plugin source string into (git_url, ref).

    Supported forms:
      - https://github.com/org/repo.git
            - https://github.com/org/repo.git?ref=main
            - https://github.com/org/repo.git@main
    """
    parsed = urlparse(source)

    if parsed.scheme in {"http", "https"}:
        query = parse_qs(parsed.query)
        if "ref" in query and query["ref"]:
            ref = query["ref"][0]
            clean_url = urlunparse(parsed._replace(query=""))
            return clean_url, ref

        # Support URL@ref shorthand for HTTPS URLs
        if "@" in source:
            base, candidate_ref = source.rsplit("@", 1)
            if base.endswith(".git") and candidate_ref:
                return base, candidate_ref

    return source, None


def install_plugin(url: str, target_dir: str = "plugins"):
    """Clone and install a plugin from a git URL using editable pip install.

    Keeps the host project's pyproject.toml unchanged; installs into the
    current environment via `pip install -e`.
    """
    git_url, ref = _parse_plugin_source(url)

    # Derive directory name from URL
    repo_name = git_url.split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    plugins_root = Path(target_dir)
    plugins_root.mkdir(exist_ok=True)

    plugin_path = plugins_root / repo_name

    if plugin_path.exists():
        print(f"Directory {plugin_path} already exists. Skipping clone.")
    else:
        clone_display = f"{git_url} (ref: {ref})" if ref else git_url
        print(f"Cloning {clone_display} into {plugin_path}...")
        try:
            clone_cmd = ["git", "clone"]
            if ref:
                clone_cmd.extend(["-b", ref, "--single-branch"])
            clone_cmd.extend([git_url, str(plugin_path)])
            subprocess.check_call(clone_cmd)
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e}")
            sys.exit(1)

    print(f"Installing {repo_name} in editable mode (pip -e)...")
    try:
        # Use the current interpreter to install into the active environment
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", str(plugin_path)]
        )
        print(f"Successfully installed {repo_name} without modifying pyproject.toml.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing plugin: {e}")
        sys.exit(1)


def develop_plugin(path: str):
    """Install an existing plugin repository in editable mode."""

    plugin_path = Path(path).expanduser().resolve()
    if not plugin_path.exists():
        print(f"Plugin path {plugin_path} does not exist.")
        sys.exit(1)

    print(f"Installing {plugin_path.name} from local path in editable mode...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", str(plugin_path)]
        )
        print("Successfully installed plugin without modifying pyproject.toml.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing plugin: {e}")
        sys.exit(1)


def list_plugins():
    """List installed plugins registered via entry points."""
    print("Installed biolm plugins:")
    # For Python 3.10+
    if hasattr(importlib.metadata, "entry_points"):
        eps = importlib.metadata.entry_points(group="biolm.plugins")
    else:
        eps = importlib.metadata.entry_points().get("biolm.plugins", [])

    if not eps:
        print("  (none)")
        return

    for ep in eps:
        dist = ep.dist
        location = "unknown"
        version = "unknown"
        if dist:
            version = dist.version
            # Try to find location for editable installs
            # direct_url.json is often present for pip installs
            direct_url = dist.read_text("direct_url.json")
            if direct_url:
                import json

                try:
                    data = json.loads(direct_url)
                    if "url" in data:
                        location = data["url"]
                except:
                    pass

        print(f"  - {ep.name} (version: {version})")
        # print(f"    Location: {location}")


def remove_plugin(plugin_name: str):
    """Uninstall a plugin."""
    # We need to find the package name associated with the plugin name
    # This is tricky because the entry point name might not match the package name.
    # But usually they are related.

    # Let's search entry points to find the distribution
    target_dist = None
    if hasattr(importlib.metadata, "entry_points"):
        eps = importlib.metadata.entry_points(group="biolm.plugins")
    else:
        eps = importlib.metadata.entry_points().get("biolm.plugins", [])

    for ep in eps:
        if ep.name == plugin_name:
            target_dist = ep.dist
            break

    if not target_dist:
        print(f"Plugin '{plugin_name}' not found.")
        return

    package_name = target_dist.metadata["Name"]
    print(f"Found plugin '{plugin_name}' in package '{package_name}'.")

    confirm = input(f"Are you sure you want to uninstall '{package_name}'? [y/N] ")
    if confirm.lower() != "y":
        print("Aborted.")
        return

    print(f"Uninstalling {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", package_name])
        print(f"Successfully uninstalled {package_name}.")
    except subprocess.CalledProcessError as e:
        print(f"Error uninstalling plugin: {e}")
        sys.exit(1)


def handle_plugin_command(args):
    parser = argparse.ArgumentParser(
        prog="biolm plugin", description="Manage biolm plugins"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Install command
    install_parser = subparsers.add_parser(
        "install", help="Install a plugin from a git URL"
    )
    install_parser.add_argument(
        "url",
        help=(
            "Git URL of the plugin repository. Optional ref supported via "
            "'?ref=<branch-or-tag>' or '@<branch-or-tag>' for HTTPS URLs."
        ),
    )
    install_parser.add_argument(
        "--dir", default="plugins", help="Directory to clone into (default: plugins)"
    )

    develop_parser = subparsers.add_parser(
        "develop",
        help="Install a local plugin repository in editable mode without cloning",
    )
    develop_parser.add_argument("path", help="Path to the plugin repository")

    # List command
    subparsers.add_parser("list", help="List installed plugins")

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove (uninstall) a plugin")
    remove_parser.add_argument("name", help="Name of the plugin to remove")

    parsed_args = parser.parse_args(args)

    if parsed_args.command == "install":
        install_plugin(parsed_args.url, parsed_args.dir)
    elif parsed_args.command == "list":
        list_plugins()
    elif parsed_args.command == "develop":
        develop_plugin(parsed_args.path)
    elif parsed_args.command == "remove":
        remove_plugin(parsed_args.name)
    else:
        parser.print_help()
