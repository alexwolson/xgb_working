"""Entry point: uv run replications [--overwrite] [--configs-dir DIR]

Runs uv run tune + uv run train for every experiment config in configs-dir,
in alphabetical order. Skips experiments whose training-config JSON already
exists (i.e. a previous run completed successfully) unless --overwrite is set.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from steel_flow.config import load_config

EXPERIMENTS_DIR = Path("config/experiments")
TRAINING_CONFIGS_DIR = Path("config")

console = Console()


def _is_completed(study_name: str) -> bool:
    """Training-config JSON is written as the last step of `uv run train`."""
    return (TRAINING_CONFIGS_DIR / f"training_config_{study_name}.json").exists()


def _run_step(label: str, cmd: list[str]) -> bool:
    """Stream a subprocess to the terminal. Returns True on success."""
    console.print(f"  [bold cyan]→[/] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        console.print(f"  [bold red]✗[/] {label} exited with code {result.returncode}")
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all replication experiments in sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run experiments even if they have already completed.",
    )
    parser.add_argument(
        "--configs-dir",
        default=str(EXPERIMENTS_DIR),
        metavar="DIR",
        help=f"Directory of experiment TOML files (default: {EXPERIMENTS_DIR}).",
    )
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)
    configs = sorted(configs_dir.glob("*.toml"))

    if not configs:
        console.print(f"[red]No TOML files found in {configs_dir}[/]")
        sys.exit(1)

    results: list[tuple[str, str, str]] = []  # (filename, study_name, status)

    for config_path in configs:
        cfg = load_config(str(config_path))
        study_name = cfg.experiment.study_name

        console.print(Rule(f"[bold]{config_path.name}[/]  [dim]{study_name}[/dim]"))

        if not args.overwrite and _is_completed(study_name):
            console.print(f"  [yellow]SKIP[/] — training config already exists")
            results.append((config_path.name, study_name, "skipped"))
            continue

        tune_ok = _run_step("tune", ["uv", "run", "tune", str(config_path)])
        if not tune_ok:
            results.append((config_path.name, study_name, "failed (tune)"))
            continue

        train_ok = _run_step("train", ["uv", "run", "train", str(config_path)])
        if not train_ok:
            results.append((config_path.name, study_name, "failed (train)"))
            continue

        results.append((config_path.name, study_name, "completed"))

    # Summary table
    console.print(Rule("[bold]Summary[/]"))
    table = Table(show_header=True, header_style="bold")
    table.add_column("Config", style="dim")
    table.add_column("Study name")
    table.add_column("Status")

    status_style = {
        "completed": "green",
        "skipped": "yellow",
    }

    for filename, study_name, status in results:
        style = status_style.get(status, "red")
        table.add_row(filename, study_name, f"[{style}]{status}[/{style}]")

    console.print(table)

    n_done = sum(1 for _, _, s in results if s == "completed")
    n_skip = sum(1 for _, _, s in results if s == "skipped")
    n_fail = sum(1 for _, _, s in results if s.startswith("failed"))
    console.print(
        f"[bold]{n_done} completed[/], [yellow]{n_skip} skipped[/], [red]{n_fail} failed[/]"
    )

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
