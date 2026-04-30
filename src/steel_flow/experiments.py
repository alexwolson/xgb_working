"""Entry point: uv run experiments [--overwrite] [--skip-replications] [--skip-binning]
                                   [--replications-dir DIR] [--binning-dir DIR]

Runs uv run tune + uv run train for two suites in sequence:
  1. Replication suite  — config/experiments/         (regression experiments)
  2. Binning suite      — config/experiments_binning/  (binned-classifier experiments)

Skips experiments whose training-config JSON already exists unless --overwrite is set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from steel_flow.config import load_config
from steel_flow.replications import _is_completed, _run_step

REPLICATIONS_DIR = Path("config/experiments")
BINNING_DIR = Path("config/experiments_binning")

console = Console()


def _run_group(
    label: str,
    configs_dir: Path,
    overwrite: bool,
) -> list[tuple[str, str, str, str]]:
    """Run all experiments in configs_dir. Returns list of (group, filename, study_name, status)."""
    configs = sorted(configs_dir.glob("*.toml"))
    if not configs:
        console.print(f"[yellow]No TOML files found in {configs_dir} — skipping group '{label}'[/]")
        return []

    results: list[tuple[str, str, str, str]] = []

    for config_path in configs:
        cfg = load_config(str(config_path))
        study_name = cfg.experiment.study_name

        console.print(Rule(f"[bold]{label}[/] · [bold]{config_path.name}[/]  [dim]{study_name}[/dim]"))

        if not overwrite and _is_completed(study_name):
            console.print("  [yellow]SKIP[/] — training config already exists")
            results.append((label, config_path.name, study_name, "skipped"))
            continue

        tune_ok = _run_step("tune", ["uv", "run", "tune", str(config_path)])
        if not tune_ok:
            results.append((label, config_path.name, study_name, "failed (tune)"))
            continue

        train_ok = _run_step("train", ["uv", "run", "train", str(config_path)])
        if not train_ok:
            results.append((label, config_path.name, study_name, "failed (train)"))
            continue

        results.append((label, config_path.name, study_name, "completed"))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full experiment suite: replications + binning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run experiments even if they have already completed.",
    )
    parser.add_argument(
        "--skip-replications",
        action="store_true",
        help="Skip the replication (regression) suite.",
    )
    parser.add_argument(
        "--skip-binning",
        action="store_true",
        help="Skip the binning suite.",
    )
    parser.add_argument(
        "--replications-dir",
        default=str(REPLICATIONS_DIR),
        metavar="DIR",
        help=f"Directory of replication TOML files (default: {REPLICATIONS_DIR}).",
    )
    parser.add_argument(
        "--binning-dir",
        default=str(BINNING_DIR),
        metavar="DIR",
        help=f"Directory of binning TOML files (default: {BINNING_DIR}).",
    )
    args = parser.parse_args()

    all_results: list[tuple[str, str, str, str]] = []

    if not args.skip_replications:
        all_results += _run_group("replications", Path(args.replications_dir), args.overwrite)

    if not args.skip_binning:
        all_results += _run_group("binning", Path(args.binning_dir), args.overwrite)

    if not all_results:
        console.print("[yellow]Nothing to run — both groups were skipped or empty.[/]")
        return

    # Summary table
    console.print(Rule("[bold]Summary[/]"))
    table = Table(show_header=True, header_style="bold")
    table.add_column("Group")
    table.add_column("Config", style="dim")
    table.add_column("Study name")
    table.add_column("Status")

    status_style = {"completed": "green", "skipped": "yellow"}

    for group, filename, study_name, status in all_results:
        style = status_style.get(status, "red")
        table.add_row(group, filename, study_name, f"[{style}]{status}[/{style}]")

    console.print(table)

    n_done = sum(1 for *_, s in all_results if s == "completed")
    n_skip = sum(1 for *_, s in all_results if s == "skipped")
    n_fail = sum(1 for *_, s in all_results if s.startswith("failed"))
    console.print(
        f"[bold]{n_done} completed[/], [yellow]{n_skip} skipped[/], [red]{n_fail} failed[/]"
    )

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
