#!/usr/bin/env python3
"""Read two JSONL loss logs, merge them by step, and save a line chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_DIR = Path("/hy-tmp/result_flash_1m/run_20260312_214955")


def pick_existing(candidates: List[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("None of these files exist:\n" + "\n".join(str(p) for p in candidates))


def read_log(path: Path) -> Dict[int, Tuple[float, int]]:
    """Return mapping: step -> (loss, epoch)."""
    data: Dict[int, Tuple[float, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc

            if "step" not in obj or "loss" not in obj:
                continue

            try:
                step = int(obj["step"])
                loss = float(obj["loss"])
            except (TypeError, ValueError):
                continue

            epoch = int(obj.get("epoch", -1))
            data[step] = (loss, epoch)
    return data


def filter_epoch(data: Dict[int, Tuple[float, int]], epoch: int) -> Dict[int, Tuple[float, int]]:
    return {step: pair for step, pair in data.items() if pair[1] == epoch}


def build_plot(first_log: Path, second_log: Path, out_file: Path) -> None:
    first = read_log(first_log)
    second = read_log(second_log)

    # Keep only epoch 1 from the first log, per data convention.
    first_epoch_1 = filter_epoch(first, epoch=1)

    # Prefer second log on overlaps (continuation run is newer).
    merged = dict(first_epoch_1)
    merged.update(second)

    if not merged:
        raise RuntimeError("No valid points found in provided logs.")

    steps = sorted(merged)
    losses = [merged[s][0] for s in steps]

    # Epoch boundary marks for a top x-axis.
    epoch_start_step: Dict[int, int] = {}
    for step in steps:
        epoch = merged[step][1]
        if epoch not in epoch_start_step:
            epoch_start_step[epoch] = step
    epoch_ticks = [epoch_start_step[e] for e in sorted(epoch_start_step)]
    epoch_labels = [f"E{e}" for e in sorted(epoch_start_step)]

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=steps, y=losses, linewidth=1.4, color="#1f77b4", ax=ax)
    ax.set_title("Training Loss vs Step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")

    # Keep step marks at bottom and epoch marks at top.
    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xlabel("Epoch")
    ax_top.set_xticks(epoch_ticks)
    ax_top.set_xticklabels(epoch_labels)

    for x in epoch_ticks:
        ax.axvline(x=x, color="#9ca3af", linewidth=0.8, alpha=0.4, linestyle="--")

    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=160)
    plt.close()

    print(f"Saved chart: {out_file}")
    print(f"Points plotted: {len(steps)}")
    print(f"Step range: {steps[0]} -> {steps[-1]}")
    print(f"Epoch-1 points from first log: {len(first_epoch_1)}")


def main() -> None:
    default_first = pick_existing([DEFAULT_DIR / "loss_log.jsonl"])
    default_second = pick_existing(
        [
            DEFAULT_DIR / "zz20260312_214955_loss_log.jsonl",
            DEFAULT_DIR / "20260312_214955_loss_log.jsonl",
        ]
    )

    parser = argparse.ArgumentParser(description="Plot merged loss logs from JSONL files.")
    parser.add_argument(
        "--first-log",
        type=Path,
        default=default_first,
        help="Path to first-epoch log file.",
    )
    parser.add_argument(
        "--second-log",
        type=Path,
        default=default_second,
        help="Path to remaining-epochs log file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "loss_line_chart.png",
        help="Output figure path (saved under project path by default).",
    )
    args = parser.parse_args()

    if not args.first_log.exists():
        raise FileNotFoundError(f"Missing first log: {args.first_log}")
    if not args.second_log.exists():
        raise FileNotFoundError(f"Missing second log: {args.second_log}")

    build_plot(args.first_log, args.second_log, args.output)


if __name__ == "__main__":
    main()
