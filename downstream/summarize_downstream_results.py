#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import csv
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate downstream checkpoint metrics and plot performance evolution.")
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[STAGE 1/3] Collecting checkpoint metrics...")
    metric_files = sorted(args.run_root.glob("checkpoint_epoch_*/metrics.json"))
    if not metric_files:
        raise FileNotFoundError(f"No metrics.json files found under: {args.run_root}")

    rows = []
    for metric_path in tqdm(metric_files, desc="summarize-metrics", leave=False):
        payload = json.loads(metric_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "pretrain_epoch": int(payload.get("pretrain_epoch", -1)),
                "checkpoint": payload.get("checkpoint_name", "unknown"),
                "rmse": float(payload["test"]["rmse"]),
                "mae": float(payload["test"]["mae"]),
                "r2": float(payload["test"]["r2"]),
                "run_dir": str(metric_path.parent),
            }
        )

    rows = sorted(rows, key=lambda x: x["pretrain_epoch"])

    print("[STAGE 2/3] Writing summary table and performance plot...")
    csv_path = args.run_root / "checkpoint_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pretrain_epoch", "checkpoint", "rmse", "mae", "r2", "run_dir"])
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.run_root / "checkpoint_comparison.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| pretrain_epoch | checkpoint | RMSE | MAE | R2 |\n")
        handle.write("|---:|---|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['pretrain_epoch']} | {row['checkpoint']} | "
                f"{row['rmse']:.6f} | {row['mae']:.6f} | {row['r2']:.6f} |\n"
            )

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = [r["pretrain_epoch"] for r in rows]
    y = [r["rmse"] for r in rows]
    sns.lineplot(x=x, y=y, marker="o", linewidth=2.0, ax=ax)
    ax.set_xlabel("Pre-training Epoch")
    ax.set_ylabel("Downstream Test RMSE")
    ax.set_title("Performance Evolution")
    fig.tight_layout()
    plot_path = args.run_root / "performance_evolution.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print("[STAGE 3/3] Done")
    print("[OK] Summary generated")
    print(f"- CSV : {csv_path}")
    print(f"- MD  : {md_path}")
    print(f"- Plot: {plot_path}")


if __name__ == "__main__":
    main()
