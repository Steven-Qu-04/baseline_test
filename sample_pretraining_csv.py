from __future__ import annotations

import argparse
import random
from pathlib import Path


def reservoir_sample_lines(input_path: Path, output_path: Path, sample_size: int, seed: int) -> None:
    rng = random.Random(seed)
    reservoir: list[str] = []

    with input_path.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline()
        if not header:
            raise RuntimeError(f"Input CSV is empty: {input_path}")

        for index, line in enumerate(handle, start=1):
            if index <= sample_size:
                reservoir.append(line)
                continue
            replacement_index = rng.randint(1, index)
            if replacement_index <= sample_size:
                reservoir[replacement_index - 1] = line

    if len(reservoir) < sample_size:
        raise RuntimeError(
            f"Requested {sample_size} rows but only found {len(reservoir)} data rows in {input_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        handle.writelines(reservoir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reservoir-sample rows from a large CSV while preserving the header.")
    parser.add_argument("--input", default="pretraining.csv", help="Source CSV path")
    parser.add_argument("--output", default="pretraining_1m.csv", help="Sampled CSV path")
    parser.add_argument("--rows", type=int, default=1_000_000, help="Number of data rows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    print(f"[Sampling] input={input_path} output={output_path} rows={args.rows} seed={args.seed}")
    reservoir_sample_lines(input_path=input_path, output_path=output_path, sample_size=args.rows, seed=args.seed)
    print(f"[Sampling] done -> {output_path}")


if __name__ == "__main__":
    main()
