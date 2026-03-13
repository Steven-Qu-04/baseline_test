from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from .lmdb_io import deserialize_data, open_lmdb


def _zeros_ratio(tensor: torch.Tensor) -> float:
    if tensor is None or tensor.numel() == 0:
        return 0.0
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    row_norm = torch.norm(tensor.float(), dim=-1)
    return float((row_norm == 0).float().mean().item())


def audit_lmdb(lmdb_path: str, sample_count: int = 512) -> dict:
    env = open_lmdb(lmdb_path, readonly=True)
    with env.begin() as txn:
        total = int((txn.get(b"length") or b"0").decode())
        if total <= 0:
            raise RuntimeError(f"No entries found in {lmdb_path}")
        interval = max(1, total // sample_count)
        inspected = 0
        zero_x = []
        zero_edge = []
        duplicate_counter = Counter()
        offline_fields = 0
        for index in range(0, total, interval):
            blob = txn.get(f"{index:012d}".encode())
            if blob is None:
                continue
            data = deserialize_data(blob)
            inspected += 1
            zero_x.append(_zeros_ratio(getattr(data, "x", None)))
            zero_edge.append(_zeros_ratio(getattr(data, "edge_attr", None)))
            molecule_id = str(getattr(data, "molecule_id", f"row:{index}"))
            duplicate_counter[molecule_id] += 1
            if hasattr(data, "offline_anchor") or hasattr(data, "offline_positives"):
                offline_fields += 1
            if inspected >= sample_count:
                break

    duplicate_rate = float(sum(1 for value in duplicate_counter.values() if value > 1) / max(1, len(duplicate_counter)))
    mean_zero_x = float(sum(zero_x) / max(1, len(zero_x)))
    mean_zero_edge = float(sum(zero_edge) / max(1, len(zero_edge)))

    evidence = {
        "inspected": inspected,
        "total_entries": total,
        "mean_zero_x_ratio": mean_zero_x,
        "mean_zero_edge_ratio": mean_zero_edge,
        "duplicate_molecule_id_rate": duplicate_rate,
        "offline_field_hits": offline_fields,
    }
    is_offline = offline_fields > 0 or mean_zero_x > 0.25 or mean_zero_edge > 0.25 or duplicate_rate > 0.05
    return {
        "branch_decision": "offline" if is_offline else "online",
        "evidence": evidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit LMDB data mode before training.")
    parser.add_argument("--lmdb", required=True)
    parser.add_argument("--output-dir", default="/hy-tmp/result")
    parser.add_argument("--report-path", default="")
    parser.add_argument("--data-mode", choices=["auto", "offline", "online"], default="auto")
    parser.add_argument("--sample-count", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.report_path) if args.report_path else Path(args.output_dir) / "data_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    audited = audit_lmdb(args.lmdb, sample_count=args.sample_count)
    if args.data_mode != "auto":
        audited["branch_decision"] = args.data_mode
        audited["override"] = True
    audited["lmdb_path"] = args.lmdb
    report_path.write_text(json.dumps(audited, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audited, sort_keys=True))


if __name__ == "__main__":
    main()
