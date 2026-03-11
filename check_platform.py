from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist

from mol_gtn.config import PipelineConfig
from mol_gtn.utils.logging import configure_logging


def run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        return completed.stdout.strip()
    except Exception as exc:
        return f"FAILED: {exc}"


def detect_nvlink(topology_output: str) -> bool:
    lines = [line.strip() for line in topology_output.splitlines() if line.strip()]
    topology_lines = [line for line in lines if line.startswith("GPU")]
    return any("\tNV" in f"\t{line}" or " NV" in line for line in topology_lines)


def peer_access_matrix(gpu_count: int) -> list[list[bool]]:
    matrix: list[list[bool]] = []
    for src in range(gpu_count):
        row = []
        for dst in range(gpu_count):
            if src == dst:
                row.append(True)
            else:
                row.append(bool(torch.cuda.can_device_access_peer(src, dst)))
        matrix.append(row)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GPU/CPU topology for DDP readiness.")
    parser.add_argument("--output-dir", default="/hy-tmp/result")
    parser.add_argument("--summary-path", default="/hy-tmp/result/platform_summary.json")
    args = parser.parse_args()

    config = PipelineConfig(output_dir=args.output_dir, platform_summary_path=args.summary_path)
    logger = configure_logging(config.log_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_count = torch.cuda.device_count()
    topo_output = run_command(["nvidia-smi", "topo", "-m"])
    gpu_list_output = run_command(["nvidia-smi", "-L"])
    summary = {
        "logical_cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": gpu_count,
        "gpu_names": [torch.cuda.get_device_name(idx) for idx in range(gpu_count)] if gpu_count > 0 else [],
        "gpu_total_memory_gb": [round(torch.cuda.get_device_properties(idx).total_memory / (1024**3), 2) for idx in range(gpu_count)] if gpu_count > 0 else [],
        "nccl_available": dist.is_nccl_available(),
        "gpu_list_output": gpu_list_output,
        "topology": topo_output,
        "has_nvlink": detect_nvlink(topo_output),
        "peer_access_matrix": peer_access_matrix(gpu_count) if gpu_count > 0 else [],
    }
    logger.info("Platform summary gpu_count=%s logical_cpu_count=%s nccl=%s nvlink=%s", summary["gpu_count"], summary["logical_cpu_count"], summary["nccl_available"], summary["has_nvlink"])
    logger.info("GPU list:\n%s", gpu_list_output)
    logger.info("Topology:\n%s", topo_output)
    Path(config.platform_summary_path).write_text(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
