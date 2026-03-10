from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch
from tqdm import tqdm

from .config import PipelineConfig
from .dataset import LmdbGraphDataset
from .features import feature_dimensions
from .model import MolecularGTN
from .utils.logging import configure_logging


def infer_collate(samples):
    return Batch.from_data_list(samples)


def run_inference(config: PipelineConfig) -> None:
    logger = configure_logging(config.log_path)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config.model_path, map_location=device, weights_only=False)
    dims = feature_dimensions()
    model = MolecularGTN(dims["atom_dim"], dims["bond_dim"], config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = LmdbGraphDataset(config.active_lmdb_path(), config)
    subset = Subset(dataset, range(min(config.num_infer, len(dataset))))
    loader = DataLoader(subset, batch_size=config.num_infer, shuffle=False, collate_fn=infer_collate)
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="infer"):
            batch = batch.to(device)
            encoded = model.encode(batch)
            embeddings = encoded.mol_embeddings.cpu()
            ptr = batch.ptr.cpu().tolist()
            for idx in range(len(ptr) - 1):
                data = subset[idx]
                row = {"molecule_id": data.molecule_id, "smiles": data.smiles}
                row.update({f"emb_{dim}": float(embeddings[idx, dim]) for dim in range(embeddings.size(1))})
                rows.append(row)
    output_path = Path(config.embeddings_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logger.info("Inference export written to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for Molecular GTN.")
    parser.add_argument("--lmdb", dest="lmdb_path", default="/hy-tmp/result/pretraining.lmdb")
    parser.add_argument("--smoke-lmdb", dest="smoke_lmdb_path", default="/hy-tmp/result/smoke_pretraining.lmdb")
    parser.add_argument("--model-path", default="/hy-tmp/result/model_final.pth")
    parser.add_argument("--embeddings-path", default="/hy-tmp/result/top10_embeddings.csv")
    parser.add_argument("--num-infer", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--lap-pe-dim", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(PipelineConfig(), **vars(args))
    run_inference(config)


if __name__ == "__main__":
    main()
