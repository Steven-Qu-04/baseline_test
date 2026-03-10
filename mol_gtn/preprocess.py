from __future__ import annotations

import argparse
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path
from queue import Empty
from typing import Iterable

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm

from .config import PipelineConfig
from .features import atom_features, bond_features_and_index, validate_feature_dimensions
from .lap_pe import compute_laplacian_positional_encoding
from .lmdb_io import open_lmdb, serialize_data
from .utils.logging import attach_queue_logger, build_queue_logging, configure_logging, ensure_output_dir
from .utils.runtime import seed_everything


SENTINEL = "__QUEUE_DONE__"


def build_data_object(row: dict, config: PipelineConfig) -> Data | None:
    smiles = row["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit returned None")
    Chem.SanitizeMol(mol)
    x = atom_features(mol)
    edge_index, edge_attr = bond_features_and_index(mol)
    validate_feature_dimensions(x, edge_attr)
    lap_pe, lap_pe_valid_mask = compute_laplacian_positional_encoding(mol.GetNumAtoms(), edge_index, config.lap_pe_dim)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        lap_pe=lap_pe,
        lap_pe_valid_mask=lap_pe_valid_mask,
        padding_mask=torch.zeros(mol.GetNumAtoms(), dtype=torch.bool),
        smiles=smiles,
        source_file=row.get("source_file", ""),
        source_line=int(row.get("source_line", -1)),
        molecule_id=f"{row.get('source_file', 'unknown')}:{row.get('source_line', -1)}",
    )
    return data


def producer(rows: list[dict], queue: mp.Queue, config: PipelineConfig, log_queue: mp.Queue | None) -> None:
    logger = attach_queue_logger(log_queue)
    for row in rows:
        try:
            data = build_data_object(row, config)
            if data is None:
                continue
            queue.put(("data", data.molecule_id, serialize_data(data)))
        except Exception as exc:
            logger.warning("Skipping invalid sample smiles=%s source=%s:%s reason=%s", row.get("smiles"), row.get("source_file"), row.get("source_line"), exc)
            queue.put(("invalid", None, None))
    queue.put((SENTINEL, None, None))


def consumer(queue: mp.Queue, lmdb_path: str, num_producers: int, batch_size: int, log_queue: mp.Queue | None) -> None:
    logger = attach_queue_logger(log_queue)
    ensure_output_dir(str(Path(lmdb_path).parent))
    env = open_lmdb(lmdb_path, readonly=False)
    done = 0
    index = 0
    invalid = 0
    txn = env.begin(write=True)
    while done < num_producers:
        try:
            message_type, key, payload = queue.get(timeout=5)
        except Empty:
            continue
        if message_type == SENTINEL:
            done += 1
            continue
        if message_type == "invalid":
            invalid += 1
            continue
        txn.put(f"{index:012d}".encode(), payload)
        index += 1
        if index % batch_size == 0:
            txn.put(b"length", str(index).encode())
            txn.put(b"invalid_count", str(invalid).encode())
            txn.commit()
            txn = env.begin(write=True)
    txn.put(b"length", str(index).encode())
    txn.put(b"invalid_count", str(invalid).encode())
    txn.commit()
    env.sync()
    env.close()
    logger.info("LMDB write complete at %s with %s entries and %s invalid rows", lmdb_path, index, invalid)


def chunk_rows(rows: list[dict], num_chunks: int) -> Iterable[list[dict]]:
    chunk_size = max(1, (len(rows) + num_chunks - 1) // num_chunks)
    for start in range(0, len(rows), chunk_size):
        yield rows[start : start + chunk_size]


def run_preprocess(config: PipelineConfig) -> None:
    seed_everything(config.seed)
    logger = configure_logging(config.log_path)
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.csv_path)
    if config.smoke_test:
        df = df.head(config.smoke_rows)
    rows = df.to_dict("records")
    lmdb_path = config.active_lmdb_path()
    if Path(lmdb_path).exists():
        Path(lmdb_path).unlink()
    log_queue, listener = build_queue_logging(config.log_path)
    queue: mp.Queue = mp.Queue(maxsize=config.queue_size)
    producers = []
    worker_count = min(config.num_workers, max(1, len(rows)))
    consumer_process = mp.Process(
        target=consumer,
        args=(queue, lmdb_path, worker_count, config.writer_batch_size, log_queue),
        name="lmdb-consumer",
    )
    consumer_process.start()
    for idx, batch in enumerate(chunk_rows(rows, worker_count)):
        process = mp.Process(
            target=producer,
            args=(batch, queue, config, log_queue),
            name=f"producer-{idx}",
        )
        producers.append(process)
        process.start()
    for process in tqdm(producers, desc="preprocess-workers"):
        process.join()
    consumer_process.join()
    listener.stop()
    logger.info("Preprocessing finished. LMDB=%s rows=%s", lmdb_path, len(rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess SMILES into LMDB-backed PyG graphs.")
    parser.add_argument("--csv", dest="csv_path", default="pretraining.csv")
    parser.add_argument("--lmdb", dest="lmdb_path", default="/hy-tmp/result/pretraining.lmdb")
    parser.add_argument("--smoke-lmdb", dest="smoke_lmdb_path", default="/hy-tmp/result/smoke_pretraining.lmdb")
    parser.add_argument("--output-dir", default="/hy-tmp/result")
    parser.add_argument("--workers", dest="num_workers", type=int, default=8)
    parser.add_argument("--queue-size", type=int, default=256)
    parser.add_argument("--writer-batch-size", type=int, default=64)
    parser.add_argument("--lap-pe-dim", type=int, default=8)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-rows", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(PipelineConfig(), **vars(args))
    run_preprocess(config)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
