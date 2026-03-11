from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import time
from dataclasses import replace
from pathlib import Path
from queue import Empty

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


TASK_SENTINEL = "__TASK_DONE__"
RESULT_SENTINEL = "__RESULT_DONE__"


def build_data_object(row: dict, config: PipelineConfig) -> Data:
    smiles = row["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit returned None")
    Chem.SanitizeMol(mol)
    x = atom_features(mol)
    edge_index, edge_attr = bond_features_and_index(mol)
    validate_feature_dimensions(x, edge_attr)
    lap_pe, lap_pe_valid_mask = compute_laplacian_positional_encoding(mol.GetNumAtoms(), edge_index, config.lap_pe_dim)
    return Data(
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


def safe_qsize(queue: mp.Queue) -> int:
    try:
        return queue.qsize()
    except (NotImplementedError, AttributeError):
        return -1


def task_feeder(
    csv_path: str,
    task_queue: mp.Queue,
    worker_count: int,
    row_limit: int | None,
    log_queue: mp.Queue | None,
) -> None:
    logger = attach_queue_logger(log_queue)
    submitted = 0
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row_limit is not None and submitted >= row_limit:
                break
            task_queue.put(row)
            submitted += 1
            if submitted % 5000 == 0:
                logger.info("Task feeder submitted %s rows (task_queue_size=%s)", submitted, safe_qsize(task_queue))
    for _ in range(worker_count):
        task_queue.put(TASK_SENTINEL)
    logger.info("Task feeder finished after submitting %s rows", submitted)


def producer_worker(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    config: PipelineConfig,
    log_queue: mp.Queue | None,
) -> None:
    logger = attach_queue_logger(log_queue)
    processed = 0
    while True:
        item = task_queue.get()
        if item == TASK_SENTINEL:
            result_queue.put((RESULT_SENTINEL, None))
            logger.info("Producer exiting after %s processed rows", processed)
            return
        try:
            data = build_data_object(item, config)
            result_queue.put(("data", serialize_data(data)))
            processed += 1
        except Exception as exc:
            logger.warning(
                "Skipping invalid sample smiles=%s source=%s:%s reason=%s",
                item.get("smiles"),
                item.get("source_file"),
                item.get("source_line"),
                exc,
            )
            result_queue.put(("invalid", None))


def consumer_writer(
    result_queue: mp.Queue,
    lmdb_path: str,
    worker_count: int,
    batch_size: int,
    log_queue: mp.Queue | None,
) -> None:
    logger = attach_queue_logger(log_queue)
    ensure_output_dir(str(Path(lmdb_path).parent))
    env = open_lmdb(lmdb_path, readonly=False)
    txn = env.begin(write=True)
    index = 0
    invalid = 0
    completed_workers = 0
    last_log_time = time.time()
    while completed_workers < worker_count:
        try:
            message_type, payload = result_queue.get(timeout=5)
        except Empty:
            continue
        if message_type == RESULT_SENTINEL:
            completed_workers += 1
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
        if time.time() - last_log_time >= 10:
            logger.info(
                "LMDB writer progress entries=%s invalid=%s result_queue_size=%s",
                index,
                invalid,
                safe_qsize(result_queue),
            )
            last_log_time = time.time()
    txn.put(b"length", str(index).encode())
    txn.put(b"invalid_count", str(invalid).encode())
    txn.commit()
    env.sync()
    env.close()
    logger.info("LMDB write complete at %s with %s entries and %s invalid rows", lmdb_path, index, invalid)


def verify_exitcodes(processes: list[mp.Process], label: str) -> None:
    failed = [process.name for process in processes if process.exitcode not in (0, None)]
    if failed:
        raise RuntimeError(f"{label} failed: {failed}")


def run_preprocess(config: PipelineConfig) -> None:
    seed_everything(config.seed)
    logger = configure_logging(config.log_path)
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = config.active_lmdb_path()
    row_limit = config.active_row_limit()
    if Path(lmdb_path).exists():
        Path(lmdb_path).unlink()

    worker_count = config.detected_preprocess_worker_count()
    task_queue_size = config.resolved_task_queue_maxsize()
    result_queue_size = config.resolved_result_queue_maxsize()
    logger.info(
        "Starting preprocess lmdb=%s worker_count=%s task_queue_maxsize=%s result_queue_maxsize=%s row_limit=%s",
        lmdb_path,
        worker_count,
        task_queue_size,
        result_queue_size,
        row_limit if row_limit is not None else "full",
    )

    log_queue, listener = build_queue_logging(config.log_path)
    context = mp.get_context("spawn")
    task_queue = context.Queue(maxsize=task_queue_size)
    result_queue = context.Queue(maxsize=result_queue_size)

    feeder = context.Process(
        target=task_feeder,
        args=(config.csv_path, task_queue, worker_count, row_limit, log_queue),
        name="task-feeder",
    )
    consumer = context.Process(
        target=consumer_writer,
        args=(result_queue, lmdb_path, worker_count, config.writer_batch_size, log_queue),
        name="lmdb-consumer",
    )
    workers = [
        context.Process(
            target=producer_worker,
            args=(task_queue, result_queue, config, log_queue),
            name=f"producer-{idx}",
        )
        for idx in range(worker_count)
    ]

    start_time = time.time()
    consumer.start()
    feeder.start()
    for worker in workers:
        worker.start()

    for worker in tqdm(workers, desc="preprocess-workers"):
        worker.join()
    feeder.join()
    consumer.join()
    listener.stop()

    verify_exitcodes([feeder], "Task feeder")
    verify_exitcodes(workers, "Producer workers")
    verify_exitcodes([consumer], "LMDB consumer")
    logger.info("Preprocessing finished. LMDB=%s elapsed_sec=%.2f", lmdb_path, time.time() - start_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess SMILES into LMDB-backed PyG graphs.")
    parser.add_argument("--csv", dest="csv_path", default="pretraining.csv")
    parser.add_argument("--lmdb", dest="lmdb_path", default="/hy-tmp/result/pretraining.lmdb")
    parser.add_argument("--smoke-lmdb", dest="smoke_lmdb_path", default="/hy-tmp/result/smoke_pretraining.lmdb")
    parser.add_argument("--medium-lmdb", dest="medium_lmdb_path", default="/hy-tmp/result/ddp_medium_50k.lmdb")
    parser.add_argument("--output-dir", default="/hy-tmp/result")
    parser.add_argument("--workers", dest="preprocess_worker_count", type=int, default=0)
    parser.add_argument("--cpu-reserve-threads", type=int, default=4)
    parser.add_argument("--task-queue-maxsize", type=int, default=0)
    parser.add_argument("--result-queue-maxsize", type=int, default=0)
    parser.add_argument("--writer-batch-size", type=int, default=64)
    parser.add_argument("--lap-pe-dim", type=int, default=8)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-rows", type=int, default=100)
    parser.add_argument("--medium-test", action="store_true")
    parser.add_argument("--medium-rows", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(PipelineConfig(), **vars(args))
    run_preprocess(config)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
