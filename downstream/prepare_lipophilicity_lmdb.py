#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import csv
import multiprocessing as mp
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Iterable

import torch
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm

from mol_gtn.features import atom_features, bond_features_and_index, validate_feature_dimensions
from mol_gtn.lap_pe import compute_laplacian_positional_encoding
from mol_gtn.lmdb_io import open_lmdb, serialize_data


def infer_columns(fieldnames: list[str], smiles_col: str | None, target_col: str | None) -> tuple[str, str]:
    lowered = {name.lower(): name for name in fieldnames}
    if smiles_col is None:
        smiles_col = lowered.get("smiles")
    if target_col is None:
        target_col = lowered.get("exp") or lowered.get("label") or lowered.get("target")
    if smiles_col is None or smiles_col not in fieldnames:
        raise ValueError(f"SMILES column not found. Available columns: {fieldnames}")
    if target_col is None or target_col not in fieldnames:
        raise ValueError(f"Target column not found. Available columns: {fieldnames}")
    return smiles_col, target_col


def load_rows(csv_path: Path, smiles_col: str | None, target_col: str | None, limit: int) -> tuple[list[tuple[int, str, float]], str, str]:
    rows: list[tuple[int, str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        smiles_key, target_key = infer_columns(reader.fieldnames, smiles_col, target_col)
        for idx, row in enumerate(reader):
            if limit > 0 and idx >= limit:
                break
            smiles = str(row[smiles_key]).strip()
            target_raw = str(row[target_key]).strip()
            if not smiles or not target_raw:
                continue
            try:
                target = float(target_raw)
            except ValueError:
                continue
            rows.append((idx, smiles, target))
    return rows, smiles_key, target_key


def build_graph(sample: tuple[int, str, float, int]) -> tuple[int, bytes | None, str | None]:
    row_id, smiles, target, lap_pe_dim = sample
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return row_id, None, "invalid_smiles"
        Chem.SanitizeMol(mol)
        x = atom_features(mol)
        edge_index, edge_attr = bond_features_and_index(mol)
        validate_feature_dimensions(x, edge_attr)
        lap_pe, lap_pe_valid_mask = compute_laplacian_positional_encoding(
            num_nodes=mol.GetNumAtoms(),
            edge_index=edge_index,
            k=lap_pe_dim,
        )
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            lap_pe=lap_pe,
            lap_pe_valid_mask=lap_pe_valid_mask,
            padding_mask=torch.zeros(mol.GetNumAtoms(), dtype=torch.bool),
            y=torch.tensor([target], dtype=torch.float32),
            smiles=smiles,
            source_row=int(row_id),
        )
        return row_id, serialize_data(data), None
    except Exception as exc:  # noqa: BLE001
        return row_id, None, str(exc)


def chunked(iterable: Iterable[tuple[int, str, float]], lap_pe_dim: int) -> Iterable[tuple[int, str, float, int]]:
    for row_id, smiles, target in iterable:
        yield (row_id, smiles, target, lap_pe_dim)


def write_lmdb(
    output_lmdb: Path,
    items: list[tuple[int, str, float]],
    lap_pe_dim: int,
    workers: int,
    chunksize: int,
) -> tuple[int, int]:
    output_lmdb.parent.mkdir(parents=True, exist_ok=True)
    if output_lmdb.exists():
        output_lmdb.unlink()

    env = open_lmdb(str(output_lmdb), readonly=False)
    txn = env.begin(write=True)

    valid_count = 0
    invalid_count = 0

    if workers <= 1:
        iterator = map(build_graph, chunked(items, lap_pe_dim))
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=workers)
        iterator = pool.imap(build_graph, chunked(items, lap_pe_dim), chunksize=chunksize)

    try:
        for row_id, blob, err in tqdm(iterator, total=len(items), desc="featurizing"):
            if blob is None:
                invalid_count += 1
                continue
            txn.put(f"{valid_count:012d}".encode(), blob)
            valid_count += 1
            if valid_count % 512 == 0:
                txn.put(b"length", str(valid_count).encode())
                txn.put(b"invalid_count", str(invalid_count).encode())
                txn.commit()
                txn = env.begin(write=True)
    finally:
        if workers > 1:
            pool.close()
            pool.join()

    txn.put(b"length", str(valid_count).encode())
    txn.put(b"invalid_count", str(invalid_count).encode())
    txn.commit()
    env.sync()
    env.close()
    return valid_count, invalid_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Lipophilicity CSV into LMDB with multi-process featurization.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-lmdb", type=Path, required=True)
    parser.add_argument("--smiles-col", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--lap-pe-dim", type=int, default=8)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    parser.add_argument("--chunksize", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="Use first N rows only. 0 means full dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[STAGE 1/3] Reading CSV and inferring columns...")
    rows, smiles_col, target_col = load_rows(args.csv, args.smiles_col, args.target_col, args.limit)
    if not rows:
        raise RuntimeError("No valid rows found in CSV after parsing.")

    print(f"[INFO] CSV          : {args.csv}")
    print(f"[INFO] Output LMDB  : {args.output_lmdb}")
    print(f"[INFO] SMILES column: {smiles_col}")
    print(f"[INFO] Target column: {target_col}")
    print(f"[INFO] Rows loaded  : {len(rows)}")
    print(f"[INFO] Workers      : {args.workers}")

    print("[STAGE 2/3] Multi-process featurization + LMDB writing...")
    valid_count, invalid_count = write_lmdb(
        output_lmdb=args.output_lmdb,
        items=rows,
        lap_pe_dim=args.lap_pe_dim,
        workers=args.workers,
        chunksize=args.chunksize,
    )
    print("[STAGE 3/3] Done")
    print(f"[OK] LMDB written. valid={valid_count} invalid={invalid_count}")


if __name__ == "__main__":
    main()
