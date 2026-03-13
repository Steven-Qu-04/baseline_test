import csv
import hashlib
import math
from itertools import islice

import torch
import torch.distributed as dist
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data

from utils import (
    FINAL_NODE_FEATURE_DIM,
    append_embedding_failure_log,
    build_augmented_node_features,
)


class PretrainingCSVDataset(IterableDataset):
    required_columns = ("smiles", "source_file", "source_line")
    node_feature_dim = FINAL_NODE_FEATURE_DIM

    def __init__(
        self,
        path,
        max_samples=None,
        *,
        shard=True,
        denoising_pos_forward=True,
        count_rows=False,
        failure_log_path="embedding_failures.csv",
    ):
        self.path = path
        self.max_samples = max_samples
        self.shard = shard
        self.denoising_pos_forward = denoising_pos_forward
        self.failure_log_path = failure_log_path
        self._logged_failures = set()
        self._row_count = None
        self._validate_columns()
        if count_rows or max_samples is not None:
            self._row_count = self._count_rows()

    def _validate_columns(self):
        with open(self.path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            missing = [column for column in self.required_columns if column not in fieldnames]
            if missing:
                raise ValueError(f"Missing required columns in {self.path}: {missing}")

    def _count_rows(self):
        with open(self.path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = reader if self.max_samples is None else islice(reader, self.max_samples)
            return sum(1 for _ in rows)

    @property
    def total_rows(self):
        if self._row_count is None:
            self._row_count = self._count_rows()
        return self._row_count

    def _dist_context(self):
        world_size = 1
        rank = 0
        if self.shard and dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()

        worker = get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0

        shard_count = world_size * num_workers if self.shard else 1
        shard_index = rank * num_workers + worker_id if self.shard else 0
        return shard_count, shard_index

    def _embed_smiles(self, row, seed):
        smiles = row["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        status = AllChem.EmbedMolecule(mol, params)
        if status == 0:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass
        else:
            failure_key = (smiles, row["source_file"], row["source_line"], "EmbedMoleculeFailed")
            if failure_key not in self._logged_failures:
                append_embedding_failure_log(
                    self.failure_log_path,
                    smiles=smiles,
                    source_file=row["source_file"],
                    source_line=row["source_line"],
                    failure_reason="EmbedMoleculeFailed",
                    fallback_used="2d",
                )
                self._logged_failures.add(failure_key)
            AllChem.Compute2DCoords(mol)

        conf = mol.GetConformer()
        pos = []
        atomic_numbers = []
        for atom_idx, atom in enumerate(mol.GetAtoms()):
            point = conf.GetAtomPosition(atom_idx)
            pos.append([point.x, point.y, point.z])
            atomic_numbers.append(atom.GetAtomicNum())

        pos = torch.tensor(pos, dtype=torch.float32)
        atom_index = torch.arange(pos.size(0), dtype=pos.dtype).unsqueeze(1)
        jitter = torch.cat(
            [
                torch.sin(atom_index * 0.37),
                torch.cos(atom_index * 0.53),
                torch.sin(atom_index * 0.71 + 0.5),
            ],
            dim=1,
        ) * 1e-3
        pos = pos + jitter
        return mol, pos, torch.tensor(atomic_numbers, dtype=torch.long)

    def _row_to_data(self, row, row_index):
        seed = int(
            hashlib.sha256(
                f"{row['smiles']}|{row['source_file']}|{row['source_line']}".encode("utf-8")
            ).hexdigest()[:8],
            16,
        ) % (2**31 - 1)
        mol, pos, atomic_numbers = self._embed_smiles(row, seed)
        natoms = pos.size(0)
        node_features = build_augmented_node_features(mol)
        data = Data(
            pos=pos,
            atomic_numbers=atomic_numbers,
            node_features=node_features,
            natoms=torch.tensor([natoms], dtype=torch.long),
            fixed=torch.zeros(natoms, dtype=torch.float32),
            forces=torch.zeros((natoms, 3), dtype=torch.float32),
            batch=torch.zeros(natoms, dtype=torch.long),
            cell=torch.eye(3, dtype=torch.float32).unsqueeze(0),
            tags=torch.full((natoms,), 2, dtype=torch.long),
            sid=torch.tensor([row_index], dtype=torch.long),
            fid=torch.tensor([0], dtype=torch.long),
            denoising_pos_forward=self.denoising_pos_forward,
            y=torch.zeros((1,), dtype=torch.float32),
        )
        data.smiles = row["smiles"]
        data.source_file = row["source_file"]
        data.source_line = int(row["source_line"])
        data.md = torch.tensor([0], dtype=torch.long)
        return data

    def _iter_rows(self, shard):
        if shard:
            shard_count, shard_index = self._dist_context()
        else:
            shard_count, shard_index = 1, 0

        with open(self.path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = reader if self.max_samples is None else islice(reader, self.max_samples)
            for row_index, row in enumerate(rows):
                if row_index % shard_count != shard_index:
                    continue
                yield row_index, row

    def __iter__(self):
        for row_index, row in self._iter_rows(shard=self.shard):
            yield self._row_to_data(row, row_index)

    def iter_unsharded(self):
        for row_index, row in self._iter_rows(shard=False):
            yield self._row_to_data(row, row_index)

    def __len__(self):
        total = self.total_rows
        if not self.shard:
            return total
        shard_count, shard_index = self._dist_context()
        if total <= shard_index:
            return 0
        return math.ceil((total - shard_index) / shard_count)
