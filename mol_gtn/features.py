from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from rdkit import Chem


ATOM_CHIRALITY = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.UNSPECIFIED,
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]
BOND_DIR = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


def _one_hot(value, choices: Sequence) -> List[float]:
    return [1.0 if value == item else 0.0 for item in choices]


def pooled_edge_features(mol: Chem.Mol) -> Tuple[List[List[float]], List[List[float]]]:
    bond_type_counts: List[List[float]] = []
    stereo_states: List[List[float]] = []
    for atom in mol.GetAtoms():
        counts = [0.0, 0.0, 0.0, 0.0]
        has_cis = False
        has_trans = False
        for bond in atom.GetBonds():
            bond_type = bond.GetBondType()
            if bond_type in BOND_TYPES:
                counts[BOND_TYPES.index(bond_type)] += 1.0
            stereo = bond.GetStereo()
            if stereo in (Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOCIS):
                has_cis = True
            if stereo in (Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOTRANS):
                has_trans = True
        stereo = [1.0 if has_cis else 0.0, 1.0 if has_trans else 0.0, 0.0]
        if not has_cis and not has_trans:
            stereo[2] = 1.0
        bond_type_counts.append(counts)
        stereo_states.append(stereo)
    return bond_type_counts, stereo_states


def atom_features(mol: Chem.Mol) -> torch.Tensor:
    bond_counts, stereo_states = pooled_edge_features(mol)
    rows: List[List[float]] = []
    for idx, atom in enumerate(mol.GetAtoms()):
        row = [
            float(atom.GetAtomicNum()),
            float(atom.GetFormalCharge()),
            float(atom.GetNumRadicalElectrons()),
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.GetValence(Chem.ValenceType.IMPLICIT)),
        ]
        row.extend(_one_hot(atom.GetChiralTag(), ATOM_CHIRALITY))
        row.extend(_one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES))
        row.extend(bond_counts[idx])
        row.extend(stereo_states[idx])
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def bond_feature_vector(bond: Chem.Bond) -> List[float]:
    row: List[float] = []
    row.extend(_one_hot(bond.GetBondType(), BOND_TYPES))
    row.extend(_one_hot(bond.GetStereo(), BOND_STEREO))
    row.extend(_one_hot(bond.GetBondDir(), BOND_DIR))
    row.append(float(bond.GetIsConjugated()))
    row.append(float(bond.IsInRing()))
    return row


def bond_features_and_index(mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_index: List[List[int]] = [[], []]
    edge_attr: List[List[float]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        feat = bond_feature_vector(bond)
        edge_index[0].extend([begin, end])
        edge_index[1].extend([end, begin])
        edge_attr.extend([feat, feat])
    if not edge_attr:
        edge_dim = len(bond_feature_vector(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, edge_dim), dtype=torch.float32),
        )
    return (
        torch.tensor(edge_index, dtype=torch.long),
        torch.tensor(edge_attr, dtype=torch.float32),
    )


def feature_dimensions() -> Dict[str, int]:
    atom_dim = 6 + len(ATOM_CHIRALITY) + len(HYBRIDIZATION_TYPES) + 4 + 3
    bond_dim = len(BOND_TYPES) + len(BOND_STEREO) + len(BOND_DIR) + 2
    return {"atom_dim": atom_dim, "bond_dim": bond_dim}


def validate_feature_dimensions(x: torch.Tensor, edge_attr: torch.Tensor) -> None:
    dims = feature_dimensions()
    if x.ndim != 2 or x.shape[1] != dims["atom_dim"]:
        raise ValueError(f"Atom feature dimension mismatch: expected {dims['atom_dim']}, got {tuple(x.shape)}")
    if edge_attr.ndim != 2 or edge_attr.shape[1] != dims["bond_dim"]:
        raise ValueError(f"Bond feature dimension mismatch: expected {dims['bond_dim']}, got {tuple(edge_attr.shape)}")
