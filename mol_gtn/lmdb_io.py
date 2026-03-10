from __future__ import annotations

import io
from typing import Any

import lmdb
import torch


def serialize_data(data: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(data, buffer)
    return buffer.getvalue()


def deserialize_data(blob: bytes) -> Any:
    buffer = io.BytesIO(blob)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def open_lmdb(path: str, readonly: bool = False, map_size: int = 1 << 40) -> lmdb.Environment:
    return lmdb.open(
        path,
        map_size=map_size,
        subdir=False,
        readonly=readonly,
        lock=not readonly,
        readahead=readonly,
        meminit=False,
        max_readers=512,
    )
