#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("algorithm") != "sha256":
        raise ValueError(f"Unsupported algorithm: {data.get('algorithm')}")
    if "files" not in data:
        raise ValueError("Manifest missing 'files'")
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default=None,
        help="manifest 路径。默认 datasets/data_tools/hash_manifest.json",
    )
    ap.add_argument(
        "--scan-extra",
        action="store_true",
        help="额外扫描 datasets 下是否存在未记录的 *.csv",
    )
    ap.add_argument(
        "--fail-on-extra",
        action="store_true",
        help="配合 --scan-extra：把 EXTRA 也当失败（CI 用）",
    )
    args = ap.parse_args()

    proj_root = project_root_from_script()
    datasets_root = (proj_root / "datasets").resolve()

    manifest_path = Path(args.manifest).resolve() if args.manifest else (proj_root / "datasets" / "data_tools" / "hash_manifest.json").resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    manifest = load_manifest(manifest_path)
    expected: Dict[str, dict] = {f["relpath_from_datasets"]: f for f in manifest["files"]}

    missing: List[str] = []
    changed: List[Tuple[str, str, str]] = []
    size_changed: List[Tuple[str, int, int]] = []
    ok: List[str] = []

    for rel, meta in expected.items():
        p = datasets_root / rel
        if not p.exists():
            missing.append(rel)
            continue

        st = p.stat()
        exp_size = int(meta.get("bytes", -1))
        if exp_size != -1 and st.st_size != exp_size:
            size_changed.append((rel, exp_size, st.st_size))

        actual = sha256_file(p)
        exp = meta["sha256"]
        if actual != exp:
            changed.append((rel, exp, actual))
        else:
            ok.append(rel)

    extra: List[str] = []
    if args.scan_extra:
        disk_csvs = sorted([p.relative_to(datasets_root).as_posix() for p in datasets_root.rglob("*.csv")])
        extra = [r for r in disk_csvs if r not in expected]

    print(f"project_root : {proj_root}")
    print(f"datasets_root : {datasets_root}")
    print(f"manifest      : {manifest_path}")
    print(f"tracked       : {len(expected)} files\n")

    if ok:
        print(f"[OK] {len(ok)} matched")
        for r in ok:
            print(f" - {r}")
        print()

    if missing:
        print(f"[MISSING] {len(missing)}")
        for r in missing:
            print(f" - {r}")
        print()

    if size_changed:
        print(f"[SIZE_CHANGED] {len(size_changed)}")
        for r, e, a in size_changed:
            print(f" - {r}  expected={e}  actual={a}")
        print()

    if changed:
        print(f"[CHANGED] {len(changed)}")
        for r, e, a in changed:
            print(f" - {r}")
            print(f"   expected: {e}")
            print(f"   actual:   {a}")
        print()

    if args.scan_extra:
        if extra:
            print(f"[EXTRA] {len(extra)} (disk 上存在但 manifest 未记录)")
            for r in extra:
                print(f" - {r}")
            print()
        else:
            print("[EXTRA] 0\n")

    failed = bool(missing or changed or size_changed)
    if args.scan_extra and args.fail_on_extra and extra:
        failed = True

    print("RESULT: " + ("FAIL" if failed else "PASS"))
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
