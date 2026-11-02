#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--minimum-valid", type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    failures = []
    for record in manifest["records"]:
        path = Path(record["output"])
        with np.load(path, allow_pickle=False) as data:
            valid = data["target_valid"].astype(bool)
            if int(valid.sum()) < args.minimum_valid:
                failures.append(f"{path}: only {int(valid.sum())} valid labels")
            if valid.any():
                uv = data["uv"][valid]
                if not (np.isfinite(uv).all() and (uv >= 0).all() and (uv <= 1).all()):
                    failures.append(f"{path}: invalid normalized coordinates")
                if not np.array_equal(data["present"][valid], np.ones(valid.sum(), dtype=np.uint8)):
                    failures.append(f"{path}: point label without object-present target")
    if failures:
        raise SystemExit("label validation failed:\n" + "\n".join(failures[:50]))
    print(json.dumps({"episodes": len(manifest["records"]), "status": "ok"}))


if __name__ == "__main__":
    main()
