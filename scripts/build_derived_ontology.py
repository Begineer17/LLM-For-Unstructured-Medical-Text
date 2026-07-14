#!/usr/bin/env python3
"""Build immutable derived ICD/RxNorm indexes without changing raw snapshots."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ontology_linking import DerivedOntologyIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology-root", default="data/ontology")
    parser.add_argument("--output", default="data/ontology/derived")
    args = parser.parse_args()

    root = Path(args.ontology_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"schema_version": 1, "indexes": {}}
    for kind in ("icd", "rxnorm"):
        raw = root / kind / "concepts.jsonl"
        index = DerivedOntologyIndex(raw, kind=kind)
        index_path = output / f"{kind}.index.jsonl"
        index.write_index(index_path)
        release = "2026" if kind == "icd" else "2026-07-06"
        manifest["indexes"][kind] = {
            **index.manifest_record(),
            "release": release,
            "index": str(index_path).replace("\\", "/"),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
