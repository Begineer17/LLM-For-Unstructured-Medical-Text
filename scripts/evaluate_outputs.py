#!/usr/bin/env python3
"""Read-only structural and linking audit for an input/output directory pair."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clinical_nlp_pipeline import ENTITY_TYPES, validate_pair  # noqa: E402


PROCEDURE_CUES = ("cắt bỏ", "phẫu thuật", "thủ thuật", "chọc hút", "nạo hạch", "stent graft", "reduced from")


def audit_pair(raw: str, data: list[dict]) -> dict[str, object]:
    types = Counter(str(item.get("type")) for item in data)
    assertions = Counter(assertion for item in data for assertion in item.get("assertions", []))
    candidate_items = [item for item in data if item.get("type") in {"CHẨN_ĐOÁN", "THUỐC"}]
    duplicates_exact = sum(
        count - 1 for count in Counter((item.get("position", [None, None])[0], item.get("position", [None, None])[1], item.get("type")) for item in data).values() if count > 1
    )
    duplicates_span = sum(
        count - 1 for count in Counter((item.get("position", [None, None])[0], item.get("position", [None, None])[1]) for item in data).values() if count > 1
    )
    heading_leaks = [item.get("text", "") for item in data if re.match(r"^\s*\d+(?:\.\d+)*[.)]?\s+", item.get("text", "")) or item.get("text", "").strip().endswith(":")]
    procedure_as_drug = [item.get("text", "") for item in data if item.get("type") == "THUỐC" and any(cue in item.get("text", "").casefold() for cue in PROCEDURE_CUES)]
    long_spans = [item.get("text", "") for item in data if len(item.get("text", "")) >= 80]
    empty_candidates = [item.get("text", "") for item in candidate_items if not item.get("candidates")]
    return {
        "entities": len(data),
        "types": dict(types),
        "assertions": dict(assertions),
        "candidate_entities": len(candidate_items),
        "nonempty_candidates": sum(bool(item.get("candidates")) for item in candidate_items),
        "empty_candidates": len(empty_candidates),
        "empty_candidate_text": empty_candidates,
        "duplicate_exact": duplicates_exact,
        "duplicate_span": duplicates_span,
        "heading_leaks": heading_leaks,
        "procedure_as_drug": procedure_as_drug,
        "long_spans_ge80": long_spans,
        "schema_errors": validate_pair(raw, data),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input_2x")
    parser.add_argument("--output", default="output_2x")
    args = parser.parse_args()
    input_dir, output_dir = Path(args.input), Path(args.output)
    report: dict[str, object] = {"files": 0, "per_file": {}, "totals": {}}
    totals = Counter()
    all_assertions = Counter()
    for input_path in sorted(input_dir.glob("*.txt"), key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem):
        output_path = output_dir / f"{input_path.stem}.json"
        if not output_path.exists():
            report["per_file"][input_path.stem] = {"missing_output": True}
            continue
        raw = input_path.read_text(encoding="utf-8")
        data = json.loads(output_path.read_text(encoding="utf-8"))
        stats = audit_pair(raw, data)
        report["per_file"][input_path.stem] = stats
        report["files"] += 1
        totals["entities"] += int(stats["entities"])
        totals["candidate_entities"] += int(stats["candidate_entities"])
        totals["nonempty_candidates"] += int(stats["nonempty_candidates"])
        totals["empty_candidates"] += int(stats["empty_candidates"])
        totals["duplicate_exact"] += int(stats["duplicate_exact"])
        totals["duplicate_span"] += int(stats["duplicate_span"])
        totals["heading_leaks"] += len(stats["heading_leaks"])
        totals["procedure_as_drug"] += len(stats["procedure_as_drug"])
        totals["long_spans_ge80"] += len(stats["long_spans_ge80"])
        for name, count in stats["types"].items():
            totals[f"type:{name}"] += count
        all_assertions.update(stats["assertions"])
    report["totals"] = {**dict(totals), "assertions": dict(all_assertions)}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
