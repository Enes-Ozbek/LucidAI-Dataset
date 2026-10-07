#!/usr/bin/env python3
"""Check and merge the team's hand-written test examples into one external test set.

    python validate_external_tests.py external_tests/*.csv

Each CSV uses the columns of external_tests/external_tests_template.csv. Rows whose author is
EXAMPLE are skipped. A row is rejected if its label is not allow/deny, its user request is not
English, it is longer than the limits used for the main data, it duplicates another row with a
different label, or its tool call already appears in train/val/test (then it would not be an
external test).

Writes data/external_test.csv and data/external_test_report.md.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from mayi_input import build_input
from prepare_dataset import LABEL_TEXT, EnglishFilter, describe, loose, norm, write_split

HERE = Path(__file__).resolve().parent
LABELS = {"allow": 0, "approve": 0, "safe": 0, "0": 0, "deny": 1, "unsafe": 1, "block": 1, "1": 1}
REQUIRED = ("author", "tool", "label")
USER_REQUEST = re.compile(r"### USER REQUEST\n(.*?)\n\n### AGENT HISTORY", re.DOTALL)


def read_rows(paths):
    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as f:  # utf-8-sig: Excel adds a BOM
            reader = csv.DictReader(f)
            missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
            if missing:
                raise SystemExit(f"{path}: missing columns {missing}; start from the template")
            for line_no, row in enumerate(reader, start=2):
                yield path.name, line_no, {k: (v or "").strip() for k, v in row.items() if k}


def main_data(folder):
    """Calls already in train/val/test, and their user requests as English reference text."""
    known, reference = {}, []
    csv.field_size_limit(10_000_000)
    for split in ("train", "val", "test"):
        with open(folder / f"{split}.csv", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                known.setdefault(loose(row["input_a"]), split)
                request = USER_REQUEST.search(row["input_b"])
                reference.append({"fields": {"user_request": request.group(1) if request else ""}})
    return known, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument("--data", type=Path, default=HERE / "data", help="folder with train/val/test.csv")
    args = parser.parse_args()

    settings = json.load(open(args.data / "stats.json", encoding="utf-8"))["settings"]
    max_call, max_input = int(settings["max_call_chars"]), int(settings["max_input_chars"])
    known, reference = main_data(args.data)
    # same English check as the main data, with the vocabulary taken from its user requests
    english = EnglishFilter(float(settings["min_en_prob"]), reference, float(settings["min_vocab_share"]))

    examples, problems, skipped = [], [], 0
    per_author, written = defaultdict(int), Counter()
    for file_name, line_no, row in read_rows(args.csv_files):
        where = f"{file_name} line {line_no}"
        if row.get("author", "").upper() == "EXAMPLE" or not any(row.values()):
            skipped += 1
            continue
        written[row["author"]] += 1
        label = LABELS.get(row["label"].lower())
        if not row["tool"]:
            problems.append(f"{where}: empty tool")
            continue
        if label is None:
            problems.append(f"{where}: label must be allow or deny, got {row['label']!r}")
            continue
        per_author[row["author"]] += 1
        fields = {"call": {"tool": row["tool"], "args": row.get("args", "")},
                  "user_request": row.get("user_request", ""),
                  "history": row.get("history") or None}
        ex = {
            "id": f"team-{row['author'].lower().replace(' ', '_')}-{per_author[row['author']]:03d}",
            "source": "team",
            "label": label,
            "fields": fields,
            "meta": {"category": row.get("category") or None, "difficulty": "unknown", "framework": None,
                     "domain": None, "gen_mode": None, "rationale": row.get("notes") or None,
                     "author": row["author"], "origin": where, "call_has_both_labels": False},
            "input_a": build_input(fields, "A"),
            "input_b": build_input(fields, "B"),
        }
        ok, why = english(ex)
        if not ok:
            problems.append(f"{where}: the user request is not English ({why})")
        elif len(ex["input_a"]) > max_call:
            problems.append(f"{where}: the tool call is {len(ex['input_a'])} chars, limit is {max_call}")
        elif len(ex["input_b"]) > max_input:
            problems.append(f"{where}: call plus request and history is {len(ex['input_b'])} chars, limit is {max_input}")
        elif loose(ex["input_a"]) in known:
            problems.append(f"{where}: the same tool call is already in {known[loose(ex['input_a'])]}; write a new one")
        else:
            examples.append(ex)

    # duplicates inside the team set: identical rows with different labels need a team decision
    labels_by_key = defaultdict(set)
    for ex in examples:
        labels_by_key[norm(ex["input_b"])].add(ex["label"])
    kept, seen = [], set()
    for ex in examples:
        key = norm(ex["input_b"])
        if len(labels_by_key[key]) > 1:
            problems.append(f"{ex['meta']['origin']}: same call, request and history as another row but a different label")
        elif key not in seen:
            seen.add(key)
            kept.append(ex)
        else:
            problems.append(f"{ex['meta']['origin']}: duplicate of an earlier row (kept the first)")

    # calls written with both labels (different request or history) are the interesting A-vs-B cases
    by_call = defaultdict(set)
    for ex in kept:
        by_call[loose(ex["input_a"])].add(ex["label"])
    for ex in kept:
        ex["meta"]["call_has_both_labels"] = len(by_call[loose(ex["input_a"])]) > 1

    write_split(kept, args.data, "external_test")
    summary = describe(kept) if kept else {"rows": 0}
    authors = Counter((ex["meta"]["author"], LABEL_TEXT[ex["label"]]) for ex in kept)
    lines = ["# External test set check", "",
             f"Kept {len(kept)} rows; skipped {skipped} example/empty rows; {len(problems)} problems.", "",
             "| author | rows written | kept allow | kept deny |", "|---|---:|---:|---:|"]
    for author in sorted(written):
        lines.append(f"| {author} | {written[author]} | {authors[(author, 'allow')]} | {authors[(author, 'deny')]} |")
    lines += ["", f"Calls written with both labels: {sum(ex['meta']['call_has_both_labels'] for ex in kept)} rows", ""]
    lines += ["## Problems", ""] + ([f"- {p}" for p in problems] or ["None."])
    (args.data / "external_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(json.dumps({k: summary.get(k) for k in ("rows", "labels", "deny_share")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
