#!/usr/bin/env python3
"""LucidAI-Dataset preparation.

download -> English + length bucket filter -> dedup -> stratified split (seed 42, repeated calls go to train) -> A/B inputs

Main data: ProCreations/auto-1b-data: proposed agent tool calls judged in context (the user's
request and the agent's earlier steps), labeled approve/deny (Apache-2.0).
Outside test sets, never used for training:
  - ProCreations/approve-or-deny: the audited benchmark built for auto-1b-data (Apache-2.0).
  - karanxa/agent-action-safety-dataset ("saroku"): tool calls labeled safe/unsafe by a
    different generator, with rules instead of a user request (Apache-2.0).

Every clean row is kept: nothing is sampled or capped unless --max-rows, --max-per-call or a
length limit asks for it. The CSVs are gzip-compressed, and a file over 49 MB is saved in parts
(train_1.csv.gz, train_2.csv.gz, ...) because GitHub refuses files over 100 MB. pandas reads
each part with pd.read_csv; pd.concat joins them.

Labels everywhere: 1 = deny, 0 = allow (approve).

Run in Colab (HuggingFace is reachable there; the main train file is 1 GB):
    pip install scikit-learn huggingface_hub pyarrow langdetect==1.0.9
    python prepare_dataset.py                    # downloads into raw/, writes data/
    python prepare_dataset.py --no-download      # use files already in raw/
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from mayi_input import build_input

HERE = Path(__file__).resolve().parent

SOURCES = {
    "auto1b": {"repo": "ProCreations/auto-1b-data", "files": ["train.parquet", "validation.parquet"]},
    "approve_or_deny": {"repo": "ProCreations/approve-or-deny", "files": ["test.parquet"]},
    "saroku": {"repo": "karanxa/agent-action-safety-dataset", "files": ["train.jsonl", "val.jsonl"]},
}
LABEL_TEXT = {0: "allow", 1: "deny"}
PARQUET_COLUMNS = ["label", "user_request", "history", "call", "category", "difficulty", "rationale",
                   "framework", "domain", "lang", "length_bucket", "is_long", "gen_mode", "subset"]
CSV_COLUMNS = ["id", "input_a", "input_b", "label", "label_text", "difficulty", "length_bucket", "category",
               "framework", "domain", "gen_mode", "call_has_both_labels", "rationale", "source"]


# ---------------------------------------------------------------- download / load

def fetch(name, raw_dir, download, files=None):
    """Return local paths of a source's files, downloading missing ones from HuggingFace."""
    spec = SOURCES[name]
    folder = raw_dir / name
    paths = []
    for filename in files or spec["files"]:
        path = folder / filename
        if not path.exists() and download:
            from huggingface_hub import hf_hub_download
            print(f"downloading {spec['repo']}/{filename}")
            hf_hub_download(spec["repo"], filename, repo_type="dataset", local_dir=folder)
        if not path.exists():
            raise SystemExit(f"missing {path}: download {spec['repo']}/{filename} or drop --no-download")
        paths.append(path)
    return paths


def read_jsonl(path, stats, args):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                stats["rows_in_files"] = stats.get("rows_in_files", 0) + 1
                yield json.loads(line)


def read_parquet(path, stats, args):
    """Yield the English rows in the kept length buckets of a parquet file, one batch at a time.

    The language and length checks use the dataset's own `lang`, `is_long` and `length_bucket`
    columns and run on whole batches before any row becomes a Python dict: the train file is
    1 GB and its long-context rows hold up to 65k tokens each.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    def add(key, n):
        stats[key] = stats.get(key, 0) + int(n)

    pf = pq.ParquetFile(path)
    columns = [c for c in PARQUET_COLUMNS if c in pf.schema_arrow.names]
    buckets = args.length_buckets.split(",")
    for batch in pf.iter_batches(batch_size=2048, columns=columns):
        english = pc.fill_null(pc.equal(batch["lang"], "English"), False)
        not_long = pc.fill_null(pc.equal(batch["is_long"], 0), False)
        in_bucket = pc.fill_null(pc.is_in(batch["length_bucket"], value_set=pa.array(buckets, type=batch["length_bucket"].type)), False)
        add("rows_in_files", batch.num_rows)
        add("dropped_not_english_lang", batch.num_rows - pc.sum(english).as_py())
        add("dropped_long_context", pc.sum(pc.and_(english, pc.invert(not_long))).as_py() or 0)
        add("dropped_length_bucket", pc.sum(pc.and_(pc.and_(english, not_long), pc.invert(in_bucket))).as_py() or 0)
        yield from batch.filter(pc.and_(pc.and_(english, not_long), in_bucket)).to_pylist()


READERS = {".jsonl": read_jsonl, ".parquet": read_parquet}


def content_id(*parts):
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12]


def from_auto1b(row, origin, source):
    label = {"approve": 0, "allow": 0, "deny": 1}.get(str(row.get("label")).strip().lower())
    if label is None or not row.get("call"):
        return None
    return {
        "id": f"{source}-{content_id(row.get('call'), row.get('user_request'), row.get('history'))}",
        "source": source,
        "label": label,
        "fields": {"call": row.get("call"), "user_request": row.get("user_request"),
                   "history": row.get("history")},
        "meta": {"category": row.get("category"), "difficulty": row.get("difficulty") or "unknown",
                 "length_bucket": row.get("length_bucket"),
                 "rationale": row.get("rationale"), "framework": row.get("framework"),
                 "domain": row.get("domain"), "gen_mode": row.get("gen_mode"), "origin_split": origin},
    }


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


USER_TASK = re.compile(r"\s*User task:\s*(.*)$", re.IGNORECASE | re.DOTALL)


def from_saroku(row, origin, source):
    """saroku has no user request: use the latest user message, else the goal, else the
    "User task:" part of its context. The rest of the context and its rules become the extra
    CONTEXT and RULES sections."""
    safe = as_bool(row.get("is_safe"))
    if safe is None or not row.get("action"):
        return None
    context = row.get("context") if isinstance(row.get("context"), str) else ""
    task = USER_TASK.search(context)
    request = row.get("user_message") or row.get("original_goal") or (task.group(1) if task else None)
    action = row["action"] if isinstance(row["action"], str) else json.dumps(row["action"])
    return {
        "id": f"saroku-{row.get('id')}",
        "source": source,
        "label": 0 if safe else 1,
        "fields": {"call": action, "user_request": request, "history": None,
                   "context": USER_TASK.sub("", context).strip() if task and request == task.group(1) else context,
                   "rules": [r for r in (row.get("constraints") or []) if isinstance(r, str)]},
        "meta": {"category": row.get("property") or "safe", "difficulty": "unknown", "length_bucket": None,
                 "rationale": row.get("reason"), "framework": None, "domain": row.get("domain"),
                 "gen_mode": None, "origin_split": origin},
    }


CONVERTERS = {"auto1b": from_auto1b, "approve_or_deny": from_auto1b, "saroku": from_saroku}


def load_source(name, args, stats, files=None):
    examples, skipped = [], 0
    for path in fetch(name, args.raw_dir, not args.no_download, files):
        for row in READERS[path.suffix](path, stats, args):
            ex = CONVERTERS[name](row, path.stem, name)
            if ex is None:
                skipped += 1
                continue
            ex["input_a"] = build_input(ex["fields"], "A")
            ex["input_b"] = build_input(ex["fields"], "B")
            examples.append(ex)
    stats["loaded"] = len(examples)
    stats["skipped_unusable"] = skipped  # rows without a label or a tool call
    return examples


# ---------------------------------------------------------------- filters

def norm(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def loose(text):
    """norm() plus digits masked and punctuation dropped, for near-duplicates."""
    text = re.sub(r"\d+", "0", norm(text))
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


CODE_TOKEN = re.compile(r"[_/\\=(){}\[\]<>@`$|]|\w\.\w|\d")
TEMPLATE_LABELS = re.compile(r"\b(agent role|environment|user task|prior steps)\s*:", re.IGNORECASE)
# common English function words that are rare in other languages; used to overrule
# langdetect, which misreads short technical English (e.g. "Clean up build folder") as Danish
ENGLISH_WORDS = set("""the to of and for with from by at are this that it all any only not please my our
your we you can will should must never without before after into if then than out now using each every
have has was were been which who what when where how or""".split())
MIN_PROSE_WORDS = 6  # below this there is too little prose to tell the language
MIN_FUNCTION_WORD_SHARE = 0.15


def natural_language(ex):
    """The prose of an example (user request, context, rules), without code-like tokens."""
    f = ex["fields"]
    parts = [f.get("user_request"), f.get("context"), *(f.get("rules") or [])]
    text = TEMPLATE_LABELS.sub(" ", " ".join(p for p in parts if isinstance(p, str)))
    return " ".join(w for w in text.split() if not CODE_TOKEN.search(w))


def prose_words(prose):
    return re.findall(r"[a-z]+", prose.lower())


def function_word_share(words):
    return sum(w in ENGLISH_WORDS for w in words) / len(words) if words else 0.0


def non_ascii_letter_ratio(text):
    letters = [ch for ch in text if ch.isalpha()]
    return sum(ord(ch) > 127 for ch in letters) / len(letters) if letters else 0.0


class EnglishFilter:
    """Keeps an example when its prose is (almost) all ASCII letters and reads as English:
    enough English function words, mostly words from the English vocabulary of `reference`, or
    langdetect gives English at least `min_prob`.

    The vocabulary is every word of the reference examples whose prose passes the function-word
    test. It is there because langdetect misreads terse technical English: it called "Purge
    re-processed dead-letter queue messages per change request" French.
    """

    def __init__(self, min_prob, reference=(), min_vocab_share=0.7):
        try:
            from langdetect import DetectorFactory, detect_langs
            from langdetect.lang_detect_exception import LangDetectException
        except ImportError:
            raise SystemExit("the English filter needs langdetect: pip install langdetect==1.0.9")
        DetectorFactory.seed = 0  # langdetect is random without a fixed seed
        self.detect, self.error, self.min_prob = detect_langs, LangDetectException, min_prob
        self.cache = {}
        self.min_vocab_share = min_vocab_share
        self.vocab = set()
        self.learn(reference)

    def learn(self, examples):
        """Add the words of examples whose prose clearly reads as English to the vocabulary."""
        for ex in examples:
            words = prose_words(natural_language(ex))
            if len(words) >= MIN_PROSE_WORDS and function_word_share(words) >= MIN_FUNCTION_WORD_SHARE:
                self.vocab.update(words)

    def __call__(self, ex):
        prose = natural_language(ex)
        if non_ascii_letter_ratio(prose) > 0.10:
            return False, "non_ascii_text"
        words = prose_words(prose)
        if len(words) < MIN_PROSE_WORDS:
            return True, "too_little_prose_to_check"
        if function_word_share(words) >= MIN_FUNCTION_WORD_SHARE:
            return True, "english_function_words"
        if self.vocab and sum(w in self.vocab for w in words) / len(words) >= self.min_vocab_share:
            return True, "english_vocabulary"
        if prose not in self.cache:
            try:
                probs = {lang.lang: lang.prob for lang in self.detect(prose)}
            except self.error:
                probs = {}
            self.cache[prose] = probs
        probs = self.cache[prose]
        if not probs or probs.get("en", 0.0) >= self.min_prob:
            return True, "langdetect_en"
        top = max(probs, key=probs.get)
        return False, f"detected_{top}"


def apply_filters(examples, args, english, stats, dropped):
    kept = []
    reasons, checks = Counter(), Counter()
    for ex in examples:
        ok, why = english(ex)
        checks[why] += 1
        if not ok:
            reasons["not_english"] += 1
            dropped.append({**ex, "dropped_because": f"not_english ({why})"})
            continue
        if args.max_call_chars and len(ex["input_a"]) > args.max_call_chars:
            reasons["call_too_long"] += 1
            dropped.append({**ex, "dropped_because": "call_too_long"})
            continue
        if args.max_input_chars and len(ex["input_b"]) > args.max_input_chars:
            reasons["input_too_long"] += 1
            dropped.append({**ex, "dropped_because": "input_too_long"})
            continue
        kept.append(ex)
    stats["english_check"] = dict(checks.most_common())  # which test decided, for every row
    stats["dropped_by_filter"] = dict(reasons)
    stats["after_filter"] = len(kept)
    return kept


# ---------------------------------------------------------------- dedup

def dedup(examples, stats, dropped):
    """1) exact duplicates of the full B input: keep one; drop every copy if their labels disagree.
    2) near duplicates (digits masked, punctuation dropped): keep the first example per label,
       so pairs that differ only in a number and carry different labels both survive."""
    labels_by_key = defaultdict(set)
    for ex in examples:
        labels_by_key[norm(ex["input_b"])].add(ex["label"])
    seen, step1 = set(), []
    counts = Counter()
    for ex in examples:
        key = norm(ex["input_b"])
        if len(labels_by_key[key]) > 1:
            counts["exact_dup_conflicting_labels"] += 1
            dropped.append({**ex, "dropped_because": "exact_duplicate_with_conflicting_labels"})
        elif key in seen:
            counts["exact_duplicate"] += 1
            dropped.append({**ex, "dropped_because": "exact_duplicate"})
        else:
            seen.add(key)
            step1.append(ex)

    seen, kept = set(), []
    for ex in step1:
        key = (loose(ex["input_b"]), ex["label"])
        if key in seen:
            counts["near_duplicate"] += 1
            dropped.append({**ex, "dropped_because": "near_duplicate"})
        else:
            seen.add(key)
            kept.append(ex)
    stats["dropped_by_dedup"] = dict(counts)
    stats["after_dedup"] = len(kept)
    return kept


# ---------------------------------------------------------------- sample and split

def group_strata(examples, strata_key):
    """Group examples by tool call (format A after loose normalisation) and give each group a
    stratum "label|<strata_key>"; strata too small to split fall back to label only, then to
    the biggest stratum."""
    groups = defaultdict(list)
    for i, ex in enumerate(examples):
        groups[loose(ex["input_a"])].append(i)
    keys = list(groups)  # insertion order = first appearance, so results are reproducible

    def stratum(key):
        members = [examples[i] for i in groups[key]]
        labels = {ex["label"] for ex in members}
        label = str(labels.pop()) if len(labels) == 1 else "mixed"
        value = Counter(str(ex["meta"].get(strata_key)) for ex in members).most_common(1)[0][0]
        return f"{label}|{value}"

    strata = [stratum(k) for k in keys]
    size = Counter(strata)
    strata = [s if size[s] >= 10 else s.split("|")[0] for s in strata]
    size = Counter(strata)
    biggest = size.most_common(1)[0][0]
    strata = [s if size[s] >= 3 else biggest for s in strata]
    return groups, keys, strata


def sample_groups(examples, n, seed, strata_key):
    """A stratified sample of about n examples, taken whole tool-call groups at a time."""
    if n <= 0 or n >= len(examples):
        return examples
    groups, keys, strata = group_strata(examples, strata_key)
    chosen, _ = train_test_split(list(range(len(keys))), train_size=n / len(examples),
                                 random_state=seed, stratify=strata)
    return [examples[i] for i in sorted(i for g in chosen for i in groups[keys[g]])]


def cap_calls(rows, cap, seed):
    """Keep at most `cap` rows per tool call (format A after loose normalisation).

    The generator wrote a few calls hundreds of times with different requests: without a cap,
    `git reset --hard HEAD` alone filled 597 rows (6.6%) of the validation split. Each label keeps
    its share of the call's rows and at least one row, so a call seen with both labels keeps both.
    Rows are picked by a seeded hash of their id, so the choice does not depend on row order.
    """
    if cap <= 0:
        return rows
    groups = defaultdict(list)
    for i, ex in enumerate(rows):
        groups[loose(ex["input_a"])].append(i)
    drop = set()
    for members in groups.values():
        if len(members) <= cap:
            continue
        by_label = defaultdict(list)
        for i in members:
            by_label[rows[i]["label"]].append(i)
        left = cap
        for j, label in enumerate(sorted(by_label, key=lambda l: (len(by_label[l]), l))):  # smaller label first
            items = by_label[label]
            quota = left if j == len(by_label) - 1 else max(1, round(cap * len(items) / len(members)))
            quota = min(quota, len(items))
            left -= quota
            items.sort(key=lambda i: hashlib.sha1(f"{seed}:{rows[i]['id']}".encode("utf-8")).hexdigest())
            drop.update(items[quota:])
    return [ex for i, ex in enumerate(rows) if i not in drop]


def split(examples, args, stats):
    """Stratified train/val/test split with a fixed seed.

    Examples with the same tool call form one group and always land in the same split, so a
    format-A model cannot pass the test by memorising a call it saw in training. Groups are
    stratified by label and difficulty.

    The generator wrote a few calls hundreds of times with different requests (`git reset --hard
    HEAD` once filled 6.6% of a validation split), and in val or test such a call would sway the
    score. Calls with more than --train-only-calls rows therefore all go to train; nothing is
    dropped, and val and test still get --val-size and --test-size of all rows. --max-per-call
    can cap every call instead.
    """
    by_call = defaultdict(list)
    for i, ex in enumerate(examples):
        by_call[loose(ex["input_a"])].append(i)
    limit = args.train_only_calls
    big = [k for k, v in by_call.items() if limit and len(v) > limit]
    big_rows = sorted(i for k in big for i in by_call[k])
    big_set = set(big_rows)
    rest_idx = [i for i in range(len(examples)) if i not in big_set]
    share = len(rest_idx) / len(examples)
    test_size, val_size = args.test_size / share, args.val_size / share
    if test_size + val_size >= 1:
        raise SystemExit(f"--train-only-calls {limit} leaves too few rows for val and test")

    groups, keys, strata = group_strata([examples[i] for i in rest_idx], "difficulty")
    idx = list(range(len(keys)))
    rest, test = train_test_split(idx, test_size=test_size, random_state=args.seed, stratify=strata)
    train, val = train_test_split(rest, test_size=val_size / (1 - test_size), random_state=args.seed,
                                  stratify=[strata[i] for i in rest])

    out = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        rows = [rest_idx[i] for g in part for i in groups[keys[g]]]
        if name == "train":
            rows += big_rows
        rows.sort()  # keep source order inside a split
        out[name] = cap_calls([examples[i] for i in rows], args.max_per_call, args.seed)

    # calls that are approved in one context and denied in another: format A cannot get these right
    mixed = {k for k, v in by_call.items() if len({examples[i]["label"] for i in v}) > 1}
    for k, v in by_call.items():
        for i in v:
            examples[i]["meta"]["call_has_both_labels"] = k in mixed
    kept = [ex for rows in out.values() for ex in rows]
    stats["call_groups"] = len(by_call)
    stats["largest_call_groups"] = sorted((len(v) for v in by_call.values()), reverse=True)[:5]
    stats["calls_to_train_only"] = len(big)
    stats["rows_to_train_only"] = len(big_rows)
    stats["calls_over_cap"] = sum(len(v) > args.max_per_call > 0 for v in by_call.values())
    stats["dropped_by_call_cap"] = len(examples) - len(kept)
    stats["in_splits"] = len(kept)
    stats["calls_with_both_labels"] = len(mixed)
    stats["rows_in_both_label_calls"] = sum(ex["meta"]["call_has_both_labels"] for ex in kept)
    return out


# ---------------------------------------------------------------- output

def length_summary(texts):
    chars = sorted(len(t) for t in texts)
    if not chars:
        return {}
    return {"min": chars[0], "median": statistics.median(chars),
            "p95": chars[int(0.95 * (len(chars) - 1))], "max": chars[-1]}


def write_csv_gz(rows, path):
    """The gzip header holds no file name or time, so reruns give the same bytes."""
    with open(path, "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as packed, \
            io.TextIOWrapper(packed, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for ex in rows:
            m = ex["meta"]
            writer.writerow([ex["id"], ex["input_a"], ex["input_b"], ex["label"], LABEL_TEXT[ex["label"]],
                             m.get("difficulty"), m.get("length_bucket"), m.get("category"), m.get("framework"),
                             m.get("domain"), m.get("gen_mode"), m.get("call_has_both_labels", False),
                             m.get("rationale"), ex["source"]])


def write_split(rows, folder, name, max_mb=0):
    """Write <name>.csv.gz, or <name>_1.csv.gz, <name>_2.csv.gz, ... when one file would be over
    max_mb (GitHub warns about files over 50 MB and refuses files over 100 MB). Returns the file names."""
    for old in [folder / f"{name}.csv", *folder.glob(f"{name}_[0-9]*.csv.gz")]:
        old.unlink(missing_ok=True)  # plain CSV or parts left by an earlier run
    path = folder / f"{name}.csv.gz"
    write_csv_gz(rows, path)
    size = path.stat().st_size
    if not max_mb or size <= max_mb * 1e6:
        return [path.name]
    path.unlink()
    parts = math.ceil(size / (0.9 * max_mb * 1e6))
    per = math.ceil(len(rows) / parts)
    names = []
    for k in range(parts):
        part = folder / f"{name}_{k + 1}.csv.gz"
        write_csv_gz(rows[k * per:(k + 1) * per], part)
        names.append(part.name)
    return names


def describe(rows):
    labels = Counter(LABEL_TEXT[ex["label"]] for ex in rows)
    return {
        "rows": len(rows),
        "labels": dict(sorted(labels.items())),
        "deny_share": round(labels["deny"] / len(rows), 4) if rows else None,
        "difficulties": dict(Counter(str(ex["meta"].get("difficulty")) for ex in rows).most_common()),
        "length_buckets": dict(Counter(str(ex["meta"].get("length_bucket")) for ex in rows).most_common()),
        "categories": dict(Counter(str(ex["meta"].get("category")) for ex in rows).most_common()),
        "frameworks": dict(Counter(str(ex["meta"].get("framework")) for ex in rows).most_common()),
        "chars_input_a": length_summary([ex["input_a"] for ex in rows]),
        "chars_input_b": length_summary([ex["input_b"] for ex in rows]),
    }


def write_report(stats, path):
    """Human-readable summary of stats.json, for the README and the presentation."""
    s = stats["auto1b"]
    lines = ["# LucidAI-Dataset report", "", "## Main data (auto-1b-data)", "",
             "| step | rows |", "|---|---:|",
             f"| rows in the files | {s['rows_in_files']} |",
             f"| dropped: user language is not English | -{s['dropped_not_english_lang']} |",
             f"| dropped: long-context rows (up to 65k tokens) | -{s['dropped_long_context']} |"]
    if s.get("dropped_length_bucket"):
        lines.append(f"| dropped: length bucket not {stats['settings']['length_buckets'].replace(',', ', ')} | "
                     f"-{s['dropped_length_bucket']} |")
    for reason, n in {**s["dropped_by_filter"], **s["dropped_by_dedup"]}.items():
        lines.append(f"| dropped: {reason.replace('_', ' ')} | -{n} |")
    lines.append(f"| clean English rows | {s['after_dedup']} |")
    if "sampled" in s:
        lines.append(f"| sampled for the splits (`--max-rows`) | {s['sampled']} |")
    cap = stats["settings"]["max_per_call"]
    if s["dropped_by_call_cap"]:
        lines.append(f"| dropped: extra rows of a tool call seen more than {cap} times (`--max-per-call`) | "
                     f"-{s['dropped_by_call_cap']} |")
    lines.append(f"| **rows in train, val and test** | **{s['in_splits']}** |")
    checks = ", ".join(f"{why.replace('_', ' ')} {n}" for why, n in s["english_check"].items())
    lines += ["", f"English check on the user requests, the test that decided each row: {checks}."]

    names = [n for n in ("train", "val", "test", "ood_approve_or_deny", "ood_saroku") if n in stats]
    lines += ["", "## Splits", "",
              "| split | rows | allow | deny | deny share | median chars A | median chars B | max chars B |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        d = stats[n]
        lines.append(f"| {n} | {d['rows']} | {d['labels'].get('allow', 0)} | {d['labels'].get('deny', 0)} | "
                     f"{d['deny_share']:.1%} | {d['chars_input_a']['median']:.0f} | "
                     f"{d['chars_input_b']['median']:.0f} | {d['chars_input_b']['max']} |")

    shared = stats["shared_calls_between_splits"]
    lines += ["", f"Tool calls shared between splits (should be 0): train/val {shared['train_val']}, "
                  f"train/test {shared['train_test']}, val/test {shared['val_test']}.",
              f"Calls that appear with both labels (only the user request or history decides): "
              f"{s['calls_with_both_labels']} calls, {s['rows_in_both_label_calls']} rows.",
              f"Most repeated tool calls: {', '.join(map(str, s['largest_call_groups']))} rows."]
    if s.get("calls_to_train_only"):
        lines.append(f"Tool calls seen more than {stats['settings']['train_only_calls']} times all went to train "
                     f"(`--train-only-calls`): {s['calls_to_train_only']} calls, {s['rows_to_train_only']} rows.")
    if s.get("calls_over_cap"):
        lines.append(f"{s['calls_over_cap']} calls were cut to {cap} rows (`--max-per-call`).")

    files = "; ".join(f"{n}: {' + '.join(stats[n]['files'])}" for n in names if "files" in stats[n])
    lines += ["", f"Files: {files}."]

    for key, column, title in (("difficulties", "difficulty", "Difficulty"),
                               ("length_buckets", "length_bucket", "Length bucket (how much agent history)"),
                               ("frameworks", "framework", "Agent framework"),
                               ("categories", "category", "Categories (each one gives the label away: error analysis only)")):
        values = {v for n in ("train", "val", "test") for v in stats[n].get(key, {})}
        lines += ["", f"## {title}", "", f"| {column} | train | val | test |", "|---|---:|---:|---:|"]
        for v in sorted(values, key=lambda v: (-stats["train"].get(key, {}).get(v, 0), v)):
            lines.append(f"| {v} | " + " | ".join(str(stats[n].get(key, {}).get(v, 0)) for n in ("train", "val", "test")) + " |")

    for name, title in (("approve_or_deny", "Outside test: approve-or-deny benchmark"),
                        ("saroku", "Outside test: saroku")):
        if name in stats:
            o = stats[name]
            parts = [f"{o['rows_in_files']} rows in the files"]
            for key in ("dropped_not_english_lang", "dropped_long_context", "dropped_length_bucket"):
                if o.get(key):
                    parts.append(f"{key.replace('dropped_', '').replace('_', ' ')} -{o[key]}")
            for reason, n in {**o["dropped_by_filter"], **o["dropped_by_dedup"]}.items():
                parts.append(f"{reason.replace('_', ' ')} -{n}")
            parts.append(f"call already in the main data -{o['dropped_overlap_with_main']}")
            if "sampled" in o:
                parts.append(f"sampled {o['sampled']}")
            lines += ["", f"## {title}", "", ", ".join(parts) + "."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def outside_test(name, args, english, main_calls, stats, dropped, sample=0):
    """Load, filter and dedup an outside test set; drop rows whose call is in the main data."""
    o = stats[name] = {}
    rows = load_source(name, args, o)
    english.learn(rows)  # saroku's terse technical prose needs its own words in the vocabulary
    rows = apply_filters(rows, args, english, o, dropped)
    rows = dedup(rows, o, dropped)
    o["dropped_overlap_with_main"] = sum(loose(ex["input_a"]) in main_calls for ex in rows)
    rows = [ex for ex in rows if loose(ex["input_a"]) not in main_calls]
    if sample and len(rows) > sample:
        rows = sample_groups(rows, sample, args.seed, "domain")
        o["sampled"] = len(rows)
    for ex in rows:
        ex["meta"].setdefault("call_has_both_labels", False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=HERE / "raw")
    parser.add_argument("--out", type=Path, default=HERE / "data")
    parser.add_argument("--no-download", action="store_true", help="only use files already in --raw-dir")
    parser.add_argument("--main-files", default=",".join(SOURCES["auto1b"]["files"]),
                        help="auto-1b-data files to use, comma separated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--length-buckets", default="short,medium,long",
                        help="auto-1b-data length buckets to keep (short, medium, long), comma separated")
    parser.add_argument("--max-call-chars", type=int, default=0,
                        help="drop rows whose format A is longer than this (0 = no limit)")
    parser.add_argument("--max-input-chars", type=int, default=0,
                        help="drop rows whose format B is longer than this (0 = no limit)")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="stratified sample of the clean main data before splitting (0 = keep all)")
    parser.add_argument("--train-only-calls", type=int, default=20,
                        help="tool calls with more rows than this all go to train (0 = split them like the rest)")
    parser.add_argument("--max-per-call", type=int, default=0,
                        help="keep at most this many rows with the same tool call (0 = no cap)")
    parser.add_argument("--min-vocab-share", type=float, default=0.7,
                        help="share of prose words that must be in the English vocabulary")
    parser.add_argument("--min-en-prob", type=float, default=0.5, help="langdetect probability of English")
    parser.add_argument("--saroku-rows", type=int, default=0,
                        help="sample the saroku outside test down to this many rows (0 = keep all)")
    parser.add_argument("--max-file-mb", type=float, default=49,
                        help="split a data file into parts above this size (GitHub warns above 50 MB and refuses "
                             "files over 100 MB; 0 = never)")
    parser.add_argument("--skip-outside-tests", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stats = {"settings": {k: str(v) for k, v in vars(args).items()}}
    dropped = []

    # main dataset
    s = stats["auto1b"] = {}
    examples = load_source("auto1b", args, s, args.main_files.split(","))
    english = EnglishFilter(args.min_en_prob, examples, args.min_vocab_share)
    s["english_vocabulary_words"] = len(english.vocab)
    examples = apply_filters(examples, args, english, s, dropped)
    examples = dedup(examples, s, dropped)
    if args.max_rows and len(examples) > args.max_rows:
        examples = sample_groups(examples, args.max_rows, args.seed, "difficulty")
        s["sampled"] = len(examples)
    parts = split(examples, args, s)
    for name, rows in parts.items():
        stats[name] = describe(rows)
        stats[name]["files"] = write_split(rows, args.out, name, args.max_file_mb)

    # leakage check: no tool call (loose format A) shared between splits
    seen = {name: {loose(ex["input_a"]) for ex in rows} for name, rows in parts.items()}
    stats["shared_calls_between_splits"] = {
        "train_val": len(seen["train"] & seen["val"]),
        "train_test": len(seen["train"] & seen["test"]),
        "val_test": len(seen["val"] & seen["test"]),
    }

    # outside test sets, kept apart from everything above
    if not args.skip_outside_tests:
        main_calls = set().union(*seen.values())
        bench = outside_test("approve_or_deny", args, english, main_calls, stats, dropped)
        stats["ood_approve_or_deny"] = describe(bench)
        stats["ood_approve_or_deny"]["files"] = write_split(bench, args.out, "ood_approve_or_deny", args.max_file_mb)
        saroku = outside_test("saroku", args, english, main_calls, stats, dropped, args.saroku_rows)
        stats["ood_saroku"] = describe(saroku)
        stats["ood_saroku"]["files"] = write_split(saroku, args.out, "ood_saroku", args.max_file_mb)

    (args.out / "dropped.jsonl").unlink(missing_ok=True)  # plain file from an earlier version
    with open(args.out / "dropped.jsonl.gz", "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as packed, \
            io.TextIOWrapper(packed, encoding="utf-8") as f:
        for ex in dropped:
            f.write(json.dumps({"id": ex["id"], "dropped_because": ex["dropped_because"],
                                "label": ex["label"], "input_b": ex["input_b"]}, ensure_ascii=False) + "\n")
    with open(args.out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    write_report(stats, args.out / "REPORT.md")

    for name in ("train", "val", "test", "ood_approve_or_deny", "ood_saroku"):
        if name in stats:
            d = stats[name]
            print(f"{name:20s} rows={d['rows']:6d}  labels={d['labels']}  deny_share={d['deny_share']}")
    print("shared tool calls between splits:", stats["shared_calls_between_splits"])


if __name__ == "__main__":
    sys.exit(main())
