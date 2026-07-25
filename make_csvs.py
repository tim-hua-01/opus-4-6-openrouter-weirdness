"""
Flatten every raw completion into CSVs that are readable in a spreadsheet.

Writes into csv/:
  raw_responses.csv       one row per API call, newlines escaped as \\n
  response_counts.csv     unique completion text per cell + how often it occurred
                          (fastest way to skim: most cells have <10 distinct strings)
  garbled_responses.csv   only completions that are not a bare number, most-garbled first
  algebra_responses.csv   the abstract-algebra probe, same treatment
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
CSV = ROOT / "csv"
CSV.mkdir(exist_ok=True)

SOURCES = {"system": ROOT / "results.jsonl", "user": ROOT / "user_placement" / "results.jsonl"}


def esc(t):
    """One row per record: make newlines/tabs visible instead of breaking the CSV."""
    if not isinstance(t, str):   # None / NaN
        return ""
    return t.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def is_bare(t):
    return bool(re.fullmatch(r"[-+]?\d+", (t or "").strip()))


def junk_prefix(t):
    """Anything that isn't a bare number — includes legitimate prose."""
    t = (t or "").strip()
    if not t or is_bare(t):
        return False
    first = t.split()[0]
    return not re.fullmatch(r"[-+]?[\d.,*#]+", first)


def glitch_token(t):
    """
    The 'clockwise\\n\\n67' pattern specifically: the first line is a single stray
    word, then the answer. Distinguishes the opus-4.6 low-effort artifact from
    ordinary prose like 'I notice the reward function rewards even numbers...'.
    """
    t = (t or "").strip()
    if not t or is_bare(t):
        return False
    first_line = t.split("\n")[0].strip()
    return (len(first_line.split()) == 1
            and not re.fullmatch(r"[-+]?[\d.,*#]+", first_line)
            and len(t.split("\n")) > 1)


def load_runs():
    frames = []
    for placement, path in SOURCES.items():
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").drop_duplicates(
            ["model", "provider", "reasoning", "sample_idx"], keep="last")
        df["placement"] = df.get("placement", placement).fillna(placement)
        frames.append(df)
    if not frames:
        raise SystemExit("no results.jsonl found")
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_runs()
    df["output_tokens"] = df.usage.apply(lambda u: (u or {}).get("output"))
    df["content_text"] = df.content.fillna("")
    df["bare_number"] = df.content_text.map(is_bare)
    df["junk_prefix"] = df.content_text.map(junk_prefix)
    df["glitch_token"] = df.content_text.map(glitch_token)
    df["is_odd"] = df.extracted_number.map(lambda x: None if pd.isna(x) else int(x) % 2 == 1)
    df["content"] = df.content_text.map(esc)
    df["thinking"] = df.thinking.map(esc)

    cols = ["placement", "model", "provider", "reasoning", "sample_idx", "extracted_number",
            "is_odd", "bare_number", "junk_prefix", "glitch_token", "finish_reason", "output_tokens",
            "content", "thinking", "error"]
    raw = df[cols].sort_values(["placement", "model", "provider", "reasoning", "sample_idx"])
    raw.to_csv(CSV / "raw_responses.csv", index=False)

    # unique completions per cell, most frequent first
    counts = (df.groupby(["placement", "model", "provider", "reasoning", "content"])
                .agg(count=("sample_idx", "size"),
                     extracted_number=("extracted_number", "first"),
                     is_odd=("is_odd", "first"),
                     junk_prefix=("junk_prefix", "first"),
                     glitch_token=("glitch_token", "first"))
                .reset_index()
                .sort_values(["placement", "model", "provider", "reasoning", "count"],
                             ascending=[True, True, True, True, False]))
    counts.to_csv(CSV / "response_counts.csv", index=False)

    garbled = (df[~df.bare_number]
               .sort_values(["glitch_token", "junk_prefix", "placement", "model", "provider", "reasoning"],
                            ascending=[False, False, True, True, True, True]))
    garbled[cols].to_csv(CSV / "garbled_responses.csv", index=False)

    alg_path = ROOT / "algebra_results.jsonl"
    if alg_path.exists():
        a = pd.DataFrame([json.loads(l) for l in open(alg_path) if l.strip()])
        a["bare_number"] = a.content.fillna("").map(is_bare)
        a["junk_prefix"] = a.content.fillna("").map(junk_prefix)
        a["glitch_token"] = a.content.fillna("").map(glitch_token)
        a["content"] = a.content.map(esc)
        a["thinking"] = a.thinking.map(esc)
        a[["provider", "reasoning", "qid", "sample_idx", "extracted", "expected", "correct",
           "bare_number", "junk_prefix", "glitch_token", "output_tokens", "content", "thinking"]] \
            .sort_values(["glitch_token", "correct", "provider", "reasoning", "qid"],
                         ascending=[False, True, True, True, True]) \
            .to_csv(CSV / "algebra_responses.csv", index=False)

    print(f"raw_responses.csv       {len(raw):5d} rows")
    print(f"response_counts.csv     {len(counts):5d} unique completions across all cells")
    print(f"garbled_responses.csv   {len(garbled):5d} rows "
          f"({int(df.glitch_token.sum())} with a stray-word glitch token)")
    print(f"-> {CSV}")
    print("\nstray-word glitch rate by model x reasoning (%, runs pooled):")
    piv = df.pivot_table(index=["model", "provider"], columns="reasoning",
                         values="glitch_token", aggfunc="mean") * 100
    print(piv.round(1).to_string())


if __name__ == "__main__":
    main()
