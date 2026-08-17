"""
Phase 1.2 — Dataset inspection.

Prints shape, columns/dtypes, candidate label column + value counts,
null checks, and sample rows for both raw CSVs. Run this and review the
output before writing prepare_data.py — do not assume schema/label names.
"""
import sys

import pandas as pd

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)

LABEL_CANDIDATES = ["label", "Label", "class", "Class", "attack_cat", "Attack",
                     "attack", "Attack_type", "attack_type", "category", "Category"]


def inspect(path):
    print("=" * 100)
    print(f"FILE: {path}")
    print("=" * 100)

    df = pd.read_csv(path, low_memory=False)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    print("Columns and dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  {col!r:40s} {dtype}")

    print("\nCandidate label columns found:")
    found_labels = [c for c in df.columns if c in LABEL_CANDIDATES]
    if not found_labels:
        # fuzzy fallback: any column with 'label' / 'class' / 'attack' in the name
        found_labels = [c for c in df.columns
                         if any(k in c.lower() for k in ["label", "class", "attack", "category"])]
    if not found_labels:
        print("  None found by name heuristic — manual inspection required.")
    for col in found_labels:
        print(f"\n  --- {col} ---")
        vc = df[col].value_counts(dropna=False)
        print(vc.to_string())
        print(f"  unique values: {df[col].nunique(dropna=False)}")

    print("\nNull/missing value check (columns with any nulls):")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print("  No nulls found.")
    else:
        print(nulls.to_string())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nSample rows (head 5):")
    print(df.head(5).to_string())

    print("\nSample rows (random 5):")
    print(df.sample(min(5, len(df)), random_state=42).to_string())

    print()
    return df, found_labels


if __name__ == "__main__":
    files = sys.argv[1:] or ["data/bccc_cleaned.csv", "data/cic_cleaned.csv"]
    results = {}
    for f in files:
        df, labels = inspect(f)
        results[f] = (df, labels)

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for f, (df, labels) in results.items():
        print(f"{f}: shape={df.shape}, candidate_labels={labels}")

    # schema comparison, useful for the Option A vs B decision in prepare_data.py
    if len(results) == 2:
        (f1, (df1, _)), (f2, (df2, _)) = results.items()
        cols1, cols2 = set(df1.columns), set(df2.columns)
        print(f"\nSchema comparison: {f1} vs {f2}")
        print(f"  Identical columns: {cols1 == cols2}")
        print(f"  Columns only in {f1}: {sorted(cols1 - cols2)}")
        print(f"  Columns only in {f2}: {sorted(cols2 - cols1)}")
        print(f"  Common columns: {len(cols1 & cols2)}")
