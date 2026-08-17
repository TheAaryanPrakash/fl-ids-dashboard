"""
Phase 1.3/1.4 — Data preparation.

DESIGN DECISION (made after reviewing inspect_data.py output):
  OPTION A. bccc_cleaned.csv (35024 rows, 104 numeric features, MQTT/IoT
  protocol-level features, 18-class multiclass label ~2000/class) and
  cic_cleaned.csv (8000 rows, 50 numeric features, network-flow-level
  features, 8-class multiclass label 1000/class) share exactly ONE column
  in common: `label`. Zero feature overlap. Option B (concatenate + single
  Dirichlet split) is not applicable — there is no shared feature space to
  concatenate. So the two files are treated as two different data
  sources/domains, each split across multiple simulated clients. This is
  Option A from instructions.md, and gives natural non-IID structure
  (different clients see different feature semantics/domains entirely).

  Consequence: since the two sources have disjoint feature spaces, a single
  global model needs one fixed input dimension shared by all clients (FedAvg
  averages parameters positionally). This script builds a UNION feature
  space (104 bccc cols + 50 cic cols = 154 cols): a bccc-sourced row keeps
  its own values in the bccc block of columns and gets zeros in the cic
  block, and vice versa for a cic-sourced row. This zero-padding trick is a
  standard, simple way to let one MLP serve heterogeneous-feature clients.

  On top of that cross-source domain shift, clients within the same source
  additionally get a Dirichlet(alpha=0.5 default) label skew over that
  source's own multiclass labels — richer, more realistic non-IID structure,
  reusing the same Dirichlet mechanism Option B would have used.

  Cleaning: both files contain exact duplicate rows (11321/35024 in bccc,
  1295/8000 in cic). These are dropped (keep first) before the train/test
  split so identical rows can't leak between a client's train partition and
  the shared test set.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BENIGN_NAMES = {"normal", "benign"}


def log(msg):
    print(f"[prepare_data] {msg}")


def load_source(path, source_name):
    df = pd.read_csv(path, low_memory=False)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_after = len(df)
    log(f"{source_name}: loaded {n_before} rows, dropped {n_before - n_after} "
        f"exact duplicates -> {n_after} rows")

    feature_cols = [c for c in df.columns if c != "label"]
    assert df[feature_cols].select_dtypes(include=["object", "str"]).shape[1] == 0, (
        f"{source_name}: found non-numeric feature columns, expected all numeric "
        f"(inspect_data.py showed none) — re-check schema."
    )

    df["label_name"] = df["label"].astype(str)
    df["label_binary"] = (~df["label_name"].str.lower().isin(BENIGN_NAMES)).astype(int)

    le = LabelEncoder()
    df["label_multiclass"] = le.fit_transform(df["label_name"])
    df["source"] = source_name

    log(f"{source_name}: binary distribution -> "
        f"{df['label_binary'].value_counts().to_dict()} (0=benign, 1=attack)")
    log(f"{source_name}: {len(le.classes_)} multiclass labels -> {list(le.classes_)}")

    return df, feature_cols, le


def split_train_test(df, feature_cols, test_frac, seed, source_name):
    class_counts = df["label_multiclass"].value_counts()
    rare = class_counts[class_counts < 2]
    if not rare.empty:
        log(f"{source_name}: WARNING classes with <2 rows cannot be stratified, "
            f"dropping {rare.index.tolist()}")
        df = df[~df["label_multiclass"].isin(rare.index)].reset_index(drop=True)

    train_df, test_df = train_test_split(
        df, test_size=test_frac, random_state=seed,
        stratify=df["label_multiclass"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    log(f"{source_name}: split into {len(train_df)} train / {len(test_df)} test "
        f"(stratified by multiclass label, test_frac={test_frac})")
    return train_df, test_df


def to_union_space(df, own_cols, all_cols):
    """Zero-pad df (which only has own_cols as real features) into the full
    union feature space all_cols, filling columns from the other source with 0."""
    other_cols = [c for c in all_cols if c not in own_cols]
    feat = pd.DataFrame(0.0, index=df.index, columns=all_cols)
    feat[own_cols] = df[own_cols].values
    feat[other_cols] = 0.0
    meta_cols = ["label_binary", "label_multiclass", "label_name", "source"]
    return pd.concat([feat, df[meta_cols].reset_index(drop=True).set_axis(feat.index)], axis=1)


def dirichlet_partition(labels, n_clients, alpha, seed):
    """Standard Dirichlet(alpha) label-skew partition. Returns a list of
    index arrays (into `labels`), one per client."""
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1
    client_indices = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        if len(idx_c) == 0:
            continue
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        cum = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, cum)
        for i, s in enumerate(splits):
            client_indices[i].extend(s.tolist())

    return [np.array(sorted(idxs)) for idxs in client_indices]


def print_client_distribution(cid, df):
    src = df["source"].iloc[0] if len(df) else "n/a"
    log(f"\n--- client_{cid} (source={src}, n={len(df)}) ---")
    log(f"  binary label counts: {df['label_binary'].value_counts().to_dict()}")
    mc = df["label_name"].value_counts()
    log(f"  multiclass label counts:\n{mc.to_string()}")


def main():
    parser = argparse.ArgumentParser(description="Prepare non-IID FL client partitions.")
    parser.add_argument("--n-clients", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5,
                         help="Dirichlet concentration for intra-source label skew")
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    if args.n_clients < 2:
        sys.exit("--n-clients must be >= 2 (need at least 1 client per source for Option A)")

    bccc_path = f"{args.data_dir}/bccc_cleaned.csv"
    cic_path = f"{args.data_dir}/cic_cleaned.csv"

    bccc_df, bccc_feats, bccc_le = load_source(bccc_path, "bccc")
    cic_df, cic_feats, cic_le = load_source(cic_path, "cic")

    bccc_train, bccc_test = split_train_test(bccc_df, bccc_feats, args.test_frac, args.seed, "bccc")
    cic_train, cic_test = split_train_test(cic_df, cic_feats, args.test_frac, args.seed, "cic")

    all_cols = sorted(set(bccc_feats) | set(cic_feats))
    assert set(bccc_feats).isdisjoint(cic_feats), "unexpected feature overlap between sources"
    log(f"\nUnion feature space: {len(bccc_feats)} (bccc) + {len(cic_feats)} (cic) "
        f"= {len(all_cols)} columns")

    bccc_train_u = to_union_space(bccc_train, bccc_feats, all_cols)
    bccc_test_u = to_union_space(bccc_test, bccc_feats, all_cols)
    cic_train_u = to_union_space(cic_train, cic_feats, all_cols)
    cic_test_u = to_union_space(cic_test, cic_feats, all_cols)

    # client allocation: bccc gets the ceiling half (it has ~4.4x more data)
    n_bccc_clients = -(-args.n_clients // 2)  # ceil
    n_cic_clients = args.n_clients - n_bccc_clients
    log(f"\nClient allocation: {n_bccc_clients} clients from bccc, "
        f"{n_cic_clients} clients from cic (total {args.n_clients})")

    bccc_labels = bccc_train["label_multiclass"].to_numpy()
    cic_labels = cic_train["label_multiclass"].to_numpy()
    bccc_parts = dirichlet_partition(bccc_labels, n_bccc_clients, args.alpha, args.seed)
    cic_parts = dirichlet_partition(cic_labels, n_cic_clients, args.alpha, args.seed + 1)

    client_dfs = []
    for part in bccc_parts:
        client_dfs.append(bccc_train_u.iloc[part].reset_index(drop=True))
    for part in cic_parts:
        client_dfs.append(cic_train_u.iloc[part].reset_index(drop=True))

    for i, cdf in enumerate(client_dfs):
        out_path = f"{args.data_dir}/client_{i}.csv"
        cdf.to_csv(out_path, index=False)
        print_client_distribution(i, cdf)
        log(f"  saved -> {out_path}")

    test_df = pd.concat([bccc_test_u, cic_test_u], ignore_index=True)
    test_df = test_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    test_path = f"{args.data_dir}/test.csv"
    test_df.to_csv(test_path, index=False)
    log(f"\n--- global test set (n={len(test_df)}) ---")
    log(f"  source counts: {test_df['source'].value_counts().to_dict()}")
    log(f"  binary label counts: {test_df['label_binary'].value_counts().to_dict()}")
    log(f"  saved -> {test_path}")

    log(f"\nDone. {len(client_dfs)} client partitions + test.csv written to {args.data_dir}/")


if __name__ == "__main__":
    main()
