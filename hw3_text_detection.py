"""
UCU AI 2026 - Homework 3: human / ai / fusion text classification.

All logic lives here; the notebook `hw3_text_detection_notebook.ipynb` only calls these
functions. The same code runs on Google Colab and locally (VS Code):
paths and device are detected automatically.

Sections
    0. Environment, paths, reproducibility
    1. Data loading
    2. EDA
    3. Near-duplicate groups and the fixed train/val split
    4. Evaluation and run saving helpers
    5. Baseline: TF-IDF + linear models
    6. Encoder: fine-tuned multilingual transformer with chunking for long texts
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import sys
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = ["human", "ai", "fusion"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
SEED = 42


# =============================================================================
# 0. Environment, paths, reproducibility
# =============================================================================
def in_colab() -> bool:
    return "google.colab" in sys.modules


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class Paths:
    root: Path    # project folder on Google Drive (or its local copy)
    data: Path    # folder the csv files are read from
    splits: Path  # groups + fixed train/val split
    runs: Path    # one subfolder per experiment


DATA_FILES = ("train.csv", "test.csv", "sample_submission.csv")


def get_paths(local_root: str | None = None,
              colab_root: str = "/content/drive/MyDrive/hw3",
              colab_local_data: str = "/content/data") -> Paths:
    """Resolve project folders.

    Colab: the project folder lives on the mounted Google Drive; csv files are
    copied once to the Colab disk because reading from Drive is slow.
    Local: `local_root` argument, else env var HW3_ROOT, else ./hw3.
    """
    if in_colab():
        root = Path(colab_root)
    else:
        root = Path(local_root or os.environ.get("HW3_ROOT", "hw3"))
    data = root / "data"
    missing = [f for f in DATA_FILES if not (data / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing in {data}: {missing}")

    if in_colab() and colab_local_data:
        local = Path(colab_local_data)
        local.mkdir(parents=True, exist_ok=True)
        for f in DATA_FILES:
            if not (local / f).exists():
                shutil.copy2(data / f, local / f)
        data = local

    paths = Paths(root=root, data=data, splits=root / "splits", runs=root / "runs")
    paths.splits.mkdir(parents=True, exist_ok=True)
    paths.runs.mkdir(parents=True, exist_ok=True)
    return paths


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_environment() -> dict:
    info = {"colab": in_colab(), "python": sys.version.split()[0]}
    try:
        import torch

        info["torch"] = torch.__version__
        info["device"] = get_device()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_mem_gb"] = round(props.total_memory / 2**30, 1)
    except ImportError:
        info["torch"] = None
    return info


# =============================================================================
# 1. Data loading
# =============================================================================
def load_data(paths: Paths):
    train = pd.read_csv(paths.data / "train.csv")
    test = pd.read_csv(paths.data / "test.csv")
    sample = pd.read_csv(paths.data / "sample_submission.csv")
    for df in (train, test):
        df["text"] = df["text"].fillna("").astype(str)
        df["n_words"] = df["text"].str.split().str.len()
        df["n_chars"] = df["text"].str.len()
    return train, test, sample


# =============================================================================
# 2. EDA
# =============================================================================
def data_overview(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """One small table with sanity checks instead of long printouts."""
    rows = {
        "rows": [len(train), len(test)],
        "unique ids": [train["id"].nunique(), test["id"].nunique()],
        "empty texts": [(train["n_words"] == 0).sum(), (test["n_words"] == 0).sum()],
        "duplicated texts (within set)": [train["text"].duplicated().sum(), test["text"].duplicated().sum()],
        "texts also in other set": [train["text"].isin(set(test["text"])).sum(),
                                    test["text"].isin(set(train["text"])).sum()],
        "missing values": [int(train.isna().sum().sum()), int(test.isna().sum().sum())],
    }
    out = pd.DataFrame(rows, index=["train", "test"]).T
    same_order = list(sample["id"]) == list(test["id"])
    out.loc["sample_submission ids == test ids (same order)"] = [same_order, ""]
    return out


def class_balance_tables(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "label": train["label"].value_counts().reindex(LABELS).to_frame("count")
                 .assign(share=lambda d: (d["count"] / d["count"].sum()).round(3)),
        "lang x label": pd.crosstab(train["lang"], train["label"])[LABELS],
        "domain x label": pd.crosstab([train["domain"], train["lang"]], train["label"])[LABELS],
    }


def plot_class_balance(train: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    train["label"].value_counts().reindex(LABELS).plot.bar(ax=axes[0], color="steelblue")
    axes[0].set_title("Label counts")
    axes[0].tick_params(axis="x", rotation=0)

    pd.crosstab(train["lang"], train["label"], normalize="index")[LABELS].plot.bar(
        ax=axes[1], stacked=True, rot=0)
    axes[1].set_title("Label share by language")
    axes[1].legend(loc="lower right", fontsize=8)

    ct = pd.crosstab(train["domain"] + " | " + train["lang"], train["label"])[LABELS]
    ct.div(ct.sum(1), axis=0).plot.barh(ax=axes[2], stacked=True, legend=False)
    for i, total in enumerate(ct.sum(1)):
        axes[2].text(1.01, i, f"n={total:,}", va="center", fontsize=8)
    axes[2].set_title("Label share by domain | language")
    axes[2].set_xlim(0, 1)
    fig.tight_layout()
    return fig


def length_table(train: pd.DataFrame) -> pd.DataFrame:
    return (train.groupby(["domain", "label"])["n_words"].median().unstack()[LABELS]
            .astype(int).rename(columns=lambda c: f"median words: {c}"))


def plot_lengths(train: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    bins = np.logspace(np.log10(10), np.log10(train["n_words"].max() + 1), 50)
    for label in LABELS:
        axes[0].hist(train.loc[train["label"] == label, "n_words"], bins=bins,
                     histtype="step", lw=1.8, label=label, density=True)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("words (log scale)")
    axes[0].set_title("Text length by label")
    axes[0].legend()

    domains = sorted(train["domain"].unique())
    width = 0.27
    for j, label in enumerate(LABELS):
        med = [train.loc[(train["domain"] == d) & (train["label"] == label), "n_words"].median()
               for d in domains]
        axes[1].bar(np.arange(len(domains)) + (j - 1) * width, med, width, label=label)
    axes[1].set_xticks(np.arange(len(domains)), domains, rotation=30, ha="right")
    axes[1].set_title("Median length (words) by domain and label")
    axes[1].legend()
    fig.tight_layout()
    return fig


def token_length_stats(texts: pd.Series, model_name: str, max_len: int = 512,
                       n: int = 3000, seed: int = SEED) -> pd.DataFrame:
    """How many texts do not fit into one encoder window."""
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(model_name)
    sample = texts.sample(min(n, len(texts)), random_state=seed).tolist()
    lengths = np.array([len(x) for x in tok(sample, add_special_tokens=False)["input_ids"]])
    content = max_len - 2
    return pd.DataFrame({
        "tokens: median": [int(np.median(lengths))],
        "tokens: 90th pct": [int(np.percentile(lengths, 90))],
        "tokens: max": [int(lengths.max())],
        f"share > {max_len}": [round(float((lengths > content).mean()), 3)],
        f"mean windows of {max_len}": [round(float(np.ceil(lengths / content).mean()), 2)],
    }, index=[model_name])


def show_examples(df: pd.DataFrame, n_chars: int = 300, seed: int = SEED) -> None:
    """One trimmed example per label and language."""
    for (label, lang), g in df.groupby(["label", "lang"]):
        row = g.sample(1, random_state=seed).iloc[0]
        text = " ".join(row["text"].split())[:n_chars]
        print(f"[{label} | {lang} | {row['domain']} | {row['n_words']} words]\n  {text}...\n")


def compare_train_test(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for col in ("lang", "domain"):
        t = pd.concat({"train": train[col].value_counts(normalize=True),
                       "test": test[col].value_counts(normalize=True)}, axis=1)
        t.index = [f"{col}={v}" for v in t.index]
        parts.append(t)
    q = pd.concat({"train": train["n_words"].quantile([0.1, 0.5, 0.9]),
                   "test": test["n_words"].quantile([0.1, 0.5, 0.9])}, axis=1)
    q.index = [f"words q{int(i * 100)}" for i in q.index]
    return pd.concat(parts + [q]).round(3)


# =============================================================================
# 3. Near-duplicate groups and the fixed split
# =============================================================================
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _doc_shingle_hashes(text: str, cache: dict, n: int, mod: int) -> np.ndarray:
    """Hashes of word n-grams of one text; only ~1/mod of them are kept (sampling)."""
    tokens = text.lower().split()
    if len(tokens) < n:
        return np.empty(0, dtype=np.uint64)
    get = cache.get
    vals = [get(t, -1) for t in tokens]
    for i, v in enumerate(vals):
        if v == -1:
            v = zlib.crc32(tokens[i].encode("utf-8", "ignore"))
            cache[tokens[i]] = v
            vals[i] = v
    w = np.asarray(vals, dtype=np.uint64)
    m = len(w) - n + 1
    h = np.zeros(m, dtype=np.uint64)
    prime = np.uint64(1099511628211)
    with np.errstate(over="ignore"):
        for k in range(n):
            h = h * prime + w[k:k + m]
        # bit mixing so that "h % mod" is a fair sample
        h ^= h >> np.uint64(33)
        h *= np.uint64(0xFF51AFD7ED558CCD)
        h ^= h >> np.uint64(33)
    return np.unique(h[h % np.uint64(mod) == 0])


def find_groups(texts: pd.Series, n: int = 5, mod: int = 8, max_df: int = 10,
                min_shared: int = 3, verbose: bool = True) -> np.ndarray:
    """Group texts that share several word 5-grams (versions of the same source text).

    1. For every text take its word 5-grams, hash them, keep a fixed 1/`mod` sample.
    2. Two texts are linked if they share >= `min_shared` sampled 5-grams.
       5-grams found in more than `max_df` texts are boilerplate and ignored.
    3. Groups = connected components of this link graph.
    Deterministic: the same data always gives the same groups.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    t0 = time.time()
    cache: dict = {}
    doc_ids, hashes = [], []
    for i, text in enumerate(texts):
        h = _doc_shingle_hashes(text, cache, n, mod)
        if len(h):
            doc_ids.append(np.full(len(h), i, dtype=np.int64))
            hashes.append(h)
    del cache
    doc = np.concatenate(doc_ids)
    hsh = np.concatenate(hashes)
    del doc_ids, hashes

    order = np.argsort(hsh, kind="stable")
    doc, hsh = doc[order], hsh[order]
    _, start, counts = np.unique(hsh, return_index=True, return_counts=True)
    run_len = np.repeat(counts, counts)
    keep = (run_len >= 2) & (run_len <= max_df)
    doc, hsh = doc[keep], hsh[keep]

    n_docs = len(texts)
    keys = []
    for off in range(1, max_df):
        same = hsh[:-off] == hsh[off:]
        if not same.any():
            break
        a, b = doc[:-off][same], doc[off:][same]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        keys.append(lo * n_docs + hi)
    keys = np.concatenate(keys) if keys else np.empty(0, dtype=np.int64)
    pairs, shared = np.unique(keys, return_counts=True)
    pairs = pairs[shared >= min_shared]
    a, b = pairs // n_docs, pairs % n_docs

    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n_docs, n_docs))
    _, groups = connected_components(graph, directed=False)
    if verbose:
        sizes = np.bincount(groups)
        print(f"find_groups: {n_docs:,} texts -> {len(sizes):,} groups "
              f"(linked pairs: {len(a):,}, max group size: {sizes.max()}, "
              f"{time.time() - t0:.0f}s)")
    return groups


def load_or_make_groups(train: pd.DataFrame, paths: Paths, recompute: bool = False) -> pd.Series:
    f = paths.splits / "groups.csv"
    if f.exists() and not recompute:
        g = pd.read_csv(f)
    else:
        g = pd.DataFrame({"id": train["id"], "group": find_groups(train["text"])})
        g.to_csv(f, index=False)
    return train[["id"]].merge(g, on="id", how="left")["group"].to_numpy()


def group_composition(train: pd.DataFrame) -> pd.DataFrame:
    """Which label combinations appear together in one group."""
    comp = train.groupby("group")["label"].agg(
        lambda s: "+".join(sorted(set(s), key=LABELS.index)))
    size = train.groupby("group").size()
    out = pd.DataFrame({"labels in group": comp, "size": size})
    return (out.groupby("labels in group")
               .agg(groups=("size", "size"), texts=("size", "sum"), max_size=("size", "max"))
               .sort_values("groups", ascending=False))


def make_split(train: pd.DataFrame, n_samples: int = 40_000, val_frac: float = 0.2,
               seed: int = SEED) -> pd.DataFrame:
    """Fixed subsample of whole groups + group-aware stratified train/val split."""
    from sklearn.model_selection import StratifiedGroupKFold

    rng = np.random.default_rng(seed)
    sizes = train.groupby("group").size()
    groups = sizes.index.to_numpy()[rng.permutation(len(sizes))]
    cum = sizes.loc[groups].cumsum().to_numpy()
    chosen = groups[: int(np.searchsorted(cum, n_samples)) + 1]
    sub = train[train["group"].isin(chosen)]

    strata = sub["label"] + "_" + sub["lang"]
    sgkf = StratifiedGroupKFold(n_splits=round(1 / val_frac), shuffle=True, random_state=seed)
    tr_idx, va_idx = next(sgkf.split(sub, strata, sub["group"]))
    split = pd.Series("train", index=sub.index)
    split.iloc[va_idx] = "val"
    return pd.DataFrame({"id": sub["id"].to_numpy(), "group": sub["group"].to_numpy(),
                         "split": split.to_numpy()})


def load_or_make_split(train: pd.DataFrame, paths: Paths, n_samples: int = 40_000,
                       recompute: bool = False, name: str = "split_v1.csv"):
    f = paths.splits / name
    if f.exists() and not recompute:
        split = pd.read_csv(f)
    else:
        split = make_split(train, n_samples=n_samples)
        split.to_csv(f, index=False)
    df = train.merge(split[["id", "split"]], on="id", how="inner")
    tr = df[df["split"] == "train"].reset_index(drop=True)
    va = df[df["split"] == "val"].reset_index(drop=True)
    return tr, va


def split_summary(tr: pd.DataFrame, va: pd.DataFrame) -> pd.DataFrame:
    overlap = len(set(tr["group"]) & set(va["group"]))
    assert overlap == 0, f"{overlap} groups are in both train and val"
    parts = []
    for col in ("label", "lang", "domain"):
        t = pd.concat({"train": tr[col].value_counts(normalize=True),
                       "val": va[col].value_counts(normalize=True)}, axis=1)
        t.index = [f"{col}={v}" for v in t.index]
        parts.append(t)
    out = pd.concat(parts).round(3)
    out.loc["rows"] = [len(tr), len(va)]
    out.loc["groups"] = [tr["group"].nunique(), va["group"].nunique()]
    return out


# ---- what makes fusion different -------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _word_ngrams(text: str, n: int = 3) -> set:
    t = text.lower().split()
    return set(zip(*[t[i:] for i in range(n)]))


def _sentences(text: str) -> list[str]:
    return [" ".join(s.lower().split()) for s in _SENT_SPLIT.split(text) if len(s.split()) >= 4]


def compare_with_human_version(train: pd.DataFrame, max_groups: int = 1000,
                               seed: int = SEED) -> pd.DataFrame:
    """For groups that contain a human text, compare every ai/fusion text with it.

    jaccard_3gram  - overlap of word 3-grams with the human version
    copied_sents   - share of sentences copied verbatim from the human version
    len_ratio      - length relative to the human version
    """
    has_human = train.groupby("group")["label"].transform(lambda s: (s == "human").any())
    multi = train[has_human & (train.groupby("group")["label"].transform("size") > 1)]
    groups = multi["group"].drop_duplicates().sample(
        min(max_groups, multi["group"].nunique()), random_state=seed)
    rows = []
    for _, g in multi[multi["group"].isin(groups)].groupby("group"):
        human = g[g["label"] == "human"].iloc[0]
        h_ngr, h_sents = _word_ngrams(human["text"]), set(_sentences(human["text"]))
        for _, r in g[g["label"] != "human"].iterrows():
            ngr, sents = _word_ngrams(r["text"]), _sentences(r["text"])
            rows.append({
                "label": r["label"],
                "jaccard_3gram": len(ngr & h_ngr) / max(1, len(ngr | h_ngr)),
                "copied_sents": np.mean([s in h_sents for s in sents]) if sents else 0.0,
                "len_ratio": r["n_words"] / max(1, human["n_words"]),
            })
    return pd.DataFrame(rows)


def plot_human_version_comparison(cmp: pd.DataFrame):
    cols = ["jaccard_3gram", "copied_sents", "len_ratio"]
    labels = [l for l in LABELS if l in set(cmp["label"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, cols):
        ax.boxplot([cmp.loc[cmp["label"] == l, col] for l in labels], showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1), labels)
        ax.set_title(f"{col} vs. human version")
    fig.tight_layout()
    return fig


def show_group_example(train: pd.DataFrame, n_chars: int = 400, seed: int = SEED) -> None:
    """Print the beginning of the human / ai / fusion versions of one source text."""
    full = train.groupby("group")["label"].agg(lambda s: set(LABELS) <= set(s))
    candidates = full[full].index
    if len(candidates) == 0:
        print("No group contains all three labels.")
        return
    gid = pd.Series(candidates).sample(1, random_state=seed).iloc[0]
    g = train[train["group"] == gid]
    for label in LABELS:
        r = g[g["label"] == label].iloc[0]
        print(f"[{label} | {r['domain']} | {r['n_words']} words]\n  "
              f"{' '.join(r['text'].split())[:n_chars]}...\n")


# =============================================================================
# 4. Evaluation and run saving
# =============================================================================
def evaluate(y_true, y_pred) -> dict:
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    out = {"macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro"))}
    for i, label in enumerate(LABELS):
        out[f"{label}_precision"] = float(p[i])
        out[f"{label}_recall"] = float(r[i])
        out[f"{label}_f1"] = float(f[i])
    return out


def per_class_table(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame({m: [metrics[f"{l}_{m}"] for l in LABELS]
                         for m in ("precision", "recall", "f1")}, index=LABELS).round(4)


def f1_by(df: pd.DataFrame, y_pred, col: str) -> pd.Series:
    from sklearn.metrics import f1_score

    y_pred = pd.Series(np.asarray(y_pred), index=df.index)
    return pd.Series({v: f1_score(g["label"], y_pred[g.index], labels=LABELS, average="macro")
                      for v, g in df.groupby(col)}, name="macro_f1").round(4)


def plot_confusion(y_true, y_pred, title: str = ""):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def scores_to_proba(scores: np.ndarray) -> np.ndarray:
    """Softmax, for models that output decision scores instead of probabilities."""
    z = scores - scores.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def make_submission(test_ids, pred_labels, sample: pd.DataFrame) -> pd.DataFrame:
    pred = pd.Series(np.asarray(pred_labels), index=np.asarray(test_ids))
    sub = pd.DataFrame({"id": sample["id"], "label": pred.reindex(sample["id"]).to_numpy()})
    assert sub["label"].notna().all(), "some test ids have no prediction"
    assert set(sub["label"]) <= set(LABELS)
    assert len(sub) == len(sample) and sub["id"].is_unique
    return sub


def save_run(run_dir: Path, metrics: dict, config: dict, val_ids, val_proba,
             test_ids, test_proba, submission: pd.DataFrame | None = None) -> None:
    """Everything later experiments need: metrics, config, val/test probabilities."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    np.save(run_dir / "val_proba.npy", np.asarray(val_proba, dtype=np.float32))
    np.save(run_dir / "test_proba.npy", np.asarray(test_proba, dtype=np.float32))
    pd.Series(np.asarray(val_ids), name="id").to_csv(run_dir / "val_ids.csv", index=False)
    pd.Series(np.asarray(test_ids), name="id").to_csv(run_dir / "test_ids.csv", index=False)
    if submission is not None:
        submission.to_csv(run_dir / "submission.csv", index=False)


def load_run(run_dir: Path) -> dict:
    return {
        "metrics": json.loads((run_dir / "metrics.json").read_text()),
        "config": json.loads((run_dir / "config.json").read_text()),
        "val_ids": pd.read_csv(run_dir / "val_ids.csv")["id"].to_numpy(),
        "val_proba": np.load(run_dir / "val_proba.npy"),
        "test_ids": pd.read_csv(run_dir / "test_ids.csv")["id"].to_numpy(),
        "test_proba": np.load(run_dir / "test_proba.npy"),
    }


def run_report(run: dict, va: pd.DataFrame, title: str = ""):
    """Per-class metrics, confusion matrix and macro-F1 by language / domain on val."""
    proba = pd.DataFrame(run["val_proba"], index=run["val_ids"]).reindex(va["id"]).to_numpy()
    pred = np.array(LABELS)[proba.argmax(1)]
    m = evaluate(va["label"], pred)
    print(f"{title}: val macro-F1 = {m['macro_f1']:.4f}")
    by = pd.concat({"lang": f1_by(va, pred, "lang"), "domain": f1_by(va, pred, "domain")})
    return per_class_table(m), by.to_frame(), plot_confusion(va["label"], pred, title)


def run_exists(paths: Paths, name: str) -> bool:
    d = paths.runs / name
    return all((d / f).exists() for f in ("metrics.json", "val_proba.npy", "test_proba.npy",
                                          "val_ids.csv", "test_ids.csv"))


def compare_runs(paths: Paths, names, va: pd.DataFrame, baseline: str | None = "baseline_tfidf",
                 reference: str | None = "encoder_xlmr_chunks") -> pd.DataFrame:
    """Same val split for every run: macro-F1, per-class F1, macro-F1 per language.

    `names` is a list of run names or a dict {run name: description}. Runs that do not
    exist yet are skipped. Adds the difference to the baseline and to the reference.
    """
    descr = names if isinstance(names, dict) else {n: "" for n in names}
    rows = {}
    for name, d in descr.items():
        if not run_exists(paths, name):
            continue
        run = load_run(paths.runs / name)
        proba = pd.DataFrame(run["val_proba"], index=run["val_ids"])
        pred = np.array(LABELS)[proba.reindex(va["id"]).to_numpy().argmax(1)]
        m = evaluate(va["label"], pred)
        row = {"description": d, "macro_f1": m["macro_f1"], **{f"F1 {l}": m[f"{l}_f1"] for l in LABELS}}
        row.update({f"macro_f1 {k}": v for k, v in f1_by(va, pred, "lang").items()})
        rows[name] = row
    out = pd.DataFrame(rows).T
    if out.empty:
        return out
    for ref, col in ((baseline, "Δ vs baseline"), (reference, "Δ vs XLM-R")):
        if ref is not None and ref in out.index:
            out[col] = out["macro_f1"].astype(float) - float(out.loc[ref, "macro_f1"])
    if not isinstance(names, dict):
        out = out.drop(columns="description")
    num = out.columns.drop("description", errors="ignore")
    out[num] = out[num].astype(float).round(4)
    return out


# =============================================================================
# 5. Baseline: TF-IDF + linear models
# =============================================================================
BASELINE_FEATURES = {
    "word": dict(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=300_000,
                 sublinear_tf=True),
    "char": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=300_000,
                 sublinear_tf=True, lowercase=False),
}

DEFAULT_BASELINE_GRID = {
    "features": ["word", "char", "word+char"],
    "models": {"logreg": [1.0, 10.0], "linsvc": [0.1, 1.0]},
}


def split_fingerprint(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame | None = None) -> str:
    """Short hash of the exact rows (and their order) of the split.

    Cached features and saved models are reused only if the fingerprint matches,
    so a changed split can never silently mix with old results.
    """
    import hashlib

    h = hashlib.md5()
    for df in (tr, va) + ((te,) if te is not None else ()):
        h.update("\n".join(map(str, df["id"])).encode())
        h.update(b"|")
    return h.hexdigest()[:10]


def make_vectorizer(kind: str):
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(**BASELINE_FEATURES[kind], dtype=np.float32)


def build_or_load_features(kind: str, tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame,
                           cache_dir: Path | None, fingerprint: str, verbose: bool = True):
    """TF-IDF of one kind for train / val / test. Fitted on the train split only.

    With `cache_dir` the three matrices are saved once (`<kind>/*.npz`) and loaded on the
    next call. `meta.json` is written last, so a half-written cache is never used.
    """
    import gc

    from scipy.sparse import load_npz, save_npz

    parts = ("train", "val", "test")
    d = cache_dir / kind if cache_dir is not None else None
    meta = {"fingerprint": fingerprint, "params": {k: str(v) for k, v in BASELINE_FEATURES[kind].items()}}
    if d is not None and (d / "meta.json").exists():
        if json.loads((d / "meta.json").read_text()) == meta:
            t0 = time.time()
            mats = tuple(load_npz(d / f"{p}.npz") for p in parts)
            if verbose:
                print(f"TF-IDF {kind}: loaded from cache, {mats[0].shape[1]:,} features "
                      f"({time.time() - t0:.0f}s)")
            return mats

    t0 = time.time()
    vec = make_vectorizer(kind)
    mats = (vec.fit_transform(tr["text"]), vec.transform(va["text"]), vec.transform(te["text"]))
    del vec
    gc.collect()
    if verbose:
        print(f"TF-IDF {kind}: {mats[0].shape[1]:,} features ({time.time() - t0:.0f}s)")
    if d is not None:
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").unlink(missing_ok=True)
        for p, m in zip(parts, mats):
            save_npz(d / f"{p}.npz", m, compressed=False)
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return mats


def load_feature_set(features: str, tr, va, te, cache_dir, fingerprint, verbose: bool = True):
    """'word', 'char' or 'word+char' -> (X_train, X_val, X_test); combined part by part to save RAM."""
    import gc

    from scipy.sparse import hstack

    kinds = features.split("+")
    mats = [list(build_or_load_features(k, tr, va, te, cache_dir, fingerprint, verbose))
            for k in kinds]
    if len(mats) == 1:
        return tuple(mats[0])
    out = []
    for i in range(3):
        out.append(hstack([m[i] for m in mats], format="csr", dtype=np.float32))
        for m in mats:
            m[i] = None  # free each part as soon as it is combined
        gc.collect()
    return tuple(out)


def make_linear_model(name: str, C: float, seed: int = SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    if name == "logreg":
        return LogisticRegression(C=C, max_iter=2000, random_state=seed)
    if name == "linsvc":
        return LinearSVC(C=C, random_state=seed)
    raise ValueError(name)


def model_scores(model, X) -> np.ndarray:
    """Class probabilities in LABELS order (softmax of decision scores for SVM)."""
    order = [list(model.classes_).index(l) for l in LABELS]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, order]
    return scores_to_proba(model.decision_function(X)[:, order])


def _model_id(features: str, model: str, C: float) -> str:
    return f"{features}_{model}_C{C}"


def _model_done(d: Path, fingerprint: str) -> bool:
    f = d / "metrics.json"
    if not (f.exists() and (d / "val_proba.npy").exists() and (d / "test_proba.npy").exists()):
        return False
    return json.loads(f.read_text()).get("fingerprint") == fingerprint


def run_baseline_grid(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame, run_dir: Path,
                      grid: dict | None = None, cache_features: bool = True,
                      verbose: bool = True) -> pd.DataFrame:
    """Grid search (feature set x model x C) on the fixed val split.

    Every model is saved to `run_dir/grid/<features>_<model>_C<C>/` right after it is
    fitted (metrics + val/test probabilities). Models already saved for the same split
    are skipped, so after a crash the call simply continues. Only one feature set is
    kept in memory at a time.
    """
    import gc

    grid = grid or DEFAULT_BASELINE_GRID
    fp = split_fingerprint(tr, va, te)
    cache_dir = run_dir / "features" if cache_features else None
    (run_dir / "grid").mkdir(parents=True, exist_ok=True)

    for fs in grid["features"]:
        todo = [(m, C) for m, Cs in grid["models"].items() for C in Cs
                if not _model_done(run_dir / "grid" / _model_id(fs, m, C), fp)]
        if not todo:
            if verbose:
                print(f"{fs}: all models already saved, skipped")
            continue
        Xtr, Xva, Xte = load_feature_set(fs, tr, va, te, cache_dir, fp, verbose)
        for model_name, C in todo:
            t0 = time.time()
            model = make_linear_model(model_name, C).fit(Xtr, tr["label"])
            fit_sec = round(time.time() - t0, 1)
            val_proba = model_scores(model, Xva)
            test_proba = model_scores(model, Xte)
            m = evaluate(va["label"], np.array(LABELS)[val_proba.argmax(1)])
            d = run_dir / "grid" / _model_id(fs, model_name, C)
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "val_proba.npy", val_proba.astype(np.float32))
            np.save(d / "test_proba.npy", test_proba.astype(np.float32))
            info = {"features": fs, "model": model_name, "C": C, "fit_sec": fit_sec,
                    "fingerprint": fp, **m}
            (d / "metrics.json").write_text(json.dumps(info, indent=2))  # written last
            if verbose:
                print(f"  {fs:10s} {model_name:7s} C={C:<5} macro-F1={m['macro_f1']:.4f} ({fit_sec}s)")
            del model, val_proba, test_proba
            gc.collect()
        del Xtr, Xva, Xte
        gc.collect()

    table = collect_baseline_grid(run_dir, fp)
    table.to_csv(run_dir / "grid.csv", index=False)
    return table


def collect_baseline_grid(run_dir: Path, fingerprint: str) -> pd.DataFrame:
    """Table of all saved models of this split, best first."""
    rows = []
    for f in sorted((run_dir / "grid").glob("*/metrics.json")):
        m = json.loads(f.read_text())
        if m.get("fingerprint") != fingerprint:
            continue
        rows.append({"model_id": f.parent.name, "features": m["features"], "model": m["model"],
                     "C": m["C"], "macro_f1": m["macro_f1"],
                     **{f"F1 {l}": m[f"{l}_f1"] for l in LABELS}, "fit_sec": m["fit_sec"]})
    return (pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
            .reset_index(drop=True).round(4))


def finalize_baseline(run_dir: Path, va: pd.DataFrame, te: pd.DataFrame,
                      sample: pd.DataFrame) -> dict:
    """Copy the best grid model to `run_dir` as the baseline run (+ submission.csv)."""
    table = pd.read_csv(run_dir / "grid.csv")
    best_id = table.iloc[0]["model_id"]
    d = run_dir / "grid" / best_id
    m = json.loads((d / "metrics.json").read_text())
    val_proba, test_proba = np.load(d / "val_proba.npy"), np.load(d / "test_proba.npy")
    sub = make_submission(te["id"], np.array(LABELS)[test_proba.argmax(1)], sample)
    config = {k: m[k] for k in ("features", "model", "C")} | {"model_id": best_id,
                                                              "tfidf": BASELINE_FEATURES}
    metrics = {k: v for k, v in m.items() if k not in ("features", "model", "C", "fit_sec")}
    save_run(run_dir, metrics, config, va["id"], val_proba, te["id"], test_proba, sub)
    return config


# =============================================================================
# 6. Encoder: multilingual transformer + chunking for long texts
# =============================================================================
@dataclass
class EncoderConfig:
    model_name: str = "xlm-roberta-base"
    max_len: int = 512           # tokens per window incl. special tokens
    overlap: int = 64            # overlap between neighbouring windows
    train_windows: int = 1       # random windows per text per epoch (resampled each epoch)
    max_eval_windows: int = 8    # windows per text at inference, evenly spaced, probs averaged
    epochs: int = 2
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    effective_batch: int = 16    # batch_size * grad_accum
    batch_size: int | None = None        # None -> chosen from GPU memory
    eval_batch_size: int | None = None
    fp16: bool = True            # mixed precision on CUDA
    freeze_embeddings: bool | None = None  # None -> True on GPUs with < 8 GB
    num_workers: int | None = None  # None -> 2 on Linux/Colab, 0 on Windows/macOS
    checkpoint_minutes: float = 15  # save last.pt this often inside an epoch (Colab sessions are short)
    seed: int = SEED
    extra: dict = field(default_factory=dict)


def _auto_batch(cfg: EncoderConfig, device: str) -> EncoderConfig:
    import torch

    mem = torch.cuda.get_device_properties(0).total_memory / 2**30 if device == "cuda" else 0
    if cfg.batch_size is None:
        if device != "cuda":
            cfg.batch_size = 4
        else:
            scale = 512 / cfg.max_len
            cfg.batch_size = int(max(2, (16 if mem >= 14 else 8 if mem >= 10 else 2) * scale))
            cfg.batch_size = min(cfg.batch_size, cfg.effective_batch)
    if cfg.eval_batch_size is None:
        cfg.eval_batch_size = cfg.batch_size * 2
    if cfg.freeze_embeddings is None:
        cfg.freeze_embeddings = device == "cuda" and mem < 8
    if cfg.num_workers is None:
        cfg.num_workers = 2 if sys.platform.startswith("linux") else 0
    return cfg


def tokenize_texts(texts, tokenizer, batch: int = 1000) -> list[np.ndarray]:
    """Full token ids without truncation and without special tokens."""
    from tqdm.auto import tqdm

    texts = list(texts)
    out = []
    for i in tqdm(range(0, len(texts), batch), desc="tokenize", leave=False):
        ids = tokenizer(texts[i:i + batch], add_special_tokens=False, truncation=False,
                        return_attention_mask=False)["input_ids"]
        out.extend(np.asarray(x, dtype=np.int32) for x in ids)
    return out


def window_starts(n_tokens: int, size: int, stride: int) -> list[int]:
    """Start positions of windows that together cover the whole text."""
    if n_tokens <= size:
        return [0]
    starts = list(range(0, n_tokens - size + 1, stride))
    if starts[-1] + size < n_tokens:
        starts.append(n_tokens - size)
    return starts


def build_windows(docs: list[np.ndarray], cfg: EncoderConfig, k: int | None,
                  rng: np.random.Generator | None = None) -> list[tuple[int, int]]:
    """(doc index, window start) pairs. k windows per doc: random if rng, else evenly spaced."""
    size = cfg.max_len - 2
    stride = size - cfg.overlap
    items = []
    for d, ids in enumerate(docs):
        starts = window_starts(len(ids), size, stride)
        if k is not None and len(starts) > k:
            if rng is not None:
                idx = np.sort(rng.choice(len(starts), size=k, replace=False))
            else:
                idx = np.unique(np.linspace(0, len(starts) - 1, k).round().astype(int))
            starts = [starts[i] for i in idx]
        items.extend((d, s) for s in starts)
    return items


class WindowDataset:
    def __init__(self, docs, items, labels, cfg: EncoderConfig, tokenizer):
        self.docs, self.items, self.labels = docs, items, labels
        self.size = cfg.max_len - 2
        self.cls, self.sep = tokenizer.cls_token_id, tokenizer.sep_token_id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d, s = self.items[i]
        ids = [self.cls] + self.docs[d][s:s + self.size].tolist() + [self.sep]
        label = -1 if self.labels is None else int(self.labels[d])
        return ids, label, d


class Collate:
    """Pads a batch of windows (a class, not a closure, so Windows DataLoader workers can pickle it)."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch

        n = max(len(b[0]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, (x, _, _) in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = 1
        labels = torch.tensor([b[1] for b in batch])
        doc = torch.tensor([b[2] for b in batch])
        return ids, mask, labels, doc


def _autocast(device: str, cfg: EncoderConfig):
    import contextlib

    import torch

    if device == "cuda" and cfg.fp16:
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def predict_windows(model, tokenizer, docs, cfg: EncoderConfig, device: str, k: int | None,
                    desc: str = "predict"):
    """Probabilities of every window -> (doc index, window start, probs (n_windows, 3))."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    items = build_windows(docs, cfg, k)
    items.sort(key=lambda it: min(len(docs[it[0]]) - it[1], cfg.max_len))  # less padding
    ds = WindowDataset(docs, items, None, cfg, tokenizer)
    dl = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False,
                    num_workers=cfg.num_workers, collate_fn=Collate(tokenizer.pad_token_id))
    probs = []
    model.eval()
    with torch.no_grad():
        for ids, mask, _, _ in tqdm(dl, desc=desc, leave=False):
            with _autocast(device, cfg):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
    doc = np.array([it[0] for it in items], dtype=np.int64)
    start = np.array([it[1] for it in items], dtype=np.int64)
    return doc, start, np.concatenate(probs).astype(np.float32)


def predict_docs(model, tokenizer, docs, cfg: EncoderConfig, device: str,
                 desc: str = "predict") -> np.ndarray:
    """Average window probabilities per text -> (n_docs, 3) in LABELS order."""
    doc, _, p = predict_windows(model, tokenizer, docs, cfg, device, cfg.max_eval_windows, desc)
    sums = np.zeros((len(docs), len(LABELS)))
    np.add.at(sums, doc, p)
    return (sums / np.bincount(doc, minlength=len(docs))[:, None]).astype(np.float32)


def _load_model_and_tokenizer(cfg: EncoderConfig):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id=LABEL2ID)
    return model, tokenizer


def train_encoder(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame,
                  cfg: EncoderConfig, run_dir: Path, resume: bool = True):
    """Fine-tune the encoder on the train split, select the best epoch on val.

    Long texts: each training step sees `train_windows` random windows of a text
    (new windows every epoch), inference averages probabilities over up to
    `max_eval_windows` evenly spaced windows that cover the text.

    Colab-safe: `last.pt` is written every `checkpoint_minutes` inside an epoch and
    after every epoch. If the session dies, calling this again continues from the
    next batch (window choice and batch order of every epoch are fixed by the seed).
    `last.pt` holds only trainable weights + their optimizer state; it is replaced in
    place and deleted when training ends. The best epoch's weights: `best_model.pt` (fp16).
    """
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import get_linear_schedule_with_warmup

    seed_everything(cfg.seed)
    device = get_device()
    run_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = run_dir / "last.pt"
    state = None
    if resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        cfg.batch_size = state["batch_size"]  # batch order must match the interrupted run
        cfg.freeze_embeddings = state["freeze_embeddings"]
    cfg = _auto_batch(cfg, device)
    grad_accum = max(1, cfg.effective_batch // cfg.batch_size)
    print(f"device={device}, batch={cfg.batch_size} x accum {grad_accum}, "
          f"freeze_embeddings={cfg.freeze_embeddings}")
    if device == "cpu":
        print("WARNING: no GPU - fine-tuning on CPU takes days. Switch the runtime to GPU.")

    model, tokenizer = _load_model_and_tokenizer(cfg)
    if cfg.freeze_embeddings:
        for p in model.base_model.embeddings.parameters():
            p.requires_grad = False
    model.to(device)

    tr_docs = tokenize_texts(tr["text"], tokenizer)
    va_docs = tokenize_texts(va["text"], tokenizer)
    tr_y = tr["label"].map(LABEL2ID).to_numpy()

    no_decay = ("bias", "LayerNorm.weight", "layernorm", "layer_norm")
    params = [p for p in model.named_parameters() if p[1].requires_grad]
    trainable = {n for n, _ in params}
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in params if not any(k in n for k in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in params if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ], lr=cfg.lr)
    n_windows = sum(min(cfg.train_windows, len(window_starts(len(d), cfg.max_len - 2,
                                                            cfg.max_len - 2 - cfg.overlap)))
                    for d in tr_docs)
    n_batches = int(np.ceil(n_windows / cfg.batch_size))
    steps_per_epoch = int(np.ceil(n_batches / grad_accum))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * steps_per_epoch * cfg.epochs), steps_per_epoch * cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and cfg.fp16))

    start_epoch, start_batch, best_f1, history = 0, 0, -1.0, []
    running, n_seen = 0.0, 0
    if state is not None:
        model.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, start_batch = state["epoch"], state["batch"]
        best_f1, history = state["best_f1"], state["history"]
        running, n_seen = state["running"], state["n_seen"]
        torch.set_rng_state(state["rng_cpu"])  # same dropout masks as an uninterrupted run
        if device == "cuda" and state["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        print(f"resumed: epoch {start_epoch + 1}, batch {start_batch}/{n_batches} "
              f"(best val macro-F1 so far {best_f1:.4f})")
        del state

    def save_last(epoch: int, batch: int) -> None:
        """Atomic save: a session that dies while writing never leaves a broken last.pt."""
        tmp = run_dir / "last.pt.tmp"
        torch.save({"model": {k: v for k, v in model.state_dict().items() if k in trainable},
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "batch": batch,
                    "best_f1": best_f1, "history": history, "running": running,
                    "n_seen": n_seen, "batch_size": cfg.batch_size,
                    "freeze_embeddings": cfg.freeze_embeddings,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None}, tmp)
        os.replace(tmp, last_ckpt)

    collate = Collate(tokenizer.pad_token_id)
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        windows = build_windows(tr_docs, cfg, cfg.train_windows, np.random.default_rng(cfg.seed + epoch))
        ds = WindowDataset(tr_docs, windows, tr_y, cfg, tokenizer)
        # fixed batch order for this epoch -> after a resume the epoch continues where it stopped
        order = np.random.default_rng(cfg.seed + 1000 + epoch).permutation(len(ds))
        first = start_batch if epoch == start_epoch else 0
        if first == 0:
            running, n_seen = 0.0, 0
        dl = DataLoader(ds, batch_size=cfg.batch_size, sampler=order[first * cfg.batch_size:].tolist(),
                        collate_fn=collate, num_workers=cfg.num_workers,
                        generator=torch.Generator().manual_seed(cfg.seed + epoch))  # keeps global RNG untouched
        model.train()
        optimizer.zero_grad(set_to_none=True)
        last_save, bad_steps = time.time(), 0
        bar = tqdm(dl, desc=f"epoch {epoch + 1}/{cfg.epochs}", initial=first, total=n_batches,
                   leave=False)
        for step, (ids, mask, labels, _) in enumerate(bar, start=first):
            with _autocast(device, cfg):
                out = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                            labels=labels.to(device))
                loss = out.loss / grad_accum
            scaler.scale(loss).backward()
            loss_value = out.loss.item()
            if not np.isfinite(loss_value):  # fp16 overflow: GradScaler skips this step
                bad_steps += 1
                if bad_steps >= 50:
                    raise RuntimeError("50 non-finite losses in a row - rerun with fp16=False")
            else:
                bad_steps = 0
                running += loss_value * len(labels)
                n_seen += len(labels)
            if (step + 1) % grad_accum == 0 or step + 1 == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if step + 1 < n_batches and time.time() - last_save > cfg.checkpoint_minutes * 60:
                    save_last(epoch, step + 1)
                    last_save = time.time()
            if step % 50 == 0:
                bar.set_postfix(loss=f"{running / max(1, n_seen):.4f}")

        train_loss = running / max(1, n_seen)
        save_last(epoch, n_batches)  # training part of the epoch is safe before the long val pass
        va_proba = predict_docs(model, tokenizer, va_docs, cfg, device, desc="val")
        m = evaluate(va["label"], np.array(LABELS)[va_proba.argmax(1)])
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "val_macro_f1": m["macro_f1"], "minutes": (time.time() - t0) / 60})
        print(f"epoch {epoch + 1}: train loss {train_loss:.4f}, "
              f"val macro-F1 {m['macro_f1']:.4f} ({history[-1]['minutes']:.1f} min)")
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            torch.save({k: v.half() if v.is_floating_point() else v
                        for k, v in model.state_dict().items()}, run_dir / "best_model.pt")
            np.save(run_dir / "val_proba.npy", va_proba)
        save_last(epoch + 1, 0)

    # --- best epoch -> final val metrics + test predictions ---
    state = torch.load(run_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    model.to(device)
    va_proba = np.load(run_dir / "val_proba.npy")
    te_proba = predict_docs(model, tokenizer, tokenize_texts(te["text"], tokenizer),
                            cfg, device, desc="test")
    metrics = evaluate(va["label"], np.array(LABELS)[va_proba.argmax(1)])
    metrics["history"] = history
    last_ckpt.unlink(missing_ok=True)  # not needed once training is finished
    return {"config": asdict(cfg), "metrics": metrics, "val_proba": va_proba,
            "test_proba": te_proba, "history": pd.DataFrame(history)}


# =============================================================================
# 7. Experiments
# =============================================================================
def save_experiment(paths: Paths, name: str, va: pd.DataFrame, te: pd.DataFrame,
                    val_proba: np.ndarray, test_proba: np.ndarray, sample: pd.DataFrame,
                    config: dict) -> dict:
    """Evaluate on val, write submission and everything else to runs/<name>/."""
    m = evaluate(va["label"], np.array(LABELS)[np.asarray(val_proba).argmax(1)])
    sub = make_submission(te["id"], np.array(LABELS)[np.asarray(test_proba).argmax(1)], sample)
    save_run(paths.runs / name, m, config, va["id"], val_proba, te["id"], test_proba, sub)
    print(f"{name}: val macro-F1 = {m['macro_f1']:.4f}")
    return m


def grouped_folds(va: pd.DataFrame, n_splits: int = 5, seed: int = SEED) -> list:
    """Folds of the validation set that keep groups whole (for models trained on val)."""
    from sklearn.model_selection import StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(va, va["label"], va["group"]))


def make_hgb(seed: int = SEED):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, early_stopping=False,
                                          random_state=seed)


def model_proba(model, X) -> np.ndarray:
    order = [list(model.classes_).index(l) for l in LABELS]
    return model.predict_proba(X)[:, order]


def cv_on_val(X_val: pd.DataFrame, y_val, folds, X_test: pd.DataFrame, make_model):
    """Models that learn from other models' val outputs are trained on val with grouped CV.

    Returns out-of-fold val probabilities (honest estimate) and test probabilities of a
    model fitted on the whole val set.
    """
    y_val = np.asarray(y_val)
    oof = np.zeros((len(X_val), len(LABELS)), dtype=np.float32)
    for tr_idx, te_idx in folds:
        model = make_model().fit(X_val.iloc[tr_idx], y_val[tr_idx])
        oof[te_idx] = model_proba(model, X_val.iloc[te_idx])
    final = make_model().fit(X_val, y_val)
    return oof, model_proba(final, X_test).astype(np.float32), final


def add_meta_columns(X: pd.DataFrame, df: pd.DataFrame, langs, domains) -> pd.DataFrame:
    """One-hot language and domain (fixed category lists, same columns for every split)."""
    X = X.reset_index(drop=True).copy()
    for v in langs:
        X[f"lang={v}"] = (df["lang"].to_numpy() == v).astype(np.float32)
    for v in domains:
        X[f"domain={v}"] = (df["domain"].to_numpy() == v).astype(np.float32)
    return X


def plot_permutation_importance(model, X: pd.DataFrame, y, title: str, top: int = 15,
                                n: int = 3000, seed: int = SEED):
    """Drop in val macro-F1 when one feature is shuffled (bigger = model relies on it more)."""
    from sklearn.inspection import permutation_importance

    idx = np.random.default_rng(seed).permutation(len(X))[:n]
    r = permutation_importance(model, X.iloc[idx], np.asarray(y)[idx], scoring="f1_macro",
                               n_repeats=3, random_state=seed)
    imp = pd.Series(r.importances_mean, index=X.columns).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(imp) + 1))
    imp.plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("drop of macro-F1 when shuffled")
    ax.set_title(title)
    fig.tight_layout()
    return fig, imp.sort_values(ascending=False)


# ---- Experiments 1-2: window-level predictions of a trained encoder ---------
def load_encoder_config(run_dir: Path) -> EncoderConfig:
    from dataclasses import fields

    conf = json.loads((run_dir / "config.json").read_text())
    names = {f.name for f in fields(EncoderConfig)}
    return EncoderConfig(**{k: v for k, v in conf.items() if k in names})


def load_trained_encoder(run_dir: Path):
    """Model + tokenizer with the best epoch's weights of a finished encoder run."""
    import torch

    cfg = load_encoder_config(run_dir)
    device = get_device()
    cfg.batch_size = cfg.eval_batch_size = None
    cfg = _auto_batch(cfg, device)
    model, tokenizer = _load_model_and_tokenizer(cfg)
    state = torch.load(run_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return model.to(device), tokenizer, cfg, device


def encoder_window_predictions(run_dir: Path, df: pd.DataFrame, split: str,
                               recompute: bool = False) -> dict:
    """Probabilities of ALL windows of every text; cached in run_dir/windows_<split>.npz.

    pos = index of the window inside its text (0 = beginning), n = windows of that text.
    """
    f = run_dir / f"windows_{split}.npz"
    ids = df["id"].astype(str).to_numpy()
    if f.exists() and not recompute:
        z = np.load(f, allow_pickle=False)
        if len(z["ids"]) == len(ids) and (z["ids"] == ids).all():
            return {k: z[k] for k in ("doc", "pos", "n", "probs")}
    model, tokenizer, cfg, device = load_trained_encoder(run_dir)
    if device == "cpu":
        print("WARNING: no GPU - window predictions on CPU are very slow.")
    docs = tokenize_texts(df["text"], tokenizer)
    doc, start, probs = predict_windows(model, tokenizer, docs, cfg, device, k=None, desc=split)
    order = np.lexsort((start, doc))
    doc, start, probs = doc[order], start[order], probs[order]
    first = np.searchsorted(doc, doc)                       # index of the doc's first window
    pos = np.arange(len(doc)) - first
    n = np.bincount(doc, minlength=len(df))[doc]
    np.savez(f, ids=ids, doc=doc, pos=pos, n=n, probs=probs)
    del model
    return {"doc": doc, "pos": pos, "n": n, "probs": probs}


def aggregate_windows(win: dict, n_docs: int, how: str) -> np.ndarray:
    """first = beginning of the text only; first_last = beginning + end; mean = all windows."""
    doc, pos, n, p = win["doc"], win["pos"], win["n"], win["probs"]
    if how == "first":
        keep = pos == 0
    elif how == "first_last":
        keep = (pos == 0) | (pos == n - 1)
    elif how == "mean":
        keep = np.ones(len(doc), dtype=bool)
    else:
        raise ValueError(how)
    sums = np.zeros((n_docs, len(LABELS)))
    np.add.at(sums, doc[keep], p[keep])
    return (sums / np.bincount(doc[keep], minlength=n_docs)[:, None]).astype(np.float32)


def window_disagreement_features(win: dict, n_docs: int) -> pd.DataFrame:
    """Per text: mean / max / min / std of each class probability over its windows,
    first and last window, max - min (how much the windows disagree), number of windows."""
    doc, pos, n, p = win["doc"], win["pos"], win["n"], win["probs"]
    cols = {}
    frame = pd.DataFrame(p, columns=LABELS)
    frame["doc"] = doc
    g = frame.groupby("doc")[LABELS]
    for stat in ("mean", "max", "min", "std"):
        agg = getattr(g, stat)().reindex(range(n_docs))
        for l in LABELS:
            cols[f"{l}_{stat}"] = agg[l].to_numpy()
    for l in LABELS:
        cols[f"{l}_range"] = cols[f"{l}_max"] - cols[f"{l}_min"]
        cols[f"{l}_std"] = np.nan_to_num(cols[f"{l}_std"])  # single window -> 0
    for name, mask in (("first", pos == 0), ("last", pos == n - 1)):
        sub = pd.DataFrame(p[mask], columns=LABELS, index=doc[mask]).reindex(range(n_docs))
        for l in LABELS:
            cols[f"{l}_{name}"] = sub[l].to_numpy()
    cols["n_windows"] = np.bincount(doc, minlength=n_docs).astype(float)
    return pd.DataFrame(cols)


# ---- Experiment 3: stylometry -------------------------------------------------
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_PUNCT = {",": "comma", ".": "period", ";": "semicolon", ":": "colon", "!": "excl", "?": "quest",
          "(": "paren", '"': "dquote", "'": "squote", "-": "hyphen", "–": "endash", "—": "emdash",
          "…": "ellipsis"}
_FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours",
                 "ich", "mich", "mir", "mein", "meine", "meiner", "wir", "uns", "unser", "unsere"}


def _style_basic(text: str) -> dict:
    sents = [s for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    sent_len = np.array([len(_WORD_RE.findall(s)) for s in sents] or [0], dtype=float)
    words = _WORD_RE.findall(text)
    wlen = np.array([len(w) for w in words] or [0], dtype=float)
    first = [w.lower() for w in words[:150]]
    return {"sent_len_mean": sent_len.mean(), "word_len_mean": wlen.mean(),
            "comma_per_sent": text.count(",") / max(1, len(sents)),
            "ttr150": len(set(first)) / max(1, len(first))}


def stylometry_features(text: str) -> dict:
    """Numbers that describe HOW a text is written, not what it is about."""
    n_chars = max(1, len(text))
    sents = [s for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    sent_len = np.array([len(_WORD_RE.findall(s)) for s in sents] or [0], dtype=float)
    words = _WORD_RE.findall(text)
    n_words = max(1, len(words))
    wlen = np.array([len(w) for w in words] or [0], dtype=float)
    low = [w.lower() for w in words]
    first300 = low[:300]
    counts = pd.Series(first300).value_counts() if first300 else pd.Series(dtype=int)
    letters = sum(c.isalpha() for c in text)
    f = {
        "n_words": len(words), "n_sents": len(sents), "log_chars": np.log1p(len(text)),
        "sent_len_mean": sent_len.mean(), "sent_len_std": sent_len.std(),
        "sent_len_cv": sent_len.std() / max(1e-6, sent_len.mean()),
        "sent_len_max": sent_len.max(), "short_sent_share": (sent_len < 8).mean(),
        "long_sent_share": (sent_len > 30).mean(),
        "word_len_mean": wlen.mean(), "word_len_std": wlen.std(),
        "long_word_share": (wlen >= 7).mean(),
        "ttr300": len(counts) / max(1, len(first300)),
        "hapax300": (counts == 1).sum() / max(1, len(first300)),
        "digit_share": sum(c.isdigit() for c in text) / n_chars,
        "upper_share": sum(c.isupper() for c in text) / max(1, letters),
        "first_person_per100": 100 * sum(w in _FIRST_PERSON for w in low) / n_words,
        "newlines_per1k": 1000 * text.count("\n") / n_chars,
        "paragraphs": text.count("\n\n") + 1,
        "quest_sent_share": np.mean([s.rstrip().endswith("?") for s in sents]) if sents else 0.0,
        "excl_sent_share": np.mean([s.rstrip().endswith("!") for s in sents]) if sents else 0.0,
    }
    for ch, name in _PUNCT.items():
        f[f"{name}_per1k"] = 1000 * text.count(ch) / n_chars
    # style inside the text: first half vs second half, and spread over 5-sentence chunks
    half = max(1, len(sents) // 2)
    a, b = _style_basic(" ".join(sents[:half])), _style_basic(" ".join(sents[half:]))
    for k in a:
        f[f"halves_diff_{k}"] = abs(a[k] - b[k])
    chunks = [sent_len[i:i + 5].mean() for i in range(0, len(sent_len), 5)]
    f["chunk_sent_len_std"] = float(np.std(chunks)) if len(chunks) > 1 else 0.0
    return f


def load_or_make_stylometry(df: pd.DataFrame, cache_file: Path, recompute: bool = False) -> pd.DataFrame:
    if cache_file.exists() and not recompute:
        X = pd.read_pickle(cache_file)
        if len(X) == len(df) and (X.index == df["id"].to_numpy()).all():
            return X.reset_index(drop=True)
    from tqdm.auto import tqdm

    X = pd.DataFrame([stylometry_features(t) for t in tqdm(df["text"], desc=cache_file.stem, leave=False)])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    X.set_axis(df["id"].to_numpy()).to_pickle(cache_file)
    return X


# ---- Experiment 4: perplexity features from an open causal LM ----------------
@dataclass
class PerplexityConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"   # open weights, multilingual (incl. German)
    max_tokens: int = 1024                  # tokens read per text
    segment: int = 64                       # tokens per segment for "inside the text" features
    batch_size: int = 4
    shard_size: int = 2000                  # texts per saved shard (resume after a crash)
    train_n: int | None = 16_000            # train texts used (None = all of the train split)


def _load_causal_lm(cfg: PerplexityConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    return model, tok


PPL_FEATURES = ["n_tokens", "lp_mean", "lp_std", "lp_min", "lp_p10", "lp_p25", "lp_p50", "lp_p75",
                "lp_p90", "ent_mean", "ent_std", "lp_over_ent", "surprise_minus_ent_mean",
                "surprise_minus_ent_std", "seg_lp_std", "seg_lp_range", "seg_lp_min", "seg_lp_max",
                "seg_ratio_std", "halves_lp_diff"]


def _ppl_doc_features(lp: np.ndarray, ent: np.ndarray, segment: int) -> list[float]:
    """lp = log-probability of every actual token, ent = entropy of the model's prediction."""
    if len(lp) < 2:
        return [len(lp)] + [np.nan] * (len(PPL_FEATURES) - 1)
    surprise = -lp
    seg_lp = np.array([lp[i:i + segment].mean() for i in range(0, len(lp), segment)])
    seg_ratio = np.array([surprise[i:i + segment].mean() / max(1e-6, ent[i:i + segment].mean())
                          for i in range(0, len(lp), segment)])
    h = len(lp) // 2
    return [len(lp), lp.mean(), lp.std(), lp.min(), *np.percentile(lp, [10, 25, 50, 75, 90]),
            ent.mean(), ent.std(), surprise.mean() / max(1e-6, ent.mean()),
            (surprise - ent).mean(), (surprise - ent).std(),
            seg_lp.std(), seg_lp.max() - seg_lp.min(), seg_lp.min(), seg_lp.max(),
            seg_ratio.std(), abs(lp[:h].mean() - lp[h:].mean())]


def _ppl_batch(model, tok, texts, cfg: PerplexityConfig, device: str) -> np.ndarray:
    import torch

    ids = tok(list(texts), add_special_tokens=False, truncation=True,
              max_length=cfg.max_tokens)["input_ids"]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    n = max(2, max(len(x) for x in ids))
    x = torch.full((len(ids), n), pad, dtype=torch.long)
    mask = torch.zeros((len(ids), n), dtype=torch.long)
    for i, seq in enumerate(ids):
        x[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, :len(seq)] = 1
    with torch.no_grad():
        logits = model(input_ids=x.to(device), attention_mask=mask.to(device)).logits
        rows = []
        for i, seq in enumerate(ids):
            L = len(seq)
            if L < 2:
                rows.append(_ppl_doc_features(np.zeros(L), np.zeros(L), cfg.segment))
                continue
            logp = torch.log_softmax(logits[i, :L - 1].float(), -1)
            target = x[i, 1:L].to(device)
            lp = logp.gather(1, target[:, None])[:, 0]
            ent = -(logp.exp() * logp).sum(-1)
            rows.append(_ppl_doc_features(lp.cpu().numpy(), ent.cpu().numpy(), cfg.segment))
            del logp
    return np.asarray(rows, dtype=np.float32)


def load_or_make_perplexity(df: pd.DataFrame, cfg: PerplexityConfig, cache_dir: Path,
                            split: str) -> pd.DataFrame:
    """Perplexity features of every text, computed in shards that are saved one by one."""
    import torch
    from tqdm.auto import tqdm

    d = cache_dir / split
    d.mkdir(parents=True, exist_ok=True)
    meta = {"model": cfg.model_name, "max_tokens": cfg.max_tokens, "segment": cfg.segment,
            "shard_size": cfg.shard_size, "ids_hash": split_fingerprint(df, df.iloc[:0])}
    meta_file = d / "meta.json"
    if meta_file.exists() and json.loads(meta_file.read_text()) != meta:
        for f in d.glob("shard_*.npy"):
            f.unlink()  # different texts or settings -> start again
    meta_file.write_text(json.dumps(meta, indent=2))

    n_shards = int(np.ceil(len(df) / cfg.shard_size))
    todo = [i for i in range(n_shards) if not (d / f"shard_{i:04d}.npy").exists()]
    if todo:
        device = get_device()
        if device == "cpu":
            print("WARNING: no GPU - perplexity features on CPU are very slow.")
        model, tok = _load_causal_lm(cfg)
        model = model.to(device, dtype=torch.float16 if device == "cuda" else torch.float32).eval()
        texts = df["text"].tolist()
        for i in tqdm(todo, desc=f"{split} shards", leave=False):
            lo, hi = i * cfg.shard_size, min(len(df), (i + 1) * cfg.shard_size)
            chunk = texts[lo:hi]
            order = np.argsort([len(t) for t in chunk])  # similar lengths in a batch -> less padding
            out = np.zeros((len(chunk), len(PPL_FEATURES)), dtype=np.float32)
            for b in range(0, len(chunk), cfg.batch_size):
                idx = order[b:b + cfg.batch_size]
                out[idx] = _ppl_batch(model, tok, [chunk[j] for j in idx], cfg, device)
            np.save(d / f"shard_{i:04d}.npy", out)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    feats = np.concatenate([np.load(d / f"shard_{i:04d}.npy") for i in range(n_shards)])
    return pd.DataFrame(feats, columns=PPL_FEATURES)


# ---- Experiment 6: thresholds (class bias) and stacking -----------------------
def _log(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0))


def tune_class_bias(proba: np.ndarray, y, grid=np.linspace(-1.5, 1.5, 31), passes: int = 2) -> np.ndarray:
    """Additive bias for log-probabilities of ai and fusion (human fixed at 0) that maximises
    macro-F1. Coordinate search; on ties the smaller bias wins."""
    from sklearn.metrics import f1_score

    y = np.asarray(y)
    lp = _log(proba)
    bias = np.zeros(len(LABELS))
    score = lambda b: f1_score(y, np.array(LABELS)[(lp + b).argmax(1)], labels=LABELS, average="macro")
    best = score(bias)
    for _ in range(passes):
        for c in (1, 2):
            for v in sorted(grid, key=abs):
                b = bias.copy()
                b[c] = v
                s = score(b)
                if s > best + 1e-9:
                    best, bias = s, b
    return bias


def apply_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return scores_to_proba(_log(proba) + bias).astype(np.float32)


def cv_class_bias(val_proba: np.ndarray, y, folds, test_proba: np.ndarray):
    """Honest estimate of thresholds: each val fold uses a bias tuned on the other folds.
    Test uses the bias tuned on the whole val set."""
    y = np.asarray(y)
    out = np.zeros_like(val_proba, dtype=np.float32)
    for tr_idx, te_idx in folds:
        out[te_idx] = apply_bias(val_proba[te_idx], tune_class_bias(val_proba[tr_idx], y[tr_idx]))
    bias = tune_class_bias(val_proba, y)
    return out, apply_bias(test_proba, bias), bias


def run_probs(paths: Paths, name: str, va: pd.DataFrame, te: pd.DataFrame):
    """Saved val / test probabilities of a run, aligned to the rows of va / te."""
    run = load_run(paths.runs / name)
    pv = pd.DataFrame(run["val_proba"], index=run["val_ids"]).reindex(va["id"]).to_numpy()
    pt = pd.DataFrame(run["test_proba"], index=run["test_ids"]).reindex(te["id"]).to_numpy()
    return pv.astype(np.float32), pt.astype(np.float32)


def stacking_inputs(paths: Paths, names: list[str], va: pd.DataFrame, te: pd.DataFrame):
    """Log-probabilities of every available run as columns (aligned to va / te by id)."""
    Xv, Xt, used = {}, {}, []
    for name in names:
        if not run_exists(paths, name):
            continue
        pv, pt = run_probs(paths, name, va, te)
        for j, l in enumerate(LABELS):
            Xv[f"{name}:{l}"], Xt[f"{name}:{l}"] = _log(pv[:, j]), _log(pt[:, j])
        used.append(name)
    return pd.DataFrame(Xv), pd.DataFrame(Xt), used


def make_stacker(seed: int = SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=seed))
