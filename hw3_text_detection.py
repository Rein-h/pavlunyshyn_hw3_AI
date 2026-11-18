"""
UCU AI 2026 - Homework 3: human / ai / fusion text classification.

All logic lives here; the notebook `hw3_text_detection.ipynb` only calls these
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
              colab_root: str = "/content/drive/MyDrive/ucu-hw3",
              colab_local_data: str = "/content/data") -> Paths:
    """Resolve project folders.

    Colab: the project folder lives on the mounted Google Drive; csv files are
    copied once to the Colab disk because reading from Drive is slow.
    Local: `local_root` argument, else env var HW3_ROOT, else ./ucu-hw3.
    """
    if in_colab():
        root = Path(colab_root)
    else:
        root = Path(local_root or os.environ.get("HW3_ROOT", "ucu-hw3"))
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


def compare_runs(paths: Paths, names: list[str], va: pd.DataFrame) -> pd.DataFrame:
    """Same val split for every run: macro-F1, per-class F1, macro-F1 per language."""
    rows = {}
    for name in names:
        run = load_run(paths.runs / name)
        proba = pd.DataFrame(run["val_proba"], index=run["val_ids"])
        pred = np.array(LABELS)[proba.reindex(va["id"]).to_numpy().argmax(1)]
        m = evaluate(va["label"], pred)
        row = {"macro_f1": m["macro_f1"], **{f"F1 {l}": m[f"{l}_f1"] for l in LABELS}}
        row.update({f"macro_f1 {k}": v for k, v in f1_by(va, pred, "lang").items()})
        rows[name] = row
    return pd.DataFrame(rows).T.round(4)


# =============================================================================
# 5. Baseline: TF-IDF + linear models
# =============================================================================
def make_vectorizer(kind: str):
    from sklearn.feature_extraction.text import TfidfVectorizer

    if kind == "word":
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                               max_features=300_000, sublinear_tf=True, dtype=np.float32)
    if kind == "char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                               max_features=300_000, sublinear_tf=True, lowercase=False,
                               dtype=np.float32)
    raise ValueError(kind)


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


DEFAULT_BASELINE_GRID = {
    "features": ["word", "char", "word+char"],
    "models": {"logreg": [1.0, 10.0], "linsvc": [0.1, 1.0]},
}


def run_baseline(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame,
                 grid: dict | None = None, verbose: bool = True):
    """Fit TF-IDF on the train split only, grid-search on the fixed val split.

    Returns (results table, best model info dict with val/test probabilities).
    """
    from scipy.sparse import hstack

    grid = grid or DEFAULT_BASELINE_GRID
    kinds = sorted({k for f in grid["features"] for k in f.split("+")})
    X = {}
    for kind in kinds:
        t0 = time.time()
        vec = make_vectorizer(kind)
        X[kind] = (vec.fit_transform(tr["text"]), vec.transform(va["text"]), vec.transform(te["text"]))
        if verbose:
            print(f"TF-IDF {kind}: {X[kind][0].shape[1]:,} features ({time.time() - t0:.0f}s)")

    def feats(fs, part):
        mats = [X[k][part] for k in fs.split("+")]
        return mats[0] if len(mats) == 1 else hstack(mats, format="csr")

    results, best = [], None
    for fs in grid["features"]:
        Xtr, Xva = feats(fs, 0), feats(fs, 1)
        for model_name, Cs in grid["models"].items():
            for C in Cs:
                t0 = time.time()
                model = make_linear_model(model_name, C).fit(Xtr, tr["label"])
                proba = model_scores(model, Xva)
                pred = np.array(LABELS)[proba.argmax(1)]
                m = evaluate(va["label"], pred)
                res = {"features": fs, "model": model_name, "C": C, "macro_f1": m["macro_f1"],
                       **{f"F1 {l}": m[f"{l}_f1"] for l in LABELS},
                       "fit_sec": round(time.time() - t0, 1)}
                results.append(res)
                if verbose:
                    print(f"  {fs:10s} {model_name:7s} C={C:<5} macro-F1={m['macro_f1']:.4f}"
                          f" ({res['fit_sec']}s)")
                if best is None or m["macro_f1"] > best["metrics"]["macro_f1"]:
                    best = {"config": {"features": fs, "model": model_name, "C": C},
                            "metrics": m, "model": model, "val_proba": proba, "features": fs}

    best["test_proba"] = model_scores(best["model"], feats(best["features"], 2))
    table = pd.DataFrame(results).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return table.round(4), best


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


def predict_docs(model, tokenizer, docs, cfg: EncoderConfig, device: str,
                 desc: str = "predict") -> np.ndarray:
    """Average window probabilities per text -> (n_docs, 3) in LABELS order."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    items = build_windows(docs, cfg, cfg.max_eval_windows)
    items.sort(key=lambda it: min(len(docs[it[0]]) - it[1], cfg.max_len))  # less padding
    ds = WindowDataset(docs, items, None, cfg, tokenizer)
    dl = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False,
                    num_workers=cfg.num_workers, collate_fn=Collate(tokenizer.pad_token_id))
    sums = np.zeros((len(docs), len(LABELS)), dtype=np.float64)
    counts = np.zeros(len(docs), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for ids, mask, _, doc in tqdm(dl, desc=desc, leave=False):
            with _autocast(device, cfg):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            p = torch.softmax(logits.float(), -1).cpu().numpy()
            np.add.at(sums, doc.numpy(), p)
            np.add.at(counts, doc.numpy(), 1)
    return (sums / counts[:, None]).astype(np.float32)


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

    Colab-safe: after every epoch `last.pt` (model + optimizer) is written to
    `run_dir`; if the session dies, calling this again resumes from it.
    Only the best epoch's weights are kept (`best_model.pt`, fp16).
    """
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import get_linear_schedule_with_warmup

    seed_everything(cfg.seed)
    device = get_device()
    cfg = _auto_batch(cfg, device)
    grad_accum = max(1, cfg.effective_batch // cfg.batch_size)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}, batch={cfg.batch_size} x accum {grad_accum}, "
          f"freeze_embeddings={cfg.freeze_embeddings}")

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
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in params if not any(k in n for k in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in params if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ], lr=cfg.lr)
    steps_per_epoch = int(np.ceil(len(tr_docs) * cfg.train_windows / cfg.batch_size / grad_accum))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * steps_per_epoch * cfg.epochs), steps_per_epoch * cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and cfg.fp16))

    start_epoch, best_f1, history = 0, -1.0, []
    last_ckpt = run_dir / "last.pt"
    if resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_f1, history = state["epoch"], state["best_f1"], state["history"]
        print(f"resumed from epoch {start_epoch} (best val macro-F1 {best_f1:.4f})")

    collate = Collate(tokenizer.pad_token_id)
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        rng = np.random.default_rng(cfg.seed + epoch)
        ds = WindowDataset(tr_docs, build_windows(tr_docs, cfg, cfg.train_windows, rng),
                           tr_y, cfg, tokenizer)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                        num_workers=cfg.num_workers,
                        generator=torch.Generator().manual_seed(cfg.seed + epoch))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running, n_seen = 0.0, 0
        bar = tqdm(dl, desc=f"epoch {epoch + 1}/{cfg.epochs}", leave=False)
        for step, (ids, mask, labels, _) in enumerate(bar):
            with _autocast(device, cfg):
                out = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                            labels=labels.to(device))
                loss = out.loss / grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0 or step + 1 == len(dl):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running += out.loss.item() * len(labels)
            n_seen += len(labels)
            if step % 50 == 0:
                bar.set_postfix(loss=f"{running / n_seen:.4f}")

        va_proba = predict_docs(model, tokenizer, va_docs, cfg, device, desc="val")
        m = evaluate(va["label"], np.array(LABELS)[va_proba.argmax(1)])
        history.append({"epoch": epoch + 1, "train_loss": running / n_seen,
                        "val_macro_f1": m["macro_f1"], "minutes": (time.time() - t0) / 60})
        print(f"epoch {epoch + 1}: train loss {running / n_seen:.4f}, "
              f"val macro-F1 {m['macro_f1']:.4f} ({history[-1]['minutes']:.1f} min)")
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            torch.save({k: v.half() if v.is_floating_point() else v
                        for k, v in model.state_dict().items()}, run_dir / "best_model.pt")
            np.save(run_dir / "val_proba.npy", va_proba)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch + 1, "best_f1": best_f1, "history": history}, last_ckpt)

    # --- best epoch -> final val metrics + test predictions ---
    state = torch.load(run_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    model.to(device)
    va_proba = np.load(run_dir / "val_proba.npy")
    te_proba = predict_docs(model, tokenizer, tokenize_texts(te["text"], tokenizer),
                            cfg, device, desc="test")
    metrics = evaluate(va["label"], np.array(LABELS)[va_proba.argmax(1)])
    metrics["history"] = history
    if last_ckpt.exists():
        last_ckpt.unlink()  # 3x model size; not needed once training is finished
    return {"config": asdict(cfg), "metrics": metrics, "val_proba": va_proba,
            "test_proba": te_proba, "history": pd.DataFrame(history)}
