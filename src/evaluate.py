"""Evaluation for Kavach AI: in-domain metrics, out-of-domain binary tests, EDA stats,
figures, and the final artifacts/metrics.json.

Run after `python -m src.train`:   python -m src.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
    precision_score, recall_score, roc_auc_score,
)

from src.preprocess import (  # noqa: E402
    CLASSES, add_features, hinglish_only, load_hinglish, load_sentinel, load_uci,
)
from src.red_flags import advisory_level, check_red_flags  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"
MODEL_PATH = ARTIFACTS / "model.joblib"
TRAIN_RESULTS_PATH = ARTIFACTS / "train_results.json"
METRICS_PATH = ARTIFACTS / "metrics.json"
BASELINE_DIR = ARTIFACTS / "baseline_with_phone"  # snapshot before Fix A (phones removed)

# Chart styling (validated reference palette, light mode)
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
# One fixed colour per class, used in every per-class figure (validated palette slots 1-5)
CLASS_COLORS = dict(zip(CLASSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))


def r4(x) -> float:
    return round(float(x), 4)


# ---------------------------------------------------------------- metrics

def multiclass_metrics(y_true, y_pred, proba) -> dict:
    return {
        "accuracy": r4(accuracy_score(y_true, y_pred)),
        "macro_f1": r4(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": r4(f1_score(y_true, y_pred, average="weighted")),
        "roc_auc_ovr": r4(roc_auc_score(y_true, proba, multi_class="ovr",
                                        labels=list(range(len(CLASSES))))),
    }


def per_class_metrics(y_true, y_pred) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(CLASSES))),
                                                 zero_division=0)
    return {c: {"precision": r4(p[i]), "recall": r4(r[i]), "f1": r4(f[i]), "support": int(s[i])}
            for i, c in enumerate(CLASSES)}


def confusion(y_true, y_pred) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))).tolist()


def binary_metrics(y_true, y_pred) -> dict:
    """Scam vs legit, positive class = scam."""
    return {
        "n": int(len(y_true)),
        "accuracy": r4(accuracy_score(y_true, y_pred)),
        "precision": r4(precision_score(y_true, y_pred, zero_division=0)),
        "recall": r4(recall_score(y_true, y_pred, zero_division=0)),
        "f1": r4(f1_score(y_true, y_pred, zero_division=0)),
    }


def predict_scam(model, texts) -> np.ndarray:
    """1 if the predicted class (argmax over 5) is any non-legit class."""
    return (model.predict(list(texts)) != CLASSES.index("legit")).astype(int)


# ---------------------------------------------------------------- OOD + EDA

def evaluate_ood(model) -> tuple[dict, dict]:
    uci, uci_dropped = load_uci()
    hing, hing_dropped = load_hinglish()
    hing_only = hinglish_only(hing, uci)
    ood = {}
    for name, df in [("hinglish", hing), ("uci_sms", uci), ("hinglish_only", hing_only)]:
        ood[name] = binary_metrics(df["label"].to_numpy(), predict_scam(model, df["text"]))
    dups = {"uci_sms": uci_dropped, "hinglish": hing_dropped}
    return ood, dups


def evaluate_rule_layer(model) -> dict:
    """False alarms of the red-flag advisory, and scam recall of ML alone vs ML OR advisory=high.
    Reported separately; the "ood" block stays ML-only."""
    from src.train import random_split  # local import: train imports this module

    def advisories(texts) -> np.ndarray:
        return np.array([advisory_level(check_red_flags(t)) for t in texts])

    def alarm_rates(texts) -> dict:
        adv = advisories(texts)
        return {"n": int(len(adv)), "pct_advisory_any": r4(100 * (adv != "none").mean()),
                "pct_advisory_high": r4(100 * (adv == "high").mean())}

    df, _ = load_sentinel()
    y = df["label"].to_numpy()
    _, te = random_split(y)
    legit_test = df.iloc[te][y[te] == CLASSES.index("legit")]["text"]
    uci, _ = load_uci()
    hing, _ = load_hinglish()
    sets = {"uci_sms": uci, "hinglish_only": hinglish_only(hing, uci)}

    recall = {}
    for name, d in sets.items():
        yt = d["label"].to_numpy()
        ml = predict_scam(model, d["text"])
        combined = ml | (advisories(d["text"]) == "high")
        recall[name] = {
            "n_scam": int(yt.sum()),
            "recall_ml_only": r4(recall_score(yt, ml, zero_division=0)),
            "recall_ml_or_high": r4(recall_score(yt, combined, zero_division=0)),
            "precision_ml_only": r4(precision_score(yt, ml, zero_division=0)),
            "precision_ml_or_high": r4(precision_score(yt, combined, zero_division=0)),
        }
    return {
        "false_alarms": {"sentinel_legit_test": alarm_rates(legit_test),
                         "uci_ham": alarm_rates(uci.loc[uci["label"] == 0, "text"])},
        "scam_recall": recall,
        "note": "advisory = 'high' if 3+ red flags, 'caution' if 2; rules in src/red_flags.py",
    }


def phone_ablation(train_results: dict, ood: dict) -> dict | None:
    """Compare against the snapshot taken before <PHONE> was removed from the model text."""
    if not (BASELINE_DIR / "metrics.json").exists():
        return None
    old = json.loads((BASELINE_DIR / "metrics.json").read_text())
    return {
        "macro_f1_random": {n: {"with_phone": old["models"][n]["macro_f1"],
                                "without_phone": train_results["models"][n]["macro_f1"]}
                            for n in train_results["models"]},
        "macro_f1_grouped": {n: {"with_phone": old["models_grouped"][n]["macro_f1"],
                                 "without_phone": train_results["models_grouped"][n]["macro_f1"]}
                             for n in train_results["models_grouped"]},
        "best_model": {"with_phone": old["best_model"], "without_phone": train_results["best_model"]},
        "ood": {k: {"with_phone": old["ood"][k], "without_phone": ood[k]} for k in ood},
    }


def eda_stats() -> dict:
    df, _ = load_sentinel()
    df = add_features(df)
    scam = df["label_name"] != "legit"
    out = {}
    for flag in ["url", "phone", "amount"]:
        out[f"pct_{flag}_scam"] = r4(100 * df.loc[scam, f"has_{flag}"].mean())
        out[f"pct_{flag}_legit"] = r4(100 * df.loc[~scam, f"has_{flag}"].mean())
    out["avg_words_scam"] = r4(df.loc[scam, "n_words"].mean())
    out["avg_words_legit"] = r4(df.loc[~scam, "n_words"].mean())
    return out


# ---------------------------------------------------------------- figures

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bar_labels(ax, bars):
    for b in bars:
        ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=INK)


def _save(fig, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(models: dict, models_grouped: dict):
    names = list(models)
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    b1 = ax.bar(x - w / 2 - 0.01, [models[n]["macro_f1"] for n in names], w,
                color=SERIES[0], label="Random split (stratified 80/20)")
    b2 = ax.bar(x + w / 2 + 0.01, [models_grouped[n]["macro_f1"] for n in names], w,
                color=SERIES[1], label="Template-grouped split (no template in both)")
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)
    lo = min(min(m["macro_f1"] for m in models.values()),
             min(m["macro_f1"] for m in models_grouped.values()))
    ymin = max(0, lo - 0.1)
    ax.set_ylim(ymin, 1.02)
    ax.set_xticks(x, names)
    ax.set_ylabel("Macro-F1 (test set)", color=INK_2)
    ax.set_title("Model comparison: macro-F1, random vs template-grouped split",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.text(0.01, -0.06, f"Note: y-axis starts at {ymin:.2f}, not 0.", color=INK_2, fontsize=9)
    _style(ax)
    _save(fig, "model_comparison.png")


def plot_confusion(cm: list[list[int]], model_name: str):
    cm = np.array(cm)
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    labels = [c.replace("_", "\n") for c in CLASSES]
    ax.set_xticks(range(len(CLASSES)), labels, fontsize=8, color=INK_2)
    ax.set_yticks(range(len(CLASSES)), labels, fontsize=8, color=INK_2)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else INK)
    ax.set_xlabel("Predicted", color=INK_2)
    ax.set_ylabel("True", color=INK_2)
    ax.set_title(f"Confusion matrix: {model_name} (template-grouped test split)",
                 color=INK, fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised recall")
    _save(fig, "confusion_matrix.png")


def plot_ood(ood: dict):
    sets = [("uci_sms", "UCI SMS"), ("hinglish", "Hinglish (full)"),
            ("hinglish_only", "Hinglish-only\n(not in UCI)")]
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(sets))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.5, 4.5), facecolor="white")
    for k, m in enumerate(metrics):
        bars = ax.bar(x + (k - 1.5) * (w + 0.01), [ood[s][m] for s, _ in sets], w,
                      color=SERIES[k], label=m.capitalize() if m != "f1" else "F1")
        _bar_labels(ax, bars)
    ax.set_xticks(x, [f"{lbl}\nn={ood[s]['n']}" for s, lbl in sets])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (positive class = scam)", color=INK_2)
    ax.set_title("Out-of-domain test: scam vs legit (scam = any non-legit class)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _style(ax)
    _save(fig, "ood_results.png")


# ---------------------------------------------------------------- main

def main():
    tr = json.loads(TRAIN_RESULTS_PATH.read_text())
    model = joblib.load(MODEL_PATH)
    best = tr["best_model"]

    ood, ood_dups = evaluate_ood(model)
    metrics = {
        "dataset": tr["dataset"],
        "models": tr["models"],
        "best_model": best,
        "per_class": tr["per_class"],
        "ood": ood,
        "eda": eda_stats(),
        "top_words": {},  # filled in Phase 3 by src/explain.py
        # --- additions beyond the contract ---
        "models_grouped": tr["models_grouped"],
        "per_class_grouped": tr["per_class_grouped"],
        "best_model_reason": tr["best_model_reason"],
        "dataset_grouped": tr["dataset_grouped"],
        "ood_duplicates_removed": ood_dups,
        "saved_model_note": tr["saved_model_note"],
        "rule_layer": evaluate_rule_layer(model),
    }
    ablation = phone_ablation(tr, ood)
    if ablation:
        metrics["phone_token_ablation"] = ablation
    if METRICS_PATH.exists():  # keep top_words if Phase 3 already filled it
        prev = json.loads(METRICS_PATH.read_text())
        metrics["top_words"] = prev.get("top_words") or {}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    plot_model_comparison(tr["models"], tr["models_grouped"])
    plot_confusion(tr["confusion_matrices"]["grouped"][best], best)
    plot_ood(ood)

    print("\n=== Out-of-domain (binary, positive = scam) ===")
    print(f"{'set':<15}{'n':>6}{'acc':>9}{'prec':>9}{'recall':>9}{'f1':>9}")
    for k, v in ood.items():
        print(f"{k:<15}{v['n']:>6}{v['accuracy']:>9.4f}{v['precision']:>9.4f}"
              f"{v['recall']:>9.4f}{v['f1']:>9.4f}")
    print(f"OOD exact duplicates dropped: {ood_dups}")
    print(f"EDA: {metrics['eda']}")

    if ablation:
        print("\n=== Fix A: macro-F1 with <PHONE> token (old) vs phones removed (new) ===")
        print(f"{'model':<22}{'random old':>11}{'new':>8}{'grouped old':>13}{'new':>8}")
        for n in ablation["macro_f1_random"]:
            r, g = ablation["macro_f1_random"][n], ablation["macro_f1_grouped"][n]
            print(f"{n:<22}{r['with_phone']:>11.4f}{r['without_phone']:>8.4f}"
                  f"{g['with_phone']:>13.4f}{g['without_phone']:>8.4f}")
        print(f"best model: {ablation['best_model']}")
        print(f"\n{'OOD set':<15}{'':<6}{'acc':>9}{'prec':>9}{'recall':>9}{'f1':>9}")
        for k, v in ablation["ood"].items():
            for tag, key in [("old", "with_phone"), ("new", "without_phone")]:
                m = v[key]
                print(f"{k if tag == 'old' else '':<15}{tag:<6}{m['accuracy']:>9.4f}"
                      f"{m['precision']:>9.4f}{m['recall']:>9.4f}{m['f1']:>9.4f}")

    rl = metrics["rule_layer"]
    print("\n=== Rule layer (separate from the model) ===")
    for k, v in rl["false_alarms"].items():
        print(f"false alarms {k:<20} n={v['n']:<5} advisory!=none {v['pct_advisory_any']:.2f}%  "
              f"advisory=high {v['pct_advisory_high']:.2f}%")
    for k, v in rl["scam_recall"].items():
        print(f"scam recall  {k:<20} n_scam={v['n_scam']:<4} ML only {v['recall_ml_only']:.4f} -> "
              f"ML or high {v['recall_ml_or_high']:.4f}   (precision {v['precision_ml_only']:.4f} -> "
              f"{v['precision_ml_or_high']:.4f})")
    print(f"Wrote {METRICS_PATH.relative_to(ROOT)} and figures to {FIGURES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
