"""Train and compare 4 TF-IDF models on two splits; save the best pipeline.

Splits:
  random  - stratified 80/20, random_state=42 (the metrics.json contract)
  grouped - StratifiedGroupKFold on template_key(text), first fold as test (~20%),
            so no template appears in both train and test
best_model is chosen by macro-F1 on the grouped split, then refit on ALL clean rows and
saved to artifacts/model.joblib (raw text in -> 5 class probabilities out).

Run:  python -m src.train   (then python -m src.evaluate)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from src.evaluate import confusion, multiclass_metrics, per_class_metrics
from src.preprocess import CLASSES, TOKEN_PATTERN, clean_series, load_sentinel, template_key

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SEED = 42


def make_features() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=TOKEN_PATTERN,
                                 lowercase=False, min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False,
                                 min_df=2, sublinear_tf=True)),
    ])


def make_models() -> dict:
    return {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(class_weight="balanced", C=10, max_iter=3000),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=1.0),
                                             method="sigmoid", cv=3),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                n_jobs=-1, random_state=SEED),
    }


def make_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("clean", FunctionTransformer(clean_series)),
        ("tfidf", make_features()),
        ("clf", clf),
    ])


def random_split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)


def grouped_split(y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, te_idx = next(sgkf.split(np.zeros(len(y)), y, groups))
    assert not set(groups[tr_idx]) & set(groups[te_idx]), "template leak between train/test"
    return tr_idx, te_idx


def run_split(X, y, tr_idx, te_idx) -> tuple[dict, dict, dict, dict]:
    results, per_class, cms, fitted = {}, {}, {}, {}
    for name, clf in make_models().items():
        pipe = make_pipeline(clf)
        t0 = time.perf_counter()
        pipe.fit(X[tr_idx], y[tr_idx])
        train_time = time.perf_counter() - t0
        proba = pipe.predict_proba(X[te_idx])
        pred = proba.argmax(axis=1)
        results[name] = {**multiclass_metrics(y[te_idx], pred, proba),
                         "train_time_s": round(train_time, 4)}
        per_class[name] = per_class_metrics(y[te_idx], pred)
        cms[name] = confusion(y[te_idx], pred)
        fitted[name] = pipe
        print(f"  {name:<20} macro-F1={results[name]['macro_f1']:.4f} "
              f"acc={results[name]['accuracy']:.4f} ({train_time:.1f}s)")
    return results, per_class, cms, fitted


def print_leaderboard(title: str, results: dict):
    print(f"\n=== Leaderboard: {title} ===")
    print(f"{'model':<22}{'accuracy':>10}{'macro_f1':>10}{'weighted_f1':>13}{'roc_auc':>10}{'time_s':>9}")
    for name, m in sorted(results.items(), key=lambda kv: -kv[1]["macro_f1"]):
        print(f"{name:<22}{m['accuracy']:>10.4f}{m['macro_f1']:>10.4f}{m['weighted_f1']:>13.4f}"
              f"{m['roc_auc_ovr']:>10.4f}{m['train_time_s']:>9.2f}")


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    df, stats = load_sentinel()
    X = df["text"].to_numpy(dtype=object)
    y = df["label"].to_numpy()
    groups = df["text"].map(template_key).to_numpy()

    # Random split (contract)
    tr_r, te_r = random_split(y)
    print(f"Random split: train={len(tr_r)} test={len(te_r)}")
    models, per_class_r, cms_r, _ = run_split(X, y, tr_r, te_r)

    # Template-grouped split
    tr_g, te_g = grouped_split(y, groups)
    print(f"Grouped split: train={len(tr_g)} test={len(te_g)} "
          f"(templates: {len(set(groups))} total, {len(set(groups[te_g]))} in test, 0 shared)")
    models_g, per_class_g, cms_g, _ = run_split(X, y, tr_g, te_g)

    print_leaderboard("random stratified 80/20", models)
    print_leaderboard("template-grouped", models_g)

    # Best = macro-F1 on grouped split; ties broken by random-split macro-F1, then train time.
    best = max(models_g, key=lambda n: (models_g[n]["macro_f1"], models[n]["macro_f1"],
                                        -models_g[n]["train_time_s"]))
    ranked = sorted(models_g, key=lambda n: -models_g[n]["macro_f1"])
    runner = ranked[1]
    reason = (f"{best} has the highest macro-F1 on the template-grouped split "
              f"({models_g[best]['macro_f1']:.4f} vs {runner} {models_g[runner]['macro_f1']:.4f}). "
              f"The grouped split is used because the random split lets near-identical templated "
              f"messages appear in both train and test, which inflates scores "
              f"(random-split macro-F1 for {best}: {models[best]['macro_f1']:.4f}).")
    print(f"\nBest model: {best}\nReason: {reason}")

    # Refit best on all clean rows for deployment
    final = make_pipeline(make_models()[best])
    final.fit(X, y)
    joblib.dump(final, ARTIFACTS / "model.joblib", compress=3)

    class_counts = {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}
    out = {
        "dataset": {**stats, "train_rows": int(len(tr_r)), "test_rows": int(len(te_r)),
                    "class_counts": class_counts},
        "dataset_grouped": {"n_templates": int(len(set(groups))), "train_rows": int(len(tr_g)),
                            "test_rows": int(len(te_g)),
                            "test_class_counts": {c: int((y[te_g] == i).sum())
                                                  for i, c in enumerate(CLASSES)}},
        "models": models,
        "models_grouped": models_g,
        "best_model": best,
        "best_model_reason": reason,
        "per_class": per_class_r[best],
        "per_class_grouped": per_class_g[best],
        "confusion_matrices": {"random": cms_r, "grouped": cms_g, "labels": CLASSES},
        "saved_model_note": (f"artifacts/model.joblib is {best} refit on all "
                             f"{stats['clean_rows']} clean rows; in-domain metrics come from "
                             f"the split-trained models; OOD metrics use the saved model."),
    }
    (ARTIFACTS / "train_results.json").write_text(json.dumps(out, indent=2))
    (ARTIFACTS / "confusion_matrices.json").write_text(
        json.dumps(out["confusion_matrices"], indent=2))
    print(f"Saved artifacts/model.joblib, train_results.json, confusion_matrices.json")


if __name__ == "__main__":
    main()
