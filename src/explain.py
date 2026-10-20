"""Explainability for Kavach AI.

explain_local(text)  - LIME on the RAW text: which words the user typed pushed the prediction.
explain_global(k)    - top word n-grams per class from the Linear SVM coefficients.

Run:  python -m src.explain   (fills metrics.json "top_words", saves top_words_per_class.png)
"""
from __future__ import annotations

import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from lime.lime_text import LimeTextExplainer  # noqa: E402
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # noqa: E402

from src.evaluate import (  # noqa: E402
    CLASS_COLORS, INK, INK_2, GRID, METRICS_PATH, _save,
)
from src.predict import load_model  # noqa: E402
from src.preprocess import CLASSES  # noqa: E402

LEGIT_IDX = CLASSES.index("legit")


def _as_pairs(exp, label: int) -> list[tuple[str, float]]:
    return [(str(w), round(float(v), 4)) for w, v in exp.as_list(label=label)]


def explain_local(text: str, num_features: int = 8, num_samples: int = 500, model=None) -> dict:
    """LIME explanation on the raw text (words match what the user typed).

    Returns {"label": predicted class, "words": [(word, weight)],
             "scam_class": most likely scam class or None, "scam_words": [(word, weight)]}.
    Positive weight = pushes toward that class. `scam_class`/`scam_words` are filled only when
    the prediction is legit, so the UI can also show what pointed toward a scam.
    """
    model = model or load_model()
    proba = model.predict_proba([text])[0]
    pred = int(np.argmax(proba))
    labels = [pred]
    scam_idx = None
    if pred == LEGIT_IDX:
        scam_idx = int(max((i for i in range(len(CLASSES)) if i != LEGIT_IDX), key=lambda i: proba[i]))
        labels.append(scam_idx)
    # fresh seeded explainer per call -> same text always gets the same explanation
    explainer = LimeTextExplainer(class_names=CLASSES, random_state=42)
    exp = explainer.explain_instance(text, model.predict_proba, labels=labels,
                                      num_features=num_features, num_samples=num_samples)
    return {
        "label": CLASSES[pred],
        "words": _as_pairs(exp, pred),
        "scam_class": CLASSES[scam_idx] if scam_idx is not None else None,
        "scam_words": _as_pairs(exp, scam_idx) if scam_idx is not None else [],
    }


_NUMERIC_RE = re.compile(r"^\d+$")


def _keep_feature(name: str, drop_stopwords: bool) -> bool:
    toks = name.split()
    if len(name) <= 1 or any(_NUMERIC_RE.match(t) for t in toks):
        return False
    # drop only if ALL words are stop words, so "help me" / "with you" survive
    return not (drop_stopwords and all(t in ENGLISH_STOP_WORDS for t in toks))


def explain_global(k: int = 15, model=None, drop_stopwords: bool = True
                   ) -> dict[str, list[tuple[str, float]]]:
    """Top-k word n-grams per class by Linear SVM coefficient (averaged over the calibrated
    folds). Uses only the word part of the TF-IDF union; drops numbers, single characters and
    (by default) n-grams made only of English stop words. Display-only: the model is unchanged."""
    model = model or load_model()
    clf = model.named_steps["clf"]
    coef = np.mean([cc.estimator.coef_ for cc in clf.calibrated_classifiers_], axis=0)
    union = model.named_steps["tfidf"]
    word_vec = dict(union.transformer_list)["word"]
    names = word_vec.get_feature_names_out()
    coef = coef[:, : len(names)]  # word block comes first in the union
    keep = np.array([_keep_feature(n, drop_stopwords) for n in names])
    out = {}
    for i, cls in enumerate(clf.classes_):
        idx = np.where(keep)[0]
        top = idx[np.argsort(-coef[i, idx])[:k]]
        out[CLASSES[int(cls)]] = [(str(names[j]), round(float(coef[i, j]), 4)) for j in top]
    return out


def plot_top_words(global_words: dict, name: str = "top_words_per_class.png"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), facecolor="white")
    axes = axes.ravel()
    for ax, cls in zip(axes, CLASSES):
        words = global_words[cls][::-1]
        ax.barh([w for w, _ in words], [v for _, v in words], color=CLASS_COLORS[cls], height=0.7)
        ax.set_title(cls, color=INK, fontsize=12, loc="left")
        ax.tick_params(colors=INK_2, labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("SVM coefficient (word n-gram TF-IDF)", color=INK_2, fontsize=9)
    axes[-1].axis("off")
    axes[-1].text(0, 0.5, "Top 15 word n-grams per class\nfrom the Linear SVM coefficients\n"
                  "(averaged over the 3 calibration folds),\nn-grams of only stop words removed.\n\nHigher = stronger evidence\n"
                  "for that class.\n<URL> / <AMOUNT> are masked tokens.",
                  color=INK_2, fontsize=11, va="center")
    fig.suptitle("Most indicative words per class", color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, name)


def write_top_words(global_words: dict, n: int = 5):
    metrics = json.loads(METRICS_PATH.read_text())
    metrics["top_words"] = {c: [w for w, _ in global_words[c][:n]] for c in CLASSES}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics["top_words"]


def main():
    gw = explain_global(15)
    plot_top_words(gw)
    top = write_top_words(gw)
    print("top_words:", json.dumps(top, indent=1))


if __name__ == "__main__":
    main()
