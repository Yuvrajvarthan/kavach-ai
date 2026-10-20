"""Generate README.md. Every number comes from artifacts/metrics.json, none typed by hand.

    python scripts/build_readme.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = json.loads((ROOT / "artifacts" / "metrics.json").read_text())

REPO = "https://github.com/Yuvrajvarthan/kavach-ai"

best = M["best_model"]
ds, dg = M["dataset"], M["dataset_grouped"]
ood, rl, ab, eda = M["ood"], M["rule_layer"], M["phone_token_ablation"], M["eda"]


def f4(x):
    return f"{x:.4f}"


def pct(x, dp=1):
    return f"{x:.{dp}f}%"


# ------------------------------------------------------------------ tables
leader = ["| Model | Macro-F1 (grouped) | Macro-F1 (random) | Accuracy (grouped) | ROC-AUC OvR (grouped) | Train time (s) |",
          "|---|---:|---:|---:|---:|---:|"]
for name in sorted(M["models"], key=lambda n: -M["models_grouped"][n]["macro_f1"]):
    g, r = M["models_grouped"][name], M["models"][name]
    star = " ⭐" if name == best else ""
    leader.append(f"| **{name}**{star} | {f4(g['macro_f1'])} | {f4(r['macro_f1'])} | {f4(g['accuracy'])} | "
                  f"{f4(g['roc_auc_ovr'])} | {r['train_time_s']:.1f} |")

per_class = ["| Class | F1 (grouped) | F1 (random) | Test support (random) |", "|---|---:|---:|---:|"]
for c, v in M["per_class"].items():
    per_class.append(f"| `{c}` | {f4(M['per_class_grouped'][c]['f1'])} | {f4(v['f1'])} | {v['support']} |")

OOD_NAMES = {"uci_sms": "UCI SMS Spam", "hinglish": "Hinglish & English (full)",
             "hinglish_only": "Hinglish-only (rows not in UCI)"}
ood_tbl = ["| Test set | n | Accuracy | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|"]
for k in ["uci_sms", "hinglish", "hinglish_only"]:
    v = ood[k]
    ood_tbl.append(f"| {OOD_NAMES[k]} | {v['n']} | {f4(v['accuracy'])} | {f4(v['precision'])} | "
                   f"{f4(v['recall'])} | {f4(v['f1'])} |")
n_scam_hing_only = rl["scam_recall"]["hinglish_only"]["n_scam"]
pct_hing_in_uci = 100 * (1 - ood["hinglish_only"]["n"] / ood["hinglish"]["n"])

drops = ", ".join(f"{n} {f4(M['models'][n]['macro_f1'])} → {f4(M['models_grouped'][n]['macro_f1'])}"
                  for n in M["models"])
pct_repeat = 100 * (1 - dg["n_templates"] / ds["clean_rows"])

abl = ["| | Macro-F1 grouped | UCI F1 | UCI recall | Hinglish-only F1 |", "|---|---:|---:|---:|---:|"]
for tag, key in [("With a `<PHONE>` token", "with_phone"), ("Phone numbers removed (final model)", "without_phone")]:
    abl.append(f"| {tag} | {f4(ab['macro_f1_grouped'][best][key])} | {f4(ab['ood']['uci_sms'][key]['f1'])} | "
               f"{f4(ab['ood']['uci_sms'][key]['recall'])} | {f4(ab['ood']['hinglish_only'][key]['f1'])} |")

fa = rl["false_alarms"]
rule_fa = ["| Legit messages | n | Advisory ≠ none | Advisory = high |", "|---|---:|---:|---:|",
           f"| Sentinel legit (test split) | {fa['sentinel_legit_test']['n']} | "
           f"{pct(fa['sentinel_legit_test']['pct_advisory_any'], 2)} | {pct(fa['sentinel_legit_test']['pct_advisory_high'], 2)} |",
           f"| UCI ham | {fa['uci_ham']['n']} | {pct(fa['uci_ham']['pct_advisory_any'], 2)} | "
           f"{pct(fa['uci_ham']['pct_advisory_high'], 2)} |"]
rule_rec = ["| Scam recall | Scams | ML only | ML **or** advisory = high |", "|---|---:|---:|---:|"]
for k, name in [("uci_sms", "UCI SMS"), ("hinglish_only", "Hinglish-only")]:
    v = rl["scam_recall"][k]
    rule_rec.append(f"| {name} | {v['n_scam']} | {f4(v['recall_ml_only'])} | {f4(v['recall_ml_or_high'])} |")

top = M["top_words"]
top_words_md = "; ".join(f"`{c}`: {', '.join(w)}" for c, w in top.items())


# ------------------------------------------------------------------ README
readme = f"""# 🛡️ Kavach AI

**Paste, screenshot or speak a suspicious SMS / WhatsApp message. Kavach tells you which kind of scam it looks like, how risky it is, which words gave it away, and what to do next.**

![Runs locally](https://img.shields.io/badge/Runs-100%25%20locally-success?logo=apple)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20qwen2.5%3A7b-black)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![Kavach AI home page](docs/screenshots/home.png)

> Data Science Lab mini project · Sem VII 2026-27 · Dept. of IT, SIES Graduate School of Technology, Nerul

**Runs fully on your laptop.** The classifier, OCR, speech-to-text and the assistant's language model
(Ollama `qwen2.5:7b`) all run locally; no API keys and no cloud services. Only the natural-sounding voice
(edge-tts) uses the internet, and without it replies are spoken by the built-in macOS voices.

## The problem

People in India receive a steady stream of fraudulent SMS and WhatsApp messages: fake bank KYC alerts,
"digital arrest" calls from people posing as police or the CBI, part-time job and investment offers that
ask for a fee, and parcel or customs scams. They are written to create panic and urgency, and many people
cannot easily tell them apart from genuine bank or delivery notifications.

Kavach gives a second opinion in seconds, explains *why*, and points to the official channels:
**call the national cyber-crime helpline 1930** or report at **[cybercrime.gov.in](https://cybercrime.gov.in)**.

## Features

- **Three ways to scan**: paste text, upload a screenshot (RapidOCR), or record / upload a voice note (faster-whisper).
- **Explainable verdict**: one of 5 classes, a scam probability (`1 − P(legit)`), a risk tier
  (High ≥ 0.80, Medium 0.50–0.80, Low < 0.50) and the words that drove the prediction, highlighted with LIME.
- **Red-flag rule layer**: 9 hand-written scam tactics in English and Hindi/Hinglish (threats, urgency,
  payment or OTP requests, "new number" impersonation, authority impersonation…). It runs in parallel to the
  model, gives its own advisory, and never changes the model's prediction.
- **"What to do now"**: vetted, per-class advice. The only contacts ever shown are 1930, cybercrime.gov.in and
  "your bank on the number printed on your card".
- **Kavach Assistant**: chat by text or voice in English, Hindi or Hinglish, with spoken replies (edge-tts) and an
  expander that shows every tool call and its raw output.
- **Model Performance page**: leaderboard, confusion matrix, out-of-domain results, top words, rule-layer and
  ablation tables, all read from `artifacts/metrics.json`.

## How it works

> **The trained model classifies; the LLM only explains.**

```mermaid
flowchart LR
    T["Text"] --> P
    S["Screenshot"] --> OCR["RapidOCR"] --> P
    V["Voice note"] --> STT["faster-whisper"] --> P
    P["preprocess.py<br/>lowercase · mask URL / amount<br/>drop phone numbers"] --> M["TF-IDF word 1-2 + char 3-5<br/>Linear SVM (calibrated)"]
    M --> VER["Class · scam probability · risk"]
    M --> LIME["LIME word highlights"]
    T --> R["Rule layer<br/>9 red flags → advisory"]
    OCR --> R
    STT --> R
    VER --> CARD["Verdict card · checklist · what to do now"]
    LIME --> CARD
    R --> CARD
```

```mermaid
flowchart LR
    Q["User question<br/>text or voice"] --> RT{{"Something to check?"}}
    RT -- "yes: message, link, incident" --> F["Code calls classify_message<br/>+ safety_guidance"]
    RT -- "no: general question" --> G["LLM may call<br/>safety_guidance / query_dataset"]
    F --> LLM["LLM explains<br/>local Ollama · qwen2.5:7b"]
    G --> LLM
    LLM --> GD{{"Guards: contradicts model?<br/>unverified number or link?<br/>invented %? wrong script?"}}
    GD -- pass --> OUT["Reply + speech<br/>edge-tts, or macOS voice offline"]
    GD -- "fail, or LLM down" --> TPL["Template answer built<br/>from tool outputs"]
```

For anything that looks like a message to check, **the code itself** runs the trained classifier before the
LLM is called, so the verdict never depends on the LLM choosing a tool. General questions ("What is a
digital arrest scam?") are never classified and get no probability. If Ollama is not running, times out
or the reply fails a guard, the user gets a template answer built from the model output and vetted guidance.

## Results

Primary data: Sentinel multiclass, {ds['clean_rows']:,} messages ({ds['duplicates_removed']} exact duplicates),
5 classes. Two test splits:

- **Random**: stratified 80/20, `random_state=42` ({ds['train_rows']:,} train / {ds['test_rows']:,} test).
- **Template-grouped**: `StratifiedGroupKFold` on a normalised template key (digits, URLs, phones and amounts
  masked), so no template appears in both train and test ({dg['train_rows']:,} / {dg['test_rows']:,};
  {dg['n_templates']:,} templates). **The best model is chosen on this split.**

All models use TF-IDF (word 1-2 grams + char 3-5 grams) and balanced class weights where supported.

{chr(10).join(leader)}

**Best model: {best}.** {M['best_model_reason']}

<details><summary>Per-class F1 ({best})</summary>

{chr(10).join(per_class)}

</details>

**Out-of-domain test (the real-world check).** Binary scam vs legit on data the model never saw
(scam = any non-legit predicted class; positive class = scam). Exact duplicates were dropped first
(UCI {M['ood_duplicates_removed']['uci_sms']}, Hinglish {M['ood_duplicates_removed']['hinglish']}).

{chr(10).join(ood_tbl)}

> ⚠️ **Hinglish-only has only n = {ood['hinglish_only']['n']} messages ({n_scam_hing_only} scams)**, so its scores have a
> wide margin of error. {pct_hing_in_uci:.1f}% of the full "Hinglish & English" set is the UCI SMS collection, so
> those two rows largely measure the same data.

| Model comparison | Confusion matrix ({best}, grouped) | Out-of-domain |
|---|---|---|
| ![Model comparison](reports/figures/model_comparison.png) | ![Confusion matrix](reports/figures/confusion_matrix.png) | ![OOD results](reports/figures/ood_results.png) |

## Honest evaluation

- **Templated training data.** Sentinel has no exact duplicates, but only {dg['n_templates']:,} distinct templates
  across {ds['clean_rows']:,} rows ({pct_repeat:.1f}% of rows repeat a template once digits, links, phones and
  amounts are masked). A random split puts near-identical messages in both train and test, which is why every
  model scores higher on it: {drops} (macro-F1, random → grouped).
- **Even the grouped split is optimistic.** Templates that differ only in a bank or agency name still count as
  different templates, so ~{M['models_grouped'][best]['macro_f1']:.3f} grouped macro-F1 is an upper bound,
  not a real-world estimate. The out-of-domain table is the realistic picture.
- **Bank- and agency-name shortcuts.** The most indicative words per class include specific names, so the
  model partly learns *who* a message names rather than the tactic. Top words: {top_words_md}.
- **No casual chat in the legit class.** Sentinel's legit messages are service notifications (banks,
  recharges, deliveries). Ordinary personal messages are often flagged as scams, hence the low out-of-domain
  precision (UCI: {f4(ood['uci_sms']['precision'])}).
- **Label noise.** Similar messages (e.g. "PNB Fraud Alert" templates, NCB impersonation, parcel/customs
  scams) appear under different classes, so the exact scam *type* is less reliable than scam vs legit.
- **Phone-number shortcut, removed.** Phone numbers occur in {pct(eda['pct_phone_scam'])} of scam rows and
  {pct(eda['pct_phone_legit'])} of legit rows, so a `<PHONE>` token teaches "phone number → legit". The
  final model never sees phone numbers (they still appear in the red-flag checklist):

{chr(10).join(abl)}

- **Rule layer.** Written from general scam-awareness guidance without looking at the out-of-domain rows.
  It raises few false alarms but adds little recall at the "high" threshold:

{chr(10).join(rule_fa)}

{chr(10).join(rule_rec)}

  The demo samples in `demo_samples/` were seen while writing the rules, so they are not an independent test of it.

Full analysis: [`notebooks/01_eda_and_modelling.ipynb`](notebooks/01_eda_and_modelling.ipynb).

## Screenshots

| Scan: text (rule layer flags a scam type the model missed) | Scan: screenshot (OCR) |
|---|---|
| ![Scan text](docs/screenshots/scan_text.png) | ![Scan screenshot](docs/screenshots/scan_screenshot.png) |
| **Model performance** | **Assistant: Hinglish voice query** |
| ![Model performance](docs/screenshots/model_performance.png) | ![Assistant voice](docs/screenshots/assistant_voice.png) |

## Tech stack

| Area | Tools |
|---|---|
| ML | scikit-learn (TF-IDF, Linear SVM, Logistic Regression, Naive Bayes, Random Forest), LIME |
| Data & plots | pandas, NumPy, matplotlib, seaborn, Plotly, Jupyter |
| App | Streamlit (multipage) |
| Speech & OCR | faster-whisper (`small`, CPU int8), edge-tts (`en-IN-NeerjaNeural`, `hi-IN-SwaraNeural`) with macOS `say` (Rishi, Lekha) offline, RapidOCR (ONNX Runtime) |
| LLM | Ollama `qwen2.5:7b` running locally (OpenAI-compatible API), with tool calling |
| Testing | pytest, Streamlit AppTest, Playwright (screenshots) |

## Run it

1. Install [Ollama](https://ollama.com/download) and start the Ollama app.
2. Then:

```bash
git clone {REPO}.git && cd kavach-ai
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b               # the Assistant's local language model (4.7 GB, one time)
streamlit run app/Home.py            # opens http://localhost:8501
```

No `.env` is needed; copy `.env.example` to `.env` only to change the model or voices. The first voice scan
downloads the Whisper `small` model once (about 460 MB).

The trained model is committed in `artifacts/`, so no training is needed. The Scan and Model Performance pages
work without any LLM; without one, the Assistant answers in offline template mode.

## Retrain from scratch

```bash
python scripts/download_data.py      # 3 datasets → data/raw/ (gitignored)
python -m src.train                  # 4 models × 2 splits, saves artifacts/model.joblib
python -m src.evaluate               # OOD, EDA, rule layer → artifacts/metrics.json + figures
python -m src.explain                # top words → metrics.json, top_words_per_class.png
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_and_modelling.ipynb
python scripts/build_readme.py       # regenerate this README from metrics.json
```

## Tests

```bash
pip install -r requirements-dev.txt && playwright install chromium
python -m pytest tests -q            # live-LLM tests are skipped if Ollama isn't running
python scripts/check_scan_pipeline.py   # all demo texts, screenshots (OCR) and audio (Whisper)
python scripts/check_assistant.py       # assistant scenarios incl. prompt injection and offline mode
python scripts/take_screenshots.py      # regenerate docs/screenshots (app must be running)
```

## Project structure

```
app/            Streamlit app: Home.py, pages/ (Scan, Model Performance, Assistant),
                llm.py, tools.py, speech.py, ocr.py, scan_core.py
src/            preprocess, train, evaluate, explain, predict, red_flags, guidance
artifacts/      model.joblib, metrics.json, confusion matrices
notebooks/      01_eda_and_modelling.ipynb
reports/figures/  report and PPT figures
demo_samples/   fictional test messages, screenshots and voice clips
scripts/        data download, README builder, pipeline checks, screenshots
tests/          pytest + Streamlit AppTest
```

## Datasets & licenses

| Dataset | Used for | License |
|---|---|---|
| [Sentinel multiclass](https://huggingface.co/datasets/Shade63/scam-classification-multiclass) (Shade63) | Training and in-domain test | License not specified on the dataset card (it says "Same as original dataset" without naming one) |
| [Hinglish & English scam text](https://huggingface.co/bolewara/hinglish-scam-text-dataset) (Anuj Bolewar) | Out-of-domain test | CC BY 4.0 |
| [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (T. A. Almeida, J. M. Gómez Hidalgo; UCI, DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84)) | Out-of-domain test | CC BY 4.0 |

The datasets are not redistributed in this repository; `scripts/download_data.py` fetches them.
All messages in `demo_samples/` are fictional.

## Limitations & future work

- **DistilBERT**: fine-tune `distilbert-base-multilingual-cased` and compare it with the Linear SVM.
- **More scam types**: utility-disconnection, "new number, send money", loan-app and sextortion scripts are thin or
  missing in the training data; today only the rule layer catches some of them.
- **Relabelling**: fix the inconsistent class boundaries (PNB fraud alerts, NCB impersonation, parcel scams).
- **A real Hinglish dataset**: the Hinglish-only test has {ood['hinglish_only']['n']} rows; real code-mixed messages
  (and casual legit chat) are needed for both training and evaluation.
- **User-feedback loop**: let users mark wrong verdicts, review them, and use them to retrain.
- A larger local LLM (when hardware allows) for more natural Hindi/Hinglish explanations; the guards stay either way.

## Team

**Group 17**: Nadar Yuvrajvarthan Murugesh · Akash Sivan Konar
Guide: **Ms. Savita Lohiya** · HOD: Prof. Leena Ladge
Data Science Lab mini project 2026-27, Department of Information Technology, SIES Graduate School of Technology, Nerul.

## Disclaimer

Kavach is an educational project. Its verdicts can be wrong; "no scam pattern detected" does not mean a message
is safe. If you think you have been defrauded, call **1930** or report at **[cybercrime.gov.in](https://cybercrime.gov.in)**.

Licensed under the [MIT License](LICENSE).
"""

(ROOT / "README.md").write_text(readme)
print(f"wrote README.md ({len(readme.splitlines())} lines)")
