import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.common import (FIGURES, SPLIT_COLORS, get_confusion, get_metrics,  # noqa: E402
                        leaderboard, page_setup)
from src.preprocess import CLASSES  # noqa: E402

page_setup("Model Performance", "📊")
M = get_metrics()
CM = get_confusion()
best = M["best_model"]
F4 = st.column_config.NumberColumn(format="%.4f")

st.title("📊 Model Performance")
st.caption("All numbers are read from artifacts/metrics.json (written by src/train.py, "
           "src/evaluate.py and src/explain.py) and match the report.")

# ------------------------------------------------------------------ leaderboard
st.header("Leaderboard")
st.markdown(f"**Best model: {best}.** {M['best_model_reason']}")
st.dataframe(leaderboard(M), hide_index=True, width="stretch",
             column_config={c: F4 for c in ["Macro-F1 (grouped)", "Macro-F1 (random)",
                                            "Accuracy (grouped)", "Accuracy (random)",
                                            "ROC-AUC (grouped)"]})
with st.expander("Full metrics per split"):
    for key, title in [("models_grouped", "Template-grouped split"), ("models", "Random stratified 80/20 split")]:
        st.markdown(f"**{title}**")
        st.dataframe(pd.DataFrame(M[key]).T, width="stretch")

models = list(M["models"])
ymin = round(min(min(m["macro_f1"] for m in M[k].values()) for k in ("models", "models_grouped")) - 0.1, 2)
fig = go.Figure()
for key, name in [("models", "Random split"), ("models_grouped", "Template-grouped split")]:
    vals = [M[key][m]["macro_f1"] for m in models]
    fig.add_bar(x=models, y=vals, name=name, marker_color=SPLIT_COLORS[name],
                text=[f"{v:.3f}" for v in vals], textposition="outside",
                hovertemplate="%{x}<br>" + name + ": %{y:.4f}<extra></extra>")
fig.update_layout(barmode="group", height=380, yaxis=dict(range=[ymin, 1.02], title="Macro-F1"),
                  legend=dict(orientation="h", y=-0.15), margin=dict(t=30, b=10))
st.subheader("Macro-F1: random vs template-grouped split")
st.plotly_chart(fig, width="stretch")
st.caption(f"y-axis starts at {ymin:.2f}, not 0. The grouped split keeps every message template "
           "on one side only, so it is the more honest in-domain estimate.")

# ------------------------------------------------------------------ confusion matrix
st.header("Confusion matrix")
c1, c2 = st.columns(2)
split = c1.radio("Split", ["grouped", "random"], horizontal=True,
                 format_func=lambda s: "Template-grouped" if s == "grouped" else "Random")
model = c2.selectbox("Model", models, index=models.index(best))
cm = pd.DataFrame(CM[split][model], index=CLASSES, columns=CLASSES)
norm = cm.div(cm.sum(axis=1), axis=0)
heat = go.Figure(go.Heatmap(
    z=norm.values, x=CLASSES, y=CLASSES, colorscale="Blues", zmin=0, zmax=1,
    text=[[f"{cm.iat[i, j]}<br>{norm.iat[i, j]:.0%}" for j in range(5)] for i in range(5)],
    texttemplate="%{text}", hovertemplate="true %{y}<br>predicted %{x}<br>%{text}<extra></extra>",
    colorbar=dict(title="Recall")))
heat.update_layout(height=470, xaxis_title="Predicted", yaxis_title="True",
                   yaxis_autorange="reversed", margin=dict(t=10))
st.plotly_chart(heat, width="stretch")
st.markdown("**Per-class metrics (best model)**")
pc = pd.DataFrame({"F1 grouped": pd.DataFrame(M["per_class_grouped"]).T["f1"],
                   "F1 random": pd.DataFrame(M["per_class"]).T["f1"],
                   "Support (random test)": pd.DataFrame(M["per_class"]).T["support"].astype(int)})
st.dataframe(pc, width="stretch")

# ------------------------------------------------------------------ OOD
st.header("Out-of-domain test (real-world check)")
st.markdown("Binary scam vs legit on data the model never trained on: a message counts as "
            "scam if the predicted class is any non-legit class (positive class = scam).")
ood = pd.DataFrame(M["ood"]).T[["n", "accuracy", "precision", "recall", "f1"]]
ood.index = ["Hinglish (full)", "UCI SMS", "Hinglish-only (not in UCI)"]
st.dataframe(ood, width="stretch", column_config={"n": st.column_config.NumberColumn(format="%d")})
pct_in_uci = 100 * (1 - M["ood"]["hinglish_only"]["n"] / M["ood"]["hinglish"]["n"])
st.warning(f"**Hinglish-only has only n = {M['ood']['hinglish_only']['n']} messages** "
           f"({M['rule_layer']['scam_recall']['hinglish_only']['n_scam']} scams), so its scores have "
           f"a wide margin of error. {pct_in_uci:.1f}% of the full 'Hinglish' set is UCI SMS, so those two rows "
           f"largely measure the same data. Exact duplicates dropped first: "
           f"UCI {M['ood_duplicates_removed']['uci_sms']}, Hinglish {M['ood_duplicates_removed']['hinglish']}.")
st.image(str(FIGURES / "ood_results.png"), width="stretch")

# ------------------------------------------------------------------ explainability
st.header("Most indicative words per class")
st.caption("Top word n-grams from the Linear SVM coefficients (n-grams made only of stop words "
           "removed for display; the model is unchanged).")
st.image(str(FIGURES / "top_words_per_class.png"), width="stretch")

# ------------------------------------------------------------------ rule layer
st.header("Rule layer (red flags)")
st.markdown("Nine hand-written scam tactics, in English and Hindi/Hinglish. **Separate from the "
            "model:** they never change its label or probabilities. Advisory = *high* for 3+ flags, "
            "*caution* for 2.")
rl = M["rule_layer"]
c1, c2 = st.columns(2)
fa = pd.DataFrame(rl["false_alarms"]).T
fa.index = ["Sentinel legit (test)", "UCI ham"]
fa.columns = ["n", "% advisory ≠ none", "% advisory = high"]
c1.markdown("**False alarms on legit messages**")
c1.dataframe(fa, width="stretch")
sr = pd.DataFrame(rl["scam_recall"]).T
sr.index = ["UCI SMS", "Hinglish-only"]
sr.columns = ["Scams", "Recall: ML only", "Recall: ML or advisory high",
              "Precision: ML only", "Precision: ML or advisory high"]
c2.markdown("**Scam recall with and without the rules**")
c2.dataframe(sr, width="stretch")
st.caption("Rules were written without looking at UCI or Hinglish-only rows. The demo samples "
           "are not an independent test of the rules.")

# ------------------------------------------------------------------ phone ablation
st.header("Fix: phone-number shortcut")
ab = M["phone_token_ablation"]
st.markdown(f"In the training data, phone numbers appear in **{M['eda']['pct_phone_scam']:.0f}%** of "
            f"scam messages and **{M['eda']['pct_phone_legit']:.1f}%** of legit ones, so the model "
            "could learn 'phone number → legit'. Phone numbers are now removed from the text the "
            "model sees (still shown in the red-flag checklist).")
c1, c2 = st.columns(2)
c1.markdown("**Macro-F1**")
c1.dataframe(pd.DataFrame({
    "Random: with phone": {m: v["with_phone"] for m, v in ab["macro_f1_random"].items()},
    "Random: removed": {m: v["without_phone"] for m, v in ab["macro_f1_random"].items()},
    "Grouped: with phone": {m: v["with_phone"] for m, v in ab["macro_f1_grouped"].items()},
    "Grouped: removed": {m: v["without_phone"] for m, v in ab["macro_f1_grouped"].items()},
}), width="stretch")
c2.markdown("**Out-of-domain F1 / recall**")
c2.dataframe(pd.DataFrame({k: {"F1 with phone": v["with_phone"]["f1"], "F1 removed": v["without_phone"]["f1"],
                               "Recall with phone": v["with_phone"]["recall"],
                               "Recall removed": v["without_phone"]["recall"]}
                           for k, v in ab["ood"].items()}).T, width="stretch")

# ------------------------------------------------------------------ limitations
st.header("Limitations")
st.info(
    "- **Templated, synthetic training data.** Many messages share a template; even the grouped "
    "split keeps templates that differ only in names or amounts, so in-domain scores are an upper bound.\n"
    "- **Bank-name shortcuts.** Top words include specific banks and agencies (e.g. *PNB*, *NCB*), "
    "so the model partly learns names rather than tactics.\n"
    "- **No casual chat in the legit class.** Legit examples are service notifications only, so "
    "ordinary personal messages are often flagged (low out-of-domain precision).\n"
    "- **Label noise.** Similar messages (PNB fraud alerts, NCB impersonation, parcel/customs scams) "
    "appear under different classes, so the scam *type* is less reliable than scam vs legit.\n"
    "- **Scam types not covered.** Utility-disconnection and 'new number, send money' scams are "
    "absent from the training data; the rule layer only partly compensates.",
    icon="⚠️")
