import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.common import CLASS_COLORS, get_metrics, leaderboard, page_setup  # noqa: E402
from src.guidance import HELPLINE, PORTAL  # noqa: E402

page_setup("Home", "🏠")
M = get_metrics()
ds = M["dataset"]
best = M["best_model"]

st.title("🛡️ Kavach AI")
st.markdown("**Paste, screenshot or speak a suspicious SMS / WhatsApp message.** Kavach tells you "
            "which kind of scam it looks like, how risky it is, which words gave it away, "
            "and what to do next.")
st.info(f"Cyber fraud through SMS, WhatsApp and calls is widespread in India: fake KYC alerts, "
        f"'digital arrest' calls, job and investment scams. If you have been defrauded, call the "
        f"national cyber-crime helpline **{HELPLINE}** or report at [cybercrime.gov.in]({PORTAL}).",
        icon="🇮🇳")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Training messages", f"{ds['clean_rows']:,}")
c2.metric("Classes", len(ds["class_counts"]))
c3.metric("Best model", best)
c4.metric("Macro-F1 (template-grouped)", f"{M['models_grouped'][best]['macro_f1']:.3f}",
          help="Scored on test messages whose template never appears in training. The random-split "
               "score is higher because near-identical templated messages land in both train and test.")
c4.caption(f"Random split: {M['models'][best]['macro_f1']:.3f}")

st.page_link("pages/1_Scan_Message.py", label="Scan a message now", icon="🔍")

left, right = st.columns([1, 1.25], gap="large")
with left:
    st.subheader("Training data: class distribution")
    counts = ds["class_counts"]
    order = sorted(counts, key=counts.get)
    total = sum(counts.values())
    fig = go.Figure(go.Bar(
        x=[counts[c] for c in order], y=order, orientation="h",
        marker_color=[CLASS_COLORS[c] for c in order],
        text=[f"{counts[c]:,} ({100 * counts[c] / total:.1f}%)" for c in order],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:,} messages<extra></extra>"))
    fig.update_layout(height=320, margin=dict(l=10, r=70, t=10, b=30),
                      xaxis_title="Messages", xaxis_range=[0, max(counts.values()) * 1.3])
    st.plotly_chart(fig, width="stretch")
    st.caption("Sentinel multiclass dataset (synthetic, Indian context). romance_parcel_sextortion "
               f"is only {100 * counts['romance_parcel_sextortion'] / total:.1f}%, so models use balanced class weights "
               "and are compared on macro-F1.")
with right:
    st.subheader("Model leaderboard")
    st.dataframe(leaderboard(M, compact=True), hide_index=True, width="stretch",
                 column_config={"Macro-F1 (grouped)": st.column_config.NumberColumn(format="%.4f"),
                                "Macro-F1 (random)": st.column_config.NumberColumn(format="%.4f")})
    st.caption(M["best_model_reason"])
    st.page_link("pages/2_Model_Performance.py", label="Full model performance", icon="📊")

st.divider()
st.caption("**How it works:** a trained TF-IDF + Linear SVM model makes every prediction. "
           "Hand-written red-flag rules are shown separately and never change the model's output. "
           "LIME highlights the words behind each prediction.  \n"
           "Data Science Lab mini project · Dept. of IT, SIES Graduate School of Technology · Group 17")
