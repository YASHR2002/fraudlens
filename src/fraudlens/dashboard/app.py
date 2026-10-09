"""FraudLens analyst dashboard (Streamlit).

Standalone by design: it imports no FraudLens code and talks to the scoring API only over HTTP
(``API_URL``), so it can be deployed separately (Docker, Hugging Face Spaces).

Run locally:  uv run streamlit run src/fraudlens/dashboard/app.py
"""

from __future__ import annotations

import math
import os
from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
# Validated palette (slots 1-2): toward fraud = orange, away from fraud = blue.
BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e6e5e1"

st.set_page_config(page_title="FraudLens", page_icon=":mag:", layout="wide")


# ----------------------------------------------------------------------------- API client


@st.cache_resource
def client() -> httpx.Client:
    # Generous timeout: /predict_explained includes an LLM call (a few seconds), and a free
    # hosting tier may need time to wake up.
    return httpx.Client(base_url=API_URL, timeout=60.0)


def api_get(path: str, **params: Any) -> Any:
    r = client().get(path, params=params or None)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def model_info() -> dict:
    return api_get("/model-info")


@st.cache_data(ttl=300, show_spinner=False)
def demo_transactions(label: str) -> pd.DataFrame:
    return pd.DataFrame(api_get("/demo/transactions", label=label, limit=1000))


# ---------------------------------------------------------------------------------- charts


def gauge(score: float, threshold: float) -> go.Figure:
    flagged = score >= threshold
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"valueformat": ".3f", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 1], "tickformat": ".1f"},
            "bar": {"color": ORANGE if flagged else BLUE, "thickness": 0.35},
            "bgcolor": "white",
            "threshold": {"line": {"color": INK2, "width": 3}, "value": threshold},
            "steps": [{"range": [threshold, 1], "color": "#fbe7de"}],
        },
    ))  # fmt: skip
    fig.update_layout(height=230, margin={"l": 20, "r": 20, "t": 20, "b": 0})
    return fig


def waterfall(result: dict) -> go.Figure:
    """SHAP waterfall in log-odds: base value -> top factors -> all other factors -> score."""
    factors = result["shap_factors"]
    score = min(max(result["score"], 1e-12), 1 - 1e-12)
    final = math.log(score / (1 - score))
    base = result["base_value_log_odds"]
    other = final - base - sum(f["shap_value"] for f in factors)
    labels = ["Base value"] + [f["label"] for f in factors] + ["All other factors", "Score"]
    values = [base] + [f["shap_value"] for f in factors] + [other, final]
    hover = ["Average model output (log-odds)"] + [f["fact"] for f in factors] + [
        "Sum of the remaining factors", f"Fraud score {result['score']:.4f}"]  # fmt: skip
    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["absolute"] + ["relative"] * (len(factors) + 1) + ["total"],
        y=labels, x=values, hovertext=hover, hoverinfo="text+x",
        increasing={"marker": {"color": ORANGE}}, decreasing={"marker": {"color": BLUE}},
        totals={"marker": {"color": INK2}}, connector={"line": {"color": GRID}},
    ))  # fmt: skip
    fig.update_layout(
        height=110 + 34 * len(labels), margin={"l": 10, "r": 10, "t": 10, "b": 30},
        yaxis={"autorange": "reversed"}, xaxis_title="Contribution to the score (log-odds)",
        plot_bgcolor="white", showlegend=False,
    )  # fmt: skip
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


# ------------------------------------------------------------------------------ the pages

REQUEST_FIELDS = ("trans_num", "cc_num", "trans_ts", "amt", "category", "dob", "city_pop",
                  "lat", "long", "merch_lat", "merch_long")  # fmt: skip


def option_label(row: pd.Series, names: dict[str, str]) -> str:
    """One line per demo transaction in the picker (card shown by its last 4 digits)."""
    when = str(row["trans_ts"])[:16].replace("T", " ")
    category = names.get(row["category"], row["category"])
    fraud = " | FRAUD" if row["is_fraud"] else ""
    return f"{when} | {category} | ${row['amt']:,.2f} | card ...{str(row['cc_num'])[-4:]}{fraud}"


def score_page(info: dict) -> None:
    st.subheader("Score a transaction")
    st.caption(
        "Demo transactions are each card's first transaction after the state snapshot "
        "(21 June 2020), so the API computes exactly the features the model was trained on. "
        "Scoring here is what-if: it does not change the API's card history."
    )
    left, right = st.columns([1, 2])
    with left:
        label = st.radio("Show", ["fraud", "legit", "all"], horizontal=True,
                         format_func=lambda s: {"fraud": "Fraud", "legit": "Legitimate",
                                                "all": "All"}[s])  # fmt: skip
        demo = demo_transactions(label)
        if demo.empty:
            st.warning("The API has no demo transactions loaded.")
            return
        names = info["categories"]
        choice = st.selectbox(
            "Transaction",
            demo.index.tolist(),
            format_func=lambda i: option_label(demo.loc[i], names),
        )
        txn = demo.loc[choice].to_dict()
        with st.expander("Edit the transaction (what-if)"):
            txn["amt"] = st.number_input("Amount (USD)", min_value=0.01, value=float(txn["amt"]),
                                         step=10.0)  # fmt: skip
            cats = sorted(names)
            txn["category"] = st.selectbox("Merchant category", cats,
                                           index=cats.index(txn["category"]),
                                           format_func=lambda c: names.get(c, c))  # fmt: skip
            hour = st.slider("Hour of day", 0, 23, int(txn["trans_ts"][11:13]))
            txn["trans_ts"] = f"{txn['trans_ts'][:11]}{hour:02d}{txn['trans_ts'][13:]}"
        actual = "Fraud" if txn["is_fraud"] else "Legitimate"
        st.markdown(f"Actual label in the data: **{actual}** (shown for the demo only)")
        go_btn = st.button("Score and explain", type="primary", width="stretch")

    with right:
        if not go_btn:
            st.info("Pick a transaction and press **Score and explain**.")
            return
        body = {k: txn[k] for k in REQUEST_FIELDS}
        body["cc_num"] = int(body["cc_num"])
        body["update_state"] = False
        with st.spinner("Scoring and asking the LLM for an analyst note..."):
            r = client().post("/predict_explained", json=body)
        if r.status_code != 200:
            st.error(f"API error {r.status_code}: {r.text[:400]}")
            return
        res = r.json()
        flagged = res["flagged"]
        g, d = st.columns([1, 1])
        with g:
            st.plotly_chart(gauge(res["score"], res["threshold"]), width="stretch")
        with d:
            st.metric("Decision", "FLAGGED for review" if flagged else "Approved")
            st.metric("Threshold", f"{res['threshold']:.4f}")
            st.caption(f"Model {res['model_name']} v{res['model_version']} | "
                       f"{res['latency_ms']:.0f} ms including explanation")  # fmt: skip
        note = res["explanation"]
        source = ("LLM: " + note["llm_model"]) if note["explanation_source"] == "llm" else (
            "template fallback (LLM unavailable)")  # fmt: skip
        st.markdown(f"#### Analyst note  \n*{source}*")
        st.write(note["summary"])
        st.markdown("\n".join(f"- {reason}" for reason in note["key_reasons"]))
        st.markdown(f"**Recommended action:** {note['recommended_action'].upper()}")
        st.markdown("#### Why this score (SHAP)")
        st.caption("Orange pushes toward fraud, blue away from it. Hover a bar for the facts.")
        st.plotly_chart(waterfall(res), width="stretch")
        with st.expander("Feature values sent to the model"):
            # Values are shown as text: the column mixes the category with numbers.
            values = {k: "-" if v is None else str(v) for k, v in res["features"].items()}
            st.dataframe(pd.Series(values, name="value").to_frame(), width="stretch")


def overview_page(info: dict) -> None:
    st.subheader("Model overview")
    m = info["metrics"]
    st.caption(f"{info['model_name']} v{info['model_version']} ({info['model_family']}), "
               f"alias @{info['alias']}, loaded from {info['source']}, trained "
               f"{(info['trained_utc'] or '')[:10]}. Threshold {info['threshold']:.4f} "
               f"(from {info['threshold_source']}).")  # fmt: skip
    st.markdown("**Test set** (fraudTest, used once for the final evaluation)")
    c = st.columns(4)
    c[0].metric("PR-AUC", f"{m.get('test_pr_auc', float('nan')):.3f}")
    c[1].metric("Recall (fraud caught)", f"{m.get('test_recall', float('nan')):.1%}")
    c[2].metric("Precision (alerts that are fraud)", f"{m.get('test_precision', float('nan')):.1%}")
    c[3].metric("Total cost", f"${m.get('test_cost', float('nan')):,.0f}")
    st.markdown("**Validation set**")
    c = st.columns(4)
    c[0].metric("PR-AUC", f"{m.get('val_pr_auc', float('nan')):.3f}")
    c[1].metric("Recall", f"{m.get('val_recall', float('nan')):.1%}")
    c[2].metric("Precision", f"{m.get('val_precision', float('nan')):.1%}")
    c[3].metric("Total cost", f"${m.get('val_cost', float('nan')):,.0f}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### What drives the model (mean |SHAP|)")
        imp = info.get("global_importance")
        if imp:
            s = pd.Series(imp).sort_values()
            fig = go.Figure(go.Bar(x=s.values, y=[k.replace("_", " ") for k in s.index],
                                   orientation="h", marker_color=BLUE))  # fmt: skip
            fig.update_layout(height=520, margin={"l": 10, "r": 10, "t": 10, "b": 30},
                              xaxis_title="Mean absolute SHAP value (log-odds)",
                              plot_bgcolor="white")  # fmt: skip
            fig.update_xaxes(gridcolor=GRID)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No global SHAP importance logged for this model.")
    with right:
        st.markdown("#### Fairness (test set)")
        fair = info.get("fairness")
        if fair:
            rows = [{"group": g.split(".", 1)[1].replace("_", " ").replace("plus", "+"),
                     "attribute": g.split(".", 1)[0].replace("_", " "),
                     "recall": v.get("recall"), "precision": v.get("precision"),
                     "false positive rate": v.get("fpr")} for g, v in fair.items()]  # fmt: skip
            df = pd.DataFrame(rows).set_index(["attribute", "group"])
            st.dataframe(df.style.format({"recall": "{:.1%}", "precision": "{:.1%}",
                                          "false positive rate": "{:.3%}"}),
                         width="stretch")  # fmt: skip
            st.caption("Gender is not a model input; age is. See docs/model_card.md for the "
                       "discussion of the gaps.")  # fmt: skip
        else:
            st.info("No fairness audit found for this model version.")
        st.markdown("#### Live state")
        stt = info["state"]
        st.write(f"{stt['cards']:,} cards in memory; snapshot cut-off "
                 f"{stt['snapshot'].get('cutoff', '?')}; "
                 f"{stt['transactions_recorded_since_start']:,} transactions recorded since "
                 "start.")  # fmt: skip
        st.write(f"LLM notes: {'on' if info['llm']['enabled'] else 'off (template fallback)'}"
                 f" ({info['llm']['model']})")  # fmt: skip


def main() -> None:
    st.title("FraudLens")
    st.caption("Explainable real-time credit card fraud detection")
    try:
        info = model_info()
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach the scoring API at {API_URL}: {exc}. Is it running? "
                 "(`docker compose --profile serve up -d`)")  # fmt: skip
        return
    tab1, tab2 = st.tabs(["Score a transaction", "Model overview"])
    with tab1:
        score_page(info)
    with tab2:
        overview_page(info)


main()
