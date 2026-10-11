# -*- coding: utf-8 -*-
"""NICU antifungal outcome calculator.

Runs the same three models that are reported in the manuscript: in-hospital
mortality, bronchopulmonary dysplasia and retinopathy of prematurity, each from
ten variables known at the moment systemic antifungal therapy is started.

Model files are produced by export_streamlit_models.py and are loaded either
from a local folder (development) or from the private model repository
(deployment). No credential is stored in this file.
"""
import io
import json
import os

import joblib
import pandas as pd
import streamlit as st

LOCAL_DIR = "streamlit_models"
REPO_NAME = os.environ.get("MODEL_REPO", "al4asus/streamlit-app")

st.set_page_config(page_title="NICU antifungal outcome calculator",
                   layout="centered", page_icon="🩺")


# --------------------------------------------------------------------------
# Model yukleme: once yerel klasor, sonra ozel repo (token st.secrets'tan)
# --------------------------------------------------------------------------
def _token():
    try:
        tok = st.secrets.get("GITHUB_TOKEN", "")
    except Exception:
        tok = ""
    return tok or os.environ.get("GITHUB_TOKEN", "")


def _from_github(path, as_json=False):
    import requests
    url = f"https://raw.githubusercontent.com/{REPO_NAME}/main/{path}"
    tok = _token()
    headers = {"Authorization": f"token {tok}"} if tok else {}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json() if as_json else io.BytesIO(r.content)


@st.cache_data(show_spinner=False)
def load_meta():
    local = os.path.join(LOCAL_DIR, "model_meta.json")
    if os.path.exists(local):
        with open(local, encoding="utf-8") as f:
            return json.load(f), "local"
    return _from_github("model_meta.json", as_json=True), "repository"


@st.cache_resource(show_spinner=False)
def load_models(meta):
    out = {}
    for key, spec in meta["models"].items():
        local = os.path.join(LOCAL_DIR, spec["file"])
        out[key] = joblib.load(local if os.path.exists(local) else _from_github(spec["file"]))
    return out


try:
    META, SOURCE = load_meta()
    MODELS = load_models(META)
except Exception as exc:                                    # noqa: BLE001
    st.error(
        "Model files could not be loaded. In deployment the calculator reads them from the "
        "private model repository, which requires GITHUB_TOKEN to be set in the app secrets. "
        f"Details: {exc}"
    )
    st.stop()

FEATURES = META["features"]
SPEC = META["field_spec"]
COHORT = META["cohort"]


# --------------------------------------------------------------------------
# Basliklar
# --------------------------------------------------------------------------
st.title("NICU antifungal outcome calculator")
st.caption(
    f"Three models estimated from {COHORT['patients']} infants who received systemic antifungal "
    f"therapy. Every input is a value known at the moment therapy is started. "
    f"Research tool: not a medical device and not a substitute for clinical judgement."
)

with st.sidebar:
    st.header("About these models")
    st.write(
        "The calculator runs the same fitted models that are reported in the manuscript, so the "
        "numbers here and the numbers in the paper describe one and the same model."
    )
    for key in ("bpd", "rop", "mortality"):
        m = META["models"][key]
        st.markdown(
            f"**{m['label']}**  \n{m['algorithm']}  \n"
            f"AUC {m['auc']:.3f} (95% CI {m['ci_low']:.3f} to {m['ci_high']:.3f})"
        )
    st.caption(
        f"Cohort: {COHORT['patients']} infants, {COHORT['courses']} antifungal courses. "
        "Discrimination was measured on out-of-fold cross-validation predictions, so no infant "
        "contributed to the model that was used to predict them."
    )
    st.caption(f"Models loaded from: {SOURCE}. Exported {META['generated']}.")


# --------------------------------------------------------------------------
# Form: alanlar meta'daki field_spec'ten uretilir
# --------------------------------------------------------------------------
def render_field(name, col):
    spec = SPEC[name]
    if spec["kind"] == "choice":
        options = spec["options"]
        labels = list(options.keys())
        default_label = next(k for k, v in options.items() if v == spec["default"])
        chosen = col.radio(spec["label"], labels, index=labels.index(default_label),
                           horizontal=True, key=name)
        return options[chosen]
    return col.number_input(spec["label"], min_value=float(spec["min"]),
                            max_value=float(spec["max"]), value=float(spec["default"]),
                            step=float(spec["step"]), key=name)


with st.form("patient"):
    st.subheader("Values at initiation of antifungal therapy")
    left, right = st.columns(2)
    values = {}
    for i, name in enumerate(FEATURES):
        values[name] = render_field(name, left if i % 2 == 0 else right)
    submitted = st.form_submit_button("Calculate", use_container_width=True)


# --------------------------------------------------------------------------
# Tahmin
# --------------------------------------------------------------------------
def predict(key, row):
    return float(MODELS[key].predict_proba(row)[:, 1][0])


def clinical_panel(key, prob):
    m = META["models"][key]
    above = prob >= m["threshold"]
    c1, c2 = st.columns([1, 2])
    c1.metric("Predicted risk", f"{prob * 100:.1f}%")
    c1.progress(min(max(prob, 0.0), 1.0))
    if above:
        c2.error(
            f"**Above the threshold of {m['threshold']:.3f}.** At this threshold the model "
            f"identified {m['sensitivity'] * 100:.0f}% of infants who developed the outcome and "
            f"{m['specificity'] * 100:.0f}% of those who did not."
        )
    else:
        c2.success(
            f"**Below the threshold of {m['threshold']:.3f}.** At this threshold the model "
            f"identified {m['sensitivity'] * 100:.0f}% of infants who developed the outcome and "
            f"{m['specificity'] * 100:.0f}% of those who did not."
        )
    c2.caption(
        f"{m['algorithm']}; AUC {m['auc']:.3f} (95% CI {m['ci_low']:.3f} to {m['ci_high']:.3f}) "
        f"from {m['n']} infants with {m['events']} events."
    )


if submitted:
    row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)

    st.markdown("---")
    st.subheader("Bronchopulmonary dysplasia")
    clinical_panel("bpd", predict("bpd", row))

    st.markdown("---")
    st.subheader("Retinopathy of prematurity")
    clinical_panel("rop", predict("rop", row))
    st.info(
        "The ROP threshold was deliberately set so that no case in the development cohort was "
        "missed. It therefore flags a large proportion of infants and is intended to support "
        "screening priority, not to rule retinopathy in or out."
    )

    st.markdown("---")
    st.subheader("In-hospital mortality")
    mort = META["models"]["mortality"]
    st.warning(
        "**Research output only, not for clinical use.** The confidence interval for this model "
        f"reaches chance level (AUC {mort['auc']:.3f}, 95% CI {mort['ci_low']:.3f} to "
        f"{mort['ci_high']:.3f}), and repeating the analysis with twenty different random "
        "partitions gave a median AUC of 0.566. Mortality in this population is determined mainly "
        "by events that occur after therapy is started, which baseline variables cannot capture. "
        "The manuscript reports this as a negative finding."
    )
    with st.expander("Show the mortality estimate anyway"):
        st.metric("Model output", f"{predict('mortality', row) * 100:.1f}%")
        st.caption(
            "No risk classification is offered for this outcome, because the model cannot be "
            "shown to perform better than chance in this cohort."
        )

    st.markdown("---")
    st.caption(
        "These models were developed on a single-centre cohort and have not been validated "
        "externally. They estimate risk for groups of infants with similar characteristics and "
        "cannot determine what will happen to an individual patient. Clinical decisions remain "
        "the responsibility of the treating team."
    )
