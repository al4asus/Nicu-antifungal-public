# NICU Antifungal — Outcome Prediction Models

Machine learning code for predicting **in-hospital mortality**, **bronchopulmonary dysplasia (BPD)** and **retinopathy of prematurity (ROP)** in neonates receiving systemic antifungal therapy.

This repository contains **source code only**. No patient data, no trained model files.

---

## Data availability

The clinical dataset supporting this study contains identifiable neonatal records and **cannot be shared publicly**. Access requires ethics committee approval and a reasonable request to the corresponding author.

Scripts expect a data file that is not distributed with this repository; they will not run without it.

---

## Method summary

All models use **only variables available at the initiation of antifungal therapy (t0)** to avoid data leakage:

`gestational age · sex · birth weight · Apgar-1 · Apgar-5 · surgery · urinary catheter · postnatal age at treatment start · first CRP · first procalcitonin`

Pipeline:

1. Ten algorithms compared (logistic regression, random forest, SVC, gradient boosting, k-NN, extremely randomized trees, AdaBoost, decision tree, MLP, XGBoost)
2. 10-fold stratified cross-validation
3. SMOTE oversampling and standardization applied **within training folds only**; test folds never resampled
4. Hyperparameters tuned by randomized search (25 configurations, 3-fold inner CV)
5. Performance from pooled **out-of-fold** predictions; 95% CI by 2000-resample bootstrap
6. Decision thresholds from Youden's J
7. Calibration curves, variable importance and decision curve analysis

---

## Repository layout

```
analysis/
  paper_primary_models.py        Main pipeline — 3 final models, all algorithms
  finalize_paper_models.py       Final model artefacts (threshold, importance, calibration)
  paper_figures_tables.py        Tables and figures (incl. decision curve analysis)
  all_models_analysis.py         Full comparison of all 10 algorithms
  per_algorithm_figures.py       Per-algorithm ROC / confusion matrix / calibration
  per_algorithm_secondary.py     Same, for exploratory escalation models
  honest_8feature_analysis.py    Leakage-free feature-set analysis
  mortality_feature_experiment.py  Feature-set sensitivity analysis for mortality
  last_development.py            Mortality model (earlier development version)
  bpd_primary_model.py           BPD model (earlier development version)
  bpd_secondary_model.py         BPD escalation model (exploratory)
  rop_primary_model.py           ROP model (earlier development version)
  rop_secondary_model.py         ROP escalation model (exploratory)

app/
  Home_Streamlit_V2.py           Streamlit risk calculator
  export_streamlit_models.py     Exports trained models for the calculator
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10, scikit-learn 1.3.2, imbalanced-learn 0.12.3, XGBoost.

---

## Configuration

The Streamlit app loads model files from a private repository and needs a GitHub token. **Never hard-code it.**

- Streamlit Cloud: *App settings → Secrets* → `GITHUB_TOKEN = "..."`
- Local: `export GITHUB_TOKEN=...`

---

## Intended use

Research use only. These models are internally validated on a small single-center cohort and have **not** been externally validated. They are not a medical device and must not be used for clinical decision-making.

---

## License

See [LICENSE](LICENSE). Copyright (c) 2025 Hasan Geredeli.
