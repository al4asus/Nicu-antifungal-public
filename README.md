# NICU Antifungal — Outcome Prediction Models

Machine learning code for predicting **in-hospital mortality**, **bronchopulmonary dysplasia (BPD)** and **retinopathy of prematurity (ROP)** in neonates receiving systemic antifungal therapy, using only information available at the moment therapy is initiated.

This repository contains **source code only**. No patient data, no trained model files.

---

## Data availability

The clinical dataset contains identifiable neonatal records and **cannot be shared publicly**. Access requires ethics committee approval and a reasonable request to the corresponding author.

Scripts expect a data file that is not distributed with this repository; they will not run without it.

---

## Cohort definition

The source dataset is organised by **treatment episode**, not by patient: an infant who received more than one antifungal agent during the same admission contributes one record per course. Nine infants contributed 20 such records.

Because the outcomes belong to the infant rather than to the individual course, all modelling is performed at the **patient level**:

- one row per infant, the **index (first)** antifungal course, taken as a whole row
- values are never mixed across courses (first CRP and first procalcitonin are measured at the start of each course and differ between courses in the same infant)
- this prevents the same infant from appearing in both a training fold and a test fold

The final model cohort is **49 infants**; descriptive treatment tables use all **60 courses**.

Postnatal age at treatment initiation is computed from the **date of birth**, not from the admission date, because infants transferred from another centre may already be receiving antifungal therapy on arrival.

---

## Predictors

Ten baseline variables, all available at treatment initiation:

`gestational age · sex · birth weight · Apgar-1 · Apgar-5 · surgery · urinary catheter · postnatal age at treatment start · first CRP · first procalcitonin`

Number of antifungal courses is recorded for reporting but is **not given to the models**, since it is not known at the prediction point.

---

## Pipeline

1. Ten algorithms compared: logistic regression, random forest, SVC, gradient boosting, k-NN, extremely randomised trees, AdaBoost, decision tree, MLP, XGBoost
2. 10-fold stratified cross-validation
3. SMOTE oversampling, standardisation and hyperparameter search applied **within training folds only**; test folds never resampled
4. Performance from pooled **out-of-fold** predictions; 95% CI by 2000-resample bootstrap
5. Thresholds from Youden's J, except ROP, where the threshold preserving 100% sensitivity is retained (screening decision)
6. Calibration curves, Brier score, variable importance and decision curve analysis

### Sensitivity analyses

- **Early severity markers** (intubation, central venous catheter, TPN, first-week inotrope) added to the baseline set
- **First-line agent**, known at treatment initiation, added as a categorical variable
- **Treatment duration**, which is *not* known at treatment initiation and is partly determined by the outcome, added deliberately to quantify how much apparent performance is gained by violating the temporal boundary of prediction
- **Random-seed stability**: the entire pipeline repeated with **20 different random seeds**, each reassigning the cross-validation folds, the synthetic oversampling and the hyperparameter search, and reseeding the algorithms themselves. This quantifies how much of the reported discrimination, and of the algorithm ranking, depends on a single data partition

---

## Repository layout

```
analysis/
  paper_primary_models.py        Cohort definition, shared pipeline, FULL vs NOSEV comparison
  finalize_paper_models.py       Final ten-variable models: threshold, importance, calibration
  paper_figures_tables.py        Tables 1-3 and Figures 1-5 (incl. decision curve analysis)
  all_models_analysis.py         Full comparison of all ten algorithms
  seed_stability_analysis.py     Repeats the entire pipeline with 20 random seeds
  seed_stability_report.py       Summary tables and figure for the seed-stability analysis
  per_algorithm_figures.py       Per-algorithm ROC / confusion matrix / calibration
  per_algorithm_secondary.py     Same, for exploratory escalation models
  drug_usage_tables.py           Antifungal agent, duration and exposure tables
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

Run the analysis scripts from the directory containing the data file. `paper_primary_models.py` is imported by the others and defines the cohort.

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

Research use only. These models are internally validated on a small single-centre cohort and have **not** been externally validated. They are not a medical device and must not be used for clinical decision-making. The mortality model in particular did not reach a level of discrimination that would support clinical use.

---

## License

See [LICENSE](LICENSE). Copyright (c) 2025 Hasan Geredeli.
