# -*- coding: utf-8 -*-
"""Hesaplayiciya gidecek modelleri paketler.

Burada model egitilmez. Paper_Primary_Models/models altindaki fitted
pipeline'lar ve FINAL_PAPER_MODELS.csv'deki esikler oldugu gibi kopyalanir,
boylece hesaplayici makaledeki modellerin aynisini calistirir.

Esikler katman-disi tahminlerden gelir; modelin kendi egitim verisi uzerinden
hesaplanan esik, agac topluluklarinda anlamsiz degerler uretiyor.

Cikti: streamlit_models/ (3 pkl + model_meta.json)
"""
import json
import os
import shutil
from datetime import date

import pandas as pd

import paper_primary_models as P

SRC = os.path.join("Paper_Primary_Models", "models")
OUT = "streamlit_models"

# (ic ad, disa aktarilan ad, arayuzde gorunen ad)
MODELS = [
    ("Mortalite", "mortality_model.pkl", "In-hospital mortality"),
    ("BPD", "bpd_model.pkl", "Bronchopulmonary dysplasia"),
    ("ROP", "rop_model.pkl", "Retinopathy of prematurity"),
]

# Arayuzdeki alan tanimlari: etiket, tip, sinirlar, varsayilan
FIELD_SPEC = {
    'gebelikhaftası': dict(label="Gestational age (weeks)", kind="number", min=20.0, max=45.0, step=0.1, default=29.0),
    'cinsiyeti':      dict(label="Sex", kind="choice", options={"Female": 1, "Male": 2}, default=2),
    'apgar5':         dict(label="Apgar score at 5 minutes", kind="number", min=0.0, max=10.0, step=1.0, default=8.0),
    'apgar1':         dict(label="Apgar score at 1 minute", kind="number", min=0.0, max=10.0, step=1.0, default=4.0),
    'dogumagirligi':  dict(label="Birth weight (g)", kind="number", min=300.0, max=5000.0, step=10.0, default=1100.0),
    'cerrahi':        dict(label="Surgery", kind="choice", options={"No": 0, "Yes": 1}, default=0),
    'idrarsondasi':   dict(label="Urinary catheter", kind="choice", options={"No": 0, "Yes": 1}, default=0),
    'PNA_t0':         dict(label="Postnatal age at antifungal initiation (days)", kind="number", min=0.0, max=200.0, step=1.0, default=25.0),
    'crp1':           dict(label="First C reactive protein (mg/L)", kind="number", min=0.0, max=500.0, step=1.0, default=50.0),
    'proc1':          dict(label="First procalcitonin (ng/mL)", kind="number", min=0.0, max=200.0, step=0.1, default=1.5),
}


def main():
    fin = pd.read_csv(os.path.join("Paper_Primary_Models", "FINAL_PAPER_MODELS.csv"),
                      encoding='utf-8-sig').set_index("Outcome")
    df = P.load_df()

    os.makedirs(OUT, exist_ok=True)
    meta = {
        "generated": date.today().isoformat(),
        "source": "Paper_Primary_Models/models/*_FINAL10.pkl",
        "cohort": {
            "patients": int(len(df)),
            "courses": int(len(P.load_df(dedup=False))),
            "description": "Patient-level cohort; one row per infant (index antifungal course).",
        },
        "features": list(P.NOSEV),
        "field_spec": FIELD_SPEC,
        "models": {},
        "notes": {
            "thresholds": "Derived from pooled out-of-fold cross-validation predictions, not from "
                          "the training data.",
            "rop_threshold": "Chosen as the highest threshold that still retains 100% sensitivity, "
                             "following the clinical decision that no case of ROP should be missed. "
                             "In this cohort that point coincides with the Youden optimum.",
            "mortality": "The lower bound of the confidence interval reaches chance level. This "
                         "model is reported as a negative finding and is not suitable for clinical "
                         "use; it is shown for research purposes only.",
        },
    }

    for name, fname, label in MODELS:
        src = os.path.join(SRC, f"{name}_FINAL10.pkl")
        if not os.path.exists(src):
            raise SystemExit("bulunamadi: " + src)
        shutil.copy2(src, os.path.join(OUT, fname))
        r = fin.loc[name]
        key = {"Mortalite": "mortality", "BPD": "bpd", "ROP": "rop"}[name]
        meta["models"][key] = {
            "file": fname,
            "label": label,
            "algorithm": str(r["Champion"]),
            "threshold": float(r["threshold"]),
            "auc": float(r["OOF_AUC"]),
            "ci_low": float(r["CI_low"]),
            "ci_high": float(r["CI_high"]),
            "sensitivity": float(r["sensitivity"]),
            "specificity": float(r["specificity"]),
            "n": int(r["n"]),
            "events": int(r["pos"]),
            "clinical_use": (key != "mortality"),
        }
        print(f"  {fname:22s} {r['Champion']:<18s} AUC {r['OOF_AUC']:.3f} "
              f"[{r['CI_low']:.3f}-{r['CI_high']:.3f}]  esik {r['threshold']:.3f}")

    with open(os.path.join(OUT, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nyazildi:", OUT)
    print("kohort:", meta["cohort"]["patients"], "bebek /", meta["cohort"]["courses"], "kur")
    print("degiskenler:", len(meta["features"]))


if __name__ == "__main__":
    main()
