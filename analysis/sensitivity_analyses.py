# -*- coding: utf-8 -*-
"""
MORTALITE DUYARLILIK ANALIZLERI (makale Tablo 3)
================================================
Dort girdi setini ayni pipeline'dan gecirir ve mortalite icin AUC'leri
karsilastirir:

  A  Baseline               nihai 10 degisken
  B  + agirlik gostergeleri entube, svk, tpn, inotrop  (t0'da biliniyor)
  C  + ilk basamak ilac     t0'da biliniyor, one-hot
  D  + tedavi suresi        t0'da BILINMIYOR, kismen sonucun kendisinden
                            kaynaklaniyor (erken olum = kisa tedavi)

D bilerek eklenmistir: tahmin sinirini ihlal etmenin AUC'yi ne kadar
yukselttigini olcmek icin. Mesru degiskenler (B, C) kayda deger bir kazanc
saglamiyor, yasak olan sagliyor.

Cikti: Paper_Figures_Tables/Table3_Sensitivity.csv
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

import paper_primary_models as P

TARGET = 'ex'
SEVERITY = ['entübe', 'svk', 'tpn', 'inotrop_severity']
DRUG_COL = 'ilaç'
DURATION_COL = 'kacgüntedavi'

DRUG_NAMES = {1: 'mikafungin', 2: 'lipozomal_ampho', 3: 'ampho_B',
              4: 'flukonazol', 5: 'kaspofungin'}


def build_sets(df):
    """Dort girdi setini (ad, DataFrame, aciklama) olarak dondurur."""
    base = df[P.NOSEV].copy()

    sev = df[P.NOSEV + SEVERITY].copy()

    drug = base.copy()
    codes = df[DRUG_COL]
    for code, name in DRUG_NAMES.items():
        drug[f'ilac_{name}'] = (codes == code).astype(int)

    dur = base.copy()
    dur[DURATION_COL] = df[DURATION_COL]

    return [
        ("Baseline ten variables", base, "referans"),
        ("Plus early severity markers", sev, "t0'da biliniyor"),
        ("Plus first line agent", drug, "t0'da biliniyor"),
        ("Plus treatment duration", dur, "t0'da BILINMIYOR"),
    ]


def run_set(name, X, y):
    X = X.copy()
    for c in X.columns:
        X[c] = X[c].fillna(X[c].median())
    keep = [c for c in X.columns if X[c].nunique() > 1]
    X = X[keep].reset_index(drop=True)

    lb, store, _ = P.championship(name, "SENS", X, y)
    champ = lb.iloc[0]['Algorithm']
    yt, yp = store[champ]
    auc = roc_auc_score(yt, yp)
    lo, hi = P.boot_ci(yt, yp)
    return dict(champion=champ, nfeat=X.shape[1], auc=auc, ci_low=lo, ci_high=hi)


def main():
    df = P.load_df()
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    y = df[TARGET].astype(int)
    print(f"Mortalite: n={len(y)}  olay={int(y.sum())}\n")

    rows = []
    base_auc = None
    for name, X, note in build_sets(df):
        r = run_set(name, X, y)
        if base_auc is None:
            base_auc = r['auc']
            delta = "referans"
        else:
            delta = f"{r['auc'] - base_auc:+.3f}"
        rows.append({
            "Model input set": name,
            "Availability at t0": note,
            "Variables": r['nfeat'],
            "Champion": r['champion'],
            "AUC": round(r['auc'], 3),
            "CI_low": round(r['ci_low'], 3),
            "CI_high": round(r['ci_high'], 3),
            "Change in AUC": delta,
        })

    out = pd.DataFrame(rows)
    path = os.path.join("Paper_Figures_Tables", "Table3_Sensitivity.csv")
    out.to_csv(path, index=False, encoding='utf-8-sig')
    print("\n" + out.to_string(index=False))
    print("\nyazildi:", path)


if __name__ == "__main__":
    main()
