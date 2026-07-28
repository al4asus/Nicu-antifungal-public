# -*- coding: utf-8 -*-
"""
NİHAİ MAKALE MODELLERİ — 10 değişkenli (severity'siz) set
=========================================================
Erken ağırlık göstergeleri (entübe/svk/tpn/inotrop) nihai modellerden çıkarılmıştır:
duyarlılık analizinde üç sonucun da hepsinde ayrım gücünü düşürmüşlerdir.

Nihai set (10): gebelikhaftası, cinsiyeti, apgar5, apgar1, dogumagirligi,
                cerrahi, idrarsondasi, PNA_t0, crp1, proc1

Bu betik, paper_primary_models.py'deki aynı pipeline'ı kullanarak NOSEV şampiyonları için
eşik / duyarlılık / özgüllük / FI / kalibrasyon / confusion matrix ve nihai modelleri üretir.
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import paper_primary_models as P

# NOSEV şampiyonları (paper_primary_models.py çıktısından)
CHAMPS = [
    ("Mortalite", 'ex',  "K-Nearest Neighbor"),
    ("BPD",       'BPD', "ExtraTrees"),
    ("ROP",       'rop', "Random Forest"),
]

def main():
    df = P.load_df()
    rows = []
    for name, tgt, champ in CHAMPS:
        X, y, dropped = P.prep(df, tgt, P.NOSEV)
        mdl, prm = P.build_models()[champ]
        yt, yp, bp = P.run_cv(X, y, mdl, prm)
        auc = roc_auc_score(yt, yp)
        lo, hi = P.boot_ci(yt, yp)
        tag = f"{name}_FINAL10"
        t, sens, spec, best = P.artifacts(name, X, y, champ, (yt, yp), tag)
        rows.append({
            'Outcome': name, 'n': len(y), 'pos': int(y.sum()), 'neg': int((y == 0).sum()),
            'nfeat': X.shape[1], 'Champion': champ,
            'OOF_AUC': round(auc, 3), 'CI_low': round(lo, 3), 'CI_high': round(hi, 3),
            'threshold': round(t, 3), 'sensitivity': round(sens, 3), 'specificity': round(spec, 3),
        })
        print(f"{name:10s} {champ:20s} AUC={auc:.3f} [{lo:.3f}-{hi:.3f}]  "
              f"esik={t:.3f} duyarlilik={sens:.3f} ozgulluk={spec:.3f}", flush=True)

    fin = pd.DataFrame(rows)
    fin.to_csv(os.path.join(P.OUT, "FINAL_PAPER_MODELS.csv"), index=False, encoding='utf-8-sig')
    print("\n" + "=" * 78)
    print("NIHAI MAKALE MODELLERI (10 degisken)")
    print("=" * 78)
    print(fin.to_string(index=False), flush=True)
    print(f"\nCiktilar: {P.OUT}/  (FI_*_FINAL10, Calibration_*_FINAL10, ConfusionMatrix_*_FINAL10, models/)")


if __name__ == "__main__":
    main()
