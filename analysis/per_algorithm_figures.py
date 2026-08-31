# -*- coding: utf-8 -*-
"""
HER ALGORİTMA İÇİN AYRI GÖRSEL (eski Results klasör formatı)
============================================================
Her outcome (mortalite/BPD/ROP, nihai 10-değişkenli set) için, 10 algoritmanın
HER BİRİ ayrı ayrı:  Detailed_ROC_*, CM_* (confusion matrix), Calibration_*
Katman-dışı (OOF) tahminlerden. Eski klasörleri bozmamak için yeni parent klasör.
Dosya adı biçimi:  <Tip>_<Algoritma>_<ModelSuffix>.png
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve
import paper_primary_models as P

PARENT = "FINAL_PerAlgorithm_Figures"
# (İngilizce ad, hedef sütun, klasör, dosya suffix)
OUTCOMES = [
    ("Mortality", 'ex',  "Ex_Results",          "Ex"),
    ("BPD",       'BPD', "BPD_Primary_Results", "BPD_Primary"),
    ("ROP",       'rop', "ROP_Primary_Results", "ROP_Primary"),
]
CHAMP = {"Mortality": P.champion_of("Mortalite"),
         "BPD": P.champion_of("BPD"),
         "ROP": P.champion_of("ROP")}


def safe(algo):
    return algo.replace(" ", "_")


def plots_for(folder, suffix, algo, yt, yp, is_champ):
    yt = np.array(yt); yp = np.array(yp)
    a = safe(algo)
    star = "  (selected model)" if is_champ else ""

    # --- ROC ---
    fpr, tpr, thr = roc_curve(yt, yp)
    auc = roc_auc_score(yt, yp)
    plt.figure(figsize=(5.2, 5))
    plt.plot(fpr, tpr, lw=2.2, color='#1f77b4', label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=.5)
    plt.xlabel('1 - Specificity'); plt.ylabel('Sensitivity')
    plt.title(f"ROC - {algo} [{suffix}]{star}", fontsize=10)
    plt.legend(loc='lower right'); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(folder, f"Detailed_ROC_{a}_{suffix}.png"), dpi=150); plt.close()

    # --- Confusion matrix (Youden) ---
    k = int(np.argmax(tpr - fpr)); t = float(thr[k])
    if not np.isfinite(t):
        t = 1.0
    pred = (yp >= t).astype(int)
    cm = confusion_matrix(yt, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    plt.figure(figsize=(4.3, 4)); plt.imshow(cm, cmap='Blues')
    for (r, c), v in np.ndenumerate(cm):
        plt.text(c, r, str(v), ha='center', va='center', fontsize=15,
                 color='white' if v > cm.max() / 2 else 'black')
    plt.xticks([0, 1], ['Predicted 0', 'Predicted 1'])
    plt.yticks([0, 1], ['Actual 0', 'Actual 1'])
    plt.title(f"CM - {algo} [{suffix}]\nthr={t:.2f}  Sens {sens:.2f}  Spec {spec:.2f}", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"CM_{a}_{suffix}.png"), dpi=150); plt.close()

    # --- Calibration ---
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], 'k--', alpha=.5, label='Perfect calibration')
    ok = False
    for strat in ('quantile', 'uniform'):
        try:
            fpos, mpred = calibration_curve(yt, yp, n_bins=5, strategy=strat)
            plt.plot(mpred, fpos, 'o-', color='#d95f0e', label=algo)
            ok = True
            break
        except Exception:
            continue
    plt.xlabel('Mean predicted probability'); plt.ylabel('Observed proportion')
    plt.title(f"Calibration - {algo} [{suffix}]", fontsize=10)
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(folder, f"Calibration_{a}_{suffix}.png"), dpi=150); plt.close()
    return ok


def main():
    df = P.load_df()
    os.makedirs(PARENT, exist_ok=True)
    total = 0
    for en, tgt, foldername, suffix in OUTCOMES:
        folder = os.path.join(PARENT, foldername)
        os.makedirs(folder, exist_ok=True)
        X, y, _ = P.prep(df, tgt, P.NOSEV)
        print(f"\n=== {en} -> {foldername}  (n={len(y)}, feat={X.shape[1]}) ===", flush=True)
        for algo, (mdl, prm) in P.build_models().items():
            yt, yp, _ = P.run_cv(X, y, mdl, prm)
            plots_for(folder, suffix, algo, yt, yp, algo == CHAMP[en])
            total += 3
            print(f"   {algo:26s} -> ROC + CM + Calibration", flush=True)
    print(f"\nToplam {total} gorsel uretildi -> {PARENT}/", flush=True)


if __name__ == "__main__":
    main()
