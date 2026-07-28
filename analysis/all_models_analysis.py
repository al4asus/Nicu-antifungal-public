# -*- coding: utf-8 -*-
"""
TÜM MODELLER — 10 algoritmanın tamamı için karşılaştırma (nihai 10-değişkenli set)
Her outcome × her algoritma: OOF-AUC + %95 GA, Youden eşiği, duyarlılık, özgüllük,
doğruluk, F1, Brier skoru. Ayrıca 10 algoritmanın ROC eğrileri (outcome başına).
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             accuracy_score, f1_score, brier_score_loss)
import paper_primary_models as P

OUT = "Paper_Figures_Tables"
os.makedirs(OUT, exist_ok=True)

OUTCOMES = [("Mortality", 'ex', "K-Nearest Neighbor"),
            ("BPD", 'BPD', "ExtraTrees"),
            ("ROP", 'rop', "Random Forest")]


def metrics_at_youden(yt, yp):
    fpr, tpr, thr = roc_curve(yt, yp)
    k = int(np.argmax(tpr - fpr))
    t = float(thr[k])
    if not np.isfinite(t):
        t = 1.0
    pred = (yp >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return t, sens, spec, accuracy_score(yt, pred), f1_score(yt, pred, zero_division=0)


def main():
    df = P.load_df()
    all_rows = []
    oof_by_outcome = {}

    for en, tgt, champ in OUTCOMES:
        X, y, _ = P.prep(df, tgt, P.NOSEV)
        oof = {}
        rows = []
        for algo, (mdl, prm) in P.build_models().items():
            yt, yp, _ = P.run_cv(X, y, mdl, prm)
            yt = np.array(yt); yp = np.array(yp)
            auc = roc_auc_score(yt, yp)
            lo, hi = P.boot_ci(yt, yp)
            t, sens, spec, acc, f1 = metrics_at_youden(yt, yp)
            brier = brier_score_loss(yt, yp)
            rows.append({'Outcome': en, 'Algorithm': algo, 'AUC': round(auc, 3),
                         'CI_low': round(lo, 3), 'CI_high': round(hi, 3),
                         'Threshold': round(t, 3), 'Sensitivity': round(sens, 3),
                         'Specificity': round(spec, 3), 'Accuracy': round(acc, 3),
                         'F1': round(f1, 3), 'Brier': round(brier, 3),
                         'Champion': 'YES' if algo == champ else ''})
            oof[algo] = (yt, yp, auc)
            print(f"{en:10s} {algo:26s} AUC={auc:.3f} [{lo:.3f}-{hi:.3f}] sens={sens:.2f} spec={spec:.2f}", flush=True)
        tb = pd.DataFrame(rows).sort_values('AUC', ascending=False).reset_index(drop=True)
        tb.to_csv(os.path.join(OUT, f"AllModels_{en}.csv"), index=False, encoding='utf-8-sig')
        all_rows.append(tb)
        oof_by_outcome[en] = (oof, champ)

    pd.concat(all_rows).to_csv(os.path.join(OUT, "AllModels_combined.csv"), index=False, encoding='utf-8-sig')

    # ROC — 10 algoritma, outcome başına panel
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax, (en, tgt, champ) in zip(axes, OUTCOMES):
        oof, _ = oof_by_outcome[en]
        order = sorted(oof.keys(), key=lambda a: oof[a][2], reverse=True)
        colors = cm.tab10(np.linspace(0, 1, 10))
        for c, algo in zip(colors, order):
            yt, yp, auc = oof[algo]
            fpr, tpr, _ = roc_curve(yt, yp)
            lw = 2.6 if algo == champ else 1.3
            lbl = f"{algo} ({auc:.3f})" + ("  ★" if algo == champ else "")
            ax.plot(fpr, tpr, lw=lw, color=c, label=lbl, alpha=0.9)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=.5)
        ax.set_title(f"{en} — all 10 algorithms", fontsize=11)
        ax.set_xlabel('1 − Specificity'); ax.set_ylabel('Sensitivity')
        ax.legend(loc='lower right', fontsize=7); ax.grid(alpha=.3)
    fig.suptitle('ROC curves of all algorithms (out-of-fold predictions) — ★ = selected final model', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "FigureS1_ROC_AllModels.png"), dpi=190, bbox_inches='tight')
    plt.close()
    print("\nFigureS1_ROC_AllModels.png yazildi.", flush=True)
    print(f"Tamamlandi -> {OUT}/ (AllModels_*.csv, FigureS1_ROC_AllModels.png)", flush=True)


if __name__ == "__main__":
    main()
