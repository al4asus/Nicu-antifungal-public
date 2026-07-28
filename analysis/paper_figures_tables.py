# -*- coding: utf-8 -*-
"""
MAKALE — FİGÜRLER ve TABLOLAR PAKETİ
====================================
Üretir:
  Tablo 1  Demografik/baseline özellikler (genel + mortaliteye göre, p değerleri)
  Tablo 2  Model performansı
  Tablo 3  Variable importance (hangi model, hangi yöntem, sıralı değerler)
  Figür 1  ROC eğrileri (3 model, katman-dışı tahminlerle)
  Figür 2  Kalibrasyon eğrileri
  Figür 3  Confusion matrix'ler (nihai eşiklerde)
  Figür 4  Decision Curve Analysis (net fayda)
  Figür 5  Variable importance grafikleri

Tüm eğriler nihai 10-değişkenli modellerin KATMAN-DIŞI (OOF) tahminlerinden hesaplanır.
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
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import paper_primary_models as P

OUT = "Paper_Figures_Tables"
os.makedirs(OUT, exist_ok=True)

CHAMPS = [("Mortality", 'ex', "K-Nearest Neighbor", "Mortalite"),
          ("BPD", 'BPD', "ExtraTrees", "BPD"),
          ("ROP", 'rop', "Random Forest", "ROP")]
COLORS = {"Mortality": "#1f77b4", "BPD": "#d95f0e", "ROP": "#2ca02c"}


# ---------------- TABLO 1: Demografik ----------------
CONT = [('gebelikhaftası', 'Gestational age, weeks'), ('dogumagirligi', 'Birth weight, g'),
        ('apgar1', 'Apgar score at 1 min'), ('apgar5', 'Apgar score at 5 min'),
        ('PNA_t0', 'Age at antifungal initiation, days'), ('crp1', 'First CRP, mg/L'),
        ('proc1', 'First procalcitonin, ng/mL'), ('time', 'Follow-up (treatment start to exit), days')]
# (sütun, etiket, "pozitif" kabul edilen kod)
# NOT: cinsiyeti veride 1/2 olarak kodlu (0/1 DEĞİL). 1 vs 2'nin hangisi erkek olduğu
# veri sözlüğünden teyit edilmeli; şimdilik kod-2 raporlanıyor.
CAT = [('cinsiyeti', 'Sex — code 2 (mapping to be confirmed)', 2),
       ('cerrahi', 'Surgery', 1), ('idrarsondasi', 'Urinary catheter', 1),
       ('entübe', 'Intubation', 1), ('svk', 'Central venous catheter', 1),
       ('tpn', 'Total parenteral nutrition', 1),
       ('inotrop_any', 'Inotrope support (first week)', 1)]


def med_iqr(s):
    s = s.dropna()
    if len(s) == 0: return "—"
    return f"{s.median():.1f} [{s.quantile(.25):.1f}–{s.quantile(.75):.1f}]"

def n_pct(s, pos=1):
    s = s.dropna()
    if len(s) == 0: return "—"
    m = (s == pos)
    return f"{int(m.sum())} ({100*m.mean():.1f}%)"

def table1(df):
    df = df.copy()
    df['inotrop_any'] = (df['inotrop_severity'] > 0).astype(int)
    alive, dead = df[df['ex'] == 0], df[df['ex'] == 1]
    rows = []
    for col, label in CONT:
        try:
            p = stats.mannwhitneyu(alive[col].dropna(), dead[col].dropna(), alternative='two-sided')[1]
            ps = f"{p:.3f}" if p >= 0.001 else "<0.001"
        except Exception:
            ps = "—"
        rows.append({'Characteristic': label, 'Overall (n=56)': med_iqr(df[col]),
                     'Survived (n=30)': med_iqr(alive[col]), 'Died (n=26)': med_iqr(dead[col]), 'p': ps})
    for col, label, pos in CAT:
        if col not in df.columns: continue
        try:
            ct = pd.crosstab(df[col] == pos, df['ex'])
            if ct.shape == (2, 2):
                p = stats.fisher_exact(ct)[1]
            else:
                p = stats.chi2_contingency(ct)[1]
            ps = f"{p:.3f}" if p >= 0.001 else "<0.001"
        except Exception:
            ps = "—"
        rows.append({'Characteristic': label, 'Overall (n=56)': n_pct(df[col], pos),
                     'Survived (n=30)': n_pct(alive[col], pos), 'Died (n=26)': n_pct(dead[col], pos), 'p': ps})
    # Outcomes
    for col, label in [('ex', 'In-hospital mortality'), ('BPD', 'BPD'), ('rop', 'ROP')]:
        rows.append({'Characteristic': label + ' (outcome)', 'Overall (n=56)': n_pct(df[col]),
                     'Survived (n=30)': '—', 'Died (n=26)': '—', 'p': '—'})
    return pd.DataFrame(rows)


# ---------------- OOF + importance ----------------
def get_oof(df, tgt, champ):
    X, y, _ = P.prep(df, tgt, P.NOSEV)
    mdl, prm = P.build_models()[champ]
    yt, yp, _ = P.run_cv(X, y, mdl, prm)
    return X, y, np.array(yt), np.array(yp)


def fit_final(X, y, champ):
    mdl, prm = P.build_models()[champ]
    Xsm, ysm = SMOTE(random_state=42).fit_resample(X, y)
    sc = StandardScaler(); Xs = sc.fit_transform(Xsm)
    s = RandomizedSearchCV(mdl, prm, n_iter=25, cv=3, n_jobs=-1, random_state=P.R, scoring='accuracy')
    s.fit(Xs, ysm)
    return Pipeline([('scaler', sc), ('model', s.best_estimator_)]), s.best_estimator_


def importance(pipe, est, X, y):
    if hasattr(est, 'feature_importances_'):
        return est.feature_importances_, "Impurity-based (mean decrease in impurity)"
    if hasattr(est, 'coef_'):
        return np.abs(est.coef_).ravel(), "Absolute standardized coefficient"
    pr = permutation_importance(pipe, X, y, n_repeats=20, random_state=P.R, scoring='roc_auc')
    return pr.importances_mean, "Permutation importance (20 repeats, AUC loss)"


# ---------------- DCA ----------------
def net_benefit(yt, yp, ths):
    n = len(yt); prev = yt.mean()
    nbm, nba = [], []
    for pt in ths:
        pred = (yp >= pt).astype(int)
        tp = int(((pred == 1) & (yt == 1)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
        w = pt / (1 - pt)
        nbm.append(tp / n - (fp / n) * w)
        nba.append(prev - (1 - prev) * w)
    return np.array(nbm), np.array(nba)


def main():
    df = P.load_df()

    # ---- Tablo 1 ----
    t1 = table1(df)
    t1.to_csv(os.path.join(OUT, "Table1_Demographics.csv"), index=False, encoding='utf-8-sig')
    print("Tablo 1 yazildi.", flush=True)

    data = {}
    for en, tgt, champ, tr in CHAMPS:
        X, y, yt, yp = get_oof(df, tgt, champ)
        pipe, est = fit_final(X, y, champ)
        imp, method = importance(pipe, est, X, y)
        lo, hi = P.boot_ci(yt, yp)
        fpr, tpr, thr = roc_curve(yt, yp)
        k = int(np.argmax(tpr - fpr)); t = float(thr[k])
        if not np.isfinite(t): t = 1.0
        data[en] = dict(X=X, y=y, yt=yt, yp=yp, champ=champ, auc=roc_auc_score(yt, yp),
                        lo=lo, hi=hi, thr=t, imp=imp, method=method, feats=list(X.columns))
        print(f"{en}: {champ} AUC={data[en]['auc']:.3f} [{lo:.3f}-{hi:.3f}] thr={t:.3f}", flush=True)

    # ---- Tablo 3: Variable importance ----
    rows = []
    for en, tgt, champ, tr in CHAMPS:
        d = data[en]
        order = np.argsort(d['imp'])[::-1]
        tot = d['imp'].sum()
        for rank, i in enumerate(order, 1):
            rows.append({'Outcome': en, 'Model': champ, 'Importance method': d['method'],
                         'Rank': rank, 'Variable': d['feats'][i],
                         'Importance': round(float(d['imp'][i]), 4),
                         'Relative (%)': round(100 * float(d['imp'][i]) / tot, 1) if tot > 0 else 0.0})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "Table3_VariableImportance.csv"), index=False, encoding='utf-8-sig')
    print("Tablo 3 yazildi.", flush=True)

    # ---- Tablo 2: performans ----
    perf = [{'Outcome': en, 'n': len(data[en]['y']), 'events': int(data[en]['y'].sum()),
             'Model': data[en]['champ'], 'AUC': round(data[en]['auc'], 3),
             'CI_low': round(data[en]['lo'], 3), 'CI_high': round(data[en]['hi'], 3),
             'Threshold': round(data[en]['thr'], 3)} for en, _, _, _ in CHAMPS]
    pd.DataFrame(perf).to_csv(os.path.join(OUT, "Table2_Performance.csv"), index=False, encoding='utf-8-sig')

    # ---- Figür 1: ROC ----
    plt.figure(figsize=(6.5, 6))
    for en, _, _, _ in CHAMPS:
        d = data[en]
        fpr, tpr, _ = roc_curve(d['yt'], d['yp'])
        plt.plot(fpr, tpr, lw=2, color=COLORS[en],
                 label=f"{en} — {d['champ']}\nAUC = {d['auc']:.3f} (95% CI {d['lo']:.3f}–{d['hi']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=.6, label='Chance')
    plt.xlabel('1 − Specificity (False positive rate)'); plt.ylabel('Sensitivity (True positive rate)')
    plt.title('ROC curves — cross-validated (out-of-fold) predictions')
    plt.legend(loc='lower right', fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "Figure1_ROC.png"), dpi=200); plt.close()

    # ---- Figür 2: Kalibrasyon ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (en, _, _, _) in zip(axes, CHAMPS):
        d = data[en]
        try:
            fpos, mpred = calibration_curve(d['yt'], d['yp'], n_bins=5, strategy='quantile')
            ax.plot(mpred, fpos, 'o-', color=COLORS[en], label=d['champ'])
        except Exception:
            pass
        ax.plot([0, 1], [0, 1], 'k--', alpha=.6, label='Perfect calibration')
        ax.set_title(f"{en}"); ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Observed proportion'); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle('Calibration curves — out-of-fold predictions', y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "Figure2_Calibration.png"), dpi=200, bbox_inches='tight'); plt.close()

    # ---- Figür 3: Confusion matrix ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (en, _, _, _) in zip(axes, CHAMPS):
        d = data[en]
        pred = (d['yp'] >= d['thr']).astype(int)
        cm = confusion_matrix(d['yt'], pred, labels=[0, 1])
        ax.imshow(cm, cmap='Blues')
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha='center', va='center', fontsize=15,
                    color='white' if v > cm.max() / 2 else 'black')
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0; spec = tn / (tn + fp) if (tn + fp) else 0
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Predicted 0', 'Predicted 1'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['Actual 0', 'Actual 1'])
        ax.set_title(f"{en} (threshold = {d['thr']:.2f})\nSens {sens:.2f} · Spec {spec:.2f}", fontsize=10)
    fig.suptitle('Confusion matrices at the selected thresholds — out-of-fold predictions', y=1.03)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "Figure3_ConfusionMatrix.png"), dpi=200, bbox_inches='tight'); plt.close()

    # ---- Figür 4: DCA ----
    ths = np.linspace(0.01, 0.95, 190)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (en, _, _, _) in zip(axes, CHAMPS):
        d = data[en]
        nbm, nba = net_benefit(d['yt'], d['yp'], ths)
        prev = d['yt'].mean()
        ax.plot(ths, nbm, lw=2, color=COLORS[en], label=f"{en} model ({d['champ']})")
        ax.plot(ths, nba, 'k-', lw=1, alpha=.7, label='Treat all')
        ax.axhline(0, color='gray', ls='--', lw=1, label='Treat none')
        ax.axvline(d['thr'], color=COLORS[en], ls=':', lw=1.2, alpha=.8,
                   label=f"Selected threshold ({d['thr']:.2f})")
        ax.set_ylim(-0.10, prev + 0.08); ax.set_xlim(0, 0.95)
        ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
        ax.set_title(f"{en} (prevalence {prev:.2f})"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.suptitle('Decision curve analysis — out-of-fold predictions', y=1.03)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "Figure4_DecisionCurve.png"), dpi=200, bbox_inches='tight'); plt.close()

    # ---- Figür 5: Variable importance ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (en, _, _, _) in zip(axes, CHAMPS):
        d = data[en]
        o = np.argsort(d['imp'])
        ax.barh([d['feats'][i] for i in o], [d['imp'][i] for i in o], color=COLORS[en])
        ax.set_title(f"{en} — {d['champ']}\n{d['method']}", fontsize=9)
        ax.set_xlabel('Importance')
    fig.suptitle('Variable importance of the final models', y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "Figure5_VariableImportance.png"), dpi=200, bbox_inches='tight'); plt.close()

    print(f"\nTamamlandi -> {OUT}/", flush=True)
    print(t1.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
