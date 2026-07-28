# -*- coding: utf-8 -*-
"""
MAKALE — 3 PRIMARY OFFLINE MODEL (nihai analiz)
================================================
Tasarım kararları:
  • Ortak değişken seti (tüm outcome'larda aynı)
  • Secondary/eskalasyon modelleri dışlandı (56 hastanın yalnızca 2'si eskale olmuştur)
  • ex = hastane-içi mortalite
  • entübe/svk/tpn/inotrop bu betikte değerlendirilir (t0'da bilinen değişkenler)

Üretilen: her outcome için şampiyon algoritma, out-of-fold (OOF) CV-AUC + bootstrap %95 GA,
Youden optimal eşik, duyarlılık/özgüllük, Feature Importance, kalibrasyon eğrisi, confusion matrix.
Ayrıca "severity değişkenli (FULL)" vs "severitysiz (NOSEV)" karşılaştırması.
"""
import os, sys, json, warnings
try:
    sys.stdout.reconfigure(encoding='utf-8')   # Windows cp1254 konsolunda Türkçe/simge çökmesini önler
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import randint, loguniform, uniform
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
import joblib
try:
    from xgboost import XGBClassifier; HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings('ignore')
os.chdir(r"C:\Users\hasan\Documents\medical-research")
OUT = "Paper_Primary_Models"
os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
R = 230

# Ortak değişken seti. TPN kohortta sabit olduğu için otomatik düşer.
FULL = ['gebelikhaftası','cinsiyeti','apgar5','apgar1','dogumagirligi','cerrahi','idrarsondasi',
        'PNA_t0','crp1','proc1','entübe','svk','tpn','inotrop_severity']
NOSEV = ['gebelikhaftası','cinsiyeti','apgar5','apgar1','dogumagirligi','cerrahi','idrarsondasi',
         'PNA_t0','crp1','proc1']

TARGETS = [("Mortalite", 'ex'), ("BPD", 'BPD'), ("ROP", 'rop')]


def load_df():
    df = pd.read_excel('micafunginson.xlsx', na_values=['#BOŞ!', '#NULL!'])
    for c in ['kullandıgıtarih1', 'takibegirdigitarih', 'taburculuktarihi']:
        df[c] = pd.to_datetime(df[c], errors='coerce')
    df['PNA_t0'] = (df['kullandıgıtarih1'] - df['takibegirdigitarih']).dt.days
    df = df[(df['PNA_t0'] >= 0) & (df['PNA_t0'] <= 200)]
    df['time'] = (df['taburculuktarihi'] - df['kullandıgıtarih1']).dt.days
    df = df[df['time'] > 0]
    df['inotrop_severity'] = df['ilkhaftainotrop'].map({0: 0, 1: 1, 2: 1, 3: 2}).fillna(0)
    return df


def prep(df, target, feats):
    d = df.dropna(subset=[target]).copy()
    y = d[target].astype(int).reset_index(drop=True)
    X = d[feats].copy()
    for c in X.columns:
        X[c] = X[c].fillna(X[c].median())
    keep = [c for c in X.columns if X[c].nunique() > 1]
    dropped = [c for c in X.columns if c not in keep]
    return X[keep].reset_index(drop=True), y, dropped


def build_models():
    m = {
        "Logistic Regression": (LogisticRegression(random_state=R, max_iter=2000),
                                {'C': loguniform(0.01, 1000), 'solver': ['liblinear']}),
        "Random Forest": (RandomForestClassifier(random_state=R),
                          {'n_estimators': randint(100, 300), 'max_depth': randint(3, 7),
                           'min_samples_split': randint(5, 15), 'max_features': ['sqrt', 'log2', None]}),
        "Support Vector Classifier": (SVC(random_state=R, probability=True),
                                      {'C': loguniform(0.1, 2000), 'gamma': loguniform(1e-5, 1e-1), 'kernel': ['rbf']}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=R),
                              {'n_estimators': randint(50, 300), 'learning_rate': loguniform(0.005, 0.1),
                               'max_depth': randint(2, 4), 'subsample': uniform(0.6, 0.4)}),
        "K-Nearest Neighbor": (KNeighborsClassifier(),
                               {'n_neighbors': randint(3, 11), 'weights': ['uniform', 'distance']}),
        "ExtraTrees": (ExtraTreesClassifier(random_state=R),
                       {'n_estimators': randint(100, 300), 'max_depth': randint(3, 7),
                        'min_samples_split': randint(5, 15), 'max_features': ['sqrt', 'log2', None]}),
        "AdaBoost": (AdaBoostClassifier(random_state=R),
                     {'n_estimators': randint(50, 200), 'learning_rate': loguniform(0.01, 0.5)}),
        "Decision Tree": (DecisionTreeClassifier(random_state=R),
                          {'criterion': ['gini', 'entropy'], 'max_depth': randint(3, 6),
                           'min_samples_split': randint(5, 20), 'min_samples_leaf': randint(2, 10)}),
        "MLP Classifier": (MLPClassifier(random_state=R, max_iter=2000, early_stopping=True),
                           {'hidden_layer_sizes': [(4,), (8,), (12,)], 'activation': ['tanh', 'relu'],
                            'solver': ['lbfgs'], 'alpha': loguniform(0.1, 100)}),
    }
    if HAS_XGB:
        m["XGBoost"] = (XGBClassifier(random_state=R, eval_metric='logloss'),
                        {'n_estimators': randint(50, 300), 'learning_rate': loguniform(0.001, 0.05),
                         'max_depth': randint(2, 4), 'subsample': uniform(0.6, 0.4),
                         'colsample_bytree': uniform(0.6, 0.4)})
    return m


def run_cv(X, y, model, params):
    skf = StratifiedKFold(10, shuffle=True, random_state=R)
    yt, yp = [], []
    bp = None
    for i, (tri, tei) in enumerate(skf.split(X, y)):
        Xtr, Xte = X.iloc[tri], X.iloc[tei]
        ytr, yte = y.iloc[tri], y.iloc[tei]
        Xsm, ysm = SMOTE(random_state=42).fit_resample(Xtr, ytr)
        sc = StandardScaler()
        Xtr_s, Xte_s = sc.fit_transform(Xsm), sc.transform(Xte)
        try:
            s = RandomizedSearchCV(model, params, n_iter=25, cv=3, n_jobs=-1, random_state=R, scoring='accuracy')
            s.fit(Xtr_s, ysm)
            est = s.best_estimator_
            if i == 0:
                bp = {k: (v.item() if hasattr(v, 'item') else v) for k, v in s.best_params_.items()}
        except Exception as e:
            est = model; est.fit(Xtr_s, ysm)
            if i == 0: bp = f"Default ({str(e)[:40]})"
        try:
            prob = est.predict_proba(Xte_s)[:, 1]
        except Exception:
            prob = est.predict(Xte_s).astype(float)
        yt.extend(yte.tolist()); yp.extend(prob.tolist())
    return np.array(yt), np.array(yp), bp


def boot_ci(yt, yp, B=2000):
    rng = np.random.default_rng(42); n = len(yt); a = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(yt[i])) > 1:
            a.append(roc_auc_score(yt[i], yp[i]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def championship(name, setname, X, y):
    print(f"\n--- {name} [{setname}]  n={len(y)} poz={int(y.sum())} neg={int((y==0).sum())} feat={X.shape[1]}", flush=True)
    rows, store, bps = [], {}, {}
    for algo, (mdl, prm) in build_models().items():
        yt, yp, bp = run_cv(X, y, mdl, prm)
        auc = roc_auc_score(yt, yp)
        rows.append({'Algorithm': algo, 'OOF_AUC': round(auc, 4),
                     'Acc@0.5': round(accuracy_score(yt, (yp >= .5).astype(int)), 4)})
        store[algo] = (yt, yp); bps[algo] = bp
        print(f"    {algo:26s} AUC={auc:.3f}", flush=True)
    lb = pd.DataFrame(rows).sort_values('OOF_AUC', ascending=False).reset_index(drop=True)
    return lb, store, bps


def artifacts(name, X, y, champ, oof, tag):
    yt, yp = oof
    fpr, tpr, thr = roc_curve(yt, yp)
    k = int(np.argmax(tpr - fpr))
    t = float(thr[k])
    if not np.isfinite(t): t = 1.0
    pred = (yp >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    mdl, prm = build_models()[champ]
    Xsm, ysm = SMOTE(random_state=42).fit_resample(X, y)
    sc = StandardScaler(); Xs = sc.fit_transform(Xsm)
    s = RandomizedSearchCV(mdl, prm, n_iter=25, cv=3, n_jobs=-1, random_state=R, scoring='accuracy')
    s.fit(Xs, ysm)
    est = s.best_estimator_
    pipe = Pipeline([('scaler', sc), ('model', est)])
    joblib.dump(pipe, os.path.join(OUT, "models", f"{tag}.pkl"))

    feats = list(X.columns)
    if hasattr(est, 'feature_importances_'):
        imp, kind = est.feature_importances_, "feature_importances_"
    elif hasattr(est, 'coef_'):
        imp, kind = np.abs(est.coef_).ravel(), "|coef| (standardize)"
    else:
        pr = permutation_importance(pipe, X, y, n_repeats=20, random_state=R, scoring='roc_auc')
        imp, kind = pr.importances_mean, "permutation"
    fi = pd.DataFrame({'Feature': feats, 'Importance': imp}).sort_values('Importance', ascending=False)
    fi.to_csv(os.path.join(OUT, f"FI_{tag}.csv"), index=False, encoding='utf-8-sig')
    plt.figure(figsize=(8, max(3.5, .45 * len(feats))))
    plt.barh(fi['Feature'][::-1], fi['Importance'][::-1], color='#2c7fb8')
    plt.title(f"Feature Importance — {champ} [{name}]\n({kind})", fontsize=10)
    plt.xlabel("Importance"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"FI_{tag}.png"), dpi=130); plt.close()

    try:
        fpos, mpred = calibration_curve(yt, yp, n_bins=5, strategy='quantile')
        plt.figure(figsize=(5.5, 5.5))
        plt.plot([0, 1], [0, 1], 'k--', alpha=.5, label='Mükemmel kalibrasyon')
        plt.plot(mpred, fpos, 'o-', color='#d95f0e', label=champ)
        plt.xlabel("Ortalama tahmin edilen olasılık"); plt.ylabel("Gözlenen pozitif oranı")
        plt.title(f"Kalibrasyon (OOF) — {name}"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"Calibration_{tag}.png"), dpi=130); plt.close()
    except Exception as e:
        print(f"    (kalibrasyon atlandı: {e})", flush=True)

    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(4.2, 3.8)); plt.imshow(cm, cmap='Blues')
    for (r_, c_), v in np.ndenumerate(cm):
        plt.text(c_, r_, str(v), ha='center', va='center', fontsize=14,
                 color='white' if v > cm.max() / 2 else 'black')
    plt.xticks([0, 1], ['Tahmin 0', 'Tahmin 1']); plt.yticks([0, 1], ['Gerçek 0', 'Gerçek 1'])
    plt.title(f"{name} — eşik={t:.3f}", fontsize=10); plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"ConfusionMatrix_{tag}.png"), dpi=130); plt.close()

    return t, sens, spec, s.best_params_


def main():
    df = load_df()
    summary = []
    for name, tgt in TARGETS:
        for setname, feats in [("FULL", FULL), ("NOSEV", NOSEV)]:
            X, y, dropped = prep(df, tgt, feats)
            if dropped:
                print(f"[{name}/{setname}] sabit oldugu icin dusen: {dropped}", flush=True)
            lb, store, bps = championship(name, setname, X, y)
            tag = f"{name}_{setname}"
            lb.to_csv(os.path.join(OUT, f"Leaderboard_{tag}.csv"), index=False, encoding='utf-8-sig')
            with open(os.path.join(OUT, f"best_params_{tag}.json"), 'w', encoding='utf-8') as f:
                json.dump(bps, f, ensure_ascii=False, indent=2, default=str)
            champ = lb.iloc[0]['Algorithm']; auc = float(lb.iloc[0]['OOF_AUC'])
            lo, hi = boot_ci(*store[champ])
            row = {'Outcome': name, 'Set': setname, 'n': len(y), 'pos': int(y.sum()),
                   'neg': int((y == 0).sum()), 'nfeat': X.shape[1], 'Champion': champ,
                   'OOF_AUC': round(auc, 3), 'CI_low': round(lo, 3), 'CI_high': round(hi, 3)}
            if setname == "FULL":
                t, sens, spec, _ = artifacts(name, X, y, champ, store[champ], tag)
                row.update({'threshold': round(t, 3), 'sensitivity': round(sens, 3),
                            'specificity': round(spec, 3)})
            summary.append(row)
            print(f"  >>> {name}/{setname} ŞAMPİYON: {champ}  AUC={auc:.3f} [{lo:.3f}–{hi:.3f}]", flush=True)

    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(OUT, "SUMMARY.csv"), index=False, encoding='utf-8-sig')
    print("\n" + "=" * 78 + "\nMAKALE — 3 PRIMARY MODEL ÖZETİ\n" + "=" * 78, flush=True)
    print(sdf.to_string(index=False), flush=True)
    print(f"\nÇıktılar: {OUT}/", flush=True)


if __name__ == "__main__":
    main()
