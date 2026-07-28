# -*- coding: utf-8 -*-
"""
DÜRÜST 8/12-DEĞİŞKEN OFFLINE ŞAMPİYON ANALİZİ
=============================================
Amaç: Canlı Streamlit calculator'ının GERÇEKTEN kullandığı 8 (Secondary'de 12)
dürüst baseline değişkenle, 10 algoritmayı 10-Fold CV + SMOTE ile yarıştırıp
her hedef için gerçek out-of-fold (OOF) CV-AUC, şampiyon, Feature Importance,
kalibrasyon eğrisi, confusion matrix ve dürüst optimal threshold üretmek.

Bu, offline araştırma kodlarının (~30 değişkenli) EKSİK bıraktığı adımdır:
8-değişkenli modelde hangi algoritmanın kazandığı ve performansı hiç ölçülmemişti.

NOT: Hiçbir canlı .pkl / meta dosyasına DOKUNMAZ. Tüm çıktılar
Honest_8Feature_Analysis/ klasörüne yazılır. Şampiyon aday modelleri
candidate_models/ altına "aday" olarak kaydedilir (deploy edilmez).
"""
import os, json, warnings
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
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             accuracy_score, f1_score, recall_score, precision_score)
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings('ignore')
os.chdir(r"C:\Users\hasan\Documents\medical-research")
OUT = "Honest_8Feature_Analysis"
os.makedirs(os.path.join(OUT, "candidate_models"), exist_ok=True)

RANDOM = 230
HONEST = ['gebelikhaftası', 'cinsiyeti', 'apgar5', 'cerrahi', 'idrarsondasi', 'PNA_t0', 'crp1', 'proc1']
SEC = ['second_antifungal_started', 'third_antifungal_started', 'antifungal_escalation', 'multiple_antifungal_lines']

# Karşılaştırma bağlamı: dağıtılmış .pkl algoritmaları + geniş-feature setinde
# seçilmiş şampiyon algoritma ve o sette raporlanan AUC değerleri
CONTEXT = {
    "Ex (Mortalite)":  {"deployed": "ExtraTrees", "wide_champion": "AdaBoost",          "wide_auc": None},
    "BPD Primary":     {"deployed": "ExtraTrees", "wide_champion": "Gradient Boosting",  "wide_auc": None},
    "BPD Secondary":   {"deployed": "RandomForest","wide_champion": "Random Forest",     "wide_auc": 0.955},
    "ROP Primary":     {"deployed": "ExtraTrees", "wide_champion": "Random Forest",      "wide_auc": 0.950},
    "ROP Secondary":   {"deployed": "RandomForest","wide_champion": "Random Forest",     "wide_auc": 0.975},
}


# --------------------------- VERİ HAZIRLIK (export_streamlit_models.py ile birebir) ---------------------------
def get_base_df():
    df = pd.read_excel('micafunginson.xlsx', na_values=['#BOŞ!', '#NULL!'])
    for col in ['kullandıgıtarih1', 'takibegirdigitarih', 'taburculuktarihi']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df['PNA_t0'] = (df['kullandıgıtarih1'] - df['takibegirdigitarih']).dt.days
    df = df[(df['PNA_t0'] >= 0) & (df['PNA_t0'] <= 200)]
    return df

def get_sec(df):
    for col in ['kullandıgıtarih2', 'kullandıgıtarih3']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df['second_antifungal_started'] = ((df['kacgunmicafungin2'].fillna(0) > 0) | (df['kullandıgıtarih2'].notna())).astype(int)
    df['third_antifungal_started'] = ((df['kacgunkullandi3'].fillna(0) > 0) | (df['kullandıgıtarih3'].notna())).astype(int)
    df['antifungal_escalation'] = ((df['second_antifungal_started'] == 1) | (df['third_antifungal_started'] == 1)).astype(int)
    df['multiple_antifungal_lines'] = df['antifungal_escalation']
    return df

def prep(df, target, feats):
    df = df.copy()
    df['time'] = (df['taburculuktarihi'] - df['kullandıgıtarih1']).dt.days
    df = df[df['time'] > 0]
    df = df.dropna(subset=[target])
    y = df[target].astype(int)
    X = df[feats].copy()
    for c in X.columns:
        X[c] = X[c].fillna(X[c].median())
    return X.reset_index(drop=True), y.reset_index(drop=True)


def build_models():
    m = {
        "Logistic Regression": (LogisticRegression(random_state=RANDOM, max_iter=2000),
                                {'C': loguniform(0.01, 1000), 'solver': ['liblinear']}),
        "Random Forest": (RandomForestClassifier(random_state=RANDOM),
                          {'n_estimators': randint(100, 300), 'max_depth': randint(3, 7),
                           'min_samples_split': randint(5, 15), 'max_features': ['sqrt', 'log2', None]}),
        "Support Vector Classifier": (SVC(random_state=RANDOM, probability=True),
                                      {'C': loguniform(0.1, 2000), 'gamma': loguniform(1e-5, 1e-1), 'kernel': ['rbf']}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=RANDOM),
                              {'n_estimators': randint(50, 300), 'learning_rate': loguniform(0.005, 0.1),
                               'max_depth': randint(2, 4), 'subsample': uniform(0.6, 0.4)}),
        "K-Nearest Neighbor": (KNeighborsClassifier(),
                               {'n_neighbors': randint(3, 11), 'weights': ['uniform', 'distance']}),
        "ExtraTrees": (ExtraTreesClassifier(random_state=RANDOM),
                       {'n_estimators': randint(100, 300), 'max_depth': randint(3, 7),
                        'min_samples_split': randint(5, 15), 'max_features': ['sqrt', 'log2', None]}),
        "AdaBoost": (AdaBoostClassifier(random_state=RANDOM),
                     {'n_estimators': randint(50, 200), 'learning_rate': loguniform(0.01, 0.5)}),
        "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM),
                          {'criterion': ['gini', 'entropy'], 'max_depth': randint(3, 6),
                           'min_samples_split': randint(5, 20), 'min_samples_leaf': randint(2, 10)}),
        "MLP Classifier": (MLPClassifier(random_state=RANDOM, max_iter=2000, early_stopping=True),
                           {'hidden_layer_sizes': [(4,), (8,), (12,)], 'activation': ['tanh', 'relu'],
                            'solver': ['lbfgs'], 'alpha': loguniform(0.1, 100)}),
    }
    if HAS_XGB:
        m["XGBoost"] = (XGBClassifier(random_state=RANDOM, eval_metric='logloss'),
                        {'n_estimators': randint(50, 300), 'learning_rate': loguniform(0.001, 0.05),
                         'max_depth': randint(2, 4), 'subsample': uniform(0.6, 0.4),
                         'colsample_bytree': uniform(0.6, 0.4)})
    return m


def evaluate_target(name, X, y):
    print(f"\n{'='*70}\n{name}  |  n={len(y)}, pozitif={int(y.sum())}, negatif={int((y==0).sum())}, feat={X.shape[1]}\n{'='*70}", flush=True)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM)
    models = build_models()
    rows, oof_store, bp_store = [], {}, {}

    for algo, (model, params) in models.items():
        yt_oof, yp_oof, fold_aucs = [], [], []
        bp = None
        for i, (tri, tei) in enumerate(skf.split(X, y)):
            Xtr, Xte = X.iloc[tri], X.iloc[tei]
            ytr, yte = y.iloc[tri], y.iloc[tei]
            Xsm, ysm = SMOTE(random_state=42).fit_resample(Xtr, ytr)
            sc = StandardScaler()
            Xtr_s, Xte_s = sc.fit_transform(Xsm), sc.transform(Xte)
            try:
                srch = RandomizedSearchCV(model, params, n_iter=25, cv=3, n_jobs=-1,
                                          random_state=RANDOM, scoring='accuracy')
                srch.fit(Xtr_s, ysm)
                bm = srch.best_estimator_
                if i == 0:
                    bp = {k: (v.item() if hasattr(v, 'item') else v) for k, v in srch.best_params_.items()}
            except Exception as e:
                bm = model
                bm.fit(Xtr_s, ysm)
                if i == 0:
                    bp = f"Default ({str(e)[:40]})"
            try:
                prob = bm.predict_proba(Xte_s)[:, 1]
            except Exception:
                prob = bm.predict(Xte_s).astype(float)
            yt_oof.extend(yte.tolist()); yp_oof.extend(prob.tolist())
            if len(set(yte.tolist())) > 1:
                fold_aucs.append(roc_auc_score(yte, prob))

        yt, yp = np.array(yt_oof), np.array(yp_oof)
        oof_auc = roc_auc_score(yt, yp)
        rows.append({
            'Algorithm': algo,
            'OOF_AUC': round(oof_auc, 4),
            'MeanFoldAUC': round(np.mean(fold_aucs), 4) if fold_aucs else np.nan,
            'StdFoldAUC': round(np.std(fold_aucs), 4) if fold_aucs else np.nan,
            'Acc@0.5': round(accuracy_score(yt, (yp >= 0.5).astype(int)), 4),
        })
        oof_store[algo] = (yt, yp)
        bp_store[algo] = bp
        print(f"  {algo:26s} OOF-AUC={oof_auc:.3f}", flush=True)

    lb = pd.DataFrame(rows).sort_values('OOF_AUC', ascending=False).reset_index(drop=True)
    return lb, oof_store, bp_store, models


def champion_artifacts(name, X, y, champ, oof, models):
    """Şampiyon için: dürüst threshold, sens/spec, FI, kalibrasyon, CM, aday model."""
    yt, yp = oof
    safe = name.replace(" ", "_").replace("(", "").replace(")", "")

    # --- Dürüst optimal threshold (OOF üzerinde Youden's J) ---
    fpr, tpr, thr = roc_curve(yt, yp)
    j = tpr - fpr
    k = int(np.argmax(j))
    opt_thr = float(thr[k])
    if not np.isfinite(opt_thr):  # roc_curve ilk eşiği +inf verebilir
        opt_thr = 1.0
    ypred = (yp >= opt_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, ypred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    # --- Aday modeli tam veride eğit (Pipeline: ölçekleme + model) ---
    base_model, params = models[champ]
    Xsm, ysm = SMOTE(random_state=42).fit_resample(X, y)
    sc = StandardScaler()
    Xsm_s = sc.fit_transform(Xsm)
    srch = RandomizedSearchCV(base_model, params, n_iter=25, cv=3, n_jobs=-1,
                              random_state=RANDOM, scoring='accuracy')
    srch.fit(Xsm_s, ysm)
    final_est = srch.best_estimator_
    pipe = Pipeline([('scaler', sc), ('model', final_est)])  # raw 8/12 feature -> predict_proba
    joblib.dump(pipe, os.path.join(OUT, "candidate_models", f"candidate_{safe}.pkl"))

    # --- Feature Importance ---
    feats = list(X.columns)
    if hasattr(final_est, 'feature_importances_'):
        imp = final_est.feature_importances_
        imp_kind = "feature_importances_"
    elif hasattr(final_est, 'coef_'):
        imp = np.abs(final_est.coef_).ravel()
        imp_kind = "|coef|"
    else:
        pr = permutation_importance(pipe, X, y, n_repeats=20, random_state=RANDOM, scoring='roc_auc')
        imp = pr.importances_mean
        imp_kind = "permutation"
    fi = pd.DataFrame({'Feature': feats, 'Importance': imp}).sort_values('Importance', ascending=False)
    fi.to_csv(os.path.join(OUT, f"FI_{safe}.csv"), index=False, encoding='utf-8-sig')
    plt.figure(figsize=(8, max(3, 0.5 * len(feats))))
    plt.barh(fi['Feature'][::-1], fi['Importance'][::-1], color='#2c7fb8')
    plt.title(f"Feature Importance — {champ} [{name}] (8/12 honest, {imp_kind})")
    plt.xlabel("Importance"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"FI_{safe}.png"), dpi=120); plt.close()

    # --- Kalibrasyon eğrisi (OOF) ---
    try:
        frac_pos, mean_pred = calibration_curve(yt, yp, n_bins=5, strategy='quantile')
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfectly calibrated')
        plt.plot(mean_pred, frac_pos, 'o-', color='#d95f0e', label=champ)
        plt.xlabel("Mean predicted probability"); plt.ylabel("Observed fraction positive")
        plt.title(f"Calibration (OOF) — {champ} [{name}]"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"Calibration_{safe}.png"), dpi=120); plt.close()
    except Exception as e:
        print(f"    (kalibrasyon atlandı: {e})", flush=True)

    # --- Confusion Matrix (dürüst threshold) ---
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap='Blues')
    for (r, c), v in np.ndenumerate(cm):
        plt.text(c, r, str(v), ha='center', va='center', fontsize=14,
                 color='white' if v > cm.max() / 2 else 'black')
    plt.xticks([0, 1], ['Pred 0', 'Pred 1']); plt.yticks([0, 1], ['True 0', 'True 1'])
    plt.title(f"CM @thr={opt_thr:.3f} — {champ} [{name}]"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"ConfusionMatrix_{safe}.png"), dpi=120); plt.close()

    return {'optimal_threshold': round(opt_thr, 4), 'sensitivity': round(sens, 4),
            'specificity': round(spec, 4), 'fi_kind': imp_kind,
            'final_best_params': {k: (v.item() if hasattr(v, 'item') else v) for k, v in srch.best_params_.items()}}


def main():
    base = get_base_df()
    sec = get_sec(get_base_df())
    configs = [
        ("Ex (Mortalite)", base, 'ex', HONEST),
        ("BPD Primary", base, 'BPD', HONEST),
        ("BPD Secondary", sec, 'BPD', HONEST + SEC),
        ("ROP Primary", base, 'rop', HONEST),
        ("ROP Secondary", sec, 'rop', HONEST + SEC),
    ]
    summary = []
    for name, d, tgt, feats in configs:
        X, y = prep(d, tgt, feats)
        lb, oof, bps, models = evaluate_target(name, X, y)
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        lb.to_csv(os.path.join(OUT, f"Leaderboard_{safe}.csv"), index=False, encoding='utf-8-sig')
        with open(os.path.join(OUT, f"best_params_{safe}.json"), 'w', encoding='utf-8') as f:
            json.dump(bps, f, ensure_ascii=False, indent=2, default=str)

        champ = lb.iloc[0]['Algorithm']
        champ_auc = float(lb.iloc[0]['OOF_AUC'])
        art = champion_artifacts(name, X, y, champ, oof[champ], models)
        ctx = CONTEXT[name]
        summary.append({
            'Model': name, 'n': len(y), 'pos': int(y.sum()), 'neg': int((y == 0).sum()),
            'nfeat': X.shape[1],
            'HONEST_champion': champ, 'HONEST_OOF_AUC': round(champ_auc, 3),
            'opt_threshold': art['optimal_threshold'],
            'sensitivity': art['sensitivity'], 'specificity': art['specificity'],
            'deployed_pkl_algo': ctx['deployed'],
            'advisor_wide_champion': ctx['wide_champion'],
            'reported_wide_AUC': ctx['wide_auc'],
        })
        print(f"  >>> ŞAMPİYON (8/12): {champ}  OOF-AUC={champ_auc:.3f}  thr={art['optimal_threshold']}  "
              f"sens={art['sensitivity']}  spec={art['specificity']}", flush=True)

    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(OUT, "SUMMARY.csv"), index=False, encoding='utf-8-sig')
    print("\n\n" + "=" * 70 + "\nÖZET TABLO\n" + "=" * 70, flush=True)
    print(sdf.to_string(index=False), flush=True)
    print(f"\nTüm çıktılar: {OUT}/  (canlı dosyalara dokunulmadı)", flush=True)


if __name__ == "__main__":
    main()
