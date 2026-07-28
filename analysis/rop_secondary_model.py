"""
ROP SECONDARY (COURSE-AWARE) MODEL
====================================
Index time : Antifungal tedavi bitişi
Outcome    : ROP (Retinopati of Prematurity)
Değişkenler: Baseline + tedavi sonrası lab + 4 antifungal escalation türevi
Kaynak     : ROP COURSE.docx + Antifungal escalation türevleri.docx
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (mean_squared_error, accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, cohen_kappa_score, matthews_corrcoef, roc_curve,
                             brier_score_loss, auc)
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform, loguniform
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================
TARGET_COL = 'rop'
SUFFIX = '_ROP_Secondary'
# ============================================================

# 1) VERİ YÜKLEME
df = pd.read_excel('micafunginson.xlsx', na_values=['#BOŞ!', '#NULL!'])

date_cols = ['kullandıgıtarih1', 'takibegirdigitarih', 'taburculuktarihi',
             'kullandıgıtarih2', 'kullandıgıtarih3']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

df['PNA_t0'] = (df['kullandıgıtarih1'] - df['takibegirdigitarih']).dt.days
df = df[(df['PNA_t0'] >= 0) & (df['PNA_t0'] <= 200)]

df['event'] = df[TARGET_COL]
df['time'] = (df['taburculuktarihi'] - df['kullandıgıtarih1']).dt.days
df = df[df['time'] > 0]
df = df.dropna(subset=[TARGET_COL])

# ============================================================
# 2) ANTİFUNGAL ESKALASYON TÜREVLERİ
# ============================================================
df['second_antifungal_started'] = 0
mask_2 = df['kullandıgıtarih2'].notna()
if 'kacgunmicafungin2' in df.columns:
    mask_2 = mask_2 | (df['kacgunmicafungin2'].fillna(0) > 0)
df.loc[mask_2, 'second_antifungal_started'] = 1

df['third_antifungal_started'] = 0
mask_3 = df['kullandıgıtarih3'].notna() if 'kullandıgıtarih3' in df.columns else pd.Series(False, index=df.index)
if 'kacgunkullandi3' in df.columns:
    mask_3 = mask_3 | (df['kacgunkullandi3'].fillna(0) > 0)
df.loc[mask_3, 'third_antifungal_started'] = 1

df['antifungal_escalation'] = ((df['second_antifungal_started'] == 1) | (df['third_antifungal_started'] == 1)).astype(int)
df['multiple_antifungal_lines'] = df['antifungal_escalation']

print(f"Antifungal Türevler: 2.hat={df['second_antifungal_started'].sum()}, "
      f"3.hat={df['third_antifungal_started'].sum()}, eskalasyon={df['antifungal_escalation'].sum()}")

# 3) KATEGORİK
df['ilaç'] = df['ilaç'].map(
    {1: 'mikafungin', 2: 'fungizon', 3: 'ambizom', 4: 'flukanazol', 5: 'kaspafungin', 6: 'nistatin'})

def categorize_ga(val):
    if pd.isna(val): return 0
    if val < 28: return 1
    elif val < 32: return 2
    elif val < 34: return 3
    elif val < 37: return 4
    else: return 5

df['gebelik_haftası_gruplu'] = df['gebelikhaftası'].apply(categorize_ga)

# ============================================================
# 4) DIŞLAMA LİSTESİ (ROP Course)
#    Hepato-renal toksisite sütunları ROP'ta YOK
#    Tedavi sonrası lab değerleri DAHİL
# ============================================================
cols_to_exclude = [
    'bebekadi', 'dogumtarihi', 'takibegirdigitarih', 'taburculuktarihi',
    'kullandıgıtarih1', 'bittiğitarih1',
    'ex',
    'bilissel6ay', 'hareket6ay', 'bilissel9ay', 'hareket9ay',
    'bilissel12ay', 'hareket12ay', 'bilissel16ay', 'hareket16ay', 'bilissel20ay',
    # ROP hedef + diğer outcome'lar
    TARGET_COL, 'BPD', 'IKK',
    # Hepato-renal (ROP modelinde yok)
    'astyüksekliği', 'altyüksekliği', 'hiperürisemi',
    'kreatininyüksekliği', 'potasyumdüşüklüğü',
    # Klinik
    'koryoretinit', 'endoftalmilt', 'endokardit',
    'üreyenmantar',
    # Tedavi süreleri (leakage)
    'kacgüntedavi', 'kacgunmicafungin2', 'kacgunkullandi3',
    'kullandıgıtarih2', 'bittiğitarih2',
    'kullandıgıtarih3', 'bittigitarih3',
    # Viral
    'viraltedavi', 'viralajan',
    # Survival
    'time', 'event',
]
# NOT: crp2, hbson vb. BU LİSTEDE YOK → Feature olarak dahil!

cols_to_drop = [c for c in cols_to_exclude if c in df.columns]
y = df[TARGET_COL].copy()

# 5) ONE-HOT ENCODE
categorical_cols = ['ilaç', 'gebelik_haftası_gruplu', 'tanı_gruplu', 'dogum_agırlıgı_gruplu',
                    'cinsiyeti', 'gebelik_tipi_gruplu', 'dogumsekli', 'nefrotoksikilaçne',
                    'antifungalkullanimi', 'tanisi']
cat_to_encode = [c for c in categorical_cols if c in df.columns]
df = df.replace([np.inf, -np.inf], np.nan)
if 'antifungalkullanimi' in df.columns:
    df = df.dropna(subset=['antifungalkullanimi'])
    df['antifungalkullanimi'] = df['antifungalkullanimi'].astype(int)
df = pd.get_dummies(df, columns=cat_to_encode, drop_first=True)

# 6) EKSİK VERİ
continuous_impute = ['gebelikhaftası', 'gebelikhaftagunu', 'dogumagirligi', 'apgar1', 'apgar5',
                     'anneyaş', 'crp1', 'proc1', 'alt1', 'ast1', 'ürikasit1', 'krea1',
                     'hb1', 'mpv1', 'wbc1', 'trombosit', 'pct1', 'potasyum1',
                     'nötrofil1', 'pdw1', 'ıg1', 'ıl61', 'PNA_t0',
                     'crp2', 'proc2', 'alt2', 'ast2', 'ürikasit2', 'krea2',
                     'hbson', 'mpvson', 'wbcson', 'trombositson', 'pctson',
                     'potasyumson', 'nötrofilson', 'pdwson', 'ıgson', 'ıl6son']
for col in continuous_impute:
    if col in df.columns: df[col] = df[col].fillna(df[col].median())
df.fillna(0, inplace=True); df[:] = np.nan_to_num(df)

# 7) X MATRİSİ
final_drop = [c for c in cols_to_drop if c in df.columns]
X = df.drop(columns=final_drop, errors='ignore')
obj_cols = X.select_dtypes(include=['object', 'datetime64']).columns.tolist()
if obj_cols:
    print(f"Otomatik düşürülen metin sütunlar: {obj_cols}")
    X = X.drop(columns=obj_cols)
y = y.loc[X.index]

print(f"\n{'='*60}")
print(f"ROP SECONDARY MODEL — {len(X.columns)} değişken ile hazır.")
print(f"  (Baseline + Post-treatment Lab + 4 Türetilmiş)")
print(f"Pozitif: {y.sum()}, Negatif: {(y==0).sum()}")
print(f"{'='*60}\n")

# 8) COX PH
cox_df = X.copy(); cox_df['time'] = df['time']; cox_df['event'] = df['event']
cph = CoxPHFitter(penalizer=0.5)
try:
    cox_df = cox_df.loc[:, (cox_df != cox_df.iloc[0]).any()]
    cph.fit(cox_df, duration_col='time', event_col='event')
    print("COX PH SUMMARY (ROP Secondary)")
    print(cph.summary[['coef', 'exp(coef)', 'p']])
    hr_table = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
    hr_table.columns = ['HR', 'Lower 95%', 'Upper 95%', 'p-value']
    hr_table.to_csv(f'Cox_Hazard_Ratios{SUFFIX}.csv', encoding='utf-8-sig')
    c_indices = []
    for _ in range(1000):
        boot = cox_df.sample(n=len(cox_df), replace=True)
        c_indices.append(concordance_index(boot['time'], -cph.predict_partial_hazard(boot), boot['event']))
    print(f"C-index: {np.mean(c_indices):.3f} [{np.percentile(c_indices,2.5):.3f}–{np.percentile(c_indices,97.5):.3f}]")
except Exception as e:
    print(f"Cox Hatası: {e}")

kmf = KaplanMeierFitter()
kmf.fit(df['time'], event_observed=df['event'], label='ROP-Free Survival (Course)')
plt.figure(figsize=(10,6)); kmf.plot_survival_function()
plt.title('KM: ROP-Free Survival (Secondary)'); plt.grid(True)
plt.savefig(f'Kaplan_Meier{SUFFIX}.png'); plt.close()

# 9) MODEL YARIŞMASI
models_and_params = {
    "Logistic Regression": (LogisticRegression(random_state=230, max_iter=2000), {'C': loguniform(0.01, 1000), 'solver': ['liblinear']}),
    "Random Forest": (RandomForestClassifier(random_state=230), {'n_estimators': randint(100,300), 'max_depth': randint(3,7), 'min_samples_split': randint(5,15), 'max_features': ['sqrt','log2',None]}),
    "Support Vector Classifier": (SVC(random_state=230, probability=True), {'C': loguniform(0.1, 2000), 'gamma': loguniform(1e-5, 1e-1), 'kernel': ['rbf']}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=230), {'n_estimators': randint(50,300), 'learning_rate': loguniform(0.005,0.1), 'max_depth': randint(2,4), 'subsample': uniform(0.6,0.4)}),
    "K-Nearest Neighbor": (KNeighborsClassifier(), {'n_neighbors': randint(3,11), 'weights': ['uniform','distance']}),
    "ExtraTrees": (ExtraTreesClassifier(random_state=230), {'n_estimators': randint(100,300), 'max_depth': randint(3,7), 'min_samples_split': randint(5,15), 'max_features': ['sqrt','log2',None]}),
    "XGBoost": (XGBClassifier(random_state=230, eval_metric='logloss'), {'n_estimators': randint(50,300), 'learning_rate': loguniform(0.001,0.05), 'max_depth': randint(2,4), 'subsample': uniform(0.6,0.4), 'colsample_bytree': uniform(0.6,0.4)}),
    "AdaBoost": (AdaBoostClassifier(random_state=230), {'n_estimators': randint(50,200), 'learning_rate': loguniform(0.01,0.5)}),
    "Decision Tree": (DecisionTreeClassifier(random_state=230), {'criterion': ['gini','entropy'], 'max_depth': randint(3,6), 'min_samples_split': randint(5,20), 'min_samples_leaf': randint(2,10)}),
    "MLP Classifier": (MLPClassifier(random_state=230, max_iter=2000, early_stopping=True), {'hidden_layer_sizes': [(4,),(8,),(12,)], 'activation': ['tanh','relu'], 'solver': ['lbfgs'], 'alpha': loguniform(0.1,100)})
}

results_list = []; all_best_params = {}; all_confusion_matrices = {}
all_oof_predictions = {}; feature_importance_accumulator = {}
n_splits = 10; skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=230)

for model_name, (model, params) in models_and_params.items():
    print(f"--- {model_name} (ROP Secondary) ---")
    tprs, aucs = [], []; mean_fpr = np.linspace(0,1,100)
    plt.figure(figsize=(8,8)); current_imp = np.zeros(X.shape[1])
    y_true_oof, y_prob_oof = [], []

    for i, (tri, tei) in enumerate(skf.split(X, y)):
        Xtr, Xte = X.iloc[tri], X.iloc[tei]; ytr, yte = y.iloc[tri], y.iloc[tei]
        sm = SMOTE(random_state=42); Xsm, ysm = sm.fit_resample(Xtr, ytr)
        sc = StandardScaler(); Xtr_s, Xte_s = sc.fit_transform(Xsm), sc.transform(Xte)

        srch = RandomizedSearchCV(model, params, n_iter=25, cv=3, n_jobs=-1, random_state=230, scoring='accuracy')
        try:
            srch.fit(Xtr_s, ysm); bm = srch.best_estimator_
            if i==0: all_best_params[model_name] = srch.best_params_
            if hasattr(bm, 'feature_importances_'): current_imp += bm.feature_importances_
        except: bm = model; all_best_params.setdefault(model_name, "Default")

        yp = bm.predict(Xte_s)
        try: ypr = bm.predict_proba(Xte_s)[:,1]
        except: ypr = yp.astype(float)
        y_true_oof.extend(yte); y_prob_oof.extend(ypr)

        fp, tp, _ = roc_curve(yte, ypr); rv = auc(fp, tp); aucs.append(rv)
        it = np.interp(mean_fpr, fp, tp); it[0]=0.0; tprs.append(it)
        plt.plot(fp, tp, lw=1, alpha=0.3)

    mt = np.mean(tprs, axis=0); mt[-1]=1.0
    plt.plot(mean_fpr, mt, 'b', lw=2, label=f'Mean AUC={auc(mean_fpr,mt):.2f}')
    plt.plot([0,1],[0,1],'r--'); plt.title(f'ROC — {model_name} [ROP Secondary]')
    plt.legend(loc='lower right'); plt.savefig(f"ROC_{model_name.replace(' ','_')}{SUFFIX}.png"); plt.close()

    if np.sum(current_imp)>0: feature_importance_accumulator[model_name] = current_imp/n_splits
    y_true_oof = np.array(y_true_oof); y_prob_oof = np.array(y_prob_oof)
    all_oof_predictions[model_name] = {'true': y_true_oof, 'prob': y_prob_oof}

    opt_thr = 0.5
    if len(np.unique(y_true_oof))>1:
        fo, to, th = roc_curve(y_true_oof, y_prob_oof)
        v = ~np.isnan(th)
        if np.sum(v)>1: opt_thr = th[v][np.argmax(to[v]-fo[v])]

    ypo = (y_prob_oof >= opt_thr).astype(int)
    cm = confusion_matrix(y_true_oof, ypo); all_confusion_matrices[model_name] = cm
    try: tn,fpp,fn,tp = cm.ravel(); spec = tn/(tn+fpp) if (tn+fpp)>0 else 0
    except: spec = np.nan

    results_list.append({
        "Model": model_name, "Optimal Threshold": opt_thr,
        "Accuracy": accuracy_score(y_true_oof, ypo),
        "Precision": precision_score(y_true_oof, ypo, zero_division=0),
        "Sensitivity": recall_score(y_true_oof, ypo, zero_division=0),
        "Specificity": spec,
        "F1 Score": f1_score(y_true_oof, ypo, zero_division=0),
        "Avg ROC-AUC": np.nanmean(aucs),
        "Brier Score": brier_score_loss(y_true_oof, y_prob_oof) if len(np.unique(y_prob_oof))>1 else np.nan
    })
    print(f"  -> Tamamlandı (AUC: {np.nanmean(aucs):.3f})")

results_df = pd.DataFrame(results_list).sort_values(by="Avg ROC-AUC", ascending=False)
print(f"\n{'='*70}")
print(f"ROP SECONDARY — FINAL MODEL COMPARISON")
print(f"{'='*70}")
print(results_df[["Model","Accuracy","Sensitivity","Specificity","F1 Score","Avg ROC-AUC","Brier Score"]].to_string())
best_name = results_df.iloc[0]["Model"]
print(f"\n>>> CHAMPION (ROP Secondary): {best_name} — AUC: {results_df.iloc[0]['Avg ROC-AUC']:.3f}")

if best_name in feature_importance_accumulator:
    imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance_accumulator[best_name]}).sort_values('Importance', ascending=False)
    plt.figure(figsize=(10,14)); sns.barplot(x='Importance', y='Feature', data=imp_df.head(30), palette='viridis')
    plt.title(f'Feature Importance — {best_name} [ROP Secondary]'); plt.tight_layout()
    plt.savefig(f"FeatureImportance{SUFFIX}.png"); plt.close()

for mn, cm in all_confusion_matrices.items():
    plt.figure(figsize=(5,4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{mn} [ROP Secondary]"); plt.tight_layout()
    plt.savefig(f"CM_{mn.replace(' ','_')}{SUFFIX}.png"); plt.close()


# Calibration Curves
for mn, pred in all_oof_predictions.items():
    yt, yp = pred['true'], pred['prob']
    if len(np.unique(yp))>1 and len(np.unique(yt))>1:
        try:
            frac, mean_pred = calibration_curve(yt, yp, n_bins=10, strategy='uniform')
            brier = results_df.loc[results_df['Model']==mn, 'Brier Score'].iloc[0]
            plt.figure(figsize=(7,6))
            plt.plot([0,1],[0,1],'k:', label='Perfect')
            plt.plot(mean_pred, frac, 's-', label=mn, markerfacecolor='red', markersize=8)
            plt.title(f'{mn} Calibration [ROP Secondary]\n(Brier: {brier:.3f})')
            plt.xlabel('Mean Predicted Prob'); plt.ylabel('Fraction of Positives')
            plt.legend(loc='upper left'); plt.grid(True); plt.tight_layout()
            plt.savefig(f"Calibration_{mn.replace(' ','_')}{SUFFIX}.png"); plt.close()
        except Exception as e:
            pass

print(f"\nROP SECONDARY MODEL ANALİZİ TAMAMLANDI.")
