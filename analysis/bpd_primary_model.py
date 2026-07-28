"""
BPD PRIMARY MODEL
=================
Index time : Antifungal başlangıcı (t0)
Outcome    : BPD (Bronkopulmoner Displazi)
Değişkenler: ~65 baseline (tedavi başlangıcında bilinen) özellik
Kaynak     : primary_model_BPD.docx
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
# YAPI AYARI
# ============================================================
TARGET_COL = 'BPD'
SUFFIX = '_BPD_Primary'

# ============================================================
# 1) VERİ YÜKLEME VE TARİH İŞLEMLERİ
# ============================================================
df = pd.read_excel('micafunginson.xlsx', na_values=['#BOŞ!', '#NULL!'])

date_cols = ['kullandıgıtarih1', 'takibegirdigitarih', 'taburculuktarihi']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# PNA_t0: Doğumdan antifungal başlangıcına kadar geçen gün
df['PNA_t0'] = (df['kullandıgıtarih1'] - df['takibegirdigitarih']).dt.days
df = df[(df['PNA_t0'] >= 0) & (df['PNA_t0'] <= 200)]

# Survival Outcome
df['event'] = df[TARGET_COL]
df['time'] = (df['taburculuktarihi'] - df['kullandıgıtarih1']).dt.days
df = df[df['time'] > 0]

# ============================================================
# 2) KATEGORİK DÖNÜŞÜMLER (orijinal kodla aynı)
# ============================================================
df['ilaç'] = df['ilaç'].map(
    {1: 'mikafungin', 2: 'fungizon', 3: 'ambizom', 4: 'flukanazol', 5: 'kaspafungin', 6: 'nistatin'})

# Gebelik haftası gruplama (ama ham değeri de koruyoruz)
def categorize_ga(val):
    if pd.isna(val): return 0
    if val < 28: return 1
    elif val < 32: return 2
    elif val < 34: return 3
    elif val < 37: return 4
    else: return 5

df['gebelik_haftası_gruplu'] = df['gebelikhaftası'].apply(categorize_ga)

# ============================================================
# 3) DIŞLAMA LİSTESİ  (primary_model_BPD.docx'a göre)
# ============================================================
cols_to_exclude = [
    # Kimlik ve altyapı tarih sütunları
    'bebekadi', 'dogumtarihi', 'takibegirdigitarih', 'taburculuktarihi',
    'kullandıgıtarih1', 'bittiğitarih1',
    # Common exclude
    'ex',
    'bilissel6ay', 'hareket6ay', 'bilissel9ay', 'hareket9ay',
    'bilissel12ay', 'hareket12ay', 'bilissel16ay', 'hareket16ay', 'bilissel20ay',
    # BPD Primary spesifik dışlamalar
    TARGET_COL,   # Hedefin kendisi
    'rop', 'IKK',
    'mantarüreme', 'üreyenmantar',
    'koryoretinit', 'endoftalmilt', 'endokardit',
    # Tedavi sonrası ölçümler (LEAKAGE — primary modelde yasak)
    'crp2', 'proc2', 'alt2', 'ast2', 'ürikasit2', 'krea2',
    'hbson', 'mpvson', 'wbcson', 'trombositson', 'pctson',
    'potasyumson', 'nötrofilson', 'pdwson', 'ıgson', 'ıl6son',
    # Tedavi tarihleri ve süreleri (leakage)
    'kullandıgıtarih2', 'bittiğitarih2', 'kacgunmicafungin2',
    'kullandıgıtarih3', 'bittigitarih3', 'kacgunkullandi3',
    'kacgüntedavi',
    # Viral sütunlar (feature listesinde yok)
    'viraltedavi', 'viralajan',
    # Survival altyapı
    'time', 'event',
]

# Sadece var olan sütunları düşür (KeyError önleme)
cols_to_drop = [c for c in cols_to_exclude if c in df.columns]
y = df[TARGET_COL].copy()

# ============================================================
# 4) KATEGORİK DEĞİŞKENLERİ ONE-HOT ENCODE
# ============================================================
categorical_cols = ['ilaç', 'gebelik_haftası_gruplu', 'tanı_gruplu', 'dogum_agırlıgı_gruplu',
                    'cinsiyeti', 'gebelik_tipi_gruplu', 'dogumsekli', 'nefrotoksikilaçne',
                    'antifungalkullanimi', 'tanisi']
# Sadece var olanları encode et
cat_to_encode = [c for c in categorical_cols if c in df.columns]

df = df.replace([np.inf, -np.inf], np.nan)
if 'antifungalkullanimi' in df.columns:
    df = df.dropna(subset=['antifungalkullanimi'])
    df['antifungalkullanimi'] = df['antifungalkullanimi'].astype(int)

df = pd.get_dummies(df, columns=cat_to_encode, drop_first=True)

# ============================================================
# 5) EKSİK VERİ (Median Imputation)
# ============================================================
continuous_impute = ['gebelikhaftası', 'gebelikhaftagunu', 'dogumagirligi', 'apgar1', 'apgar5',
                     'anneyaş', 'crp1', 'proc1', 'alt1', 'ast1', 'ürikasit1', 'krea1',
                     'hb1', 'mpv1', 'wbc1', 'trombosit', 'pct1', 'potasyum1',
                     'nötrofil1', 'pdw1', 'ıg1', 'ıl61', 'PNA_t0']
for col in continuous_impute:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

df.fillna(0, inplace=True)
df[:] = np.nan_to_num(df)

# ============================================================
# 6) X MATRİSİNİ OLUŞTUR (Dışlanan sütunları temizle)
# ============================================================
# One-hot sonrası güncellenen sütunlardan dışlanması gerekenleri çıkar
final_drop = [c for c in cols_to_drop if c in df.columns]
X = df.drop(columns=final_drop, errors='ignore')
# Kalan string/object sütunları otomatik düşür (kankültüründeüreyenbakteri vb.)
obj_cols = X.select_dtypes(include=['object', 'datetime64']).columns.tolist()
if obj_cols:
    print(f"Otomatik düşürülen metin sütunlar: {obj_cols}")
    X = X.drop(columns=obj_cols)
# y zaten yukarıda alındı, indexleri hizala
y = y.loc[X.index]

print(f"\n{'='*60}")
print(f"BPD PRIMARY MODEL — {len(X.columns)} değişken ile hazır.")
print(f"Pozitif (BPD=1): {y.sum()}, Negatif (BPD=0): {(y==0).sum()}")
print(f"{'='*60}\n")

# ============================================================
# 7) COX PH MODELİ + BOOTSTRAP C-INDEX
# ============================================================
cox_df = X.copy()
cox_df['time'] = df['time']
cox_df['event'] = df['event']
cph = CoxPHFitter(penalizer=0.5)

try:
    cox_df = cox_df.loc[:, (cox_df != cox_df.iloc[0]).any()]
    cph.fit(cox_df, duration_col='time', event_col='event')
    print("COX PH SUMMARY (BPD Primary)")
    print(cph.summary[['coef', 'exp(coef)', 'p']])

    hr_table = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
    hr_table.columns = ['Hazard Ratio', 'Lower 95% CI', 'Upper 95% CI', 'p-value']
    hr_table.to_csv(f'Cox_Hazard_Ratios{SUFFIX}.csv', encoding='utf-8-sig')
    print(f"-> Cox HR tablosu kaydedildi: Cox_Hazard_Ratios{SUFFIX}.csv")

    # Bootstrap C-index
    n_boot = 1000
    c_indices = []
    for _ in range(n_boot):
        boot = cox_df.sample(n=len(cox_df), replace=True)
        c_val = concordance_index(boot['time'], -cph.predict_partial_hazard(boot), boot['event'])
        c_indices.append(c_val)
    print(f"Optimism-Corrected C-index: {np.mean(c_indices):.3f}")
    print(f"95% CI: [{np.percentile(c_indices, 2.5):.3f} – {np.percentile(c_indices, 97.5):.3f}]")
except Exception as e:
    print(f"Cox Hatası: {e}")

# ============================================================
# 8) KAPLAN-MEIER EĞRİSİ
# ============================================================
kmf = KaplanMeierFitter()
kmf.fit(df['time'], event_observed=df['event'], label='BPD-Free Survival')
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Kaplan-Meier: BPD-Free Survival')
plt.xlabel('Days from Antifungal Start')
plt.ylabel('Probability')
plt.grid(True)
plt.savefig(f'Kaplan_Meier{SUFFIX}.png')
plt.close()
print(f"-> KM eğrisi kaydedildi: Kaplan_Meier{SUFFIX}.png")

# ============================================================
# 9) TABLE 1: BPD vs Non-BPD Karşılaştırma
# ============================================================
numeric_cols_for_table1 = [c for c in X.columns if df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
if len(numeric_cols_for_table1) > 0:
    table1 = df.groupby(TARGET_COL)[numeric_cols_for_table1[:30]].mean().T
    table1.columns = ['Non-BPD (Mean)', 'BPD (Mean)']
    table1.to_csv(f'Table1_Comparison{SUFFIX}.csv', encoding='utf-8-sig')
    print(f"-> Table 1 kaydedildi: Table1_Comparison{SUFFIX}.csv")

# ============================================================
# 10) MODEL YARIŞMASI (10-Fold Stratified CV + SMOTE)
# ============================================================
models_and_params = {
    "Logistic Regression": (
        LogisticRegression(random_state=230, max_iter=2000),
        {'C': loguniform(0.01, 1000), 'solver': ['liblinear']}),
    "Random Forest": (
        RandomForestClassifier(random_state=230),
        {'n_estimators': randint(100, 300), 'max_depth': randint(3, 7),
         'min_samples_split': randint(5, 15), 'max_features': ['sqrt', 'log2', None]}),
    "Support Vector Classifier": (
        SVC(random_state=230, probability=True),
        {'C': loguniform(0.1, 2000), 'gamma': loguniform(1e-5, 1e-1), 'kernel': ['rbf']}),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=230),
        {'n_estimators': randint(50, 300), 'learning_rate': loguniform(0.005, 0.1),
         'max_depth': randint(2, 4), 'subsample': uniform(0.6, 0.4)}),
    "K-Nearest Neighbor": (
        KNeighborsClassifier(),
        {'n_neighbors': randint(3, 11), 'weights': ['uniform', 'distance']}),
    "ExtraTrees": (
        ExtraTreesClassifier(random_state=230),
        {'n_estimators': randint(100, 300), 'max_depth': randint(3, 7),
         'min_samples_split': randint(5, 15), 'max_features': ['sqrt', 'log2', None]}),
    "XGBoost": (
        XGBClassifier(random_state=230, eval_metric='logloss'),
        {'n_estimators': randint(50, 300), 'learning_rate': loguniform(0.001, 0.05),
         'max_depth': randint(2, 4), 'subsample': uniform(0.6, 0.4), 'colsample_bytree': uniform(0.6, 0.4)}),
    "AdaBoost": (
        AdaBoostClassifier(random_state=230),
        {'n_estimators': randint(50, 200), 'learning_rate': loguniform(0.01, 0.5)}),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=230),
        {'criterion': ['gini', 'entropy'], 'max_depth': randint(3, 6),
         'min_samples_split': randint(5, 20), 'min_samples_leaf': randint(2, 10)}),
    "MLP Classifier": (
        MLPClassifier(random_state=230, max_iter=2000, early_stopping=True),
        {'hidden_layer_sizes': [(4,), (8,), (12,)],
         'activation': ['tanh', 'relu'], 'solver': ['lbfgs'],
         'alpha': loguniform(0.1, 100)})
}

results_list = []
all_best_params = {}
all_confusion_matrices = {}
all_oof_predictions = {}
feature_importance_accumulator = {}
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=230)

for model_name, (model, params) in models_and_params.items():
    print(f"--- {model_name} (BPD Primary) ---")
    fold_metrics = []
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(8, 8))
    current_importances = np.zeros(X.shape[1])
    y_true_oof, y_prob_oof = [], []

    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_sm)
        X_test_sc = scaler.transform(X_test)

        search = RandomizedSearchCV(model, params, n_iter=25, cv=3, n_jobs=-1, random_state=230, scoring='accuracy')
        try:
            search.fit(X_train_sc, y_train_sm)
            best = search.best_estimator_
            if i == 0: all_best_params[model_name] = search.best_params_
            if hasattr(best, 'feature_importances_'):
                current_importances += best.feature_importances_
        except Exception as e:
            print(f"  WARNING fold {i+1}: {e}")
            best = model
            if i == 0: all_best_params[model_name] = "Default"

        y_pred = best.predict(X_test_sc)
        try:
            y_prob = best.predict_proba(X_test_sc)[:, 1]
        except:
            y_prob = y_pred.astype(float)

        y_true_oof.extend(y_test); y_prob_oof.extend(y_prob)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)
        aucs.append(roc_auc_val)
        interp_tpr = np.interp(mean_fpr, fpr, tpr); interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i+1} (AUC={roc_auc_val:.2f})' if i < 3 else "")

        fold_metrics.append({
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Sensitivity": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_val,
            "Kappa": cohen_kappa_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred)
        })

    # ROC Plot
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    mean_auc_val = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, 'b', lw=2, alpha=0.8,
             label=r'Mean ROC (AUC=%0.2f $\pm$ %0.2f)' % (mean_auc_val, std_auc))
    std_tpr = np.std(tprs, axis=0)
    plt.fill_between(mean_fpr, np.maximum(mean_tpr-std_tpr,0), np.minimum(mean_tpr+std_tpr,1),
                     color='grey', alpha=0.2, label=r'$\pm$ 1 std')
    plt.plot([0,1],[0,1],'r--',lw=2,alpha=0.8,label='Chance')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'ROC (10-Fold) — {model_name} [BPD Primary]')
    plt.legend(loc='lower right', fontsize='small')
    plt.savefig(f"ROC_{model_name.replace(' ','_')}{SUFFIX}.png"); plt.close()

    if np.sum(current_importances) > 0:
        feature_importance_accumulator[model_name] = current_importances / n_splits

    y_true_oof = np.array(y_true_oof); y_prob_oof = np.array(y_prob_oof)
    all_oof_predictions[model_name] = {'true': y_true_oof, 'prob': y_prob_oof}

    # Optimal Threshold (Youden's J)
    opt_thr = 0.5
    if len(np.unique(y_true_oof)) > 1:
        fpr_o, tpr_o, thr_o = roc_curve(y_true_oof, y_prob_oof)
        valid = ~np.isnan(thr_o)
        if np.sum(valid) > 1:
            j = tpr_o[valid] - fpr_o[valid]
            opt_thr = thr_o[valid][np.argmax(j)]

    y_pred_opt = (y_prob_oof >= opt_thr).astype(int)
    cm = confusion_matrix(y_true_oof, y_pred_opt)
    all_confusion_matrices[model_name] = cm
    try:
        tn, fp, fn, tp = cm.ravel()
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
    except:
        spec = np.nan

    results_list.append({
        "Model": model_name, "Optimal Threshold": opt_thr,
        "Accuracy": accuracy_score(y_true_oof, y_pred_opt),
        "Precision": precision_score(y_true_oof, y_pred_opt, zero_division=0),
        "Sensitivity": recall_score(y_true_oof, y_pred_opt, zero_division=0),
        "Specificity": spec,
        "F1 Score": f1_score(y_true_oof, y_pred_opt, zero_division=0),
        "MSE": mean_squared_error(y_true_oof, y_pred_opt),
        "Kappa": cohen_kappa_score(y_true_oof, y_pred_opt),
        "MCC": matthews_corrcoef(y_true_oof, y_pred_opt),
        "Brier Score": brier_score_loss(y_true_oof, y_prob_oof) if len(np.unique(y_prob_oof))>1 else np.nan,
        "Avg ROC-AUC": np.nanmean(aucs)
    })
    print(f"  -> Tamamlandı (AUC: {np.nanmean(aucs):.3f})")

# ============================================================
# 11) SONUÇ RAPORU
# ============================================================
results_df = pd.DataFrame(results_list).sort_values(by="Avg ROC-AUC", ascending=False)
print(f"\n\n{'='*70}")
print(f"  BPD PRIMARY — FINAL MODEL COMPARISON")
print(f"{'='*70}")
cols_show = ["Model","Accuracy","Precision","Sensitivity","Specificity","F1 Score","Avg ROC-AUC","Brier Score"]
print(results_df[cols_show].to_string())

# Champion model
best_name = results_df.iloc[0]["Model"]
print(f"\n>>> CHAMPION (BPD Primary): {best_name} — AUC: {results_df.iloc[0]['Avg ROC-AUC']:.3f}")

# Feature Importance
if best_name in feature_importance_accumulator:
    imp_df = pd.DataFrame({'Feature': X.columns,
                           'Importance': feature_importance_accumulator[best_name]}).sort_values('Importance', ascending=False)
    plt.figure(figsize=(10, 12))
    sns.barplot(x='Importance', y='Feature', data=imp_df.head(25), palette='viridis')
    plt.title(f'Feature Importance — {best_name} [BPD Primary]')
    plt.tight_layout()
    plt.savefig(f"FeatureImportance{SUFFIX}.png"); plt.close()
    print(f"-> Feature importance kaydedildi: FeatureImportance{SUFFIX}.png")

# Confusion Matrices
for mn, cm in all_confusion_matrices.items():
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred: 0","Pred: 1"], yticklabels=["True: 0","True: 1"])
    ot = results_df.loc[results_df['Model']==mn, 'Optimal Threshold'].iloc[0]
    plt.title(f"{mn}\nThr≈{ot:.3f} [BPD Primary]")
    plt.tight_layout()
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
            plt.title(f'{mn} Calibration [BPD Primary]\n(Brier: {brier:.3f})')
            plt.xlabel('Mean Predicted Prob'); plt.ylabel('Fraction of Positives')
            plt.legend(loc='upper left'); plt.grid(True); plt.tight_layout()
            plt.savefig(f"Calibration_{mn.replace(' ','_')}{SUFFIX}.png"); plt.close()
        except: pass

print(f"\n{'='*60}")
print(f"BPD PRIMARY MODEL ANALİZİ TAMAMLANDI.")
print(f"{'='*60}")
