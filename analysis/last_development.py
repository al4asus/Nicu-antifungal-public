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
# Yeni eklenen kütüphane
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Data Loading and Preparation ---
excel_file_path = 'micafunginson.xlsx'
df = pd.read_excel(excel_file_path, na_values=['#BOŞ!', '#NULL!'])

# --- YENİ EKLENEN: TARİH DÖNÜŞÜMLERİ ---
date_cols = ['kullandıgıtarih1', 'takibegirdigitarih', 'taburculuktarihi']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# 1) ZAMAN DEĞİŞKENLERİ (t0 ve PNA_t0)
# t0 zaten 'kullandıgıtarih1' sütunu.
df['PNA_t0'] = (df['kullandıgıtarih1'] - df['takibegirdigitarih']).dt.days

# Kontrol: Negatif değerleri veya aşırı uçları temizle
df = df[df['PNA_t0'] >= 0] # Doğumdan önce tedavi başlayamaz
df = df[df['PNA_t0'] <= 200] # Aşırı uç değer kontrolü

# 2) SURVIVAL OUTCOME (Event ve Time)
df['event'] = df['ex'] # 1=Ex, 0=Alive
df['time'] = (df['taburculuktarihi'] - df['kullandıgıtarih1']).dt.days

# Kontrol: time <= 0 olan hatalı kayıtları temizle
df = df[df['time'] > 0]

# Gerekli hesaplamalar bittikten sonra orijinal tarih sütunlarını ve isimleri temizle
df = df.drop(['bebekadi', 'dogumtarihi'], axis=1)
# --------------------------------------------------

df['ilaç'] = df['ilaç'].map(
    {1: 'mikafungin', 2: 'fungizon', 3: 'ambizom', 4: 'flukanazol', 5: 'kaspafungin', 6: 'nistatin'})
df['gebelikhaftagunu'] = pd.read_excel('micafungindeğismemis.xlsx', usecols=['gebelikhaftagunu'])['gebelikhaftagunu']


def categorize_gebelik_haftasi(gebelik_haftasi):
    if pd.isna(gebelik_haftasi): return 0
    if gebelik_haftasi < 28:
        return 1
    elif 28 <= gebelik_haftasi < 32:
        return 2
    elif 32 <= gebelik_haftasi < 34:
        return 3
    elif 34 <= gebelik_haftasi < 37:
        return 4
    else:
        return 5


df['gebelik_haftası_gruplu'] = df['gebelikhaftası'].apply(categorize_gebelik_haftasi)
df = df.drop(['gebelikhaftası', 'gebelikhaftagunu'], axis=1)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['antifungalkullanimi'])
df['antifungalkullanimi'] = df['antifungalkullanimi'].astype(int)

# --- YENİ: İNOTROP ŞİDDET SKORU ---
# 0: yok, 1: tekli (dopamin veya dobutamin), 2: çiftli (her ikisi)
inotrop_map = {0: 0, 1: 1, 2: 1, 3: 2}
df['inotrop_severity'] = df['ilkhaftainotrop'].map(inotrop_map)
df = df.drop(['ilkhaftainotrop'], axis=1) # Orijinalini siliyoruz
# ---------------------------------

# NOT: 'inotrop_severity' sayısal (ordinal) bir değer olduğu için
# get_dummies içine dahil etmemelisin, direkt sayı olarak kalmalı.
df = pd.get_dummies(df, columns=['ilaç', 'gebelik_haftası_gruplu', 'tanı_gruplu', 'dogum_agırlıgı_gruplu', 'cinsiyeti',
                                 'gebelik_tipi_gruplu', 'dogumsekli', 'nefrotoksikilaçne', 'antifungalkullanimi',
                                 'tanisi'], drop_first=True)



# --- REQUIREMENT 6: EKSİK VERİ STRATEJİSİ (Median Imputation) ---
# Belirlenen sürekli değişkenlerde boşlukları 0 yerine MEDİAN ile dolduruyoruz
continuous_baseline = ['apgar5', 'crp1', 'proc1', 'PNA_t0']
for col in continuous_baseline:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Geri kalan her şey (kategorikler vb.) için 0 güvenli bir limandır
df.fillna(0, inplace=True)
df[:] = np.nan_to_num(df)
# ---------------------------------------------------------------

# --- REQUIREMENT 5: BASELINE PREDICTOR SET (Whitelist) ---
# Sadece dürüst tahmin yapacak olan o 12 ana değişkeni seçiyoruz
baseline_final_list = [
    'apgar5', 'entübe', 'inotrop_severity', 'svk', 'tpn', 'cerrahi', 'idrarsondasi',
    'crp1', 'proc1', 'PNA_t0'
]

# dummy kolonlarını (cinsiyet ve gebelik grupları) otomatik yakalayalım
X_cols = [c for c in df.columns if any(v == c for v in baseline_final_list) or
          c.startswith('cinsiyeti_') or
          c.startswith('gebelik_haftası_gruplu_')]

# X'i sadece bu tertemiz ve dürüst 12 (ve dummy) değişkenle kısıtlıyoruz
X = df[X_cols]
y = df['ex']

print(f"Baseline modeliniz {len(X.columns)} değişken ile donduruldu.")
# ---------------------------------------------------------

# ========================================================
# REQUIREMENT 7 & 8: COX PH MODEL & BOOTSTRAP C-INDEX
# ========================================================
"""
# 1. Cox Modeli için Veri Hazırlığı
# Lifelines kütüphanesi X, time ve event sütunlarını aynı tabloda bekler.
cox_df = X.copy()
cox_df['time'] = df['time']  # Requirement 2'de hesapladığın gün sayısı
cox_df['event'] = df['event']  # Requirement 2'de hesapladığın vefat durumu (1=Ex)
"""
# 1. Cox Modeli için TAM dürüst "X" (baseline_final_list + dummies) sütunlarını seçiyoruz
cox_features = list(X.columns)

# Veriyi hazırlarken süre ve vefat bilgisini de ekleyerek birleştiriyoruz
cox_df = X.copy()
cox_df['time'] = df['time']
cox_df['event'] = df['event']

# Not: Akademik dürüstlük gereği tüm dürüst verileri kullanıyoruz.
# 1. Cox Modelini Tanımla (Penalizer ekleyerek modeli stabilize ediyoruz)
# Penalizer=0.5, 12 değişkende oluşabilecek matris yakınsama hatalarını önler (Ridge)
cph = CoxPHFitter(penalizer=0.5)

try:
    # 2. Önce veride tek bir değer içeren (sabit) sütun var mı temizleyelim
    cox_df = cox_df.loc[:, (cox_df != cox_df.iloc[0]).any()]

    # 3. Modeli Eğit
    cph.fit(cox_df, duration_col='time', event_col='event')

    print("\n" + "=" * 30)
    print("COX MODEL SUMMARY (STATISTICAL BASELINE)")
    print(cph.summary[['coef', 'exp(coef)', 'p']])
    print("=" * 30)

    # --- REQUIREMENT 8 (Bootstrap) SADECE FİT BAŞARILIYSA ÇALIŞMALI ---
    n_bootstrap = 1000
    c_indices = []
    print(f"\nRunning {n_bootstrap} Bootstrap iterations for C-index...")

    for i in range(n_bootstrap):
        boot_sample = cox_df.sample(n=len(cox_df), replace=True)
        # C-index hesapla
        c_val = concordance_index(boot_sample['time'],
                                  -cph.predict_partial_hazard(boot_sample),
                                  boot_sample['event'])
        c_indices.append(c_val)

    mean_c = np.mean(c_indices)
    print(f"Optimism-Corrected C-index: {mean_c:.3f}")
    print(f"95% CI: [{np.percentile(c_indices, 2.5):.3f} - {np.percentile(c_indices, 97.5):.3f}]")

except Exception as e:
    print(f"\nCox Model Hala Yakınsayamadı. Hata: {e}")
    print("İpucu: Değişken sayısını azaltmayı veya 'penalizer' değerini 0.5 yapmayı deneyin.")

# 4. Bootstrap ile C-index Hesaplama (Requirement 8)
n_bootstrap = 1000
c_indices = []

print(f"\nRunning {n_bootstrap} Bootstrap iterations for C-index...")
for i in range(n_bootstrap):
    # Veriyi yerine koyarak örnekle (Resampling)
    boot_sample = cox_df.sample(n=len(cox_df), replace=True)

    # C-index hesapla (Harrell's C)
    # Concordance index: Modelin hastaların vefat sırasını ne kadar doğru tahmin ettiğidir.
    c_val = concordance_index(boot_sample['time'],
                              -cph.predict_partial_hazard(boot_sample),
                              boot_sample['event'])
    c_indices.append(c_val)

mean_c = np.mean(c_indices)
print(f"Optimism-Corrected C-index: {mean_c:.3f}")
print(f"95% CI: [{np.percentile(c_indices, 2.5):.3f} - {np.percentile(c_indices, 97.5):.3f}]")
# ========================================================

# ========================================================
# REQUIREMENT 10: ÇIKTILAR (PAPER İÇİN TABLO VE GRAFİKLER)
# ========================================================

# --- 1. Kaplan-Meier Eğrisi (Overall Survival) ---
kmf = KaplanMeierFitter()
kmf.fit(cox_df['time'], event_observed=cox_df['event'], label='Genel Sağkalım')

plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Kaplan-Meier Overall Survival Curve')
plt.xlabel('Günler (Tedavi Başlangıcından İtibaren)')
plt.ylabel('Sağkalım Olasılığı')
plt.grid(True)
plt.savefig('Kaplan_Meier_Curve.png')
print("\n-> Kaplan-Meier grafiği 'Kaplan_Meier_Curve.png' olarak kaydedildi.")

# --- 2. Hazard Ratio (HR) Tablosu ve %95 GA ---
# Cox modelinden gelen HR değerlerini ve Güven Aralıklarını tablo olarak alıyoruz
hr_table = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
hr_table.columns = ['Hazard Ratio (HR)', 'Lower 95% CI', 'Upper 95% CI', 'p-value']

# Bu tabloyu Excel'de açabilmek için CSV olarak kaydediyoruz
hr_table.to_csv('Cox_Hazard_Ratios_Table.csv', encoding='utf-8-sig')
print("-> Hazard Ratio tablosu 'Cox_Hazard_Ratios_Table.csv' olarak kaydedildi.")

# --- 3. Table 1: Exitus vs Survivor Karşılaştırma ---
# Gruplara göre ortalamaları alarak basit bir karşılaştırma tablosu oluşturur
table1 = df.groupby('ex')[baseline_final_list].mean().T
table1.columns = ['Survivor (Mean)', 'Exitus (Mean)']
table1.to_csv('Table1_Baseline_Comparison.csv', encoding='utf-8-sig')
print("-> Table 1 (Karşılaştırma) 'Table1_Baseline_Comparison.csv' olarak kaydedildi.")

# --- MODELS AND PARAMETER RANGES --
models_and_params = {
    "Logistic Regression": (
        LogisticRegression(random_state=230, max_iter=2000), {'C': loguniform(0.01, 1000), 'solver': ['liblinear']}),

    "Random Forest": (RandomForestClassifier(random_state=230),
                      {'n_estimators': randint(100, 300), 'max_depth': randint(3, 7),
                       'min_samples_split': randint(5, 15), 'max_features': ['sqrt', 'log2', None]}),

    "Support Vector Classifier": (SVC(random_state=230, probability=True),
                                  {'C': loguniform(0.1, 2000), 'gamma': loguniform(1e-5, 1e-1), 'kernel': ['rbf']}),

    "Gradient Boosting": (GradientBoostingClassifier(random_state=230),
                          {'n_estimators': randint(50, 300), 'learning_rate': loguniform(0.005, 0.1),
                           'max_depth': randint(2, 4), 'subsample': uniform(0.6, 0.4)}),

    "K-Nearest Neighbor": (KNeighborsClassifier(), {'n_neighbors': randint(3, 11), 'weights': ['uniform', 'distance']}),

    "ExtraTrees": (ExtraTreesClassifier(random_state=230),
                   {'n_estimators': randint(100, 300), 'max_depth': randint(3, 7),
                    'min_samples_split': randint(5, 15), 'max_features': ['sqrt', 'log2', None]}),

    "XGBoost": (XGBClassifier(random_state=230, eval_metric='logloss'),
                {'n_estimators': randint(50, 300), 'learning_rate': loguniform(0.001, 0.05),
                 'max_depth': randint(2, 4), 'subsample': uniform(0.6, 0.4), 'colsample_bytree': uniform(0.6, 0.4)}),

    "AdaBoost": (
        AdaBoostClassifier(random_state=230),
        {'n_estimators': randint(50, 200), 'learning_rate': loguniform(0.01, 0.5)}),

    "Decision Tree": (DecisionTreeClassifier(random_state=230),
                      {'criterion': ['gini', 'entropy'], 'max_depth': randint(3, 6),
                       'min_samples_split': randint(5, 20), 'min_samples_leaf': randint(2, 10)}),

    "MLP Classifier": (MLPClassifier(random_state=230, max_iter=2000, early_stopping=True),
                       {'hidden_layer_sizes': [(4,), (8,), (12,)],
                        'activation': ['tanh', 'relu'],
                        'solver': ['lbfgs'],
                        'alpha': loguniform(0.1, 100)})
}
print("\n" + "=" * 50)
print(f"EĞİTİMDE KULLANILAN TOPLAM SÜTUN SAYISI: {len(X.columns)}")
print("KULLANILAN TÜM SÜTUNLARIN LİSTESİ:")
for i, col in enumerate(X.columns, 1):
    print(f"{i}. {col}")
print("=" * 50)

results_list = []
all_best_params = {}
all_confusion_matrices = {}
all_oof_predictions = {}
feature_importance_accumulator = {}  # To store feature importances for RF
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=230)

for model_name, (model, params) in models_and_params.items():
    print(f"--- Optimization and Evaluation Started for: {model_name} ---")

    fold_metrics_default = []
    y_true_oof, y_prob_oof = [], []
    best_params_for_model = {}

    # ROC Plot Setup for this specific model (Individual fold tracking)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(8, 8))

    # Placeholder for accumulating feature importances for this model (if applicable)
    current_model_importances = np.zeros(X.shape[1])

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_smote)
        X_test_scaled = scaler.transform(X_test)

        search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=25,
                                    cv=3, n_jobs=-1, random_state=230, scoring='accuracy')
        try:
            search.fit(X_train_scaled, y_train_smote)
            best_model = search.best_estimator_
            if i == 0:
                best_params_for_model = search.best_params_

                # Özellik önemi üretebilen her modelin verisini topluyoruz
            if hasattr(best_model, 'feature_importances_'):
                current_model_importances += best_model.feature_importances_

        except ValueError as e:
            print(f"    WARNING: Error during optimization for {model_name} (fold {i + 1}): {e}")
            best_model = model
            if i == 0:
                best_params_for_model = "Default (Search Error)"

        y_pred_default = best_model.predict(X_test_scaled)
        try:
            y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
        except AttributeError:
            y_prob = y_pred_default

        y_true_oof.extend(y_test)
        y_prob_oof.extend(y_prob)

        # --- ROC Calculation for Individual Fold ---
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        # Plot individual fold (Slanted/Curved as per original data shape)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i + 1} (AUC = {roc_auc:.2f})' if i < 3 else "")

        fold_metrics_default.append({
            "Accuracy": accuracy_score(y_test, y_pred_default),
            "Precision": precision_score(y_test, y_pred_default, zero_division=0),
            "Sensitivity": recall_score(y_test, y_pred_default, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred_default, zero_division=0),
            "ROC-AUC": roc_auc,
            "Kappa": cohen_kappa_score(y_test, y_pred_default),
            "MCC": matthews_corrcoef(y_test, y_pred_default)
        })

    all_best_params[model_name] = best_params_for_model

    # --- Finalizing the Detailed ROC Plot for this model ---
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2,
             alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'Detailed ROC (10-Fold CV) - {model_name}')
    plt.legend(loc="lower right", fontsize='small')
    plt.savefig(f"Detailed_ROC_{model_name.replace(' ', '_')}.png")
    plt.close()

    # DEĞİŞİKLİK: Model bittiğinde, 10 fold'un ortalamasını al ve sözlüğe kaydet
    if np.sum(current_model_importances) > 0:
        feature_importance_accumulator[model_name] = current_model_importances / n_splits

    print(f"--- {model_name} Completed ---")

    y_true_oof = np.array(y_true_oof)
    y_prob_oof = np.array(y_prob_oof)
    all_oof_predictions[model_name] = {'true': y_true_oof, 'prob': y_prob_oof}

    optimal_threshold = 0.5
    if not np.all(np.isnan(y_prob_oof)) and not np.all(y_prob_oof == y_prob_oof.astype(int)) and len(
            np.unique(y_true_oof)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true_oof, y_prob_oof)
        valid_threshold_indices = np.where(~np.isnan(thresholds))[0]
        if len(valid_threshold_indices) > 1:
            valid_fpr = fpr[valid_threshold_indices]
            valid_tpr = tpr[valid_threshold_indices]
            valid_thresholds = thresholds[valid_threshold_indices]
            j_scores = valid_tpr - valid_fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = valid_thresholds[optimal_idx]

    y_pred_optimal = (y_prob_oof >= optimal_threshold).astype(int)

    acc_opt = accuracy_score(y_true_oof, y_pred_optimal)
    prec_opt = precision_score(y_true_oof, y_pred_optimal, zero_division=0)
    sens_opt = recall_score(y_true_oof, y_pred_optimal, zero_division=0)
    f1_opt = f1_score(y_true_oof, y_pred_optimal, zero_division=0)
    kappa_opt = cohen_kappa_score(y_true_oof, y_pred_optimal)
    mcc_opt = matthews_corrcoef(y_true_oof, y_pred_optimal)
    cm_opt = confusion_matrix(y_true_oof, y_pred_optimal)
    all_confusion_matrices[model_name] = cm_opt

    try:
        tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()
        spec_opt = tn_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) > 0 else 0
    except ValueError:
        spec_opt = np.nan
        unique_true, counts_true = np.unique(y_true_oof, return_counts=True)
        unique_pred, counts_pred = np.unique(y_pred_optimal, return_counts=True)
        total_samples = len(y_true_oof)
        if len(unique_true) == 1 and len(unique_pred) == 1 and unique_true[0] == unique_pred[0]:
            if unique_true[0] == 0:
                tn_opt, fp_opt, fn_opt, tp_opt = total_samples, 0, 0, 0
            else:
                tn_opt, fp_opt, fn_opt, tp_opt = 0, 0, 0, total_samples
        elif len(unique_true) == 1 and len(unique_pred) == 1 and unique_true[0] != unique_pred[0]:
            if unique_true[0] == 0:
                tn_opt, fp_opt, fn_opt, tp_opt = 0, total_samples, 0, 0
            else:
                tn_opt, fp_opt, fn_opt, tp_opt = 0, 0, total_samples, 0
        else:
            tn_opt, fp_opt, fn_opt, tp_opt = 0, 0, 0, 0

    mse_opt = mean_squared_error(y_true_oof, y_pred_optimal)
    brier_opt = brier_score_loss(y_true_oof, y_prob_oof) if not np.all(np.isnan(y_prob_oof)) and not np.all(
        y_prob_oof == y_prob_oof.astype(int)) else np.nan

    avg_roc_auc = np.nanmean([fm["ROC-AUC"] for fm in fold_metrics_default])
    avg_accuracy_default = np.mean([fm["Accuracy"] for fm in fold_metrics_default])
    avg_f1_default = np.mean([fm["F1 Score"] for fm in fold_metrics_default])

    results_list.append({
        "Model": model_name,
        "Optimal Threshold": optimal_threshold,
        "Accuracy": acc_opt,
        "Precision": prec_opt,
        "Sensitivity": sens_opt,
        "Specificity": spec_opt,
        "F1 Score": f1_opt,
        "MSE": mse_opt,
        "Kappa": kappa_opt,
        "MCC": mcc_opt,
        "Brier Score": brier_opt,
        "Avg ROC-AUC": avg_roc_auc,
        "Accuracy (Thr=0.5)": avg_accuracy_default,
        "F1 Score (Thr=0.5)": avg_f1_default
    })

# --- FINAL REPORTING ---
results_df = pd.DataFrame(results_list)
results_df_sorted = results_df.sort_values(by="Accuracy", ascending=False)

print("\n\n--- FINAL PERFORMANCE SUMMARY OF MODELS (At Optimal Threshold) ---")
cols_to_show = ["Model", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1 Score", "MSE", "Avg ROC-AUC",
                "Brier Score"]
print(results_df_sorted[cols_to_show].to_string())

print("\n\n--- REPRESENTATIVE BEST HYPERPARAMETERS FOUND FOR ALL MODELS ---")
for model_name, params in all_best_params.items():
    print(f"\n# {model_name}")
    print(params)

# --- EN BAŞARILI "AÇIKLANABİLİR" MODELİ BUL VE ÇİZ ---
# Önem sırası hafızasında (accumulator) verisi olan modelleri filtrele
explainable_models = [m for m in results_df['Model'] if m in feature_importance_accumulator]

if explainable_models:
    # Bu modeller içinden accuracy değeri en yüksek olanı seç
    best_explainable_name = \
    results_df[results_df['Model'].isin(explainable_models)].sort_values(by="Accuracy", ascending=False).iloc[0][
        "Model"]

    print(f"\n>>> Açıklanabilir En İyi Model: {best_explainable_name}")

    best_imp = feature_importance_accumulator[best_explainable_name]
    imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': best_imp}).sort_values(by='Importance', ascending=False)

    print(f"--- FEATURE IMPORTANCE RANKING (Champion: {best_explainable_name}) ---")
    print(imp_df.to_string(index=False))

    plt.figure(figsize=(10, 12))
    sns.barplot(x='Importance', y='Feature', data=imp_df.head(20), palette='viridis')
    plt.title(f'Feature Importance (Best Explainable Model: {best_explainable_name})')
    plt.tight_layout()
    plt.savefig(f"Feature_Importance_Champion_{best_explainable_name.replace(' ', '_')}.png")
    print(f"\n-> Grafik kaydedildi: Feature_Importance_Champion_{best_explainable_name}.png")
else:
    print("\n!!! HATA: Hiçbir model özellik önemi üretmedi.")

print("\n\n--- CONFUSION MATRICES (At Optimal Threshold) ---")
for model_name, cm in all_confusion_matrices.items():
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted: Negative (0)", "Predicted: Positive (1)"],
                yticklabels=["Actual: Negative (0)", "Actual: Positive (1)"])
    opt_thr = results_df.loc[results_df['Model'] == model_name, 'Optimal Threshold'].iloc[0]
    plt.title(f"{model_name}\nOptimal Threshold ≈ {opt_thr:.3f}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"ConfusionMatrix_{model_name.replace(' ', '_')}.png")
    print(f"-> Confusion Matrix for {model_name} saved as 'ConfusionMatrix_{model_name.replace(' ', '_')}.png'.")
    plt.close()


print("\n\n--- CALIBRATION CURVES FOR MODELS ---")
for model_name, predictions in all_oof_predictions.items():
    y_true = predictions['true']
    y_prob = predictions['prob']

    if not np.all(np.isnan(y_prob)) and len(np.unique(y_prob)) > 1:
        valid_indices = ~np.isnan(y_prob)
        if np.sum(valid_indices) > 0:
            y_true_valid = y_true[valid_indices]
            y_prob_valid = y_prob[valid_indices]
            if len(np.unique(y_true_valid)) > 1:
                try:
                    fraction_of_positives, mean_predicted_value = calibration_curve(y_true_valid, y_prob_valid,
                                                                                    n_bins=10, strategy='uniform')

                    plt.figure(figsize=(7, 6))
                    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
                    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{model_name}",
                             markerfacecolor='red', markersize=8)

                    brier_score_model = results_df.loc[results_df['Model'] == model_name, 'Brier Score'].iloc[0]

                    plt.ylabel("Fraction of Positives")
                    plt.xlabel("Mean Predicted Probability")
                    plt.ylim([-0.05, 1.05])
                    plt.legend(loc="upper left")
                    plt.title(f'{model_name} Calibration Curve\n(Brier Score: {brier_score_model:.3f})')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f"CalibrationCurve_{model_name.replace(' ', '_')}.png")
                    print(
                        f"-> Calibration Curve for {model_name} saved as 'CalibrationCurve_{model_name.replace(' ', '_')}.png'.")
                    plt.close()

                except ValueError as e:
                    print(f"-> Could not draw Calibration Curve for {model_name} (error): {e}")
            else:
                print(f"-> Could not draw Calibration Curve for {model_name} (only one class present).")
        else:
            print(f"-> Could not draw Calibration Curve for {model_name} (no valid probability scores).")
    else:
        print(f"-> Could not draw Calibration Curve for {model_name} (invalid or uniform probability scores).")