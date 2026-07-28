import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve
import warnings

warnings.filterwarnings('ignore')

# 8 Dürüst Değişken (Mortalite modelindekilerle birebir örtüşen)
HONEST_FEATURES = ['gebelikhaftası', 'cinsiyeti', 'apgar5', 'cerrahi', 'idrarsondasi', 'PNA_t0', 'crp1', 'proc1']

# 4 Antifungal Eskalasyon Türevi (Secondary modeller için)
SEC_FEATURES = ['second_antifungal_started', 'third_antifungal_started', 'antifungal_escalation', 'multiple_antifungal_lines']

def get_base_df():
    df = pd.read_excel('micafunginson.xlsx', na_values=['#BOŞ!', '#NULL!'])
    date_cols = ['kullandıgıtarih1', 'takibegirdigitarih', 'taburculuktarihi']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df['PNA_t0'] = (df['kullandıgıtarih1'] - df['takibegirdigitarih']).dt.days
    df = df[(df['PNA_t0'] >= 0) & (df['PNA_t0'] <= 200)]
    return df

def get_secondary_derivatives(df):
    date_cols_sec = ['kullandıgıtarih2', 'kullandıgıtarih3']
    for col in date_cols_sec:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df['second_antifungal_started'] = ((df['kacgunmicafungin2'].fillna(0) > 0) | (df['kullandıgıtarih2'].notna())).astype(int)
    df['third_antifungal_started'] = ((df['kacgunkullandi3'].fillna(0) > 0) | (df['kullandıgıtarih3'].notna())).astype(int)
    df['antifungal_escalation'] = ((df['second_antifungal_started'] == 1) | (df['third_antifungal_started'] == 1)).astype(int)
    df['multiple_antifungal_lines'] = df['antifungal_escalation']
    return df

def prep_dataset(df, target_col, features):
    if target_col == 'BPD':
        df['event'] = df['BPD']
    elif target_col == 'rop':
        df['event'] = df['rop']

    df['time'] = (df['taburculuktarihi'] - df['kullandıgıtarih1']).dt.days
    df = df[df['time'] > 0]
    df = df.dropna(subset=[target_col])
    
    y = df[target_col].copy().astype(int)
    
    X = df[features].copy()
    # Eksikleri medyan ile doldur
    for c in X.columns:
        X[c] = X[c].fillna(X[c].median())
    
    return X, y

def train_and_export(model, X, y, filename):
    sm = SMOTE(random_state=42)
    Xsm, ysm = sm.fit_resample(X, y)
    
    model.fit(Xsm, ysm)
    
    y_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, thr = roc_curve(y, y_prob)
    opt_idx = np.argmax(tpr - fpr)
    opt_threshold = float(thr[opt_idx])
    
    joblib.dump(model, filename)
    print(f"{filename} oluşturuldu. Thr: {opt_threshold}")
    return X.columns.tolist(), opt_threshold

meta_registry = {}
df_base = get_base_df()
df_sec = get_secondary_derivatives(df_base.copy())

print("BPD Primary İşleniyor...")
X_bp, y_bp = prep_dataset(df_base.copy(), 'BPD', HONEST_FEATURES)
model_bp = ExtraTreesClassifier(random_state=230, n_estimators=200, max_depth=5)
feat_bp, thr_bp = train_and_export(model_bp, X_bp, y_bp, 'bpd_primary_model.pkl')
meta_registry['bpd_primary'] = {'features': feat_bp, 'optimal_threshold': thr_bp}

print("BPD Secondary İşleniyor...")
X_bs, y_bs = prep_dataset(df_sec.copy(), 'BPD', HONEST_FEATURES + SEC_FEATURES)
model_bs = RandomForestClassifier(random_state=230, n_estimators=200)
feat_bs, thr_bs = train_and_export(model_bs, X_bs, y_bs, 'bpd_secondary_model.pkl')
meta_registry['bpd_secondary'] = {'features': feat_bs, 'optimal_threshold': thr_bs}

print("ROP Primary İşleniyor...")
X_rp, y_rp = prep_dataset(df_base.copy(), 'rop', HONEST_FEATURES)
model_rp = ExtraTreesClassifier(random_state=230, n_estimators=200)
feat_rp, thr_rp = train_and_export(model_rp, X_rp, y_rp, 'rop_primary_model.pkl')
meta_registry['rop_primary'] = {'features': feat_rp, 'optimal_threshold': thr_rp}

print("ROP Secondary İşleniyor...")
X_rs, y_rs = prep_dataset(df_sec.copy(), 'rop', HONEST_FEATURES + SEC_FEATURES)
model_rs = RandomForestClassifier(random_state=230, n_estimators=200)
feat_rs, thr_rs = train_and_export(model_rs, X_rs, y_rs, 'rop_secondary_model.pkl')
meta_registry['rop_secondary'] = {'features': feat_rs, 'optimal_threshold': thr_rs}

with open('streamlit_meta_new.json', 'w', encoding='utf-8') as f:
    json.dump(meta_registry, f, indent=4, ensure_ascii=False)
    
print("Tüm dürüst klinik modeller başarıyla export edildi!")
