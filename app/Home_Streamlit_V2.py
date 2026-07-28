import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import joblib
import requests

# =========================
# AYARLAR VE GITHUB YAPILANDIRMASI
# =========================
# Token is read from Streamlit secrets or the environment; never hard-code it.
# Streamlit Cloud: add GITHUB_TOKEN under App settings -> Secrets
# Local: set the GITHUB_TOKEN environment variable
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", os.environ.get("GITHUB_TOKEN", ""))
REPO_NAME = "al4asus/streamlit-app"
KM_PATH = "Kaplan_Meier_Curve.png"

@st.cache_resource(show_spinner=False)
def load_file_from_github(token: str, repo_name: str, file_path: str, is_json=False):
    url = f"https://raw.githubusercontent.com/{repo_name}/main/{file_path}"
    headers = {"Authorization": f"token {token}"} if token else {}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    if is_json:
        return r.json()
    return BytesIO(r.content)

@st.cache_data(show_spinner=False)
def get_metadatas():
    meta_mortality = load_file_from_github(GITHUB_TOKEN, REPO_NAME, "model_meta.json", is_json=True)
    meta_new = load_file_from_github(GITHUB_TOKEN, REPO_NAME, "streamlit_meta_new.json", is_json=True)
    return meta_mortality, meta_new

@st.cache_resource(show_spinner=False)
def get_all_models():
    models = {}
    with st.spinner("AI Modelleri Yükleniyor (Mortalite, BPD, ROP)... Lütfen bekleyin."):
        # Mortalite
        models['mortality'] = joblib.load(load_file_from_github(GITHUB_TOKEN, REPO_NAME, "model.pkl"))
        # Yeni BPD ve ROP modelleri
        models['bpd_primary'] = joblib.load(load_file_from_github(GITHUB_TOKEN, REPO_NAME, "bpd_primary_model.pkl"))
        models['bpd_secondary'] = joblib.load(load_file_from_github(GITHUB_TOKEN, REPO_NAME, "bpd_secondary_model.pkl"))
        models['rop_primary'] = joblib.load(load_file_from_github(GITHUB_TOKEN, REPO_NAME, "rop_primary_model.pkl"))
        models['rop_secondary'] = joblib.load(load_file_from_github(GITHUB_TOKEN, REPO_NAME, "rop_secondary_model.pkl"))
    return models

st.set_page_config(page_title="NICU Fungal Infection Predictor", layout="centered", page_icon="🩺")

# BAŞLIK
st.title("NICU Fungal Infection: Complication & Mortality Predictor")

# MODELLERİ YÜKLE
try:
    models = get_all_models()
    meta_mortality, meta_new = get_metadatas()
except Exception as e:
    st.error(f"Sistem dosyaları yüklenemedi. Github deponuzda tüm .pkl ve .json dosyalarının (bpd, rop vb.) tam olduğuna emin olun. Hata: {e}")
    st.stop()

# =========================
# YAN MENÜ (SIDEBAR)
# =========================
st.sidebar.title("Navigation")
st.sidebar.markdown("Predictive AI Models")
app_mode = st.sidebar.radio("Select Prediction Target:", 
                            ["💀 Mortality (Ex)", "🫁 Bronchopulmonary Dysplasia (BPD)", "👁️ Retinopathy of Prematurity (ROP)"])

# Seçilen arayüze göre model tanımlamaları
active_model_key = None
active_threshold = 0.5
active_features = []
is_secondary = False

if "Mortality" in app_mode:
    st.header("Mortality Prediction")
    active_model_key = 'mortality'
    active_threshold = float(meta_mortality.get("optimal_threshold", 0.5))
    active_features = meta_mortality.get("features", [])
    st.info("Uses admission (t0) clinical features to predict mortality risk.")
else:
    if "BPD" in app_mode:
        st.header("BPD Complication Risk")
        prefix = "bpd"
    else:
        st.header("ROP Complication Risk")
        prefix = "rop"
        
    phase_selection = st.radio("Clinical Phase:", ["Phase 1: At Diagnosis (Primary t0)", "Phase 2: Post-Treatment (Secondary Course)"], horizontal=True)
    is_secondary = "Phase 2" in phase_selection
    
    active_model_key = f"{prefix}_secondary" if is_secondary else f"{prefix}_primary"
    active_threshold = float(meta_new[active_model_key]['optimal_threshold'])
    active_features = meta_new[active_model_key]['features']
    
st.markdown("---")

# =========================
# ORTAK FORMLAR (8 DÜRÜST DEĞİŞKEN)
# =========================
with st.form("clinical_form"):
    st.subheader("Baseline Diagnostics (t0)")
    c1, c2 = st.columns(2)
    with c1:
        gebelikhaftası = st.number_input("Gestational Age (weeks):", min_value=20.0, max_value=45.0, value=28.0, step=0.1)
        cinsiyet_input = st.radio("Gender:", ["Female (0)", "Male (1)"])
        apgar5 = st.number_input("5-Minute Apgar:", min_value=0.0, max_value=10.0, value=7.0, step=1.0)
        cerrahi_input = st.radio("Surgical Intervention?:", ["No (0)", "Yes (1)"])
    with c2:
        PNA_t0 = st.number_input("Postnatal Age at Antifungal (Days):", min_value=0.0, max_value=200.0, value=5.0)
        crp1 = st.number_input("Initial CRP (mg/L):", min_value=0.0, max_value=500.0, value=10.0, step=1.0)
        proc1 = st.number_input("Initial Procalcitonin (ng/mL):", min_value=0.0, max_value=200.0, value=2.0, step=0.2)
        sonda_input = st.radio("Urinary Catheter?:", ["No (0)", "Yes (1)"])

    cinsiyeti = 1 if "Male" in cinsiyet_input else 0
    cerrahi = 1 if "Yes" in cerrahi_input else 0
    idrarsondasi = 1 if "Yes" in sonda_input else 0

    # TEDAVİ SEYRİ SORULARI (Sadece Secondary ise görünür/alınır)
    sec_anti_2, sec_anti_3, esc, mult = 0, 0, 0, 0
    if is_secondary and "Mortality" not in app_mode:
        st.markdown("---")
        st.subheader("Treatment Escalation (Course Variables)")
        st.error("Select these only if the patient required treatment escalation or prolonged therapy.")
        
        c3, c4 = st.columns(2)
        with c3:
            anti2_in = st.checkbox("Was a 2nd Antifungal Line Started?")
            anti3_in = st.checkbox("Was a 3rd Antifungal Line Started?")
        
        sec_anti_2 = 1 if anti2_in else 0
        sec_anti_3 = 1 if anti3_in else 0
        esc = 1 if (sec_anti_2 or sec_anti_3) else 0
        mult = esc  # aynı matematiksel karşılık
    
    submitted = st.form_submit_button("Predict Result")

# =========================
# TAHMİN MONTE CARLO SİMÜLASYONU
# =========================
if submitted:
    try:
        model = models[active_model_key]
        n_bootstrap = 10_000
        rng = np.random.default_rng(42)

        # Baseline Gürültüleri
        b_gebelik = np.clip(gebelikhaftası + rng.normal(0, 0.2, n_bootstrap), 20, 45)
        b_pna = np.clip(PNA_t0 + rng.normal(0, 1.0, n_bootstrap), 0, 200)
        b_crp = np.clip(crp1 + rng.normal(0, 1.5, n_bootstrap), 0, None)
        b_proc = np.clip(proc1 + rng.normal(0, 0.3, n_bootstrap), 0, None)
        b_apgar = np.clip(np.round(apgar5 + rng.normal(0, 0.5, n_bootstrap)), 0, 10)

        # DataFrame Hazırlığı
        input_data = {
            "gebelikhaftası": b_gebelik,
            "cinsiyeti": np.full(n_bootstrap, cinsiyeti),
            "apgar5": b_apgar,
            "cerrahi": np.full(n_bootstrap, cerrahi),
            "idrarsondasi": np.full(n_bootstrap, idrarsondasi),
            "PNA_t0": b_pna,
            "crp1": b_crp,
            "proc1": b_proc
        }
        
        # Eğer Secondary ise eskalasyon değişkenlerini de ekle
        if is_secondary and "Mortality" not in app_mode:
            input_data["second_antifungal_started"] = np.full(n_bootstrap, sec_anti_2)
            input_data["third_antifungal_started"] = np.full(n_bootstrap, sec_anti_3)
            input_data["antifungal_escalation"] = np.full(n_bootstrap, esc)
            input_data["multiple_antifungal_lines"] = np.full(n_bootstrap, mult)
            
        sim_df = pd.DataFrame(input_data)
        
        # Meta'dan gelen feature sırasına zorla
        if active_features:
            sim_df = sim_df[active_features]
            
        # Model Tahmini
        preds = model.predict_proba(sim_df)[:, 1]
        
        # %95 Güven Aralığı Çıkarımı (Prediction Interval)
        lower = float(np.percentile(preds, 2.5))
        upper = float(np.percentile(preds, 97.5))
        mean_pred = float(np.mean(preds))

        # =========================
        # EKRAN ÇIKTILARI
        # =========================
        st.markdown("---")
        st.subheader("Prediction Results")
        
        is_high_risk = mean_pred >= active_threshold
        risk_color = "red" if is_high_risk else "green"
        
        if "Mortality" in app_mode:
            lbl_safe = "Survival Probability"
            lbl_risk = "Mortality Risk"
        elif "BPD" in app_mode:
            lbl_safe = "BPD-Free Probability"
            lbl_risk = "BPD Risk"
        else:
            lbl_safe = "ROP-Free Probability"
            lbl_risk = "ROP Risk"

        c1, c2 = st.columns(2)
        with c1:
            st.metric(label=lbl_safe, value=f"{(1.0 - mean_pred)*100:.1f}%")
        with c2:
            st.metric(label=lbl_risk, value=f"{mean_pred*100:.1f}%")

        if is_high_risk:
            st.error(f"⚠️ HIGH RISK: The predicted probability is above the clinical threshold ({active_threshold*100:.1f}%).")
        else:
            st.success(f"Low Risk: The predicted probability is under the clinical threshold.")

        st.info(f"🔹 **95% Confidence (Prediction) Interval:** [{lower*100:.1f}% — {upper*100:.1f}%]")

        # Grafik
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot([lower*100, upper*100], [0, 0], linewidth=8, solid_capstyle='round', label='95% CI', color='#3498db')
        ax.plot(mean_pred*100, 0, 'o', markersize=12, label=f'Risk: {mean_pred*100:.1f}%', color=risk_color)
        ax.axvline(x=active_threshold*100, color='gray', linestyle='--', label=f'Threshold ({active_threshold*100:.1f}%)')
        ax.set_xlim(0, 100); ax.set_ylim(-1, 1); ax.set_yticks([])
        ax.set_xlabel('Event Probability (%)', fontsize=12)
        ax.set_title(f'Statistical Risk Distribution ({app_mode.split(" ")[1]})', fontsize=14, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction Error: Lütfen tüm özelliklerin doğru yüklendiğinden emin olun. Hata: {e}")

st.markdown("---")
if "Mortality" in app_mode:
    with st.expander("Genişletilmiş Veri: Genel Popülasyon Sağkalımı (Kaplan-Meier Eğrisi)"):
        st.info("Aşağıdaki eğri, seçtiğimiz modelin eğitildiği genel sağkalım tablosunu göstermektedir.")
        try:
            km_data = load_file_from_github(GITHUB_TOKEN, REPO_NAME, KM_PATH)
            st.image(km_data, use_container_width=True)
        except Exception as e:
            pass

st.caption("Warning: This model is a clinical decision support tool. It should not replace medical professional judgment.")
