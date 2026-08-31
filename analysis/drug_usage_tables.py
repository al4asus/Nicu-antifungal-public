# -*- coding: utf-8 -*-
"""
İLAÇ KULLANIM TABLOLARI
=======================
Klinik ekibin talebi: "hangi hastalar kaç gün neleri kullanmış" + profilaksi/steroid.

Üretir:
  Table_DrugSummary.csv   Ajan bazında: hasta sayısı, medyan süre [ÇAA], min-maks
  Table_Exposures.csv     Profilaksi / steroid / antibiyotik maruziyetleri (n, %)
  Table_PatientDrugs.csv  Hasta bazında satır satır (Ek materyal; kimlik bilgisi YOK)
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import os
import pandas as pd
import numpy as np
import paper_primary_models as P

OUT = "Paper_Figures_Tables"
os.makedirs(OUT, exist_ok=True)

DRUG = {1: 'Micafungin', 2: 'Amphotericin B deoxycholate', 3: 'Liposomal amphotericin B',
        4: 'Fluconazole', 5: 'Caspofungin', 6: 'Nystatin'}


def med_iqr(s):
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return "—"
    return f"{s.median():.0f} [{s.quantile(.25):.0f}–{s.quantile(.75):.0f}]"


def main():
    df = P.load_df(dedup=False).copy()      # TUM kurler (tedavi tanimlayici tablolar)
    cohort = P.load_df()                     # bebek basina tek satir (maruziyet oranlari)
    df['bittiğitarih1'] = pd.to_datetime(df['bittiğitarih1'], errors='coerce')
    # Tedavi suresi: kayitli 'kacgüntedavi' esas alinir (tarih yazim hatalarina dayanikli);
    # bu alan bossa tarih farkina donulur.
    dur_dates = (df['bittiğitarih1'] - df['kullandıgıtarih1']).dt.days
    df['duration_1'] = df['kacgüntedavi'].where(df['kacgüntedavi'].notna(), dur_dates)
    # Tutarsizliklari raporla (tarih farki ile kayitli sure 5 gunden fazla ayrisiyorsa)
    diff = (dur_dates - df['kacgüntedavi']).abs()
    bad = df[(diff > 5) & df['kacgüntedavi'].notna()]
    if len(bad):
        print("UYARI - tedavi suresi tutarsizligi (Excel satiri: kayitli sure vs tarih farki):")
        for i, r in bad.iterrows():
            print(f"   Excel satir {r['excel_satir']}: kacgüntedavi={r['kacgüntedavi']:.0f} gun, "
                  f"tarihlerden {dur_dates.loc[i]:.0f} gun "
                  f"({r['kullandıgıtarih1'].date()} -> {r['bittiğitarih1'].date()})")
        print()
    df['agent_1'] = df['ilaç'].map(DRUG)
    df['duration_2'] = df['kacgunmicafungin2']
    n = len(df)
    print(f"Kohort n={n}\n")

    # ---- 1) Ajan bazinda ozet ----
    rows = []
    for code, name in DRUG.items():
        sub = df[df['ilaç'] == code]
        if len(sub) == 0:
            continue
        d = sub['duration_1'].dropna()
        rows.append({
            'Antifungal agent': name,
            'Courses, n (%)': f"{len(sub)} ({100*len(sub)/n:.1f}%)",
            'Duration, days — median [IQR]': med_iqr(d),
            'Duration, days — min–max': f"{d.min():.0f}–{d.max():.0f}" if len(d) else "—",
        })
    summ = pd.DataFrame(rows).sort_values('Courses, n (%)',
                                          key=lambda s: s.str.split(' ').str[0].astype(int),
                                          ascending=False)
    alld = df['duration_1'].dropna()
    summ.loc[len(summ)] = {'Antifungal agent': 'All courses',
                           'Courses, n (%)': f"{n} (100%)",
                           'Duration, days — median [IQR]': med_iqr(alld),
                           'Duration, days — min–max': f"{alld.min():.0f}–{alld.max():.0f}"}
    summ.to_csv(os.path.join(OUT, "Table_DrugSummary.csv"), index=False, encoding='utf-8-sig')
    print("=== Ajan bazinda ===")
    print(summ.to_string(index=False), flush=True)

    # ---- 2) Maruziyetler ----
    nc = len(cohort)

    def npct(col, pos=1):
        s = cohort[col].dropna()
        k = int((s == pos).sum())
        return f"{k} ({100*k/len(s):.1f}%)" if len(s) else "—"

    exp = pd.DataFrame([
        {'Exposure': 'Fluconazole prophylaxis', 'n (%)': npct('flukanazolprofilaksisi')},
        {'Exposure': 'Postnatal systemic steroid', 'n (%)': npct('steroidkullanimi')},
        {'Exposure': 'Antenatal steroid', 'n (%)': npct('antenatalsteroid', 1)},
        {'Exposure': 'Multiple antibiotic courses', 'n (%)': npct('cokluantibiyotikkullanimi')},
        {'Exposure': 'More than one antifungal course',
         'n (%)': f"{int((cohort['kur_sayisi'] >= 2).sum())} ({100*(cohort['kur_sayisi'] >= 2).mean():.1f}%)"},
        {'Exposure': 'Three antifungal courses',
         'n (%)': f"{int((cohort['kur_sayisi'] >= 3).sum())} ({100*(cohort['kur_sayisi'] >= 3).mean():.1f}%)"},
    ])
    exp.to_csv(os.path.join(OUT, "Table_Exposures.csv"), index=False, encoding='utf-8-sig')
    print("\n=== Maruziyetler ===")
    print(exp.to_string(index=False), flush=True)

    # ---- 3) Hasta bazinda (kimlik bilgisi olmadan) ----
    pat = pd.DataFrame({
        'Patient': df['hasta_id'].map({h: f"P{j+1:02d}" for j, h in enumerate(sorted(cohort['hasta_id']))}).values,
        'Course no.': df.groupby('hasta_id').cumcount().values + 1,
        'GA, weeks': df['gebelikhaftası'].values,
        'Birth weight, g': df['dogumagirligi'].values,
        'Sex': np.where(df['cinsiyeti'].values == 2, 'M', 'F'),
        'Age at start, days': df['PNA_t0'].values,
        'Agent': df['agent_1'].values,
        'Duration, days': df['duration_1'].values,
        'Second-line, days': df['duration_2'].values,
        'Fluconazole prophylaxis': np.where(df['flukanazolprofilaksisi'].values == 1, 'Yes', 'No'),
        'Systemic steroid': np.where(df['steroidkullanimi'].values == 1, 'Yes', 'No'),
        'Died': np.where(df['ex'].values == 1, 'Yes', 'No'),
        'BPD': np.where(df['BPD'].values == 1, 'Yes', 'No'),
        'ROP': np.where(df['rop'].values == 1, 'Yes', 'No'),
    })
    pat.to_csv(os.path.join(OUT, "Table_PatientCourses.csv"), index=False, encoding='utf-8-sig')
    print(f"\n=== Hasta bazinda tablo: {len(pat)} satir -> Table_PatientCourses.csv ===")
    print(pat.head(6).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
