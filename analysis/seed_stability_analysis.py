# -*- coding: utf-8 -*-
"""Cok-tohumlu kararlilik analizi (multi-seed stability analysis).

Amac: Bildirilen AUC'lerin ve sampiyon algoritma secimlerinin ne kadari
gercek sinyal, ne kadari tek bir rastgele bolunmenin sansi?

Ana analizde her sey tek bir tohuma (paper_primary_models.R = 230) bagli:
  1. 10-katli capraz dogrulamanin bolunmesi
  2. SMOTE'un urettigi sentetik ornekler
  3. RandomizedSearchCV'nin denedigi 25 hiperparametre kombinasyonu
  4. Algoritmalarin kendi ic rastgeleligi (random_state)

Bu betik ayni pipeline'i N farkli tohumla tekrar eder ve iki dagilim raporlar:
  (a) Sampiyon AUC  - her tohumda kazanan algoritmanin AUC'si
  (b) Sabit model AUC - makalede bildirilen algoritmanin AUC'si (asil onemli olan)

ANA ANALIZI DEGISTIRMEZ. Ciktilar Seed_Stability/ altina yazilir; makaleye
duyarlilik analizi olarak eklenir.
"""
import os
import sys
import time
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

import paper_primary_models as P  # load_df, prep, build_models, NOSEV, TARGETS

OUT = "Seed_Stability"
os.makedirs(OUT, exist_ok=True)

# Ana analizin tohumu (230) referans olarak ILK sirada tutulur.
SEEDS = [230, 7, 13, 21, 42, 77, 101, 123, 202, 314,
         404, 512, 606, 777, 808, 909, 1001, 1234, 2024, 3407]

# Makalede bildirilen sampiyonlar (referans tohumdan). Sabit model dagilimi
# bunlar uzerinden hesaplanir.
REPORTED = {"Mortalite": "Gradient Boosting",
            "BPD": "Random Forest",
            "ROP": "Random Forest"}


def run_cv_seeded(X, y, model, params, seed):
    """paper_primary_models.run_cv ile ayni pipeline, ama TUM rastgelelik
    kaynaklari `seed` ile kontrol edilir (SMOTE dahil)."""
    skf = StratifiedKFold(10, shuffle=True, random_state=seed)
    yt, yp = [], []
    for tri, tei in skf.split(X, y):
        Xtr, Xte = X.iloc[tri], X.iloc[tei]
        ytr, yte = y.iloc[tri], y.iloc[tei]
        Xsm, ysm = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
        sc = StandardScaler()
        Xtr_s, Xte_s = sc.fit_transform(Xsm), sc.transform(Xte)
        try:
            s = RandomizedSearchCV(model, params, n_iter=25, cv=3, n_jobs=-1,
                                   random_state=seed, scoring='accuracy')
            s.fit(Xtr_s, ysm)
            est = s.best_estimator_
        except Exception:
            est = model
            est.fit(Xtr_s, ysm)
        try:
            prob = est.predict_proba(Xte_s)[:, 1]
        except Exception:
            prob = est.predict(Xte_s).astype(float)
        yt.extend(yte.tolist())
        yp.extend(prob.tolist())
    return roc_auc_score(np.array(yt), np.array(yp))


def main():
    t_start = time.time()
    df = P.load_df()
    print(f"Kohort: {len(df)} bebek", flush=True)
    print(f"Tohumlar: {SEEDS}", flush=True)
    print(f"Degisken seti: NOSEV ({len(P.NOSEV)}) -> {P.NOSEV}\n", flush=True)

    rows = []
    long_path = os.path.join(OUT, "seed_results_long.csv")

    for si, seed in enumerate(SEEDS, 1):
        # build_models() modul duzeyindeki R'yi okur; algoritmalarin kendi
        # random_state'i de tohumla birlikte degissin diye gecici olarak ayarla.
        P.R = seed
        models = P.build_models()

        for name, col in P.TARGETS:
            X, y, dropped = P.prep(df, col, P.NOSEV)
            t0 = time.time()
            for algo, (mdl, prm) in models.items():
                auc = run_cv_seeded(X, y, mdl, prm, seed)
                rows.append({"seed": seed, "outcome": name, "algorithm": algo,
                             "auc": round(float(auc), 4),
                             "n": int(len(y)), "events": int(y.sum())})
            best = max([r for r in rows if r["seed"] == seed and r["outcome"] == name],
                       key=lambda r: r["auc"])
            print(f"[{si:2d}/{len(SEEDS)}] seed={seed:<5d} {name:<10s} "
                  f"kazanan={best['algorithm']:<26s} AUC={best['auc']:.3f} "
                  f"({time.time()-t0:.0f} sn)", flush=True)

        # Her tohumdan sonra ara sonucu diske yaz (uzun kosuda guvenlik)
        pd.DataFrame(rows).to_csv(long_path, index=False, encoding='utf-8-sig')

    P.R = 230  # modulu eski haline dondur

    long = pd.DataFrame(rows)
    long.to_csv(long_path, index=False, encoding='utf-8-sig')

    # ---- Ozet 1: her tohumda kazanan algoritma ve AUC'si
    champ = (long.sort_values("auc", ascending=False)
                 .groupby(["outcome", "seed"], as_index=False).first())
    champ.to_csv(os.path.join(OUT, "champion_per_seed.csv"),
                 index=False, encoding='utf-8-sig')

    # ---- Ozet 2: algoritma kac tohumda kazandi
    counts = (champ.groupby(["outcome", "algorithm"]).size()
                   .reset_index(name="wins")
                   .sort_values(["outcome", "wins"], ascending=[True, False]))
    counts["of_seeds"] = len(SEEDS)
    counts.to_csv(os.path.join(OUT, "champion_counts.csv"),
                  index=False, encoding='utf-8-sig')

    # ---- Ozet 3: outcome basina dagilim (hem sampiyon hem sabit model)
    summ = []
    for name, _ in P.TARGETS:
        c = champ[champ.outcome == name]["auc"]
        fixed_algo = REPORTED[name]
        f = long[(long.outcome == name) & (long.algorithm == fixed_algo)]["auc"]
        ref = long[(long.outcome == name) & (long.algorithm == fixed_algo) &
                   (long.seed == 230)]["auc"]
        summ.append({
            "outcome": name,
            "reported_model": fixed_algo,
            "reported_auc_seed230": float(ref.iloc[0]) if len(ref) else np.nan,
            "champion_auc_median": round(float(c.median()), 3),
            "champion_auc_min": round(float(c.min()), 3),
            "champion_auc_max": round(float(c.max()), 3),
            "champion_auc_iqr": f"{c.quantile(.25):.3f}-{c.quantile(.75):.3f}",
            "fixed_auc_median": round(float(f.median()), 3),
            "fixed_auc_min": round(float(f.min()), 3),
            "fixed_auc_max": round(float(f.max()), 3),
            "fixed_auc_iqr": f"{f.quantile(.25):.3f}-{f.quantile(.75):.3f}",
            "reported_model_wins": int((champ[(champ.outcome == name)]["algorithm"]
                                        == fixed_algo).sum()),
            "n_seeds": len(SEEDS),
        })
    summary = pd.DataFrame(summ)
    summary.to_csv(os.path.join(OUT, "summary_by_outcome.csv"),
                   index=False, encoding='utf-8-sig')

    # ---- Figur: tohumlar arasi AUC dagilimi
    labels = {"Mortalite": "Mortality", "BPD": "BPD", "ROP": "ROP"}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    data, ticks = [], []
    for i, (name, _) in enumerate(P.TARGETS):
        fixed_algo = REPORTED[name]
        f = long[(long.outcome == name) & (long.algorithm == fixed_algo)]["auc"].values
        data.append(f)
        ticks.append(f"{labels[name]}\n({fixed_algo})")
    bp = ax.boxplot(data, labels=ticks, widths=0.5, patch_artist=True,
                    medianprops=dict(color='black', lw=1.6))
    for patch in bp['boxes']:
        patch.set_facecolor('#cfe2f3')
        patch.set_edgecolor('#3d5a80')
    for i, f in enumerate(data, 1):
        x = np.random.default_rng(1).normal(i, 0.045, len(f))
        ax.plot(x, f, 'o', ms=4, color='#3d5a80', alpha=.55, zorder=3)
    for i, (name, _) in enumerate(P.TARGETS, 1):
        r = summary[summary.outcome == name]["reported_auc_seed230"].iloc[0]
        ax.plot([i - .3, i + .3], [r, r], color='#c1121f', lw=2, zorder=4,
                label='Reported analysis (seed 230)' if i == 1 else None)
    ax.axhline(0.5, ls='--', color='grey', lw=1, label='Chance level')
    ax.set_ylabel('Out-of-fold AUC')
    ax.set_ylim(0.35, 1.0)
    ax.set_title(f'Stability of model performance across {len(SEEDS)} random seeds',
                 fontsize=11)
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "FigureS2_SeedStability.png"), dpi=300)
    fig.savefig(os.path.join("Paper_Figures_Tables", "FigureS2_SeedStability.png"), dpi=300)
    plt.close(fig)

    print("\n" + "=" * 78, flush=True)
    print("OZET (sabit = makalede bildirilen algoritma)", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nKAZANMA SAYILARI", flush=True)
    print(counts.to_string(index=False), flush=True)
    print(f"\nToplam sure: {(time.time()-t_start)/60:.1f} dk", flush=True)
    print(f"Ciktilar -> {OUT}/", flush=True)


if __name__ == "__main__":
    main()
