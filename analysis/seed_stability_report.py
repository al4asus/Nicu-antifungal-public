# -*- coding: utf-8 -*-
"""seed_results_long.csv'den ozet tablolar ve figur uretir.

Koşudan ayri tutuldu ki raporlama bicimi degistiginde 70 dakikalik analiz
bastan calistirilmasin.

Figurdeki referans cizgisi FINAL_PAPER_MODELS.csv'den okunur. Kararlilik
kosusu SMOTE'u da tohuma bagladigi icin (ana analizde sabit 42) tohum=230
satiri ana analizle birebir ayni degil.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
SS = "Seed_Stability"
LONG = os.path.join(SS, "seed_results_long.csv")

long = pd.read_csv(LONG, encoding='utf-8-sig')
fin = pd.read_csv(os.path.join("Paper_Primary_Models", "FINAL_PAPER_MODELS.csv"),
                  encoding='utf-8-sig')
MAIN = {r['Outcome']: {'model': r['Champion'], 'auc': float(r['OOF_AUC'])}
        for _, r in fin.iterrows()}
ORDER = ["Mortalite", "BPD", "ROP"]
EN = {"Mortalite": "Mortality", "BPD": "BPD", "ROP": "ROP"}

seeds = sorted(long['seed'].unique())
n_seeds = len(seeds)
print(f"{n_seeds} tohum, {len(long)} satir")

# --- her tohumda kazanan
champ = (long.sort_values("auc", ascending=False)
             .groupby(["outcome", "seed"], as_index=False).first())
champ.to_csv(os.path.join(SS, "champion_per_seed.csv"), index=False, encoding='utf-8-sig')

counts = (champ.groupby(["outcome", "algorithm"]).size().reset_index(name="wins"))
counts["of_seeds"] = n_seeds
counts["outcome"] = pd.Categorical(counts["outcome"], ORDER, ordered=True)
counts = counts.sort_values(["outcome", "wins"], ascending=[True, False])
counts.to_csv(os.path.join(SS, "champion_counts.csv"), index=False, encoding='utf-8-sig')

# --- ozet
rows = []
for name in ORDER:
    m = MAIN[name]
    f = long[(long.outcome == name) & (long.algorithm == m['model'])]["auc"]
    c = champ[champ.outcome == name]["auc"]
    wins = int((champ[champ.outcome == name]["algorithm"] == m['model']).sum())
    rows.append({
        "outcome": name,
        "reported_model": m['model'],
        "reported_auc_main": f"{m['auc']:.3f}",
        "fixed_auc_median": f"{f.median():.3f}",
        "fixed_auc_min": f"{f.min():.3f}",
        "fixed_auc_max": f"{f.max():.3f}",
        "fixed_auc_iqr": f"{f.quantile(.25):.3f}–{f.quantile(.75):.3f}",
        "champion_auc_median": f"{c.median():.3f}",
        "champion_auc_min": f"{c.min():.3f}",
        "champion_auc_max": f"{c.max():.3f}",
        "reported_model_wins": wins,
        "n_seeds": n_seeds,
    })
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(SS, "summary_by_outcome.csv"), index=False, encoding='utf-8-sig')

# --- figur
fig, ax = plt.subplots(figsize=(7.5, 5))
data, ticks = [], []
for name in ORDER:
    m = MAIN[name]
    data.append(long[(long.outcome == name) & (long.algorithm == m['model'])]["auc"].values)
    ticks.append(f"{EN[name]}\n({m['model']})")
bp = ax.boxplot(data, tick_labels=ticks, widths=0.5, patch_artist=True,
                medianprops=dict(color='black', lw=1.6))
for patch in bp['boxes']:
    patch.set_facecolor('#cfe2f3'); patch.set_edgecolor('#3d5a80')
rng = np.random.default_rng(1)
for i, f in enumerate(data, 1):
    ax.plot(rng.normal(i, 0.045, len(f)), f, 'o', ms=4, color='#3d5a80', alpha=.55, zorder=3)
for i, name in enumerate(ORDER, 1):
    r = MAIN[name]['auc']
    ax.plot([i - .3, i + .3], [r, r], color='#c1121f', lw=2.2, zorder=4,
            label='Reported analysis' if i == 1 else None)
ax.axhline(0.5, ls='--', color='grey', lw=1, label='Chance level')
ax.set_ylabel('Out-of-fold AUC')
ax.set_ylim(0.30, 1.0)
ax.set_title(f'Stability of model performance across {n_seeds} random seeds', fontsize=11)
ax.legend(loc='lower right', fontsize=8, frameon=False)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(SS, "FigureS2_SeedStability.png"), dpi=300)
fig.savefig(os.path.join("Paper_Figures_Tables", "FigureS2_SeedStability.png"), dpi=300)
plt.close(fig)

print("\n" + summary.to_string(index=False))
print("\n" + counts.to_string(index=False))
