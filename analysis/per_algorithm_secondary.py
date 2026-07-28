# -*- coding: utf-8 -*-
"""
SECONDARY modeller için her-algoritma görselleri (keşifsel/ek — makaleye girmiyor).
Feature set = nihai 10 baseline + 4 eskalasyon değişkeni (sabit olanlar otomatik düşer).
NOT: Kohortta yalnızca 2/56 hasta eskale olduğu için eskalasyon değişkenleri ~sabittir;
bu modeller Primary'nin neredeyse aynısıdır ve yalnızca bütünlük için üretilmiştir.
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import paper_primary_models as P
from per_algorithm_figures import plots_for, PARENT

SEC = ['second_antifungal_started', 'third_antifungal_started',
       'antifungal_escalation', 'multiple_antifungal_lines']
FEATS = P.NOSEV + SEC
OUTCOMES = [("BPD", 'BPD', "BPD_Secondary_Results", "BPD_Secondary"),
            ("ROP", 'rop', "ROP_Secondary_Results", "ROP_Secondary")]


def load_sec():
    df = P.load_df()
    df['second_antifungal_started'] = ((df['kacgunmicafungin2'].fillna(0) > 0) | (df['kullandıgıtarih2'].notna())).astype(int)
    df['third_antifungal_started'] = ((df['kacgunkullandi3'].fillna(0) > 0) | (df['kullandıgıtarih3'].notna())).astype(int)
    df['antifungal_escalation'] = ((df['second_antifungal_started'] == 1) | (df['third_antifungal_started'] == 1)).astype(int)
    df['multiple_antifungal_lines'] = df['antifungal_escalation']
    return df


def main():
    df = load_sec()
    esc = int(df['antifungal_escalation'].sum())
    print(f"Kohortta eskalasyon yapan: {esc}/{len(df)} (bu yuzden Secondary ~ Primary)\n", flush=True)
    total = 0
    for en, tgt, foldername, suffix in OUTCOMES:
        folder = os.path.join(PARENT, foldername)
        os.makedirs(folder, exist_ok=True)
        X, y, dropped = P.prep(df, tgt, FEATS)
        print(f"=== {en} Secondary -> {foldername}  (n={len(y)}, feat={X.shape[1]}, dusen sabit={dropped})", flush=True)
        # OOF topla, sonra sampiyonu (en yuksek AUC) isaretle
        oof = {}
        for algo, (mdl, prm) in P.build_models().items():
            yt, yp, _ = P.run_cv(X, y, mdl, prm)
            oof[algo] = (np.array(yt), np.array(yp), roc_auc_score(np.array(yt), np.array(yp)))
        champ = max(oof, key=lambda a: oof[a][2])
        for algo in oof:
            yt, yp, auc = oof[algo]
            plots_for(folder, suffix, algo, yt, yp, algo == champ)
            total += 3
            print(f"   {algo:26s} AUC={auc:.3f}" + ("  <- en yuksek" if algo == champ else ""), flush=True)
    print(f"\nToplam {total} gorsel -> {PARENT}/ (Secondary)", flush=True)


if __name__ == "__main__":
    main()
