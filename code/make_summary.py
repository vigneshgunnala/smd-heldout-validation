"""
Summary table, RMSE and the headline figure for the SMD study.

Reuses the data preparation and model from smd_pipeline.py unchanged, then
reports random-split vs leave-one-station-out results for the weather-only
model, plus the 2018 drought check at the withheld Moore Park station.

Usage:
    python code/make_summary.py --data-dir ./data --out-dir ./output
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from smd_pipeline import (CHEAP_FEATURES, FULL_FEATURES, TARGET, build_dataset,
                          make_model)

STATIONS = {"Athenry": "dly1875.csv", "Valentia": "dly2275.csv", "MoorePark": "dly575.csv"}


def scores(y, p):
    return dict(r2=r2_score(y, p), mae=mean_absolute_error(y, p),
                rmse=float(np.sqrt(mean_squared_error(y, p))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", default="./output")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = build_dataset(args.data_dir, STATIONS)
    rows = []

    for label, feats in (("weather-only", CHEAP_FEATURES), ("full", FULL_FEATURES)):
        # Random 3-fold split (same folds as smd_pipeline.py)
        y_all, p_all = [], []
        for tr_i, te_i in KFold(3, shuffle=True, random_state=0).split(data):
            tr, te = data.iloc[tr_i], data.iloc[te_i]
            p_all.append(make_model().fit(tr[feats], tr[TARGET]).predict(te[feats]))
            y_all.append(te[TARGET].values)
        rows.append(dict(features=label, protocol="Random 3-fold split",
                         **scores(np.concatenate(y_all), np.concatenate(p_all))))

        # Leave-one-station-out
        held = []
        for st in STATIONS:
            tr, te = data[data.station != st], data[data.station == st]
            pred = make_model().fit(tr[feats], tr[TARGET]).predict(te[feats])
            s = scores(te[TARGET], pred)
            held.append(s)
            rows.append(dict(features=label, protocol=f"Hold out {st}", **s))
            if label == "weather-only" and st == "MoorePark":
                m = te.year.values == 2018
                drought = scores(te[TARGET].values[m], pred[m])
        rows.append(dict(features=label, protocol="Station-held-out mean",
                         **{k: np.mean([h[k] for h in held]) for k in ("r2", "mae", "rmse")}))

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "summary_random_vs_heldout.csv"), index=False)

    w = table[table.features == "weather-only"].set_index("protocol")
    rnd, mean = w.loc["Random 3-fold split"], w.loc["Station-held-out mean"]
    full_mean = table[(table.features == "full") &
                      (table.protocol == "Station-held-out mean")].iloc[0]

    print(f"Modelling set: {len(data)} station-days")
    print(table.round(3).to_string(index=False))
    print(f"\nMAE higher on held-out stations by {100 * (mean.mae / rnd.mae - 1):.1f}%")
    print(f"RMSE higher on held-out stations by {100 * (mean.rmse / rnd.rmse - 1):.1f}%")
    print(f"Weather-only vs full instrumentation, held-out MAE gap: "
          f"{mean.mae - full_mean.mae:.2f} mm")
    print(f"Moore Park 2018 (withheld): R2 {drought['r2']:.3f}, MAE {drought['mae']:.2f} mm")

    # Figure: random split vs each held-out station
    order = ["Random 3-fold split", "Hold out Valentia", "Hold out Athenry",
             "Hold out MoorePark", "Station-held-out mean"]
    labels = ["Random split\n(usual test)", "Held-out:\nValentia", "Held-out:\nAthenry",
              "Held-out:\nMoore Park", "Held-out\nmean"]
    colors = ["#9aa3ad", "#7fa7c9", "#7fa7c9", "#7fa7c9", "#1f5f8b"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(6.4, 2.5), dpi=300)
    bars = ax.bar(labels, [w.loc[o, "mae"] for o in order], color=colors, width=0.62)
    for b, o in zip(bars, order):
        ax.text(b.get_x() + b.get_width() / 2, w.loc[o, "mae"] + 0.06,
                f"{w.loc[o, 'mae']:.2f} mm\nR² {w.loc[o, 'r2']:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#222")
    ax.axhline(rnd.mae, color="#9aa3ad", ls="--", lw=0.8)
    ax.set_ylabel("Mean absolute error (mm)")
    ax.set_ylim(0, 3.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0, labelsize=7.8)
    pct = 100 * (mean.mae / rnd.mae - 1)
    ax.set_title(f"Same model, same inputs: error on unseen stations is {pct:.0f}% "
                 f"higher than a random split suggests", fontsize=8.5, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_random_vs_heldout.png"))


if __name__ == "__main__":
    main()
