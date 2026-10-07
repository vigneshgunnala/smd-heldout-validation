"""
Sensor-free prediction of soil moisture deficit from Met Eireann daily records.

Predicts Met Eireann's operational soil moisture deficit (smd_wd, well-drained)
from basic weather variables, and quantifies the accuracy cost of dropping
agrometeorological instrumentation.

Design rules enforced here:
  1. The target is an independently produced Met Eireann product, never derived
     from the model's own inputs.
  2. No lagged or autoregressive form of the target is used as a feature.
     SMD has lag-1 autocorrelation ~0.98; including it would make the task
     trivial and uninformative.
  3. Evaluation uses leave-one-station-out and leave-one-year-out. A random
     split is computed only to demonstrate the inflation it causes.

Data: Met Eireann daily synoptic station files, licensed CC BY 4.0.
      https://cli.fusio.net/cli/climate_data/showdata.php
      Stations: Athenry (1875), Valentia Observatory (2275), Moore Park (575).

Usage:
    python smd_pipeline.py --data-dir ./data --out-dir ./output
"""

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

TARGET = "smd_wd"
YEAR_MIN, YEAR_MAX = 2015, 2023  # window where all stations report SMD

NUMERIC = ["maxtp", "mintp", "rain", "wdsp", "hm", "hg", "soil", "pe", "evap",
           "smd_wd", "smd_md", "smd_pd", "glorad", "sun", "cbl", "gmin"]

CHEAP_FEATURES = [
    "rain", "maxtp", "mintp", "meantp", "trange", "wdsp", "lat", "lon",
    "rain_3d", "rain_7d", "rain_14d", "rain_30d",
    "tmean_7d", "tmean_14d", "tmean_30d",
    "wdsp_7d", "wdsp_14d", "wdsp_30d",
    "dry_7d", "dry_14d", "dry_30d", "days_since_rain",
    "doy_sin", "doy_cos",
]
INSTRUMENT_FEATURES = ["pe", "evap", "glorad", "soil", "pe_7d", "pe_14d", "pe_30d"]
FULL_FEATURES = CHEAP_FEATURES + INSTRUMENT_FEATURES


def make_model():
    return RandomForestRegressor(
        n_estimators=60, min_samples_leaf=3, max_depth=16,
        n_jobs=2, random_state=42,
    )


def read_station(path, station_name):
    """Parse a Met Eireann daily CSV: variable-length metadata header, then data."""
    with open(path, encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()

    header_row = next(i for i, ln in enumerate(lines) if ln.strip().startswith("date,"))
    meta = "".join(lines[:5])
    lat = float(re.search(r"Latitude:\s*(-?[\d.]+)", meta).group(1))
    lon = float(re.search(r"Longitude:\s*(-?[\d.]+)", meta).group(1))

    df = pd.read_csv(path, skiprows=header_row, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("ind")]  # quality flags, not used
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")

    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
        else:
            df[col] = np.nan

    df["station"] = station_name
    df["lat"] = lat
    df["lon"] = lon
    return df[["station", "date", "lat", "lon"] + NUMERIC]


def engineer(group):
    """Derive predictors from weather inputs only. Never touches the target."""
    g = group.sort_values("date").copy()
    g["meantp"] = (g.maxtp + g.mintp) / 2
    g["trange"] = g.maxtp - g.mintp

    for w in (3, 7, 14, 30):
        g[f"rain_{w}d"] = g.rain.rolling(w, min_periods=1).sum()
        g[f"tmean_{w}d"] = g.meantp.rolling(w, min_periods=1).mean()
        g[f"wdsp_{w}d"] = g.wdsp.rolling(w, min_periods=1).mean()
    for w in (7, 14, 30):
        g[f"dry_{w}d"] = (g.rain.fillna(0) < 0.2).rolling(w, min_periods=1).sum()
        g[f"pe_{w}d"] = g.pe.rolling(w, min_periods=1).sum()

    streak, out = 0, []
    for r in g.rain.fillna(0).values:
        streak = 0 if r >= 1.0 else streak + 1
        out.append(streak)
    g["days_since_rain"] = out

    doy = g.date.dt.dayofyear
    g["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    g["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return g


def climatology_baseline(train, test):
    """Day-of-year mean SMD from the training set."""
    lut = train.groupby("doy")[TARGET].mean()
    return test.doy.map(lut).fillna(train[TARGET].mean()).values


def water_balance_baseline(train, test):
    """Conventional bucket model: SMD_t = clip(SMD_t-1 + PE_seasonal - rain, 0, cap)."""
    pe_lut = train.groupby("doy")["pe"].mean()
    pe_mean = train["pe"].mean()
    cap = train[TARGET].quantile(0.99)

    test = test.copy()
    test["_pos"] = np.arange(len(test))
    preds = np.zeros(len(test))

    for _, g in test.groupby("station"):
        g = g.sort_values("date")
        state = 0.0
        pe_vals = g.doy.map(pe_lut).fillna(pe_mean).values
        rain_vals = g.rain.fillna(0).values
        positions = g["_pos"].values
        for k in range(len(g)):
            state = min(max(state + pe_vals[k] - rain_vals[k], 0), cap)
            preds[positions[k]] = state
    return preds


def evaluate(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)


def build_dataset(data_dir, station_files):
    frames = [read_station(os.path.join(data_dir, fn), name)
              for name, fn in station_files.items()]
    data = pd.concat(frames, ignore_index=True)
    data = data[data.date.notna()].sort_values(["station", "date"]).reset_index(drop=True)
    data = (data.groupby("station", group_keys=False)[data.columns.tolist()]
                .apply(engineer).reset_index(drop=True))

    data["year"] = data.date.dt.year
    data["doy"] = data.date.dt.dayofyear
    data = data[(data.year >= YEAR_MIN) & (data.year <= YEAR_MAX)]
    return data.dropna(subset=[TARGET] + FULL_FEATURES).reset_index(drop=True)


def run_all(data):
    records = []

    # Naive random split, included only to quantify its optimism
    kf = KFold(3, shuffle=True, random_state=0)
    for label, feats in (("cheap", CHEAP_FEATURES), ("full", FULL_FEATURES)):
        truths, preds = [], []
        for tr_idx, te_idx in kf.split(data):
            tr, te = data.iloc[tr_idx], data.iloc[te_idx]
            preds.append(make_model().fit(tr[feats], tr[TARGET]).predict(te[feats]))
            truths.append(te[TARGET].values)
        mae, r2 = evaluate(np.concatenate(truths), np.concatenate(preds))
        records.append(dict(protocol="Random split (naive)", model=f"RF {label}",
                            mae=mae, r2=r2))

    # Leave-one-station-out
    for station in data.station.unique():
        tr, te = data[data.station != station], data[data.station == station]
        for label, feats in (("cheap", CHEAP_FEATURES), ("full", FULL_FEATURES)):
            mae, r2 = evaluate(te[TARGET],
                               make_model().fit(tr[feats], tr[TARGET]).predict(te[feats]))
            records.append(dict(protocol=f"Hold out {station}", model=f"RF {label}",
                                mae=mae, r2=r2))
        mae, r2 = evaluate(te[TARGET], climatology_baseline(tr, te))
        records.append(dict(protocol=f"Hold out {station}",
                            model="Baseline: climatology", mae=mae, r2=r2))
        mae, r2 = evaluate(te[TARGET], water_balance_baseline(tr, te))
        records.append(dict(protocol=f"Hold out {station}",
                            model="Baseline: water balance", mae=mae, r2=r2))

    # Leave-one-year-out
    yearly = []
    for year in sorted(data.year.unique()):
        tr, te = data[data.year != year], data[data.year == year]
        for label, feats in (("cheap", CHEAP_FEATURES), ("full", FULL_FEATURES)):
            mae, r2 = evaluate(te[TARGET],
                               make_model().fit(tr[feats], tr[TARGET]).predict(te[feats]))
            yearly.append(dict(year=year, model=f"RF {label}", mae=mae, r2=r2))
        mae, r2 = evaluate(te[TARGET], climatology_baseline(tr, te))
        yearly.append(dict(year=year, model="Baseline: climatology", mae=mae, r2=r2))

    return pd.DataFrame(records), pd.DataFrame(yearly)


def feature_importance(data, held_out="MoorePark"):
    tr, te = data[data.station != held_out], data[data.station == held_out]
    model = make_model().fit(tr[CHEAP_FEATURES], tr[TARGET])
    result = permutation_importance(model, te[CHEAP_FEATURES], te[TARGET],
                                    n_repeats=5, random_state=0, n_jobs=2)
    imp = pd.DataFrame({"feature": CHEAP_FEATURES,
                        "importance": result.importances_mean})
    imp = imp.sort_values("importance", ascending=False)
    imp["pct"] = 100 * imp.importance / imp.importance.sum()
    return imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", default="./output")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    station_files = {
        "Athenry": "dly1875.csv",
        "Valentia": "dly2275.csv",
        "MoorePark": "dly575.csv",
    }

    data = build_dataset(args.data_dir, station_files)
    print(f"Modelling set: {data.shape[0]} station-days, "
          f"{data.station.nunique()} stations, {YEAR_MIN}-{YEAR_MAX}")

    main_results, yearly_results = run_all(data)
    importance = feature_importance(data)

    data.to_csv(os.path.join(args.out_dir, "model_set.csv"), index=False)
    main_results.to_csv(os.path.join(args.out_dir, "results_main.csv"), index=False)
    yearly_results.to_csv(os.path.join(args.out_dir, "results_year.csv"), index=False)
    importance.to_csv(os.path.join(args.out_dir, "importance.csv"), index=False)

    print("\n--- Main results ---")
    print(main_results.to_string(index=False))
    print("\n--- Leave-one-year-out means ---")
    print(yearly_results.groupby("model")[["mae", "r2"]].mean().round(3).to_string())
    print("\n--- Top predictors (weather-only model) ---")
    print(importance.head(8).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
