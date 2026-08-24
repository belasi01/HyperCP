"""
compare_es_ed0_cops_hypersas.py - Direct near-simultaneous comparison of surface
irradiance (HyperSAS Es vs COPS Ed0+) at native per-scan sampling, averaged over a
short window (default 5 s) around each shared timestamp. Motivation (Simon): the Rrs
mismatch found in compare_hypersas_L3.py might trace back to a surface-irradiance
disagreement between the two independent sensors -- this isolates that directly (no
Rho correction, no NIR correction, no COPS extrapolation-to-surface fit).

Restricted to stations/moments where a HyperSAS ensemble and a COPS cast actually
overlap (or come within --max-offset-min minutes) in real time -- see
_TimeOffsetAnalysis/HyperSAS_COPS_TimeOffsets.csv. Most GreenEdge stations don't
(HyperSAS was often run hours before/after the COPS cast, confirmed by Simon) --
expect only a handful of usable moments.

Sources (per-scan, not the ensemble-binned L2 products):
- HyperSAS: IRRADIANCE/ES in HyperSAS/L1BQC/<station>_L1BQC.hdf (hyperspectral,
  Datetag/Timetag2-keyed, same decoding as compare_hypersas_cops.load_hypersas_ensembles).
- COPS: Ed0(time, wavelength) read directly from the raw cast .tsv via
  pycops.io.raw.read_cast() -- the processed .nc only keeps a single per-cast
  extrapolated scalar (ed0_value_at_0) and *normalized* correction ratios
  (ed0_correction*), not the raw absolute Ed0(t) needed here.

Usage:
    conda activate hypercp
    python compare_es_ed0_cops_hypersas.py [--window-s 5] [--max-offset-min 10]
"""
import os
import sys
import argparse

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MY_DIR = os.path.dirname(os.path.abspath(__file__))
if MY_DIR not in sys.path:
    sys.path.insert(0, MY_DIR)
PYCOPS_SRC = "/Users/simonbelanger/PythonProjects/pyCOPS/src"
if PYCOPS_SRC not in sys.path:
    sys.path.insert(0, PYCOPS_SRC)

import compare_hypersas_cops as chc  # noqa: E402  (also gives us chc.cpc = compare_pysas_cops)
from pycops.io.raw import read_cast  # noqa: E402

TIME_OFFSETS_CSV = os.path.join(chc.GREENEDGE_ROOT, "_TimeOffsetAnalysis", "HyperSAS_COPS_TimeOffsets.csv")
OUT_DIR = os.path.join(chc.GREENEDGE_ROOT, "_ES_Comparison")
WINDOW_S = 5
MAX_OFFSET_MIN = 10  # au-dela, pas assez simultane pour etre informatif sur l'irradiance


def load_hypersas_es_raw(station_path):
    """IRRADIANCE/ES du L1BQC (par scan, pas ensemble L2) -> DataFrame indexe par Datetime,
    colonnes = longueurs d'onde (nm). None si le fichier n'existe pas."""
    label = os.path.basename(station_path)
    fpath = os.path.join(station_path, "HyperSAS", "L1BQC", f"{label}_L1BQC.hdf")
    if not os.path.exists(fpath):
        return None
    with h5py.File(fpath, "r") as f:
        if "IRRADIANCE/ES" not in f:
            return None
        es = f["IRRADIANCE/ES"][...]
    datetag = es["Datetag"].astype(int)
    timetag2 = es["Timetag2"].astype(int)
    wl_names = [n for n in es.dtype.names if n not in ("Datetag", "Timetag2")]
    waves = np.array([float(w) for w in wl_names])
    dts = []
    for d, t in zip(datetag, timetag2):
        year, doy = d // 1000, d % 1000
        base = pd.Timestamp(year=year, month=1, day=1, tz="UTC") + pd.Timedelta(days=doy - 1)
        h, m, s, ms = int(t // 10000000), int((t // 100000) % 100), int((t // 1000) % 100), int(t % 1000)
        dts.append(base + pd.Timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms))
    values = np.vstack([es[w] for w in wl_names]).T
    return pd.DataFrame(values, columns=waves, index=pd.DatetimeIndex(dts, name="Datetime"))


def load_cops_ed0_raw(cops_dir):
    """Concatene Ed0(t, wavelength) de tous les casts .tsv de cops_dir -> DataFrame indexe par
    Datetime (UTC), colonnes = longueurs d'onde standard COPS (19 bandes)."""
    frames = []
    for fname in sorted(os.listdir(cops_dir)):
        if not fname.endswith("_URC.tsv"):  # exclut GPS_*.tsv et autres fichiers non-cast
            continue
        try:
            ds = read_cast(os.path.join(cops_dir, fname))
        except Exception as e:
            print(f"    lecture {fname} echouee: {e}")
            continue
        idx = pd.DatetimeIndex(ds["time"].values, name="Datetime")
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        frames.append(pd.DataFrame(ds["Ed0"].values, columns=ds["wavelength"].values, index=idx))
    if not frames:
        return None
    return pd.concat(frames).sort_index()


def average_window(df, center, window_s):
    half = pd.Timedelta(seconds=window_s / 2)
    sub = df[(df.index >= center - half) & (df.index <= center + half)]
    if len(sub) == 0:
        return None, 0
    return sub.mean(axis=0), len(sub)


def main(window_s=WINDOW_S, max_offset_min=MAX_OFFSET_MIN):
    os.makedirs(OUT_DIR, exist_ok=True)
    offsets = pd.read_csv(TIME_OFFSETS_CSV, parse_dates=["EnsembleTime_UTC", "COPS_start", "COPS_end"])
    offsets["AbsOffsetMin"] = offsets["OffsetFromCOPS_s"].abs() / 60
    close = offsets[offsets["AbsOffsetMin"] <= max_offset_min].copy()
    print(f"{len(close)} ensemble(s) HyperSAS a moins de {max_offset_min} min d'un cast COPS, "
          f"sur {close['Station'].nunique()} station(s) : {sorted(close['Station'].unique())}\n")

    rows = []
    for station in sorted(close["Station"].unique()):
        station_path = os.path.join(chc.GREENEDGE_ROOT, station)
        cops_dir = chc.cpc.find_cops_dir(station_path)
        es_df = load_hypersas_es_raw(station_path)
        ed0_df = load_cops_ed0_raw(cops_dir)
        if es_df is None or ed0_df is None:
            print(f"[SKIP] {station}: ES HyperSAS (L1BQC) ou Ed0 COPS (raw) indisponible.")
            continue

        for _, row in close[close["Station"] == station].iterrows():
            center = row["EnsembleTime_UTC"]
            es_mean, n_es = average_window(es_df, center, window_s)
            ed0_mean, n_ed0 = average_window(ed0_df, center, window_s)
            if es_mean is None or ed0_mean is None:
                print(f"[SKIP] {station} @ {center}: fenetre {window_s}s vide (HyperSAS n={n_es}, COPS n={n_ed0}).")
                continue

            es_wl, es_vals = es_mean.index.to_numpy(dtype=float), es_mean.to_numpy(dtype=float)
            ed0_wl, ed0_vals = ed0_mean.index.to_numpy(dtype=float), ed0_mean.to_numpy(dtype=float)
            resampled = np.interp(ed0_wl, es_wl, es_vals)
            mask = ~np.isnan(resampled) & ~np.isnan(ed0_vals) & (ed0_vals > 0)
            if mask.sum() < 3:
                print(f"[SKIP] {station} @ {center}: pas assez de bandes valides pour comparer.")
                continue

            ratio = resampled[mask] / ed0_vals[mask]
            bias_pct = float(np.mean(ratio - 1) * 100)

            rows.append({
                "Station": station, "Timestamp_UTC": center,
                "OffsetFromCOPS_min": round(row["OffsetFromCOPS_s"] / 60, 2),
                "N_scans_HyperSAS": n_es, "N_scans_COPS": n_ed0,
                "Biais_moyen_%": round(bias_pct, 2),
                "Ratio_min": round(float(ratio.min()), 3), "Ratio_max": round(float(ratio.max()), 3),
            })

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(ed0_wl[mask], ed0_vals[mask], "o-", color="black", label=f"COPS Ed0 (n={n_ed0} scans)")
            ax.plot(ed0_wl[mask], resampled[mask], "s--", color="steelblue", label=f"HyperSAS Es (n={n_es} scans)")
            ax.set_xlabel("Longueur d'onde (nm)")
            ax.set_ylabel("Irradiance (unites natives instrument)")
            ax.set_title(f"Es/Ed0 -- {station} @ {center:%H:%M:%S} UTC (offset {row['OffsetFromCOPS_s'] / 60:.1f} min)")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()
            fig_path = os.path.join(OUT_DIR, f"Es_{station}_{center:%H%M%S}.png")
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        print("Aucune comparaison exploitable (aucune fenetre commune avec des scans des deux cotes).")
        return
    csv_path = os.path.join(OUT_DIR, "Es_Ed0_comparison.csv")
    df_out.to_csv(csv_path, index=False)
    print(df_out.to_string(index=False))
    print(f"\nCSV : {csv_path}")
    print(f"Biais moyen global (HyperSAS/COPS - 1, %) : {df_out['Biais_moyen_%'].mean():.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window-s", type=float, default=WINDOW_S)
    p.add_argument("--max-offset-min", type=float, default=MAX_OFFSET_MIN)
    args = p.parse_args()
    main(args.window_s, args.max_offset_min)
