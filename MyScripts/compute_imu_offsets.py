"""
compute_imu_offsets.py - Calcule, pour chaque date disponible au niveau L1A (données
IMU brutes, non corrigées), la médiane de ROLL et PITCH (SATTHS1500A.tdf), et en
déduit un offset recommandé (= -médiane, arrondi à 0.1°) à utiliser avec
`run_pySAS006_processing.py --roll-offset --pitch-offset` (ou `ROLL_OFFSET`/
`PITCH_OFFSET` dans pipeline_config.env pour une période stable).

Le biais IMU n'est pas une constante sur toute la campagne -- il a sauté plusieurs
fois, généralement de façon groupée (ROLL et PITCH bougent ensemble), vraisemblablement
suite à des chocs/repositionnements physiques du capteur (voir MyScripts/README_pipeline.md
"Known quirks / open items"). Un écart-type élevé sur une journée (colonne
roll_std_raw/pitch_std_raw, seuil par défaut 2.0°) indique une journée de transition
(saut en cours de journée) où un offset journalier unique est une approximation
grossière -- envisager un retraitement par plage horaire (--time) dans ce cas.

Usage :
    conda activate hypercp
    python compute_imu_offsets.py                          # toutes les dates disponibles
    python compute_imu_offsets.py --start 20260701 --end 20260807
    python compute_imu_offsets.py --out /path/to/file.csv
"""
import os
import sys
import re
import glob
import collections
import argparse
import csv

import numpy as np
import h5py

MY_DIR = os.path.dirname(os.path.abspath(__file__))
if MY_DIR not in sys.path:
    sys.path.insert(0, MY_DIR)


def load_pipeline_config(config_path):
    config = {}
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                config[key.strip()] = val.strip()
    return config


ENV = load_pipeline_config(os.path.join(MY_DIR, "pipeline_config.env"))
MAIN_DATA_PATH = ENV["MAIN_DATA_PATH"]
PYSAS_PATH = os.path.join(MAIN_DATA_PATH, "pySAS")
L1A_DIR = os.path.join(PYSAS_PATH, "L1A")
DEFAULT_OUT = os.path.join(PYSAS_PATH, "IMU_Roll_Pitch_Offsets.csv")

TRANSITION_STD_THRESHOLD = 2.0  # degres


def compute_daily_stats(l1a_dir, start=None, end=None):
    files_by_date = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(l1a_dir, "*_L1A.hdf"))):
        m = re.search(r'_(\d{8})_\d{6}_L1A\.hdf$', os.path.basename(f))
        if not m:
            continue
        date = m.group(1)
        if start and date < start:
            continue
        if end and date > end:
            continue
        files_by_date[date].append(f)

    rows = []
    for date in sorted(files_by_date):
        rolls, pitches = [], []
        n_ok = 0
        for f in files_by_date[date]:
            try:
                with h5py.File(f, "r") as h5f:
                    g = h5f.get("SATTHS1500A.tdf")
                    if g is None:
                        continue
                    r = g["ROLL"][...]["NONE"]
                    p = g["PITCH"][...]["NONE"]
                    r = r[np.isfinite(r) & (np.abs(r) < 180)]
                    p = p[np.isfinite(p) & (np.abs(p) < 180)]
                    if len(r) and len(p):
                        rolls.append(r)
                        pitches.append(p)
                        n_ok += 1
            except (OSError, KeyError) as e:
                print(f"  ⚠️  {os.path.basename(f)} ignoré ({e})")

        n_files = len(files_by_date[date])
        if not rolls:
            rows.append({"date": date, "n_files": n_files, "n_ok": n_ok,
                        "roll_median_raw": None, "pitch_median_raw": None,
                        "roll_std_raw": None, "pitch_std_raw": None,
                        "roll_offset_recommended": None, "pitch_offset_recommended": None,
                        "note": "Aucune donnée SATTHS1500A exploitable"})
            continue

        rolls = np.concatenate(rolls)
        pitches = np.concatenate(pitches)
        roll_med = float(np.median(rolls))
        pitch_med = float(np.median(pitches))
        roll_std = float(np.std(rolls))
        pitch_std = float(np.std(pitches))

        note = ""
        if roll_std > TRANSITION_STD_THRESHOLD or pitch_std > TRANSITION_STD_THRESHOLD:
            note = ("Jour de transition (écart-type élevé -- saut de biais en cours de "
                    "journée) : offset journalier = approximation grossière, envisager "
                    "un retraitement par plage horaire (--time)")

        rows.append({
            "date": date, "n_files": n_files, "n_ok": n_ok,
            "roll_median_raw": round(roll_med, 2), "pitch_median_raw": round(pitch_med, 2),
            "roll_std_raw": round(roll_std, 2), "pitch_std_raw": round(pitch_std, 2),
            "roll_offset_recommended": round(-roll_med, 1), "pitch_offset_recommended": round(-pitch_med, 1),
            "note": note,
        })
    return rows


def write_csv(rows, out_path):
    fieldnames = ["date", "n_files", "n_ok", "roll_median_raw", "pitch_median_raw",
                 "roll_std_raw", "pitch_std_raw", "roll_offset_recommended",
                 "pitch_offset_recommended", "note"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ CSV écrit : {out_path} ({len(rows)} date(s))")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=None, help="Date de début AAAAMMJJ (incluse)")
    parser.add_argument("--end", default=None, help="Date de fin AAAAMMJJ (incluse)")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Fichier CSV de sortie (défaut: {DEFAULT_OUT})")
    args = parser.parse_args()

    rows = compute_daily_stats(L1A_DIR, start=args.start, end=args.end)
    write_csv(rows, args.out)
