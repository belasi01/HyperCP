"""
plot_negative_vs_cloud.py - Density plot du % de valeurs Rrs negatives sur l'ensemble
du spectre en fonction du ratio Li(750)/Es(750) (CloudRatio, indicateur de nebulosite,
Ruddick 2006 / IOCCG Protocols), agrege sur toute la campagne (ou les dates choisies).

Sert a visualiser a grande echelle le biais deja documente (mémoire
rho_sky_lut_low_wind_bias) : le LUT rho_sky (M99/Z17) surestime sous vent faible +
ciel nuageux, produisant du Rrs negatif independamment de toute autre cause (ex.
ombrage du navire, voir detect_ship_shadow.py). Reutilise _cloud_ratio et
_pct_negative_bands de detect_ship_shadow.py (meme formule, pas de duplication).

Usage :
    conda activate hypercp
    python plot_negative_vs_cloud.py                   # toute la campagne (modele M99NN)
    python plot_negative_vs_cloud.py --date 20260816 --date 20260815
    python plot_negative_vs_cloud.py --model Z17NN --gridsize 40
"""
import os
import sys
import re
import glob
import argparse

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

MY_DIR = os.path.dirname(os.path.abspath(__file__))
if MY_DIR not in sys.path:
    sys.path.insert(0, MY_DIR)

import detect_ship_shadow as dss  # noqa: E402 -- réutilise _cloud_ratio / _pct_negative_bands

PYSAS_PATH = dss.PYSAS_PATH
OUT_DIR = dss.OUT_DIR

# Rampe séquentielle bleue à teinte unique (magnitude = densité de points), même
# palette que detect_ship_shadow.py (dataviz skill: références/palette.md).
SEQ_BLUE_CMAP = dss.SEQ_BLUE_CMAP
INK_PRIMARY = dss.INK_PRIMARY
INK_SECONDARY = dss.INK_SECONDARY
GRIDLINE = dss.GRIDLINE
CHART_SURFACE = dss.CHART_SURFACE


def gather(dates, model):
    l2_dir = os.path.join(PYSAS_PATH, model, "L2")
    if dates:
        files = []
        for d in dates:
            files.extend(sorted(glob.glob(os.path.join(l2_dir, f"*{d}*_L2.hdf"))))
    else:
        files = sorted(glob.glob(os.path.join(l2_dir, "*_L2.hdf")))

    if not files:
        raise FileNotFoundError(f"Aucun L2 {model} trouvé dans {l2_dir}")

    cloud_ratios, pct_negs = [], []
    for fp in files:
        try:
            with h5py.File(fp, "r") as h5f:
                cr = dss._cloud_ratio(h5f)
                pn = dss._pct_negative_bands(h5f, min_nm=0)
                if cr is None or pn is None:
                    continue
                n = min(len(cr), len(pn))
                cloud_ratios.append(cr[:n])
                pct_negs.append(pn[:n])
        except (OSError, KeyError) as e:
            print(f"⚠️  {os.path.basename(fp)} ignoré ({e})")

    cloud_ratios = np.concatenate(cloud_ratios) if cloud_ratios else np.array([])
    pct_negs = np.concatenate(pct_negs) if pct_negs else np.array([])
    valid = np.isfinite(cloud_ratios) & np.isfinite(pct_negs)
    print(f"📋 {len(files)} fichier(s) L2 {model}, {valid.sum()} enregistrement(s) exploitable(s)")
    return cloud_ratios[valid], pct_negs[valid]


def plot_density(cloud_ratio, pct_neg, out_path, label, gridsize):
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    hb = ax.hexbin(cloud_ratio, pct_neg, gridsize=gridsize, cmap=SEQ_BLUE_CMAP,
                   mincnt=1, extent=(0, max(0.5, float(np.nanmax(cloud_ratio))), 0, 100))

    ax.set_xlabel("CloudRatio -- Li(750)/Es(750)", color=INK_SECONDARY)
    ax.set_ylabel("% de bandes Rrs négatives (spectre complet)", color=INK_SECONDARY)
    ax.set_title(f"Rrs négatif vs nébulosité -- {label} ({len(cloud_ratio)} enregistrements)",
                color=INK_PRIMARY, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", color=GRIDLINE, linewidth=0.8)
    ax.tick_params(colors=INK_SECONDARY)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)

    cbar = fig.colorbar(hb, ax=ax, pad=0.02)
    cbar.set_label("Nombre d'enregistrements (densité)", color=INK_SECONDARY)
    cbar.ax.yaxis.set_tick_params(color=INK_SECONDARY, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE)
    plt.close(fig)
    print(f"📊 Figure : {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", action="append", default=None, help="Date AAAAMMJJ (répétable) -- défaut: toute la campagne")
    parser.add_argument("--model", default="M99NN", help="Modèle L2 source (défaut: M99NN, géométrie/CloudRatio indépendants du modèle)")
    parser.add_argument("--gridsize", type=int, default=35, help="Résolution du hexbin (défaut: 35)")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    cloud_ratio, pct_neg = gather(args.date, args.model)
    label = ", ".join(args.date) if args.date else "toute la campagne"
    suffix = "_".join(args.date) if args.date else "AllDates"
    out_path = os.path.join(args.out_dir, f"NegativeRrs_vs_CloudRatio_{suffix}.png")
    os.makedirs(args.out_dir, exist_ok=True)
    plot_density(cloud_ratio, pct_neg, out_path, label, args.gridsize)
