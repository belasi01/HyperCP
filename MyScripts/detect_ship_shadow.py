"""
detect_ship_shadow.py - Detecte, sur les fichiers L2 deja produits, les casts pySAS a
risque de contamination par l'ombre du navire sur Lt (capteur monte fixe a la proue de
l'Amundsen).

Distinction importante avec le filtre relAz deja present dans HyperCP
(bL1aqcCleanSunAngle / fL1aqcSunAngleMin-Max, Source/ProcessL1aqc.py) : ce filtre compare
l'azimut solaire au POINTAGE du capteur (cap + decalage mecanique de montage) pour eviter
le glint (fenetre de Mobley 1999). Ce script-ci calcule un angle different, "ShipRelAz" :
le cap du NAVIRE (proxy: COURSE GPS) relatif a l'azimut solaire -- la geometrie pertinente
pour l'ombrage de la coque/superstructure, qui n'a pas d'equivalent existant dans HyperCP.

Routine autonome, hors pipeline HyperCP : relit uniquement les HDF L2 deja produits
(groupe ANCILLARY -- COURSE, SOLAR_AZ, SZA -- deja disponibles sans retraitement), ne
modifie rien. Rapport + figure diagnostique seulement (v1) ; un mode de suppression des
ensembles flagues des HDF L2 est une extension future explicite, pas construite ici.

Usage :
    conda activate hypercp
    python detect_ship_shadow.py --date 20260816
    python detect_ship_shadow.py --all
    python detect_ship_shadow.py --date 20260816 --elev-safe 40 --max-restrict 100 --cloud-flag 0.05
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
from scipy.interpolate import interp1d

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
GEOM_MODEL = "M99NN"  # géométrie/ancillaire identiques quel que soit le modèle de rho
OUT_DIR = os.path.join(PYSAS_PATH, "AnalysisComparison")

# Paramètres par défaut du seuil dynamique -- heuristiques de départ, à affiner
DEFAULT_ELEV_SAFE = 40.0       # élévation (°) au-dessus de laquelle aucune restriction
DEFAULT_MAX_RESTRICT = 100.0   # seuil ShipRelAz (°) le plus restrictif, près de l'horizon
DEFAULT_CLOUD_FLAG = 0.05      # même convention que fL1bqcCloudFlag (Source/ConfigFile.py)
DEFAULT_CLOUD_RELIEF = 20.0    # assouplissement du seuil (°) si ciel couvert

NEG_BAND_MIN_NM = 650.0        # bande rouge/NIR utilisée pour le signal de validation croisée


def _normalize180(angle):
    a = np.asarray(angle, dtype=np.float64)
    a = np.mod(a + 180.0, 360.0) - 180.0
    return a


def max_shiprelaz(elevation, cloud_ratio, elev_safe, max_restrict, cloud_flag, cloud_relief):
    """Seuil dynamique (°) au-dela duquel abs(ShipRelAz) est considere a risque
    d'ombrage. Decroit lineairement de 180 deg (a elev_safe) vers max_restrict (a 0 deg
    d'elevation), puis assoupli si ciel couvert (cloud_ratio > cloud_flag)."""
    elevation = np.clip(elevation, 0.0, None)
    frac = np.clip(elevation / elev_safe, 0.0, 1.0)
    threshold = max_restrict + frac * (180.0 - max_restrict)
    relief = np.where(np.asarray(cloud_ratio) > cloud_flag, cloud_relief, 0.0)
    return np.minimum(threshold + relief, 180.0)


def _cloud_ratio(h5f):
    """Reprend exactement la formule de extract_l2_qc_tables.py::extract_cast_metadata_and_qc
    (Li(750)/Es(750), Ruddick 2006 / IOCCG Protocols)."""
    if "/IRRADIANCE/ES_HYPER" not in h5f or "/RADIANCE/LI_HYPER" not in h5f:
        return None
    es_raw = h5f["/IRRADIANCE/ES_HYPER"][...]
    li_raw = h5f["/RADIANCE/LI_HYPER"][...]
    es_wl = sorted([c for c in es_raw.dtype.names if re.match(r'^[\d.]+$', c)], key=float)
    li_wl = sorted([c for c in li_raw.dtype.names if re.match(r'^[\d.]+$', c)], key=float)
    if not es_wl or not li_wl:
        return None
    es_x = np.array([float(w) for w in es_wl])
    li_x = np.array([float(w) for w in li_wl])
    es_interp = interp1d(es_x, np.array([es_raw[w] for w in es_wl]), axis=0)
    li_interp = interp1d(li_x, np.array([li_raw[w] for w in li_wl]), axis=0)
    es750 = es_interp(750.0)
    li750 = li_interp(750.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(es750 != 0, li750 / es750, np.nan)


def _pct_negative_bands(h5f, min_nm=NEG_BAND_MIN_NM):
    """% de bandes >min_nm negatives dans Rrs_HYPER, par enregistrement -- signal de
    validation croisee (pas le critere de detection). min_nm=0 -> spectre complet."""
    if "/REFLECTANCE/Rrs_HYPER" not in h5f:
        return None
    rrs_raw = h5f["/REFLECTANCE/Rrs_HYPER"][...]
    wl_names = [n for n in rrs_raw.dtype.names if re.match(r'^[\d.]+$', n)]
    bands = [n for n in wl_names if float(n) > min_nm]
    if not bands:
        return None
    arr = np.vstack([rrs_raw[n] for n in bands]).T.astype(np.float64)  # (N, nBands)
    return 100.0 * np.mean(arr < 0, axis=1)


def _pct_negative_red_nir(h5f):
    return _pct_negative_bands(h5f, min_nm=NEG_BAND_MIN_NM)


def process_date(date_str, params):
    l2_dir = os.path.join(PYSAS_PATH, GEOM_MODEL, "L2")
    files = sorted(glob.glob(os.path.join(l2_dir, f"*{date_str}*_L2.hdf")))
    if not files:
        print(f"⚠️  Aucun L2 {GEOM_MODEL} trouvé pour {date_str}")
        return None

    rows = []
    for fp in files:
        try:
            with h5py.File(fp, "r") as h5f:
                anc = h5f.get("ANCILLARY")
                if anc is None or "COURSE" not in anc or "SOLAR_AZ" not in anc or "SZA" not in anc:
                    continue
                datetag = anc["COURSE"]["Datetag"][...]
                timetag2 = anc["COURSE"]["Timetag2"][...]
                course = anc["COURSE"]["TRUE"][...].astype(np.float64)
                solar_az = anc["SOLAR_AZ"]["SOLAR_AZ"][...].astype(np.float64)
                sza = anc["SZA"]["SZA"][...].astype(np.float64)
                lat = anc["LATITUDE"]["LATITUDE"][...] if "LATITUDE" in anc else np.full(len(course), np.nan)
                lon = anc["LONGITUDE"]["LONGITUDE"][...] if "LONGITUDE" in anc else np.full(len(course), np.nan)

                elevation = 90.0 - sza
                ship_relaz = np.abs(_normalize180(course - solar_az))
                cloud_ratio = _cloud_ratio(h5f)
                if cloud_ratio is None:
                    cloud_ratio = np.full(len(course), np.nan)
                pct_neg = _pct_negative_red_nir(h5f)
                if pct_neg is None:
                    pct_neg = np.full(len(course), np.nan)

                n = len(course)
                cloud_ratio = cloud_ratio[:n] if len(cloud_ratio) >= n else np.pad(cloud_ratio, (0, n - len(cloud_ratio)), constant_values=np.nan)
                pct_neg = pct_neg[:n] if len(pct_neg) >= n else np.pad(pct_neg, (0, n - len(pct_neg)), constant_values=np.nan)

                threshold = max_shiprelaz(elevation, cloud_ratio,
                                          params["elev_safe"], params["max_restrict"],
                                          params["cloud_flag"], params["cloud_relief"])
                flag = ship_relaz > threshold

                for i in range(n):
                    rows.append({
                        "Filename": os.path.basename(fp),
                        "Datetag": int(datetag[i]),
                        "Timetag2": int(timetag2[i]),
                        "Latitude": float(lat[i]),
                        "Longitude": float(lon[i]),
                        "ShipRelAz": round(float(ship_relaz[i]), 2),
                        "Elevation": round(float(elevation[i]), 2),
                        "CloudRatio": round(float(cloud_ratio[i]), 4) if np.isfinite(cloud_ratio[i]) else None,
                        "Threshold": round(float(threshold[i]), 2),
                        "ShadowRisk": bool(flag[i]),
                        "PctNegativeRedNIR": round(float(pct_neg[i]), 1) if np.isfinite(pct_neg[i]) else None,
                    })
        except (OSError, KeyError) as e:
            print(f"⚠️  {os.path.basename(fp)} ignoré ({e})")

    if not rows:
        print(f"⚠️  Aucune donnée exploitable pour {date_str}")
        return None

    return rows


def write_csv(rows, date_str, out_dir):
    import csv
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ShipShadow_{date_str}.csv")
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ CSV : {out_path} ({len(rows)} enregistrement(s), "
         f"{sum(r['ShadowRisk'] for r in rows)} flagué(s))")
    return out_path


# Palette (voir dataviz skill: références/palette.md). Deux magnitudes continues
# affichées côte à côte -> chacune sa propre rampe séquentielle à teinte unique (bleu
# pour la 1ère, orange -- 2e teinte catégorielle -- pour la 2e, jamais la même rampe
# pour deux grandeurs physiques différentes). Rouge de statut réservé pour le flag.
_SEQ_BLUE_HEX = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
                "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
_SEQ_ORANGE_HEX = ["#fce3d5", "#fad2b8", "#f7c09a", "#f4ae7c", "#f19c5f", "#ee8a48",
                   "#eb6834", "#d1592c", "#b64b24", "#9c3d1d", "#812f16", "#67210f", "#4c1608"]
SEQ_BLUE_CMAP = mcolors.LinearSegmentedColormap.from_list("seq_blue", _SEQ_BLUE_HEX)
SEQ_ORANGE_CMAP = mcolors.LinearSegmentedColormap.from_list("seq_orange", _SEQ_ORANGE_HEX)
SEQ_BLUE_CMAP.set_bad("#c3c2b7")
SEQ_ORANGE_CMAP.set_bad("#c3c2b7")
STATUS_CRITICAL = "#d03b3b"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
CHART_SURFACE = "#fcfcfb"
CLOUD_RATIO_VMAX = 0.5  # borne fixe (max observé sur la campagne ~0.45) pour comparer les dates entre elles


def _scatter_panel(ax, ship_relaz, elevation, color_vals, flag, cmap, vmin, vmax):
    m_ok, m_flag = ~flag, flag
    sc = ax.scatter(ship_relaz[m_ok], elevation[m_ok], c=color_vals[m_ok], cmap=cmap,
                    marker="o", s=40, vmin=vmin, vmax=vmax, edgecolors=INK_MUTED, linewidths=0.4,
                    label="Non flagué")
    ax.scatter(ship_relaz[m_flag], elevation[m_flag], c=color_vals[m_flag], cmap=cmap,
              marker="^", s=90, vmin=vmin, vmax=vmax, edgecolors=STATUS_CRITICAL, linewidths=1.4,
              label="Flagué (risque ombrage)")
    ax.set_xlabel("ShipRelAz -- |cap navire − azimut solaire| (°)", color=INK_SECONDARY)
    ax.grid(True, linestyle="--", color=GRIDLINE, linewidth=0.8)
    ax.tick_params(colors=INK_SECONDARY)
    ax.set_facecolor(CHART_SURFACE)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    return sc


def plot_diagnostic(rows, date_str, out_dir, cloud_flag):
    ship_relaz = np.array([r["ShipRelAz"] for r in rows])
    elevation = np.array([r["Elevation"] for r in rows])
    flag = np.array([r["ShadowRisk"] for r in rows])
    pct_neg = np.array([r["PctNegativeRedNIR"] if r["PctNegativeRedNIR"] is not None else np.nan for r in rows])
    cloud_ratio = np.array([r["CloudRatio"] if r["CloudRatio"] is not None else np.nan for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True, facecolor=CHART_SURFACE)

    sc_left = _scatter_panel(axes[0], ship_relaz, elevation, cloud_ratio, flag,
                             SEQ_BLUE_CMAP, 0, CLOUD_RATIO_VMAX)
    axes[0].set_title(f"CloudRatio (Li750/Es750) -- seuil nuageux={cloud_flag:g}",
                      color=INK_PRIMARY, fontsize=11)
    axes[0].set_ylabel("Élévation solaire (°)", color=INK_SECONDARY)
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    cbar_left = fig.colorbar(sc_left, ax=axes[0], pad=0.02)
    cbar_left.set_label("CloudRatio", color=INK_SECONDARY)
    cbar_left.ax.yaxis.set_tick_params(color=INK_SECONDARY, labelcolor=INK_SECONDARY)

    sc_right = _scatter_panel(axes[1], ship_relaz, elevation, pct_neg, flag,
                              SEQ_ORANGE_CMAP, 0, 100)
    axes[1].set_title(f"% bandes >{NEG_BAND_MIN_NM:.0f} nm négatives ({GEOM_MODEL})",
                      color=INK_PRIMARY, fontsize=11)
    cbar_right = fig.colorbar(sc_right, ax=axes[1], pad=0.02)
    cbar_right.set_label(f"% bandes >{NEG_BAND_MIN_NM:.0f} nm négatives", color=INK_SECONDARY)
    cbar_right.ax.yaxis.set_tick_params(color=INK_SECONDARY, labelcolor=INK_SECONDARY)

    n_unknown_cloud = int((~np.isfinite(cloud_ratio)).sum())
    suptitle = f"Détection ombrage navire -- {date_str} ({len(rows)} cast(s))"
    if n_unknown_cloud:
        suptitle += f"  -- {n_unknown_cloud} cast(s) à CloudRatio inconnu (gris, gauche)"
    fig.suptitle(suptitle, color=INK_PRIMARY, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(out_dir, f"ShipShadow_{date_str}.png")
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE)
    plt.close(fig)
    print(f"📊 Figure : {out_path}")


def available_dates():
    l2_dir = os.path.join(PYSAS_PATH, GEOM_MODEL, "L2")
    dates = set()
    for fp in glob.glob(os.path.join(l2_dir, "*_L2.hdf")):
        m = re.search(r'_(\d{8})_\d{6}_L2\.hdf$', os.path.basename(fp))
        if m:
            dates.add(m.group(1))
    return sorted(dates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", action="append", default=None, help="Date AAAAMMJJ (répétable)")
    parser.add_argument("--all", action="store_true", help=f"Toutes les dates avec L2 {GEOM_MODEL}")
    parser.add_argument("--elev-safe", type=float, default=DEFAULT_ELEV_SAFE)
    parser.add_argument("--max-restrict", type=float, default=DEFAULT_MAX_RESTRICT)
    parser.add_argument("--cloud-flag", type=float, default=DEFAULT_CLOUD_FLAG)
    parser.add_argument("--cloud-relief", type=float, default=DEFAULT_CLOUD_RELIEF)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    if args.all:
        dates = available_dates()
    elif args.date:
        dates = args.date
    else:
        parser.error("Spécifier --date <AAAAMMJJ> (répétable) ou --all")

    params = {
        "elev_safe": args.elev_safe,
        "max_restrict": args.max_restrict,
        "cloud_flag": args.cloud_flag,
        "cloud_relief": args.cloud_relief,
    }

    for date_str in dates:
        print(f"\n=== {date_str} ===")
        rows = process_date(date_str, params)
        if rows:
            write_csv(rows, date_str, args.out_dir)
            plot_diagnostic(rows, date_str, args.out_dir, args.cloud_flag)
