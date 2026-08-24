"""
compare_hypersas_cops.py - GreenEdge 2016 equivalent of compare_pysas_cops.py, for
HyperSAS vs COPS comparison per station.

Key difference from the pySAS version: pySAS logs continuously all day, so a cast
falling inside the COPS cast window is the normal case and the tight-window match in
compare_pysas_cops.py::find_pysas_casts_in_window works directly. GreenEdge HyperSAS
was run in separate measurement bursts at a station (sometimes well before/after the
COPS profile, per Simon), so a tight-window match would silently return nothing for
many stations. This script's current scope is deliberately narrower: a temporal-gap
DIAGNOSTIC (report + plot) to see, across all stations, how far HyperSAS bursts
actually sit from the COPS window before deciding on a matching policy -- the
generate-comparison-figures part of compare_pysas_cops.py is not ported yet.

Station layout (already collated, per Simon -- nothing here copies files, unlike
compare_pysas_cops.py's copy_pysas_files/copy_camera_photos):
    <GREENEDGE_ROOT>/<date>_Station<name>/
        cops/select.cops.dat, cops/nc/*.nc          (COPS reference, same format as pySAS)
        HyperSAS/<model>NN/L2/<station>_L2.hdf      (produced by run_greenedge_processing.py)

Usage:
    conda activate hypercp
    python compare_hypersas_cops.py --station 20160609_StationG100
    python compare_hypersas_cops.py --all
"""
import os
import sys
import re
import glob
import argparse

import h5py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MY_DIR = os.path.dirname(os.path.abspath(__file__))
if MY_DIR not in sys.path:
    sys.path.insert(0, MY_DIR)

# Reuses the COPS-side reading exactly as-is from compare_pysas_cops.py (same file
# format/convention for both projects) instead of duplicating it.
import compare_pysas_cops as cpc  # noqa: E402


def load_pipeline_config(config_path):
    config = {}
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, val = line.split("=", 1)
            config[key.strip()] = val.strip()
    return config


env = load_pipeline_config(os.path.join(MY_DIR, "pipeline_config_greenedge.env"))
GREENEDGE_ROOT = env["GREENEDGE_ROOT"]
GEOM_MODEL = "M99NN"  # ensemble timing is identical across models, one is enough
BURST_GAP_MIN = 10  # minutes -- gap threshold separating two distinct HyperSAS "bursts"


def parse_station_label(station_path):
    """GreenEdge folder names are '<date>_Station<name>' OR '<date>_Transit<name>'
    (e.g. 20160604_TransitGE1) -- compare_pysas_cops.parse_station_date only matches
    '_Station', so this can't just reuse it."""
    base = os.path.basename(os.path.normpath(station_path))
    m = re.match(r"(\d{8})_", base)
    if not m:
        raise ValueError(f"Impossible d'extraire la date du nom de dossier station : {base}")
    return m.group(1), base


def load_hypersas_ensembles(station_path):
    """Horodatages (+ Datetag/Timetag2 pour retrouver les spectres plus tard) de tous
    les ensembles L2 HyperSAS de la station (un seul fichier, modèle GEOM_MODEL -- la
    géométrie temporelle est indépendante du modèle rho). Retourne un DataFrame trié
    par Datetime, ou None si pas encore traité."""
    l2_path = os.path.join(station_path, "HyperSAS", GEOM_MODEL, "L2",
                           f"{os.path.basename(station_path)}_L2.hdf")
    if not os.path.exists(l2_path):
        return None
    with h5py.File(l2_path, "r") as f:
        rrs = f["/REFLECTANCE/Rrs_HYPER"][...]
        datetag = rrs["Datetag"].astype(int)
        timetag2 = rrs["Timetag2"].astype(int)

    rows = []
    for d, t in zip(datetag, timetag2):
        year, doy = d // 1000, d % 1000
        base = pd.Timestamp(year=year, month=1, day=1, tz="UTC") + pd.Timedelta(days=doy - 1)
        h, m, s, ms = t // 10000000, (t // 100000) % 100, (t // 1000) % 100, t % 1000
        dt = base + pd.Timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)
        rows.append({"Datetime": dt, "Datetag": int(d), "Timetag2": int(t)})
    return pd.DataFrame(rows).sort_values("Datetime").reset_index(drop=True)


def offset_from_window(times, window_start, window_end):
    """Secondes entre chaque horodatage et la fenêtre COPS -- 0 si dedans, sinon
    distance au bord le plus proche (signe: + si après la fenêtre, - si avant)."""
    offsets = []
    for t in times:
        if t < window_start:
            offsets.append((t - window_start).total_seconds())
        elif t > window_end:
            offsets.append((t - window_end).total_seconds())
        else:
            offsets.append(0.0)
    return np.array(offsets)


def group_bursts(times, gap_min=BURST_GAP_MIN):
    """Regroupe les horodatages en 'séries' distinctes -- nouvelle série dès qu'un
    écart dépasse gap_min minutes. Répond directement à 'parfois on a lancé des
    séries de mesures HyperSAS à différentes heures' -- pour voir combien de séries
    existent et à quelle heure, pas juste une liste plate de timestamps."""
    if len(times) == 0:
        return []
    bursts, current = [], [times[0]]
    for t in times[1:]:
        if (t - current[-1]).total_seconds() > gap_min * 60:
            bursts.append(current)
            current = [t]
        else:
            current.append(t)
    bursts.append(current)
    return bursts


def analyze_station(station, burst_gap_min=BURST_GAP_MIN):
    station_path = os.path.join(GREENEDGE_ROOT, station)
    date_str, label = parse_station_label(station_path)

    cops_dir = cpc.find_cops_dir(station_path)
    casts = cpc.load_selected_cops_casts(cops_dir)
    window_start = min(c["start"] for c in casts)
    window_end = max(c["end"] for c in casts)

    ens_df = load_hypersas_ensembles(station_path)
    if ens_df is None:
        print(f"⚠️  [{label}] Pas de L2 HyperSAS ({GEOM_MODEL}) trouvé -- station pas encore traitée.")
        return None
    if len(ens_df) == 0:
        print(f"⚠️  [{label}] L2 HyperSAS présent mais 0 ensemble (aucun cast valide).")
        return None
    times = ens_df["Datetime"]

    offsets = offset_from_window(times, window_start, window_end)
    bursts = group_bursts(times, gap_min=burst_gap_min)

    rows = []
    for t, off in zip(times, offsets):
        rows.append({
            "Station": label, "EnsembleTime_UTC": t, "COPS_start": window_start,
            "COPS_end": window_end, "OffsetFromCOPS_s": round(float(off), 1),
            "InWindow": bool(off == 0),
        })
    df = pd.DataFrame(rows)

    n_in = int((offsets == 0).sum())
    print(f"📍 [{label}] fenêtre COPS {window_start.time()} -> {window_end.time()} "
         f"({len(casts)} cast(s)) | {len(times)} ensemble(s) HyperSAS en {len(bursts)} "
         f"série(s), {n_in} dans la fenêtre, écart min hors-fenêtre "
         f"{np.min(np.abs(offsets[offsets != 0])) / 60:.1f} min"
         if n_in < len(times) else f"tous dans la fenêtre")

    return df, casts, window_start, window_end, times, bursts, label, ens_df, station_path


def plot_station_timeline(out_path, casts, window_start, window_end, times, bursts, label):
    fig, ax = plt.subplots(figsize=(10, 3))

    ax.axvspan(window_start, window_end, color="steelblue", alpha=0.25, label="Fenêtre COPS")
    for c in casts:
        ax.axvline(c["start"], color="steelblue", linewidth=1, alpha=0.6)

    colors = plt.cm.tab10.colors
    for i, burst in enumerate(bursts):
        ax.scatter(burst, [1] * len(burst), color=colors[i % len(colors)], s=60, zorder=5,
                  label=f"Série {i + 1} (n={len(burst)})")

    ax.set_yticks([])
    ax.set_xlabel("Heure UTC")
    ax.set_title(f"Chronologie HyperSAS vs COPS -- {label}")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


N_CLOSEST = 3
FLAG_GAP_HOURS = 1.0  # au-delà, on flague -- pas de limite dure : "on sait que c'est la même station"


def select_closest_ensembles(ens_df, window_start, window_end, n=N_CLOSEST, flag_hours=FLAG_GAP_HOURS):
    """Les n ensembles HyperSAS temporellement les plus proches de la fenêtre COPS,
    sans limite de temps stricte (même station physique, donc toujours comparable en
    principe) -- mais on flague chaque ensemble retenu à plus de flag_hours de la
    fenêtre, pour garder une trace explicite de la confiance temporelle plutôt que de
    l'exclure silencieusement."""
    offsets = offset_from_window(ens_df["Datetime"], window_start, window_end)
    df = ens_df.assign(OffsetFromCOPS_s=offsets, AbsOffset_s=np.abs(offsets))
    closest = df.sort_values("AbsOffset_s").head(n).sort_values("Datetime").reset_index(drop=True)
    closest["Flagged_1h"] = closest["AbsOffset_s"] > flag_hours * 3600
    return closest


def get_hypersas_rrs_spectrum(station_path, method, timetag2):
    """Même logique que rrs_explorer_app.get_rrs_spectrum, mais un seul fichier par
    méthode par station (pas de recherche par nom de fichier de cast)."""
    fpath = os.path.join(station_path, "HyperSAS", method, "L2",
                         f"{os.path.basename(station_path)}_L2.hdf")
    if not os.path.exists(fpath):
        return None, None
    with h5py.File(fpath, "r") as h5f:
        rrs_path = "/REFLECTANCE/Rrs_HYPER"
        if rrs_path not in h5f:
            return None, None
        rrs_raw = h5f[rrs_path][...]
        colnames = rrs_raw.dtype.names
        wavelengths = np.array([float(w) for w in colnames[2:]])
        timetag2_vector = rrs_raw[colnames[1]]
        line_indices = np.where(timetag2_vector == timetag2)[0]
        if len(line_indices) == 0:
            return None, None
        line_idx = line_indices[0]
        spectrum = np.array([rrs_raw[w][line_idx] for w in colnames[2:]])
        return wavelengths, spectrum


def get_hypersas_rrs_uncertainty(station_path, method, timetag2):
    fpath = os.path.join(station_path, "HyperSAS", method, "L2",
                         f"{os.path.basename(station_path)}_L2.hdf")
    if not os.path.exists(fpath):
        return None, None
    with h5py.File(fpath, "r") as h5f:
        unc_path = "/REFLECTANCE/Rrs_HYPER_unc"
        if unc_path not in h5f:
            return None, None
        unc_raw = h5f[unc_path][...]
        colnames = unc_raw.dtype.names
        wavelengths = np.array([float(w) for w in colnames[2:]])
        timetag2_vector = unc_raw[colnames[1]]
        line_indices = np.where(timetag2_vector == timetag2)[0]
        if len(line_indices) == 0:
            return None, None
        line_idx = line_indices[0]
        unc = np.array([unc_raw[w][line_idx] for w in colnames[2:]])
        return wavelengths, unc


def gather_hypersas_spectra(station_path, closest):
    specs = {m: [] for m in cpc.rea.METHODS}
    wavelengths = None
    for row in closest.itertuples():
        for method in cpc.rea.METHODS:
            wl, spec = get_hypersas_rrs_spectrum(station_path, method, row.Timetag2)
            if wl is None:
                continue
            if wavelengths is None:
                wavelengths = wl
            specs[method].append(spec)
    return wavelengths, specs


def gather_hypersas_uncertainty(station_path, closest, method):
    uncs = []
    wavelengths = None
    for row in closest.itertuples():
        wl, unc = get_hypersas_rrs_uncertainty(station_path, method, row.Timetag2)
        if wl is None:
            continue
        if wavelengths is None:
            wavelengths = wl
        uncs.append(unc)
    return wavelengths, uncs


def run_comparison(station_path, label, casts, window_start, window_end, ens_df):
    """Sélectionne les N_CLOSEST ensembles HyperSAS (peu importe l'écart, flag si
    >FLAG_GAP_HOURS) et produit les mêmes figures/stats que compare_pysas_cops.py,
    avec instrument_label='HyperSAS' pour des titres/labels corrects."""
    closest = select_closest_ensembles(ens_df, window_start, window_end)
    n_flagged = int(closest["Flagged_1h"].sum())
    print(f"    🔗 {N_CLOSEST} ensemble(s) HyperSAS retenu(s) (écarts : "
         f"{[round(v / 60, 1) for v in closest['OffsetFromCOPS_s']]} min)"
         + (f" -- ⚠️ {n_flagged} flagué(s) (> {FLAG_GAP_HOURS}h)" if n_flagged else ""))

    dest_dir = os.path.join(station_path, "HyperSAS_vs_COPS")
    os.makedirs(dest_dir, exist_ok=True)
    closest.to_csv(os.path.join(dest_dir, f"SelectedEnsembles_{label}.csv"), index=False)

    cops_means = {variant: cpc.average_cops_rrs(casts, variant) for variant in cpc.COPS_VARIANTS}
    wl, specs = gather_hypersas_spectra(station_path, closest)
    if wl is None:
        print(f"    ⚠️  Aucun spectre HyperSAS lisible pour les ensembles retenus.")
        return

    fig1_path = os.path.join(dest_dir, f"Rrs_comparison_{label}.png")
    cpc.plot_spectra_comparison(fig1_path, casts, cops_means, wl, specs, label, instrument_label="HyperSAS")

    fig2_path = os.path.join(dest_dir, f"Scatter_vs_COPS_{label}.png")
    csv_path = os.path.join(dest_dir, f"Stats_vs_COPS_{label}.csv")
    df_stats = cpc.plot_scatter_and_stats(fig2_path, csv_path, cops_means, wl, specs, instrument_label="HyperSAS")
    print(df_stats.to_string(index=False))

    best_by_variant = cpc.pick_best_method(df_stats, instrument_label="HyperSAS")
    for variant, best_method in best_by_variant.items():
        cops_wl, cops_mean = cops_means[variant]
        spec_mean = np.nanmean(np.vstack(specs[best_method]), axis=0)
        unc_wl, uncs = gather_hypersas_uncertainty(station_path, closest, best_method)
        unc_mean = np.nanmean(np.vstack(uncs), axis=0) if uncs else None
        if unc_mean is not None and not np.array_equal(unc_wl, wl):
            unc_mean = np.interp(wl, unc_wl, unc_mean)

        fig3_path = os.path.join(dest_dir, f"Rrs_BestMethod_{variant}_{label}.png")
        cpc.plot_best_method_uncertainty(fig3_path, variant, best_method, casts, cops_mean,
                                         wl, spec_mean, unc_mean, label, instrument_label="HyperSAS")
        print(f"    🏆 Meilleure méthode vs COPS {variant} : {best_method} -> {fig3_path}")


def discover_stations():
    stations = []
    for cops_dir in sorted(glob.glob(os.path.join(GREENEDGE_ROOT, "*", "cops"))):
        stations.append(os.path.basename(os.path.dirname(cops_dir)))
    return stations


def main(stations, out_dir, burst_gap_min=BURST_GAP_MIN, do_compare=True):
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    for station in stations:
        try:
            result = analyze_station(station, burst_gap_min=burst_gap_min)
        except Exception as e:
            print(f"❌ [{station}] {e}")
            continue
        if result is None:
            continue
        df, casts, window_start, window_end, times, bursts, label, ens_df, station_path = result
        all_rows.append(df)
        plot_station_timeline(os.path.join(out_dir, f"Timeline_{label}.png"),
                              casts, window_start, window_end, times, bursts, label)

        if do_compare:
            try:
                run_comparison(station_path, label, casts, window_start, window_end, ens_df)
            except Exception as e:
                print(f"    ❌ [{label}] Comparaison échouée : {e}")

    if not all_rows:
        print("⚠️  Aucune station exploitable (COPS + L2 HyperSAS).")
        return

    df_all = pd.concat(all_rows, ignore_index=True)
    csv_path = os.path.join(out_dir, "HyperSAS_COPS_TimeOffsets.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\n📋 Rapport global : {csv_path}")

    # Résumé par station: écart minimal (le meilleur candidat de correspondance),
    # utile pour juger d'un coup d'oeil quelles stations ont un vrai near-simultané
    # vs celles où tout HyperSAS est loin de COPS.
    summary = (df_all.groupby("Station")["OffsetFromCOPS_s"]
              .apply(lambda s: np.min(np.abs(s)))
              .sort_values(ascending=False))
    print("\n--- Écart minimal |HyperSAS - COPS| par station (secondes) ---")
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.abs(df_all["OffsetFromCOPS_s"]) / 60, bins=30, color="steelblue")
    ax.set_xlabel("|Écart HyperSAS - fenêtre COPS| (minutes)")
    ax.set_ylabel("Nombre d'ensembles")
    ax.set_title("Distribution des écarts temporels HyperSAS vs COPS -- toutes stations")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    hist_path = os.path.join(out_dir, "HyperSAS_COPS_TimeOffsets_Histogram.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"📊 Histogramme global : {hist_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--station", action="append", default=None, help="Nom de dossier station (répétable)")
    parser.add_argument("--all", action="store_true", help="Toutes les stations avec un dossier cops/")
    parser.add_argument("--out-dir", default=os.path.join(GREENEDGE_ROOT, "_TimeOffsetAnalysis"))
    parser.add_argument("--burst-gap-min", type=float, default=BURST_GAP_MIN,
                        help=f"Écart (min) séparant deux séries HyperSAS distinctes (défaut: {BURST_GAP_MIN})")
    parser.add_argument("--no-compare", dest="compare", action="store_false",
                        help="Diagnostic temporel seul (timeline/CSV/histogramme), sans générer les "
                             "figures de comparaison Rrs vs COPS.")
    parser.set_defaults(compare=True)
    args = parser.parse_args()

    if not args.station and not args.all:
        parser.error("Spécifier --station <nom> (répétable) ou --all")

    stations = discover_stations() if args.all else args.station
    main(stations, args.out_dir, burst_gap_min=args.burst_gap_min, do_compare=args.compare)
