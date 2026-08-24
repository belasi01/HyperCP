"""
compare_hypersas_L3.py - Compare HyperSAS M99NIR/Z17NIR Rrs against the final COPS L3
database (GreenEdge2016.csv, Rrs_XXX_mean columns already resolved per-cast between
loess/linear -- see pycops/processing/database.py's use of each cast's own
"rrs_0p_recommended" .nc variable), across every GreenEdge 2016 station that has a
HyperSAS/COPS matchup (compare_hypersas_cops.py --all), to decide which of the two
Rho-correction methods tracks the final COPS reference best overall.

Deliberately does NOT reuse compare_hypersas_cops.py's Stats_vs_COPS_<label>.csv: those
compare against the two RAW per-cast nc variants (rrs_0p_linear/rrs_0p_loess) uniformly,
ignoring each cast's own select.cops.dat method choice -- not the same reference as L3,
which blends per-cast (loess for some casts, linear for others, at stations like G100/
G102/G512 where casts disagree). This script reads L3 directly instead.

Reuses compare_hypersas_cops.get_hypersas_rrs_spectrum() to read a station's HyperSAS L2
HDF5, and the already-computed SelectedEnsembles_<label>.csv (from
compare_hypersas_cops.py --all) for which HyperSAS ensembles to average -- no
re-selection here, so this stays consistent with the existing per-station comparison.

Usage:
    conda activate hypercp
    python compare_hypersas_L3.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MY_DIR = os.path.dirname(os.path.abspath(__file__))
if MY_DIR not in sys.path:
    sys.path.insert(0, MY_DIR)

import compare_hypersas_cops as chc  # noqa: E402

L3_CSV = "/Users/simonbelanger/Data/GreenEdge/L3/cops/GreenEdge2016.csv"
METHODS = ["M99NIR", "Z17NIR"]
OUT_DIR = os.path.join(chc.GREENEDGE_ROOT, "_L3_Comparison")

# G107 (offset temporel de 4.8h avec le cast COPS -- voir SelectedEnsembles_*.csv) s'est avere
# etre un spectre de glace de mer, pas un matchup eau libre valide (confirme par Simon) --
# exclu de la comparaison Rrs plutot que traite comme un "outlier" statistique ordinaire.
EXCLUDED_STATIONS = {"20160611_StationG107", "20160630_StationG507"}

# Marqueur distinct par station (matplotlib n'a qu'un nombre fini de marqueurs lisibles ;
# recycle si plus de stations que de marqueurs, improbable ici avec 11).
STATION_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "h", "<", ">", "p", "8"]


def load_l3_spectrum(df_l3, station_code):
    """(wavelengths, Rrs) tries depuis GreenEdge2016.csv pour un code station (ex. "G102"),
    ou (None, None) si absent. NaN retires (bandes UV/NIR non toujours mesurees par COPS)."""
    row = df_l3[df_l3["station_id"] == station_code]
    if row.empty:
        return None, None
    row = row.iloc[0]
    wl_cols = [c for c in df_l3.columns if re.match(r"^Rrs_\d+_mean$", c)]
    waves = np.array([float(c.split("_")[1]) for c in wl_cols])
    values = row[wl_cols].to_numpy(dtype=float)
    order = np.argsort(waves)
    waves, values = waves[order], values[order]
    mask = ~np.isnan(values)
    return waves[mask], values[mask]


def load_hypersas_mean_spectrum(station_path, method, selected_csv_path):
    """Moyenne des spectres HyperSAS (methode donnee) sur les ensembles deja retenus par
    compare_hypersas_cops.py (SelectedEnsembles_<label>.csv) -- (None, None) si indisponible."""
    if not os.path.exists(selected_csv_path):
        return None, None
    closest = pd.read_csv(selected_csv_path)
    specs, wl = [], None
    for timetag2 in closest["Timetag2"]:
        w, spec = chc.get_hypersas_rrs_spectrum(station_path, method, int(timetag2))
        if w is None:
            continue
        wl = w
        specs.append(spec)
    if not specs:
        return None, None
    return wl, np.nanmean(np.vstack(specs), axis=0)


def compute_stats(l3_wl, l3_rrs, hs_wl, hs_rrs):
    """Reechantillonne HyperSAS (hyperspectral) sur la grille (plus grossiere) de L3, puis
    bias/RMSD/R² -- meme formule que compare_pysas_cops.plot_scatter_and_stats. wl retournee
    (meme masque que x/y) pour permettre de colorer les points par longueur d'onde ensuite."""
    resampled = np.interp(l3_wl, hs_wl, hs_rrs)
    mask = ~np.isnan(resampled) & ~np.isnan(l3_rrs)
    n = int(mask.sum())
    if n < 2:
        return None
    x, y, wl = l3_rrs[mask], resampled[mask], l3_wl[mask]
    bias = float(np.mean(y - x))
    rmsd = float(np.sqrt(np.mean((y - x) ** 2)))
    r = np.corrcoef(x, y)[0, 1]
    mape = float(np.mean(np.abs((y - x) / x)) * 100)
    return dict(n=n, bias=bias, rmsd=rmsd, r2=float(r ** 2) if np.isfinite(r) else None, mape=mape, x=x, y=y, wl=wl)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df_l3 = pd.read_csv(L3_CSV)

    stations = sorted(
        d for d in os.listdir(chc.GREENEDGE_ROOT)
        if os.path.isdir(os.path.join(chc.GREENEDGE_ROOT, d, "HyperSAS_vs_COPS"))
        and d not in EXCLUDED_STATIONS
    )
    print(f"{len(stations)} station(s) avec matchup HyperSAS/COPS trouvee(s) "
          f"(exclu : {sorted(EXCLUDED_STATIONS)}) : {stations}\n")

    per_station_rows = []
    pooled = {m: {"x": [], "y": [], "wl": [], "station": []} for m in METHODS}

    for label in stations:
        station_path = os.path.join(chc.GREENEDGE_ROOT, label)
        m = re.match(r"\d{8}_Station(.+)$", label)
        if not m:
            print(f"[SKIP] {label}: nom de dossier inattendu.")
            continue
        code = m.group(1)

        l3_wl, l3_rrs = load_l3_spectrum(df_l3, code)
        if l3_wl is None:
            print(f"[SKIP] {label}: pas de ligne L3 pour le code station '{code}'.")
            continue

        selected_csv = os.path.join(station_path, "HyperSAS_vs_COPS", f"SelectedEnsembles_{label}.csv")

        for method in METHODS:
            hs_wl, hs_rrs = load_hypersas_mean_spectrum(station_path, method, selected_csv)
            if hs_wl is None:
                print(f"[SKIP] {label}/{method}: pas de spectre HyperSAS exploitable.")
                continue
            stats = compute_stats(l3_wl, l3_rrs, hs_wl, hs_rrs)
            if stats is None:
                continue
            per_station_rows.append({
                "Station": label, "Methode": method, "N": stats["n"],
                "Biais (HyperSAS-L3)": round(stats["bias"], 5),
                "RMSD": round(stats["rmsd"], 5),
                "R2": round(stats["r2"], 4) if stats["r2"] is not None else None,
                "MAPE_%": round(stats["mape"], 2),
            })
            pooled[method]["x"].extend(stats["x"].tolist())
            pooled[method]["y"].extend(stats["y"].tolist())
            pooled[method]["wl"].extend(stats["wl"].tolist())
            pooled[method]["station"].extend([label] * len(stats["x"]))

    df_per_station = pd.DataFrame(per_station_rows)
    csv_path = os.path.join(OUT_DIR, "Stats_vs_L3_per_station.csv")
    df_per_station.to_csv(csv_path, index=False)
    print(f"\n=== Stats par station (vs COPS L3) ===\n{df_per_station.to_string(index=False)}")
    print(f"\nCSV : {csv_path}")

    summary_rows = []
    for method in METHODS:
        x, y = np.array(pooled[method]["x"]), np.array(pooled[method]["y"])
        if len(x) < 2:
            continue
        bias = float(np.mean(y - x))
        rmsd = float(np.sqrt(np.mean((y - x) ** 2)))
        r = np.corrcoef(x, y)[0, 1]
        mape = float(np.mean(np.abs((y - x) / x)) * 100)
        n_stations = df_per_station[df_per_station["Methode"] == method]["Station"].nunique()
        summary_rows.append({
            "Methode": method, "N_stations": n_stations, "N_points": len(x),
            "Biais (HyperSAS-L3)": round(bias, 5), "RMSD": round(rmsd, 5),
            "R2": round(float(r ** 2), 4) if np.isfinite(r) else None,
            "MAPE_%": round(mape, 2),
        })
    df_summary = pd.DataFrame(summary_rows).sort_values("RMSD")
    print(f"\n=== Stats globales poolees (toutes stations, tous points spectraux) ===\n{df_summary.to_string(index=False)}")
    summary_csv = os.path.join(OUT_DIR, "Stats_vs_L3_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nResume global : {summary_csv}")

    if len(df_summary) == 2:
        winner = df_summary.iloc[0]["Methode"]
        print(f"\n>>> Meilleure methode globale vs COPS L3 (RMSD le plus faible) : {winner}")

    # Couleur = longueur d'onde (job "magnitude/ordre" -> une seule teinte, claire->foncee,
    # jamais un arc-en-ciel) ; marqueur = station (job "identite", encodage secondaire donc pas
    # de conflit avec la couleur). Meme echelle de couleur (vmin/vmax) et memes marqueurs sur les
    # deux panneaux pour rester comparables, une seule colorbar et une seule legende partagees.
    all_wl = np.concatenate([pooled[m]["wl"] for m in METHODS if pooled[m]["wl"]])
    wl_min, wl_max = (float(all_wl.min()), float(all_wl.max())) if len(all_wl) else (400, 700)
    stations_seen = sorted({s for m in METHODS for s in pooled[m]["station"]})
    marker_of = {s: STATION_MARKERS[i % len(STATION_MARKERS)] for i, s in enumerate(stations_seen)}

    fig, axes = plt.subplots(1, len(METHODS), figsize=(7.5 * len(METHODS), 7), squeeze=False)
    axes = axes[0]
    scatter_ref = None
    for ax, method in zip(axes, METHODS):
        x = np.array(pooled[method]["x"])
        y = np.array(pooled[method]["y"])
        wl = np.array(pooled[method]["wl"])
        station = np.array(pooled[method]["station"])
        if len(x) == 0:
            continue
        for s in stations_seen:
            sel = station == s
            if not sel.any():
                continue
            scatter_ref = ax.scatter(
                x[sel], y[sel], c=wl[sel], cmap="viridis", vmin=wl_min, vmax=wl_max,
                marker=marker_of[s], s=45, alpha=0.85, edgecolors="none",
            )
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        pad = 0.05 * (lims[1] - lims[0]) if lims[1] > lims[0] else 0.01
        lims = [lims[0] - pad, lims[1] + pad]
        ax.plot(lims, lims, "k--", linewidth=1, label="1:1")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(r"$R_{rs}$ COPS L3 (sr$^{-1}$)")
        ax.set_ylabel(rf"$R_{{rs}}$ HyperSAS {method} (sr$^{{-1}}$)")
        ax.set_title(f"{method} vs COPS L3 ({len(stations_seen)} stations, G107 exclue)")
        ax.grid(True, linestyle="--", alpha=0.4)

    # Legende des stations (marqueur, sans couleur -- neutre/gris pour ne pas dupliquer l'encodage
    # couleur=longueur d'onde) partagee sous les deux panneaux.
    station_handles = [
        plt.Line2D([0], [0], marker=marker_of[s], color="dimgray", linestyle="", markersize=8, label=s)
        for s in stations_seen
    ]
    station_handles.append(plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1, label="1:1"))
    fig.legend(handles=station_handles, loc="lower center", ncol=min(6, len(station_handles)),
               fontsize=8, bbox_to_anchor=(0.5, -0.06))

    if scatter_ref is not None:
        cbar = fig.colorbar(scatter_ref, ax=axes, orientation="vertical", fraction=0.03, pad=0.02)
        cbar.set_label("Longueur d'onde (nm)")

    fig_path = os.path.join(OUT_DIR, "Scatter_vs_L3_pooled.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatterplot : {fig_path}")


if __name__ == "__main__":
    main()
