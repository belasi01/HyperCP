"""
compare_pysas_cops.py - Compare pySAS L2 Rrs (7 sky-correction methods) against a
COPS in-water profiler station, to help identify which method best matches the COPS
reference.

Given a station folder (.../L2/YYYYMMDD_StationID/, produced when processing CTD/IOP/
COPS casts for that station), this:
1. Reads the station's cops/ subfolder (select.cops.dat + nc/*.nc) to get the COPS
   reference Rrs spectrum (averaged over the casts flagged "selected") and the exact
   UTC time window the COPS casts covered.
2. Finds the pySAS L2 casts (from all 7 methods) whose timestamp falls in that window,
   using the day's L2_Methods_Quotes_<date>.csv (same source as rrs_explorer_app.py).
3. Copies (or symlinks) the corresponding pySAS L2 HDF5 files into a new
   .../L2/YYYYMMDD_StationID/pySAS/<method>/ subfolder, alongside the other
   instruments' data for that station.
4. Produces a comparison figure (Rrs vs wavelength, COPS + all 7 pySAS methods),
   a scatterplot (pySAS vs COPS per method, resampled onto COPS's wavelength grid),
   and a stats table (bias/RMSD/R² per method vs COPS) -- all saved under that
   pySAS/ subfolder.

Usage:
    conda activate hypercp
    python compare_pysas_cops.py /path/to/Amundsen_2026/L2/20260808_StationCS1-1/
    python compare_pysas_cops.py <station_path> --copy   # real copies instead of symlinks
"""
import os
import sys
import re
import glob
import shutil
import argparse

import h5py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

MY_DIR = os.path.dirname(os.path.abspath(__file__))
if MY_DIR not in sys.path:
    sys.path.insert(0, MY_DIR)

# Reuse the L2 reading/caching logic already written for the Rrs Explorer (METHODS,
# METHOD_COLORS, load_casts, get_rrs_spectrum, BASE_PATH) instead of duplicating it.
import rrs_explorer_app as rea  # noqa: E402
import Source.utils.dating as dating  # noqa: E402


def parse_station_date(station_path):
    base = os.path.basename(os.path.normpath(station_path))
    m = re.match(r"(\d{8})_Station", base)
    if not m:
        raise ValueError(f"Impossible d'extraire la date du nom de dossier station : {base}")
    return m.group(1)


def find_cops_dir(station_path):
    for name in sorted(os.listdir(station_path)):
        full = os.path.join(station_path, name)
        if os.path.isdir(full) and name.lower().startswith("cops"):
            return full
    raise FileNotFoundError(f"Aucun sous-dossier cops/COPS trouvé dans {station_path}")


# Les deux variantes d'extrapolation vers la surface (0+) que pyCOPS produit -- on les
# compare toutes les deux plutôt que de se fier uniquement au choix par défaut de
# select.cops.dat (3e champ), pour voir si le choix d'extrapolation influence le
# classement des méthodes pySAS.
COPS_VARIANTS = ["linear", "loess"]


def load_selected_cops_casts(cops_dir):
    """Lit select.cops.dat (format: filename;flag;rrs_variant;...) et retourne, pour
    chaque cast retenu (flag == 1), sa fenêtre temporelle UTC et ses spectres Rrs
    (rrs_0p_linear ET rrs_0p_loess, indépendamment de la variante choisie par défaut
    dans select.cops.dat)."""
    select_path = os.path.join(cops_dir, "select.cops.dat")
    nc_dir = os.path.join(cops_dir, "nc")
    casts = []
    with open(select_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(";")
            if len(parts) < 2:
                continue
            fname, flag = parts[0].strip(), parts[1].strip()
            if flag != "1":
                continue
            nc_path = os.path.join(nc_dir, os.path.splitext(fname)[0] + ".nc")
            if not os.path.exists(nc_path):
                print(f"⚠️  Fichier nc manquant pour le cast sélectionné : {nc_path}")
                continue
            with xr.open_dataset(nc_path) as ds:
                cast = {
                    "file": fname,
                    "start": pd.Timestamp(ds.time.values.min()).tz_localize("UTC"),
                    "end": pd.Timestamp(ds.time.values.max()).tz_localize("UTC"),
                    "wavelength": ds.wavelength.values.copy(),
                    "lat": ds.attrs.get("latitude"),
                    "lon": ds.attrs.get("longitude"),
                }
                missing = []
                for variant in COPS_VARIANTS:
                    var_name = f"rrs_0p_{variant}"
                    if var_name in ds:
                        cast[f"rrs_{variant}"] = ds[var_name].values.copy()
                    else:
                        missing.append(var_name)
                if missing:
                    print(f"⚠️  {fname}: variable(s) manquante(s) {missing}")
                casts.append(cast)
    if not casts:
        raise ValueError(f"Aucun cast COPS retenu (flag=1) dans {select_path}")
    return casts


def average_cops_rrs(casts, variant):
    key = f"rrs_{variant}"
    wl_ref = casts[0]["wavelength"]
    specs = [c[key] for c in casts if key in c and np.array_equal(c["wavelength"], wl_ref)]
    if len(specs) < len(casts):
        print(f"⚠️  Certains casts COPS n'ont pas de spectre {key} utilisable -- ignorés dans la moyenne.")
    return wl_ref, np.nanmean(np.vstack(specs), axis=0)


def find_pysas_casts_in_window(date_str, window_start, window_end):
    df_casts = rea.load_casts(date_str)
    cast_dt = [
        pd.Timestamp(dating.timeTag2ToDateTime(
            dating.dateTagToDateTime(int(row.Datetag)), int(row.Timetag2)))
        for row in df_casts.itertuples()
    ]
    df_casts = df_casts.assign(_dt=cast_dt)
    mask = (df_casts["_dt"] >= window_start) & (df_casts["_dt"] <= window_end)
    return df_casts[mask].drop(columns=["_dt"]).reset_index(drop=True)


def copy_pysas_files(df_window, station_path, use_symlink=True):
    dest_root = os.path.join(station_path, "pySAS")
    for method in rea.METHODS:
        method_dest = os.path.join(dest_root, method)
        os.makedirs(method_dest, exist_ok=True)
        for fname in df_window["Filename"].unique():
            src = os.path.join(rea.BASE_PATH, method, "L2", fname)
            if not os.path.exists(src):
                continue
            dst = os.path.join(method_dest, fname)
            if os.path.exists(dst) or os.path.islink(dst):
                continue
            if use_symlink:
                try:
                    os.symlink(src, dst)
                except OSError:
                    # e.g. CIFS/network share without symlink support
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
    return dest_root


def gather_pysas_spectra(df_window):
    """Spectres pySAS (grille native) pour chaque cast/méthode dans la fenêtre COPS."""
    specs = {m: [] for m in rea.METHODS}
    wavelengths = None
    for row in df_window.itertuples():
        for method in rea.METHODS:
            wl, spec = rea.get_rrs_spectrum(method, row.Filename, row.Timetag2)
            if wl is None:
                continue
            if wavelengths is None:
                wavelengths = wl
            specs[method].append(spec)
    return wavelengths, specs


def get_rrs_uncertainty(method, filename, timetag2):
    """Incertitude combinée calculée par HyperCP (budget FRM/PIU, Source/PIU/), stockée
    à côté du spectre Rrs sous /REFLECTANCE/Rrs_HYPER_unc avec la même structure
    (Datetag/Timetag2 + une colonne par longueur d'onde) -- même logique de lecture que
    rea.get_rrs_spectrum, juste un chemin de dataset différent."""
    fpath = os.path.join(rea.BASE_PATH, method, "L2", filename)
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


def gather_pysas_uncertainty(df_window, method):
    """Incertitudes pySAS (grille native) pour un cast/méthode donnée dans la fenêtre COPS."""
    uncs = []
    wavelengths = None
    for row in df_window.itertuples():
        wl, unc = get_rrs_uncertainty(method, row.Filename, row.Timetag2)
        if wl is None:
            continue
        if wavelengths is None:
            wavelengths = wl
        uncs.append(unc)
    return wavelengths, uncs


def pick_best_method(df_stats):
    """Meilleure méthode par variante COPS (RMSD la plus faible, |biais| comme
    départage) -- ex: pour trancher entre 'loess vs Z17SimSpec (RMSD=0.00019)' et
    d'autres méthodes à RMSD quasi identique."""
    best = {}
    for variant, group in df_stats.groupby("Référence COPS"):
        row = group.assign(_abs_bias=group["Biais (pySAS-COPS)"].abs()) \
                    .sort_values(["RMSD", "_abs_bias"]).iloc[0]
        best[variant] = row["Méthode"]
    return best


# Style distinct par variante COPS, partagé entre les deux figures.
COPS_VARIANT_STYLE = {
    "linear": dict(color="black", linestyle="-", marker="o"),
    "loess": dict(color="dimgray", linestyle="--", marker="s"),
}


def plot_spectra_comparison(out_path, cops_casts, cops_means, pysas_wl, pysas_specs, station_label):
    """cops_means: {variant: (wavelength, mean_spectrum)}. Affiche chaque cast pySAS et
    COPS valide en trait fin, et la moyenne de chaque méthode/variante en trait plein."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for method in rea.METHODS:
        casts = pysas_specs[method]
        if not casts:
            continue
        color = rea.METHOD_COLORS[method]
        for spec in casts:
            ax.plot(pysas_wl, spec, color=color, alpha=0.2, linewidth=0.7)
        mean_spec = np.nanmean(np.vstack(casts), axis=0)
        ax.plot(pysas_wl, mean_spec, color=color, linewidth=2, label=f"{method} (n={len(casts)})")

    for variant in COPS_VARIANTS:
        key = f"rrs_{variant}"
        style = COPS_VARIANT_STYLE[variant]
        for cast in cops_casts:
            if key in cast:
                ax.plot(cast["wavelength"], cast[key], color=style["color"],
                        linestyle=style["linestyle"], alpha=0.35, linewidth=1)
        if variant in cops_means:
            wl, mean_spec = cops_means[variant]
            ax.plot(wl, mean_spec, color=style["color"], linestyle=style["linestyle"],
                    marker=style["marker"], linewidth=2.5, markersize=6,
                    label=f"COPS {variant} (n={len(cops_casts)})", zorder=10)

    ax.set_xlabel("Longueur d'onde (nm)")
    ax.set_ylabel(r"$R_{rs}$ (sr$^{-1}$)")
    ax.set_title(f"Comparaison Rrs pySAS vs COPS -- {station_label}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter_and_stats(out_fig_path, out_csv_path, cops_means, pysas_wl, pysas_specs):
    """Un panneau de scatterplot par variante COPS (linear, loess), stats calculées
    séparément contre chacune."""
    variants = [v for v in COPS_VARIANTS if v in cops_means]
    fig, axes = plt.subplots(1, len(variants), figsize=(7 * len(variants), 7), squeeze=False)
    axes = axes[0]
    records = []

    for ax, variant in zip(axes, variants):
        cops_wl, cops_rrs = cops_means[variant]
        all_vals = [cops_rrs]
        for method in rea.METHODS:
            casts = pysas_specs[method]
            if not casts:
                continue
            mean_spec = np.nanmean(np.vstack(casts), axis=0)
            resampled = np.interp(cops_wl, pysas_wl, mean_spec)
            mask = ~np.isnan(resampled) & ~np.isnan(cops_rrs)
            n = int(mask.sum())
            if n < 2:
                continue
            x, y = cops_rrs[mask], resampled[mask]
            bias = float(np.mean(y - x))
            rmsd = float(np.sqrt(np.mean((y - x) ** 2)))
            r = np.corrcoef(x, y)[0, 1]
            records.append({
                "Référence COPS": variant, "Méthode": method, "N": n,
                "Biais (pySAS-COPS)": round(bias, 5), "RMSD": round(rmsd, 5),
                "R²": round(float(r ** 2), 4) if np.isfinite(r) else None,
            })
            color = rea.METHOD_COLORS[method]
            ax.scatter(x, y, color=color, label=method, alpha=0.8, s=40)
            all_vals.append(y)

        lims = [np.nanmin(np.concatenate(all_vals)), np.nanmax(np.concatenate(all_vals))]
        pad = 0.05 * (lims[1] - lims[0]) if lims[1] > lims[0] else 0.01
        lims = [lims[0] - pad, lims[1] + pad]
        ax.plot(lims, lims, color="black", linestyle="--", linewidth=1, label="1:1")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(rf"$R_{{rs}}$ COPS {variant} (sr$^{{-1}}$)")
        ax.set_ylabel(r"$R_{rs}$ pySAS (sr$^{-1}$)")
        ax.set_title(f"pySAS vs COPS ({variant})")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_fig_path, dpi=150)
    plt.close(fig)

    df_stats = pd.DataFrame(records).sort_values(["Référence COPS", "RMSD"])
    df_stats.to_csv(out_csv_path, index=False)
    return df_stats


def plot_best_method_uncertainty(out_path, variant, best_method, cops_casts, cops_mean,
                                  pysas_wl, pysas_spec_mean, pysas_unc_mean, station_label):
    """Spectre COPS (moyenne +/- écart-type inter-casts) vs meilleure méthode pySAS
    (moyenne +/- incertitude combinée HyperCP, Rrs_HYPER_unc) -- pour voir si la
    référence COPS tombe dans la bande d'incertitude calculée par HyperCP."""
    key = f"rrs_{variant}"
    cops_wl = cops_casts[0]["wavelength"]
    cops_stack = np.vstack([c[key] for c in cops_casts if key in c])
    cops_std = np.nanstd(cops_stack, axis=0)

    fig, ax = plt.subplots(figsize=(9, 6))
    style = COPS_VARIANT_STYLE[variant]
    ax.fill_between(cops_wl, cops_mean - cops_std, cops_mean + cops_std,
                     color=style["color"], alpha=0.15)
    ax.plot(cops_wl, cops_mean, color=style["color"], linestyle=style["linestyle"],
            marker=style["marker"], linewidth=2.5, markersize=6,
            label=f"COPS {variant} (± écart-type, n={len(cops_stack)})")

    color = rea.METHOD_COLORS[best_method]
    if pysas_unc_mean is not None:
        ax.fill_between(pysas_wl, pysas_spec_mean - pysas_unc_mean, pysas_spec_mean + pysas_unc_mean,
                         color=color, alpha=0.2)
        unc_label = " (± incertitude HyperCP)"
    else:
        unc_label = " (incertitude indisponible)"
    ax.plot(pysas_wl, pysas_spec_mean, color=color, linewidth=2.5,
            label=f"pySAS {best_method}{unc_label}")

    ax.set_xlabel("Longueur d'onde (nm)")
    ax.set_ylabel(r"$R_{rs}$ (sr$^{-1}$)")
    ax.set_title(f"Meilleure méthode ({best_method}) vs COPS {variant} -- {station_label}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(station_path, use_symlink=True):
    station_path = os.path.abspath(station_path)
    date_str = parse_station_date(station_path)
    station_label = os.path.basename(station_path)

    cops_dir = find_cops_dir(station_path)
    casts = load_selected_cops_casts(cops_dir)
    window_start = min(c["start"] for c in casts)
    window_end = max(c["end"] for c in casts)
    cops_means = {variant: average_cops_rrs(casts, variant) for variant in COPS_VARIANTS}
    print(f"📍 {station_label} | fenêtre COPS UTC : {window_start} -> {window_end} "
          f"({len(casts)} cast(s) retenu(s))")

    df_window = find_pysas_casts_in_window(date_str, window_start, window_end)
    if df_window.empty:
        raise ValueError("Aucun cast pySAS trouvé dans la fenêtre temporelle COPS -- "
                          "vérifier que la date a été traitée jusqu'à extract_l2_qc_tables.py.")
    print(f"🔗 {len(df_window)} cast(s) pySAS trouvé(s) dans cette fenêtre.")

    dest_root = copy_pysas_files(df_window, station_path, use_symlink=use_symlink)
    print(f"📂 Données pySAS {'liées' if use_symlink else 'copiées'} dans : {dest_root}")

    pysas_wl, pysas_specs = gather_pysas_spectra(df_window)
    if pysas_wl is None:
        raise ValueError("Aucun spectre Rrs pySAS lisible pour les casts de la fenêtre.")

    fig1_path = os.path.join(dest_root, f"Rrs_comparison_{station_label}.png")
    plot_spectra_comparison(fig1_path, casts, cops_means, pysas_wl, pysas_specs, station_label)
    print(f"📈 Figure de comparaison : {fig1_path}")

    fig2_path = os.path.join(dest_root, f"Scatter_vs_COPS_{station_label}.png")
    csv_path = os.path.join(dest_root, f"Stats_vs_COPS_{station_label}.csv")
    df_stats = plot_scatter_and_stats(fig2_path, csv_path, cops_means, pysas_wl, pysas_specs)
    print(f"📊 Scatterplot : {fig2_path}")
    print(f"📋 Stats : {csv_path}")
    print(df_stats.to_string(index=False))

    best_by_variant = pick_best_method(df_stats)
    for variant, best_method in best_by_variant.items():
        cops_wl, cops_mean = cops_means[variant]
        pysas_spec_mean = np.nanmean(np.vstack(pysas_specs[best_method]), axis=0)
        unc_wl, uncs = gather_pysas_uncertainty(df_window, best_method)
        pysas_unc_mean = np.nanmean(np.vstack(uncs), axis=0) if uncs else None
        if pysas_unc_mean is not None and not np.array_equal(unc_wl, pysas_wl):
            pysas_unc_mean = np.interp(pysas_wl, unc_wl, pysas_unc_mean)

        fig3_path = os.path.join(dest_root, f"Rrs_BestMethod_{variant}_{station_label}.png")
        plot_best_method_uncertainty(fig3_path, variant, best_method, casts, cops_mean,
                                      pysas_wl, pysas_spec_mean, pysas_unc_mean, station_label)
        print(f"🏆 Meilleure méthode vs COPS {variant} : {best_method} -> {fig3_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("station_path", help="Chemin du dossier station (.../L2/YYYYMMDD_StationID/)")
    parser.add_argument("--copy", action="store_true",
                         help="Copier les fichiers L2 pySAS au lieu de créer des liens symboliques.")
    args = parser.parse_args()
    main(args.station_path, use_symlink=not args.copy)
