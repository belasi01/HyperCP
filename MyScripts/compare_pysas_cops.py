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
3. Copies (real copies by default -- see --symlink) the corresponding pySAS L2 HDF5
   files into a new .../L2/YYYYMMDD_StationID/pySAS/<method>/ subfolder, alongside the
   other instruments' data for that station. Also copies any all-sky image / 360
   mosaic whose timestamp falls in the same window, into AS_Camera/ and Mosaic360/
   subfolders next to it (see copy_camera_photos()).
4. Produces a comparison figure (Rrs vs wavelength, COPS + all 7 pySAS methods),
   a scatterplot (pySAS vs COPS per method, resampled onto COPS's wavelength grid),
   and a stats table (bias/RMSD/R² per method vs COPS) -- all saved under that
   pySAS/ subfolder.

Usage:
    conda activate hypercp
    python compare_pysas_cops.py /path/to/Amundsen_2026/L2/20260808_StationCS1-1/
    python compare_pysas_cops.py <station_path> --symlink   # links instead of real copies
"""
import os
import sys
import re
import glob
import shutil
import argparse

from datetime import datetime

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


def _copy_or_link(src, dst, use_symlink):
    """copyfile() plutôt que copy2() : copy2 essaie de préserver les xattrs (ex.
    com.apple.provenance sur macOS), ce que l'OS refuse pour un script -- lève
    "Operation not permitted" après que le contenu ait déjà été copié correctement."""
    if os.path.exists(dst) or os.path.islink(dst):
        return False
    if use_symlink:
        try:
            os.symlink(src, dst)
        except OSError:
            # e.g. CIFS/network share without symlink support
            shutil.copyfile(src, dst)
    else:
        shutil.copyfile(src, dst)
    return True


def copy_pysas_files(df_window, station_path, use_symlink=False):
    dest_root = os.path.join(station_path, "pySAS")
    for method in rea.METHODS:
        method_dest = os.path.join(dest_root, method)
        os.makedirs(method_dest, exist_ok=True)
        for fname in df_window["Filename"].unique():
            src = os.path.join(rea.BASE_PATH, method, "L2", fname)
            if not os.path.exists(src):
                continue
            _copy_or_link(src, os.path.join(method_dest, fname), use_symlink)
    return dest_root


def copy_camera_photos(station_path, date_str, window_start, window_end, use_symlink=False):
    """Copie (ou lie) les photos all-sky et mosaïques 360 dont l'horodatage tombe dans
    [window_start, window_end] (bornes UTC tz-aware), à côté du dossier pySAS/ de la
    station. Cherche d'abord dans le cache local déjà synchronisé (AS_Camera/,
    Mosaic360/, peuplé par sync_allsky_camera.py / sync_360_camera.py pour les casts
    M99NIR valides), sinon directement sur le montage SMB source -- pour couvrir aussi
    les horodatages qui ne correspondent à aucun cast pySAS valide (ex. HS6/COPS a
    tourné pendant un trou de QC pySAS). Utilisé par compare_pysas_cops.py et
    compare_pysas_hs6.py."""
    import sync_allsky_camera as sac
    import sync_360_camera as s360

    as_dest = os.path.join(station_path, "AS_Camera")
    mo_dest = os.path.join(station_path, "Mosaic360")

    n_as = 0
    as_day_dir = sac._find_day_dir(rea.CAMERA_DIR, date_str) or sac._find_day_dir(sac.CAMERA_SRC_ROOT, date_str)
    if as_day_dir:
        pattern = re.compile(rf"{date_str}(\d{{6}})_\d+\.jpg$")
        for fname in sorted(os.listdir(as_day_dir)):
            m = pattern.match(fname)
            if not m:
                continue
            frame_dt = pd.to_datetime(f"{date_str}{m.group(1)}", format="%Y%m%d%H%M%S", utc=True)
            if not (window_start <= frame_dt <= window_end):
                continue
            os.makedirs(as_dest, exist_ok=True)
            if _copy_or_link(os.path.join(as_day_dir, fname), os.path.join(as_dest, fname), use_symlink):
                n_as += 1

    n_mo = 0
    index = s360._index_mosaics(rea.MOSAIC_DIR, date_str) or s360._index_mosaics(s360.CAMERA360_SRC_ROOT, date_str)
    for entry_dt, path in index:
        if not (window_start <= entry_dt <= window_end):
            continue
        os.makedirs(mo_dest, exist_ok=True)
        if _copy_or_link(path, os.path.join(mo_dest, os.path.basename(path)), use_symlink):
            n_mo += 1

    print(f"📷 {n_as} image(s) all-sky, 🌐 {n_mo} mosaïque(s) 360 dans la fenêtre -> {as_dest} / {mo_dest}")
    return n_as, n_mo


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


def pick_best_method(df_stats, instrument_label="pySAS"):
    """Meilleure méthode par variante COPS (RMSD la plus faible, |biais| comme
    départage) -- ex: pour trancher entre 'loess vs Z17SimSpec (RMSD=0.00019)' et
    d'autres méthodes à RMSD quasi identique."""
    bias_col = f"Biais ({instrument_label}-COPS)"
    best = {}
    for variant, group in df_stats.groupby("Référence COPS"):
        row = group.assign(_abs_bias=group[bias_col].abs()) \
                    .sort_values(["RMSD", "_abs_bias"]).iloc[0]
        best[variant] = row["Méthode"]
    return best


# Style distinct par variante COPS, partagé entre les deux figures.
COPS_VARIANT_STYLE = {
    "linear": dict(color="black", linestyle="-", marker="o"),
    "loess": dict(color="dimgray", linestyle="--", marker="s"),
}


def plot_spectra_comparison(out_path, cops_casts, cops_means, pysas_wl, pysas_specs, station_label,
                            instrument_label="pySAS"):
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
    ax.set_title(f"Comparaison Rrs {instrument_label} vs COPS -- {station_label}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter_and_stats(out_fig_path, out_csv_path, cops_means, pysas_wl, pysas_specs,
                           instrument_label="pySAS"):
    """Un panneau de scatterplot par variante COPS (linear, loess), stats calculées
    séparément contre chacune."""
    bias_col = f"Biais ({instrument_label}-COPS)"
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
                bias_col: round(bias, 5), "RMSD": round(rmsd, 5),
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
        ax.set_ylabel(rf"$R_{{rs}}$ {instrument_label} (sr$^{{-1}}$)")
        ax.set_title(f"{instrument_label} vs COPS ({variant})")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_fig_path, dpi=150)
    plt.close(fig)

    df_stats = pd.DataFrame(records).sort_values(["Référence COPS", "RMSD"])
    df_stats.to_csv(out_csv_path, index=False)
    return df_stats


def plot_best_method_uncertainty(out_path, variant, best_method, cops_casts, cops_mean,
                                  pysas_wl, pysas_spec_mean, pysas_unc_mean, station_label,
                                  instrument_label="pySAS"):
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
            label=f"{instrument_label} {best_method}{unc_label}")

    ax.set_xlabel("Longueur d'onde (nm)")
    ax.set_ylabel(r"$R_{rs}$ (sr$^{-1}$)")
    ax.set_title(f"Meilleure méthode ({best_method}) vs COPS {variant} -- {station_label}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(station_path, use_symlink=False):
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

    copy_camera_photos(station_path, date_str, window_start, window_end, use_symlink=use_symlink)

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


# =============================================================================
# Ed0 (COPS, capteur de référence surface) vs Es (pySAS) -- comparaison systématique
# à la seconde près, indépendante de la comparaison Rrs ci-dessus.
#
# Contrairement à Rrs (qui utilise la fenêtre COPS complète et les ensembles L2 5 min),
# ici on veut la simultanéité fine : le capteur Ed0 du COPS (instrument de surface
# séparé du profil EdZ/LuZ, logue en continu pendant CHAQUE cast, bon ou mauvais) est
# comparé au Es pySAS au niveau L1BQC (dark déjà soustrait, scan par scan, ~1 Hz,
# PAS encore moyenné en ensembles 5 min) -- les deux binnés à 5 secondes.
# =============================================================================

BAND_WIDTH_NM = 10.0  # largeur assumée des bandes COPS -- moyenne Es pySAS sur +/- 5 nm
BIN_SECONDS = 5


def _parse_cops_datetime(s):
    """'DateTimeUTC' est presque toujours 'MM/DD/YYYY HH:MM:SS.mmm PM', mais la
    dernière ligne d'un cast omet parfois les millisecondes -- pandas 1.5.3 n'a pas
    de parsing multi-format (format="mixed", pandas>=2.0), fallback manuel."""
    try:
        return datetime.strptime(s, "%m/%d/%Y %I:%M:%S.%f %p")
    except ValueError:
        return datetime.strptime(s, "%m/%d/%Y %I:%M:%S %p")


def load_cops_ed0_series(cops_dir, only_selected=False):
    """Vraie irradiance Ed0 (µW/(cm² nm), capteur de référence surface), lue depuis le
    fichier COPS BRUT (.csv/.tsv à la racine de cops/, colonnes 'Ed0<wl>'), PAS le .nc
    traité -- vérifié que ed0_correction/_raw/_smoothed dans le .nc ne sont que des
    FACTEURS DE CORRECTION sans dimension (~0.79-1.28, moyenne ~1.0, probablement
    appliqués au profil EdZ/LuZ pour compenser les variations d'éclairement de surface
    pendant la descente), pas l'irradiance elle-même -- confirmé en comparant les
    magnitudes au Es pySAS (le .nc donnait un ratio Es/Ed0 de 46 à 105, alors que le
    brut donne des magnitudes du même ordre, ~90-110 µW/(cm² nm) à 443 nm des deux
    côtés). 'DateTimeUTC' est déjà en UTC (vérifié contre le 'time' du .nc, identique
    à la ms près). Par défaut TOUS les casts (pas seulement ceux retenus dans
    select.cops.dat) -- Ed0 est un capteur de surface continu, sa comparaison à Es ne
    dépend pas de la qualité du profil EdZ/LuZ de ce cast-là."""
    selected_files = None
    if only_selected:
        selected_files = set()
        with open(os.path.join(cops_dir, "select.cops.dat")) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(";")
                if len(parts) >= 2 and parts[1].strip() == "1":
                    selected_files.add(parts[0].strip())

    # Seulement les fichiers _URC (cast calibré, unités d'ingénierie µW/(cm² nm)) --
    # PAS _URU (même dossier, mêmes noms de colonnes "Ed0<wl>" mais tension brute non
    # calibrée "(xV)") ni _URE (housekeeping) : les inclure contaminerait la série
    # calibrée avec des valeurs de magnitude totalement différente sous le même nom de
    # colonne. Les dossiers cops/ routiniers (par station) ne contiennent que des
    # _URC.csv donc ce filtre ne change rien pour eux ; les dossiers d'expérience dédiée
    # (ex. Es_experiment/) contiennent aussi _URU/_URE/_LOG côte à côte.
    raw_files = sorted(glob.glob(os.path.join(cops_dir, "*_URC.csv")) +
                       glob.glob(os.path.join(cops_dir, "*_URC.tsv")))
    frames = []
    for raw_path in raw_files:
        fname = os.path.basename(raw_path)
        if selected_files is not None and fname not in selected_files:
            continue
        df = pd.read_csv(raw_path, sep=None, engine="python", encoding="latin-1")
        df.columns = [c.strip().strip('"') for c in df.columns]
        ed0_cols = [c for c in df.columns if re.match(r'^Ed0(\d+)(?:\s|$)', c)]
        if "DateTimeUTC" not in df.columns or not ed0_cols:
            # Le dossier cops/ contient aussi d'autres .csv (ex: GPS_<date>.csv) sans
            # rapport avec les casts -- silencieusement ignorés plutôt que plantés sur
            # un format de date différent.
            continue
        # Format majoritairement "MM/DD/YYYY HH:MM:SS.mmm PM", mais quelques lignes
        # (souvent la dernière d'un cast) omettent les millisecondes -- pandas 1.5.3
        # n'a pas format="mixed" (ajouté en 2.0), fallback manuel par ligne.
        times = pd.DatetimeIndex(
            [_parse_cops_datetime(s) for s in df["DateTimeUTC"].astype(str).str.strip('"')]
        ).tz_localize("UTC")
        for col in df.columns:
            m = re.match(r'^Ed0(\d+)(?:\s|$)', col)
            if not m:
                continue
            frames.append(pd.DataFrame({
                "Datetime": times, "Wavelength": float(m.group(1)), "Ed0": df[col].values,
            }))
    if not frames:
        return pd.DataFrame(columns=["Datetime", "Wavelength", "Ed0"])
    return pd.concat(frames, ignore_index=True)


def load_pysas_es_series(date_str, window_start, window_end):
    """Toutes les trames ES par scan (niveau L1BQC -- dark déjà soustrait, PAS encore
    binné en ensembles) des fichiers pySAS du jour, filtrées sur [window_start,
    window_end]. Longueurs d'onde natives HyperOCR (~3.3 nm de résolution)."""
    pattern = os.path.join(rea.BASE_PATH, "L1BQC", f"*{date_str}*_L1BQC.hdf")
    frames = []
    for fp in sorted(glob.glob(pattern)):
        with h5py.File(fp, "r") as h5f:
            if "/IRRADIANCE/ES" not in h5f:
                continue
            es_raw = h5f["/IRRADIANCE/ES"][...]
            wl_names = [n for n in es_raw.dtype.names if re.match(r'^[\d.]+$', n)]
            datetag = es_raw["Datetag"].astype(int)
            timetag2 = es_raw["Timetag2"].astype(int)
            times = pd.DatetimeIndex([
                dating.timeTag2ToDateTime(dating.dateTagToDateTime(d), t)
                for d, t in zip(datetag, timetag2)
            ])
            mask = (times >= window_start) & (times <= window_end)
            if not mask.any():
                continue
            for wl_name in wl_names:
                frames.append(pd.DataFrame({
                    "Datetime": times[mask], "Wavelength": float(wl_name),
                    "Es": es_raw[wl_name][mask],
                }))
    if not frames:
        return pd.DataFrame(columns=["Datetime", "Wavelength", "Es"])
    return pd.concat(frames, ignore_index=True)


def load_pysas_ancillary_series(date_str, window_start, window_end):
    """SZA (et Li(750)/Es(750) comme indicateur de nébulosité, même formule que
    extract_l2_qc_tables.py/detect_ship_shadow.py) par scan, pour contextualiser
    chaque bin Ed0/Es -- teste l'hypothèse ciel dégagé + SZA élevé -> écart plus fort."""
    pattern = os.path.join(rea.BASE_PATH, "L1BQC", f"*{date_str}*_L1BQC.hdf")
    frames = []
    for fp in sorted(glob.glob(pattern)):
        with h5py.File(fp, "r") as h5f:
            if "/ANCILLARY/SZA" not in h5f:
                continue
            sza_raw = h5f["/ANCILLARY/SZA"][...]
            datetag = sza_raw["Datetag"].astype(int)
            timetag2 = sza_raw["Timetag2"].astype(int)
            times = pd.DatetimeIndex([
                dating.timeTag2ToDateTime(dating.dateTagToDateTime(d), t)
                for d, t in zip(datetag, timetag2)
            ])
            mask = (times >= window_start) & (times <= window_end)
            if not mask.any():
                continue
            sza = sza_raw["SZA"][mask]

            cloud_ratio = np.full(mask.sum(), np.nan)
            if "/IRRADIANCE/ES" in h5f and "/RADIANCE/LI" in h5f:
                from scipy.interpolate import interp1d
                es_raw = h5f["/IRRADIANCE/ES"][...][mask]
                li_raw = h5f["/RADIANCE/LI"][...][mask]
                es_wl = sorted([c for c in es_raw.dtype.names if re.match(r'^[\d.]+$', c)], key=float)
                li_wl = sorted([c for c in li_raw.dtype.names if re.match(r'^[\d.]+$', c)], key=float)
                if es_wl and li_wl:
                    es750 = interp1d([float(w) for w in es_wl], np.array([es_raw[w] for w in es_wl]), axis=0)(750.0)
                    li750 = interp1d([float(w) for w in li_wl], np.array([li_raw[w] for w in li_wl]), axis=0)(750.0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        cloud_ratio = np.where(es750 != 0, li750 / es750, np.nan)

            frames.append(pd.DataFrame({"Datetime": times[mask], "SZA": sza, "CloudRatio": cloud_ratio}))
    if not frames:
        return pd.DataFrame(columns=["Datetime", "SZA", "CloudRatio"])
    return pd.concat(frames, ignore_index=True)


def bin_5s(df, time_col="Datetime", group_cols=None, value_cols=("Value",), bin_seconds=BIN_SECONDS):
    """Floor Datetime à des bins de bin_seconds, moyenne des value_cols par
    (bin, *group_cols)."""
    out = df.copy()
    out["Bin"] = out[time_col].dt.floor(f"{bin_seconds}s")
    group = ["Bin"] + list(group_cols or [])
    return out.groupby(group, as_index=False)[list(value_cols)].mean()


def match_ed0_es(cops_dir, date_str, label, band_width=BAND_WIDTH_NM, bin_seconds=BIN_SECONDS):
    """Coeur du matching Ed0 (COPS) vs Es (pySAS) pour un dossier cops/ et une date
    donnés -- indépendant de l'arborescence L2/station : partagé par
    match_ed0_es_for_station (dossier station .../L2/.../cops/) et les scripts
    d'expérience dédiée (ex. compare_es_ed0_experiment.py, dossier cops/ autonome hors
    arborescence L2). Ed0 COPS (tous casts, brut calibré _URC) vs Es pySAS (L1BQC par
    scan, moyenné sur +/- band_width/2 nm autour de chaque longueur d'onde COPS), tous
    deux binnés à bin_seconds, appariés sur le bin temporel commun. Retourne un
    DataFrame long [Station, Bin, Wavelength_COPS, Ed0, Es, SZA, CloudRatio]."""
    ed0_df = load_cops_ed0_series(cops_dir)
    if ed0_df.empty:
        print(f"⚠️  [{label}] Aucune série Ed0 exploitable.")
        return pd.DataFrame()

    window_start = ed0_df["Datetime"].min() - pd.Timedelta(minutes=1)
    window_end = ed0_df["Datetime"].max() + pd.Timedelta(minutes=1)

    es_df = load_pysas_es_series(date_str, window_start, window_end)
    if es_df.empty:
        print(f"⚠️  [{label}] Aucune donnée Es pySAS (L1BQC) dans la fenêtre {window_start} -> {window_end}.")
        return pd.DataFrame()

    anc_df = load_pysas_ancillary_series(date_str, window_start, window_end)
    anc_binned = bin_5s(anc_df, group_cols=[], value_cols=["SZA", "CloudRatio"], bin_seconds=bin_seconds) \
        if not anc_df.empty else pd.DataFrame(columns=["Bin", "SZA", "CloudRatio"])

    ed0_binned = bin_5s(ed0_df, group_cols=["Wavelength"], value_cols=["Ed0"], bin_seconds=bin_seconds)
    ed0_binned = ed0_binned.rename(columns={"Wavelength": "Wavelength_COPS"})

    records = []
    for cops_wl in sorted(ed0_binned["Wavelength_COPS"].unique()):
        es_band = es_df[(es_df["Wavelength"] >= cops_wl - band_width / 2) &
                        (es_df["Wavelength"] <= cops_wl + band_width / 2)]
        if es_band.empty:
            continue
        es_band_binned = bin_5s(es_band, group_cols=[], value_cols=["Es"], bin_seconds=bin_seconds)

        ed0_wl = ed0_binned[ed0_binned["Wavelength_COPS"] == cops_wl][["Bin", "Ed0"]]
        merged = ed0_wl.merge(es_band_binned, on="Bin", how="inner")
        if merged.empty:
            continue
        merged = merged.merge(anc_binned, on="Bin", how="left")
        merged["Wavelength_COPS"] = cops_wl
        merged["Station"] = label
        records.append(merged)

    if not records:
        print(f"⚠️  [{label}] Aucun bin de {bin_seconds}s simultané entre Ed0 et Es.")
        return pd.DataFrame()
    df = pd.concat(records, ignore_index=True)
    print(f"📍 [{label}] {len(df)} bin(s) de {bin_seconds}s Ed0/Es apparié(s) sur "
         f"{df['Wavelength_COPS'].nunique()} longueur(s) d'onde COPS.")
    return df


def match_ed0_es_for_station(station_path, band_width=BAND_WIDTH_NM, bin_seconds=BIN_SECONDS):
    """Enveloppe de match_ed0_es pour un dossier station .../L2/YYYYMMDD_StationID/ --
    en dérive cops_dir/date_str/label plutôt que de les recevoir directement."""
    station_path = os.path.abspath(station_path)
    date_str = parse_station_date(station_path)
    label = os.path.basename(station_path)
    cops_dir = find_cops_dir(station_path)
    return match_ed0_es(cops_dir, date_str, label, band_width=band_width, bin_seconds=bin_seconds)


def plot_ed0_es_per_wavelength(df, out_dir, label="toutes stations"):
    """Un scatterplot Es (pySAS) vs Ed0 (COPS) par longueur d'onde COPS, coloré par
    SZA -- pour vérifier l'hypothèse Es > Ed0, écart plus fort à ciel dégagé + SZA
    élevé. Retourne le DataFrame de stats par longueur d'onde."""
    os.makedirs(out_dir, exist_ok=True)
    wavelengths = sorted(df["Wavelength_COPS"].unique())
    n = len(wavelengths)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)

    stats_rows = []
    for ax, wl in zip(axes.flat, wavelengths):
        sub = df[df["Wavelength_COPS"] == wl].dropna(subset=["Ed0", "Es"])
        if sub.empty:
            ax.axis("off")
            continue
        sc = ax.scatter(sub["Ed0"], sub["Es"], c=sub["SZA"], cmap="plasma", s=12, alpha=0.7)
        lims = [min(sub["Ed0"].min(), sub["Es"].min()), max(sub["Ed0"].max(), sub["Es"].max())]
        ax.plot(lims, lims, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f"{wl:.0f} nm (n={len(sub)})", fontsize=9)
        ax.tick_params(labelsize=7)

        diff = sub["Es"] - sub["Ed0"]
        ratio = sub["Es"] / sub["Ed0"]
        clear = sub["CloudRatio"] < 0.05
        stats_rows.append({
            "Wavelength_COPS": wl, "N": len(sub),
            "Bias_Es-Ed0": float(diff.mean()), "Ratio_Es/Ed0": float(ratio.mean()),
            "Corr_diff_vs_SZA": float(np.corrcoef(sub["SZA"], diff)[0, 1]) if sub["SZA"].notna().sum() > 2 else None,
            "Bias_clear_sky": float(diff[clear].mean()) if clear.any() else None,
            "Bias_cloudy": float(diff[~clear].mean()) if (~clear).any() else None,
        })

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.colorbar(sc, ax=axes, shrink=0.6, label="SZA (°)")
    fig.suptitle(f"Es (pySAS, L1BQC) vs Ed0 (COPS, brut) -- bins {BIN_SECONDS}s -- {label}", fontweight="bold")
    out_path = os.path.join(out_dir, "Ed0_vs_Es_per_wavelength.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"📊 Figure : {out_path}")

    df_stats = pd.DataFrame(stats_rows).sort_values("Wavelength_COPS")
    csv_path = os.path.join(out_dir, "Ed0_vs_Es_stats.csv")
    df_stats.to_csv(csv_path, index=False)
    print(f"📋 Stats : {csv_path}")
    print(df_stats.to_string(index=False))

    plot_stats_vs_wavelength(df_stats, out_dir, label=label)
    return df_stats


# Palette catégorielle (dataviz skill: références/palette.md) -- 3 séries identité
# (biais global / ciel dégagé / nuageux), teintes fixes 1-2-3, jamais recyclées.
_CAT_BLUE = "#2a78d6"
_CAT_ORANGE = "#eb6834"
_CAT_AQUA = "#1baf7a"
_INK_SECONDARY = "#52514e"
_GRIDLINE = "#e1e0d9"
_CHART_SURFACE = "#fcfcfb"


def plot_stats_vs_wavelength(df_stats, out_dir, label="toutes stations"):
    """Biais (global/ciel dégagé/nuageux), ratio et corrélation au SZA en fonction de
    la longueur d'onde -- pour lire le comportement spectral d'un coup d'oeil plutôt
    que colonne par colonne dans le CSV."""
    wl = df_stats["Wavelength_COPS"].values

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, facecolor=_CHART_SURFACE)

    ax = axes[0]
    ax.set_facecolor(_CHART_SURFACE)
    ax.axhline(0, color=_INK_SECONDARY, linewidth=0.8)
    ax.plot(wl, df_stats["Bias_Es-Ed0"], color=_CAT_BLUE, marker="o", linewidth=2, label="Biais global")
    ax.plot(wl, df_stats["Bias_clear_sky"], color=_CAT_ORANGE, marker="o", linewidth=2, label="Ciel dégagé")
    ax.plot(wl, df_stats["Bias_cloudy"], color=_CAT_AQUA, marker="o", linewidth=2, label="Nuageux")
    ax.set_ylabel("Biais Es-Ed0")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", color=_GRIDLINE)

    ax = axes[1]
    ax.set_facecolor(_CHART_SURFACE)
    ax.axhline(1, color=_INK_SECONDARY, linewidth=0.8)
    ax.plot(wl, df_stats["Ratio_Es/Ed0"], color=_CAT_BLUE, marker="o", linewidth=2)
    ax.set_ylabel("Ratio Es/Ed0")
    ax.grid(True, linestyle="--", color=_GRIDLINE)

    ax = axes[2]
    ax.set_facecolor(_CHART_SURFACE)
    ax.axhline(0, color=_INK_SECONDARY, linewidth=0.8)
    ax.plot(wl, df_stats["Corr_diff_vs_SZA"], color=_CAT_BLUE, marker="o", linewidth=2)
    ax.set_ylabel("Corrélation (Es-Ed0) vs SZA")
    ax.set_xlabel("Longueur d'onde COPS (nm)")
    ax.grid(True, linestyle="--", color=_GRIDLINE)

    fig.suptitle(f"Ed0 vs Es -- statistiques spectrales -- {label}", fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "Ed0_vs_Es_stats_vs_wavelength.png")
    fig.savefig(out_path, dpi=150, facecolor=_CHART_SURFACE)
    plt.close(fig)
    print(f"📊 Figure : {out_path}")


def discover_stations_with_cops():
    """Racine L2 dérivée de rea.MAIN_DATA_PATH (.../Amundsen_2026/L1/ -> .../L2/) --
    toutes les stations ayant un sous-dossier cops/, pour --ed0-vs-es --all."""
    l2_root = os.path.join(os.path.dirname(os.path.normpath(rea.MAIN_DATA_PATH)), "L2")
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(l2_root, "*", "cops")))


def compare_ed0_es(station_paths, out_dir):
    """Point d'entrée agrégé -- une ou plusieurs stations, un seul jeu de figures/stats
    combinant tous les bins Ed0/Es appariés (plus de puissance statistique pour juger
    l'effet SZA/nébulosité que station par station)."""
    all_dfs = []
    for station_path in station_paths:
        try:
            df = match_ed0_es_for_station(station_path)
        except Exception as e:
            print(f"❌ [{os.path.basename(station_path)}] {e}")
            continue
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("⚠️  Aucune donnée Ed0/Es appariée sur les stations fournies.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    df_all.to_csv(os.path.join(out_dir, "Ed0_vs_Es_matched_bins.csv"), index=False)
    plot_ed0_es_per_wavelength(df_all, out_dir, label=f"{df_all['Station'].nunique()} station(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("station_path", nargs="?", default=None,
                         help="Chemin du dossier station (.../L2/YYYYMMDD_StationID/) -- mode Rrs par défaut, "
                              "ou une station unique pour --ed0-vs-es.")
    parser.add_argument("--symlink", action="store_true",
                         help="Lier symboliquement au lieu de copier (économise l'espace disque, mais "
                              "les liens deviennent inutilisables sans accès au montage SMB -- ex. après "
                              "la fin de la campagne. Copie réelle par défaut.)")
    parser.add_argument("--ed0-vs-es", action="store_true",
                         help="Mode comparaison Ed0 (COPS, brut) vs Es (pySAS, L1BQC par scan) binnée à 5s, "
                              "au lieu de la comparaison Rrs par défaut.")
    parser.add_argument("--station", action="append", default=None,
                         help="Nom de dossier station sous .../L2/ (répétable) -- pour --ed0-vs-es.")
    parser.add_argument("--all", action="store_true",
                         help="Toutes les stations avec un dossier cops/ -- pour --ed0-vs-es.")
    parser.add_argument("--out-dir", default=None,
                         help="Dossier de sortie pour --ed0-vs-es (défaut: <MAIN_DATA_PATH>/pySAS/Ed0_vs_Es/).")
    args = parser.parse_args()

    if args.ed0_vs_es:
        if args.all:
            station_paths = discover_stations_with_cops()
        elif args.station:
            l2_root = os.path.join(os.path.dirname(os.path.normpath(rea.MAIN_DATA_PATH)), "L2")
            station_paths = [os.path.join(l2_root, s) for s in args.station]
        elif args.station_path:
            station_paths = [args.station_path]
        else:
            parser.error("--ed0-vs-es nécessite station_path, --station (répétable) ou --all")
        out_dir = args.out_dir or os.path.join(rea.MAIN_DATA_PATH, "pySAS", "Ed0_vs_Es")
        compare_ed0_es(station_paths, out_dir)
    else:
        if not args.station_path:
            parser.error("station_path requis pour le mode Rrs (par défaut) -- ou utiliser --ed0-vs-es")
        main(args.station_path, use_symlink=args.symlink)
