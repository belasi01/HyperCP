"""
compare_pysas_hs6.py - Extrait les casts pySAS L2 (7 methodes de correction du ciel)
correspondant a la fenetre temporelle d'un profil HS6/IOP (pyIOPs) d'une station, pour
les avoir disponibles a cote des autres instruments de la station.

Contrairement a COPS (Rrs vs Rrs, voir compare_pysas_cops.py), le HS6 mesure la
retrodiffusion (bb) -- il n'y a pas de comparaison directe possible avec du Rrs. Ce
script se limite donc a :
1. Lire le(s) fichier(s) HS6 .nc du dossier iops/ de la station et leur fenetre UTC
   (attributs globaux time_window_start/time_window_stop, ecrits par pyIOPs).
2. Trouver les casts pySAS (toutes methodes) dont le timestamp tombe dans cette
   fenetre, via le L2_Methods_Quotes_<date>.csv du jour (meme source que
   rrs_explorer_app.py).
3. Copier (copies reelles par defaut -- voir --symlink) les fichiers L2 pySAS
   correspondants dans .../pySAS/<methode>/, a cote des autres instruments de la
   station. Copier aussi toute image all-sky / mosaique 360 dont l'horodatage tombe
   dans la meme fenetre, dans AS_Camera/ et Mosaic360/ (cf. compare_pysas_cops.py
   ::copy_camera_photos, reutilisee ici).
4. Sauvegarder une figure du spectre Rrs pySAS moyen (toutes methodes, casts
   individuels en trait fin) sur la fenetre -- pour contexte visuel uniquement, ce
   n'est pas une comparaison au HS6.

Usage:
    conda activate hypercp
    python compare_pysas_hs6.py /path/to/Amundsen_2026/L2/20260724_StationRAM5/
    python compare_pysas_hs6.py <station_path> --symlink   # liens plutot que des copies reelles
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

MY_DIR = os.path.dirname(os.path.abspath(__file__))
if MY_DIR not in sys.path:
    sys.path.insert(0, MY_DIR)

# Reutilise la logique deja ecrite pour la comparaison COPS (parsing du nom de
# station, recherche des casts pySAS dans une fenetre, copie, extraction des
# spectres) plutot que de la dupliquer.
import rrs_explorer_app as rea  # noqa: E402
import compare_pysas_cops as cpc  # noqa: E402


def find_iops_dir(station_path):
    for name in sorted(os.listdir(station_path)):
        full = os.path.join(station_path, name)
        if os.path.isdir(full) and name.lower().startswith("iops"):
            return full
    raise FileNotFoundError(f"Aucun sous-dossier iops/ trouvé dans {station_path}")


def load_hs6_windows(iops_dir):
    """Lit la fenêtre temporelle UTC (et la position) de chaque fichier HS6 .nc du
    dossier iops/ de la station -- normalement un seul profil, mais on gère aussi le
    cas de plusieurs profils répétés à la même station (fenêtre globale = union)."""
    nc_files = sorted(glob.glob(os.path.join(iops_dir, "HS_*.nc")))
    if not nc_files:
        raise FileNotFoundError(f"Aucun fichier HS6 (HS_*.nc) dans {iops_dir}")

    windows = []
    for nc_path in nc_files:
        with xr.open_dataset(nc_path) as ds:
            windows.append({
                "file": os.path.basename(nc_path),
                "start": pd.Timestamp(ds.attrs["time_window_start"]),
                "end": pd.Timestamp(ds.attrs["time_window_stop"]),
                "lat": ds.attrs.get("latitude"),
                "lon": ds.attrs.get("longitude"),
            })
    return windows


def plot_pysas_context(out_path, pysas_wl, pysas_specs, station_label, window_start, window_end):
    """Spectres Rrs pySAS (casts individuels en trait fin, moyenne par méthode en
    trait plein) sur la fenêtre du profil HS6 -- contexte visuel, pas de comparaison
    au HS6 (grandeurs physiques différentes : Rrs vs bb)."""
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

    ax.set_xlabel("Longueur d'onde (nm)")
    ax.set_ylabel(r"$R_{rs}$ (sr$^{-1}$)")
    ax.set_title(
        f"Rrs pySAS pendant le profil HS6 -- {station_label}\n"
        f"{window_start:%H:%M:%S} -> {window_end:%H:%M:%S} UTC"
    )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(station_path, use_symlink=False):
    station_path = os.path.abspath(station_path)
    date_str = cpc.parse_station_date(station_path)
    station_label = os.path.basename(station_path)

    iops_dir = find_iops_dir(station_path)
    windows = load_hs6_windows(iops_dir)
    window_start = min(w["start"] for w in windows)
    window_end = max(w["end"] for w in windows)
    print(f"📍 {station_label} | fenêtre HS6 UTC : {window_start} -> {window_end} "
          f"({len(windows)} profil(s))")

    df_window = cpc.find_pysas_casts_in_window(date_str, window_start, window_end)
    if df_window.empty:
        raise ValueError("Aucun cast pySAS trouvé dans la fenêtre temporelle HS6 -- "
                          "vérifier que la date a été traitée jusqu'à extract_l2_qc_tables.py.")
    print(f"🔗 {len(df_window)} cast(s) pySAS trouvé(s) dans cette fenêtre.")

    dest_root = cpc.copy_pysas_files(df_window, station_path, use_symlink=use_symlink)
    print(f"📂 Données pySAS {'liées' if use_symlink else 'copiées'} dans : {dest_root}")

    cpc.copy_camera_photos(station_path, date_str, window_start, window_end, use_symlink=use_symlink)

    pysas_wl, pysas_specs = cpc.gather_pysas_spectra(df_window)
    if pysas_wl is None:
        raise ValueError("Aucun spectre Rrs pySAS lisible pour les casts de la fenêtre.")

    fig_path = os.path.join(dest_root, f"Rrs_pySAS_context_HS6_{station_label}.png")
    plot_pysas_context(fig_path, pysas_wl, pysas_specs, station_label, window_start, window_end)
    print(f"📈 Figure de contexte pySAS : {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("station_path", help="Chemin du dossier station (.../L2/YYYYMMDD_StationID/)")
    parser.add_argument("--symlink", action="store_true",
                         help="Lier symboliquement au lieu de copier (économise l'espace disque, mais "
                              "les liens deviennent inutilisables sans accès au montage SMB -- ex. après "
                              "la fin de la campagne. Copie réelle par défaut.)")
    args = parser.parse_args()
    main(args.station_path, use_symlink=args.symlink)
