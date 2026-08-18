"""
match_pysas_leg1_stations.py - Pour les stations d'un leg donne ou des echantillons
HPLC ET POC-PON (CHN) ont ete filtres (Amundsen2026_Ardyna_MasterSheet.xlsx), trouve
les casts pySAS L2 correspondants et cree/peuple le dossier station .../L2/
YYYYMMDD_StationID/pySAS/<methode>/ -- contrairement a compare_pysas_cops.py et
compare_pysas_hs6.py, le dossier station n'existe pas encore : c'est ce script qui le
cree, a partir des metadonnees du MasterSheet (pas d'un cast COPS/HS6 deja traite).

Pour chaque station :
1. Fenetre CTD = [start_rosette_time_utc, end_rosette_time_utc] (colonnes du
   MasterSheet), position = latitude/longitude du MasterSheet (format "N 54°48,662" /
   "W 052°57,464", ou juste "58°52,087" sans lettre -- toute la mission etant dans
   l'hemisphere nord-ouest, le signe est applique sans avoir besoin de la lettre).
2. Match temporel : cast(s) pySAS dont l'horodatage tombe dans la fenetre CTD (meme
   logique que compare_pysas_cops.py::find_pysas_casts_in_window).
3. A defaut, match spatial (repli) : cast(s) pySAS a moins de --max-distance-km de la
   station ET a moins de --max-offset-hours de la fenetre CTD. Seuils fixes avec Simon :
   1 km / 3h.
4. Conflit entre stations : un meme cast pySAS peut satisfaire le critere spatial de
   PLUSIEURS stations proches (ex. HT2-1000 et HT2-1500). Il n'est alors attribue qu'a
   la station dont la fenetre CTD est la plus proche dans le temps (voir
   compute_all_matches) -- jamais duplique.
5. Si rien ne satisfait ni l'un ni l'autre (ou si la journee n'a pas ete traitee dans
   HyperCP jusqu'a extract_l2_qc_tables.py -- pas de L2_Methods_Quotes_<date>.csv), la
   station est rapportee sans match, aucun dossier n'est cree pour elle.
6. Pour chaque station matchee, sauvegarde aussi Rrs_mean_std_<station>.png (Rrs moyen
   +/- ecart-type par methode, sur les casts pySAS retenus) dans le dossier pySAS/.

Par defaut le script ne fait qu'un rapport (aucun dossier cree, aucune copie) --
relancer avec --apply pour creer les dossiers et copier les fichiers pySAS L2. En mode
--apply, le dossier pySAS/ de chaque station est synchronise EXACTEMENT sur le match
courant (sync_pysas_files) : les fichiers qui ne font plus partie du match (ex. apres
resserrement des seuils) sont retires, pas seulement les nouveaux ajoutes. Si une
station qui avait un match sous un seuil plus permissif n'en a plus, son dossier pySAS/
(et le dossier station, s'il devient vide) est supprime (remove_station_pysas).

Usage:
    conda activate hypercp
    python match_pysas_leg1_stations.py                 # rapport seulement
    python match_pysas_leg1_stations.py --apply          # cree les dossiers + copie
    python match_pysas_leg1_stations.py --apply --symlink
"""
import os
import re
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MY_DIR = os.path.dirname(os.path.abspath(__file__))
if MY_DIR not in sys.path:
    sys.path.insert(0, MY_DIR)

# rrs_explorer_app insere la racine HyperCP dans sys.path (necessaire pour Source.*) --
# doit donc etre importe avant Source.utils.dating.
import rrs_explorer_app as rea  # noqa: E402
import Source.utils.dating as dating  # noqa: E402
import compare_pysas_cops as cpc  # noqa: E402

MASTERSHEET_PATH = os.path.expanduser("~/Data/Amundsen_2026/Log/Amundsen2026_Ardyna_MasterSheet.xlsx")
L2_ROOT = os.path.expanduser("~/Data/Amundsen_2026/L2")

# Corrections connues d'erreurs de saisie dans le MasterSheet (station, cast) -> date
# correcte. RA01/002 y est date 2026-08-13 (qui correspond en realite a la station
# CS2-1, Leg 2) ; Simon confirme que ce cast a ete fait le 2026-07-12.
DATE_OVERRIDES = {
    ("RA01", "002"): "2026-07-12",
}


def parse_dm(coord):
    """"N 54°48,662" / "W 052°57,464" / "58°52,087" (sans lettre) -> degres decimaux
    non signes (le signe est applique par l'appelant selon lat/lon, cf. read_leg_casts)."""
    m = re.search(r"(\d+)\s*°?\s*(\d+[.,]\d+)", str(coord))
    if not m:
        raise ValueError(f"Coordonnee illisible : {coord!r}")
    deg, minute = float(m.group(1)), float(m.group(2).replace(",", "."))
    return deg + minute / 60


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def station_folder_name(date, station):
    clean = re.sub(r"\s+", "", str(station))
    return f"{date:%Y%m%d}_Station{clean}"


def read_leg_casts(mastersheet_path, leg):
    """Casts (station, cast) du leg donne presents A LA FOIS dans HPLC et POC-PON,
    avec fenetre temporelle UTC et position issues du MasterSheet."""
    cols = ["station", "cast", "date", "latitude", "longitude",
            "start_rosette_time_utc", "end_rosette_time_utc"]

    hplc = pd.read_excel(mastersheet_path, sheet_name="HPLC", header=0)
    poc = pd.read_excel(mastersheet_path, sheet_name="POC-PON", header=0)
    hplc.columns = [str(c).strip() for c in hplc.columns]
    poc.columns = [str(c).strip() for c in poc.columns]

    hplc_casts = hplc.loc[hplc["leg"] == leg, cols].drop_duplicates(subset=["station", "cast"])
    poc_casts = poc.loc[poc["leg"] == leg, cols].drop_duplicates(subset=["station", "cast"])

    merged = pd.merge(hplc_casts, poc_casts, on=["station", "cast"], suffixes=("", "_poc"))

    mismatches = merged[merged["date"] != merged["date_poc"]]
    if not mismatches.empty:
        print("⚠️  Dates HPLC/POC-PON discordantes pour ces casts (verifier le MasterSheet) :")
        print(mismatches[["station", "cast", "date", "date_poc"]].to_string(index=False))

    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_poc")])

    for (station, cast), new_date in DATE_OVERRIDES.items():
        idx = (merged["station"] == station) & (merged["cast"].astype(str) == cast)
        if idx.any():
            merged.loc[idx, "date"] = pd.Timestamp(new_date)

    merged["lat"] = merged["latitude"].apply(parse_dm)
    merged["lon"] = -merged["longitude"].apply(parse_dm)

    def build_window(row):
        d = pd.Timestamp(row["date"]).normalize()
        start = (d + pd.to_timedelta(str(row["start_rosette_time_utc"]))).tz_localize("UTC")
        end = (d + pd.to_timedelta(str(row["end_rosette_time_utc"]))).tz_localize("UTC")
        if end < start:
            end += pd.Timedelta(days=1)
        return start, end

    starts, ends = zip(*merged.apply(build_window, axis=1))
    merged["window_start"], merged["window_end"] = starts, ends

    return merged.reset_index(drop=True)


def load_pysas_day(date_str):
    try:
        return rea.load_casts(date_str)
    except FileNotFoundError:
        return None


def gather_candidate_casts(window_start, window_end, tolerance):
    """Concatene les casts pySAS des journees couvrant [window_start-tol, window_end+tol]
    (>1 journee seulement si la fenetre ou la tolerance chevauche minuit UTC)."""
    dates = sorted({
        (window_start - tolerance).date(), window_start.date(),
        window_end.date(), (window_end + tolerance).date(),
    })
    frames, available_dates = [], []
    for d in dates:
        date_str = d.strftime("%Y%m%d")
        df = load_pysas_day(date_str)
        if df is None:
            continue
        available_dates.append(date_str)
        frames.append(df)
    if not frames:
        return None, available_dates

    df_all = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Filename", "Timetag2"])
    cast_dt = [
        pd.Timestamp(dating.timeTag2ToDateTime(
            dating.dateTagToDateTime(int(row.Datetag)), int(row.Timetag2)))
        for row in df_all.itertuples()
    ]
    return df_all.assign(_dt=cast_dt), available_dates


def compute_all_matches(casts, max_dist_km, max_offset):
    """Match temporel/spatial pour chaque station (voir docstring du module), avec
    resolution des conflits : si un meme cast pySAS satisfait le critere spatial de
    plusieurs stations en attente, il n'est retenu que par celle dont la fenetre CTD
    est temporellement la plus proche -- jamais copie dans plus d'un dossier station.

    Retourne {index_ligne_casts: (statut, df_match_ou_None, diagnostic)}."""
    results = {}
    candidate_rows = []  # (idx, Filename, offset, dist) -- pour arbitrage inter-stations

    for idx, row in enumerate(casts.itertuples()):
        df_all, available_dates = gather_candidate_casts(row.window_start, row.window_end, max_offset)
        diag = {"available_dates": available_dates, "best_distance_km": None, "best_offset": None}
        if df_all is None:
            results[idx] = ("no_data", None, diag)
            continue

        temporal = df_all[(df_all["_dt"] >= row.window_start) & (df_all["_dt"] <= row.window_end)]
        if not temporal.empty:
            results[idx] = ("temporal", temporal.drop(columns=["_dt"]).reset_index(drop=True), diag)
            continue

        dist = haversine_km(row.lat, row.lon, df_all["Latitude"].astype(float), df_all["Longitude"].astype(float))
        offset = df_all["_dt"].apply(lambda dt: max(row.window_start - dt, dt - row.window_end, pd.Timedelta(0)))
        diag["best_distance_km"] = round(float(dist.min()), 3)
        diag["best_offset"] = offset.loc[dist.idxmin()]

        mask = (dist <= max_dist_km) & (offset <= max_offset)
        candidates = df_all[mask].assign(_offset=offset[mask])
        results[idx] = ("spatial_pending", candidates, diag)
        for filename, off in zip(candidates["Filename"], candidates["_offset"]):
            candidate_rows.append((idx, filename, off))

    if candidate_rows:
        df_cand = pd.DataFrame(candidate_rows, columns=["idx", "Filename", "offset"])
        winners = df_cand.loc[df_cand.groupby("Filename")["offset"].idxmin()]
        for idx, group in winners.groupby("idx"):
            _, candidates, diag = results[idx]
            final = (candidates[candidates["Filename"].isin(set(group["Filename"]))]
                     .drop(columns=["_offset"]).reset_index(drop=True))
            results[idx] = ("spatial", final, diag)

    for idx, (status, candidates, diag) in results.items():
        if status == "spatial_pending":
            results[idx] = ("none", None, diag)

    return results


MANIFEST_NAME = ".match_pysas_leg_manifest.json"


def _load_manifest(dest_root):
    path = os.path.join(dest_root, MANIFEST_NAME)
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


def _save_manifest(dest_root, filenames):
    with open(os.path.join(dest_root, MANIFEST_NAME), "w") as f:
        json.dump(sorted(filenames), f)


def sync_pysas_files(df_match, station_path, use_symlink=False):
    """Comme cpc.copy_pysas_files, mais retire aussi les fichiers qui ne font plus
    partie du match courant -- SEULEMENT ceux que ce script avait lui-meme ajoutes lors
    d'un run precedent (suivi via .match_pysas_leg_manifest.json dans pySAS/). Ne touche
    jamais aux fichiers ajoutes par un autre script (ex. compare_pysas_cops.py,
    compare_pysas_hs6.py) pour la meme station -- important car plusieurs scripts
    peuvent peupler le meme dossier pySAS/<methode>/ pour une station donnee, chacun
    avec sa propre fenetre temporelle de match."""
    dest_root = os.path.join(station_path, "pySAS")
    os.makedirs(dest_root, exist_ok=True)
    previously_managed = _load_manifest(dest_root)
    wanted = set(df_match["Filename"].unique())
    to_remove = previously_managed - wanted

    for method in rea.METHODS:
        method_dest = os.path.join(dest_root, method)
        os.makedirs(method_dest, exist_ok=True)
        for fname in to_remove:
            fpath = os.path.join(method_dest, fname)
            if os.path.exists(fpath) or os.path.islink(fpath):
                os.remove(fpath)
        for fname in wanted:
            src = os.path.join(rea.BASE_PATH, method, "L2", fname)
            if os.path.exists(src):
                cpc._copy_or_link(src, os.path.join(method_dest, fname), use_symlink)

    _save_manifest(dest_root, wanted)
    return dest_root


def remove_station_pysas(station_path):
    """Retire uniquement les fichiers geres par ce script (voir manifeste dans
    sync_pysas_files) -- jamais ceux ajoutes par un autre script pour la meme station.
    pySAS/<methode>/ n'est supprime que s'il devient vide, pySAS/ que s'il ne reste plus
    que le manifeste, et le dossier station que s'il devient completement vide."""
    dest_root = os.path.join(station_path, "pySAS")
    if not os.path.isdir(dest_root):
        return False

    previously_managed = _load_manifest(dest_root)
    for method in rea.METHODS:
        method_dest = os.path.join(dest_root, method)
        if not os.path.isdir(method_dest):
            continue
        for fname in previously_managed:
            fpath = os.path.join(method_dest, fname)
            if os.path.exists(fpath) or os.path.islink(fpath):
                os.remove(fpath)
        if not os.listdir(method_dest):
            os.rmdir(method_dest)

    manifest_path = os.path.join(dest_root, MANIFEST_NAME)
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    if os.path.isdir(dest_root) and not os.listdir(dest_root):
        os.rmdir(dest_root)
    if os.path.isdir(station_path) and not os.listdir(station_path):
        os.rmdir(station_path)
    return bool(previously_managed)


def plot_rrs_mean_std(out_path, pysas_wl, pysas_specs, station_label):
    """Rrs moyen +/- ecart-type (bande ombree) par methode, sur les casts pySAS
    retenus pour la station -- pas de casts individuels (voir plot_pysas_context dans
    compare_pysas_hs6.py pour cette variante)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for method in rea.METHODS:
        casts = pysas_specs[method]
        if not casts:
            continue
        arr = np.vstack(casts)
        mean_spec = np.nanmean(arr, axis=0)
        std_spec = np.nanstd(arr, axis=0)
        color = rea.METHOD_COLORS[method]
        ax.fill_between(pysas_wl, mean_spec - std_spec, mean_spec + std_spec, color=color, alpha=0.15)
        ax.plot(pysas_wl, mean_spec, color=color, linewidth=2, label=f"{method} (n={len(casts)})")

    ax.set_xlabel("Longueur d'onde (nm)")
    ax.set_ylabel(r"$R_{rs}$ (sr$^{-1}$)")
    ax.set_title(f"Rrs pySAS moyen ± écart-type -- {station_label}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(apply_changes, max_dist_km, max_offset_hours, leg, use_symlink):
    max_offset = pd.Timedelta(hours=max_offset_hours)
    casts = read_leg_casts(MASTERSHEET_PATH, leg)
    matches = compute_all_matches(casts, max_dist_km, max_offset)

    rows = []
    for idx, row in enumerate(casts.itertuples()):
        status, df_match, diag = matches[idx]

        dest = None
        if apply_changes:
            station_path = os.path.join(L2_ROOT, station_folder_name(row.date, row.station))
            if status in ("temporal", "spatial"):
                os.makedirs(station_path, exist_ok=True)
                dest = sync_pysas_files(df_match, station_path, use_symlink=use_symlink)

                pysas_wl, pysas_specs = cpc.gather_pysas_spectra(df_match)
                if pysas_wl is not None:
                    station_label = station_folder_name(row.date, row.station)
                    fig_path = os.path.join(dest, f"Rrs_mean_std_{station_label}.png")
                    plot_rrs_mean_std(fig_path, pysas_wl, pysas_specs, station_label)
            else:
                remove_station_pysas(station_path)

        rows.append({
            "station": row.station,
            "cast": row.cast,
            "date": row.date.date(),
            "fenêtre_UTC": f"{row.window_start:%H:%M}-{row.window_end:%H:%M}",
            "statut": status,
            "n_pySAS": 0 if df_match is None else len(df_match),
            "dist_km (@ecart)": (
                None if diag["best_distance_km"] is None
                else f"{diag['best_distance_km']} km (@ {diag['best_offset']})"
            ),
            "dossier": dest,
        })

    df_report = pd.DataFrame(rows)
    print(df_report.to_string(index=False))

    n_ok = df_report["statut"].isin(["temporal", "spatial"]).sum()
    n_no_data = (df_report["statut"] == "no_data").sum()
    mode = "dossiers crees et fichiers copiés" if apply_changes else "mode rapport -- relancer avec --apply pour copier"
    print(f"\n{n_ok}/{len(df_report)} station(s) avec un match pySAS ({mode}).")
    if n_no_data:
        print(f"{n_no_data} station(s) sans aucune donnée pySAS traitée pour leur date (à retraiter dans HyperCP).")

    return df_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leg", type=int, default=1, help="Leg a traiter (defaut: 1)")
    parser.add_argument("--apply", action="store_true",
                         help="Creer les dossiers station et copier les fichiers pySAS (defaut: rapport seulement)")
    parser.add_argument("--symlink", action="store_true",
                         help="Lier symboliquement au lieu de copier (voir compare_pysas_cops.py)")
    parser.add_argument("--max-distance-km", type=float, default=1.0,
                         help="Distance max pour un match spatial de repli (defaut: 1.0 km)")
    parser.add_argument("--max-offset-hours", type=float, default=3.0,
                         help="Ecart temporel max tolere pour un match spatial de repli (defaut: 3h)")
    args = parser.parse_args()
    main(args.apply, args.max_distance_km, args.max_offset_hours, args.leg, args.symlink)
