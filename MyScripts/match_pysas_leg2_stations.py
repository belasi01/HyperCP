"""
match_pysas_leg2_stations.py - Meme exercice que match_pysas_leg1_stations.py, pour le
Leg 2, mais avec une source de casts differente : pas de MasterSheet Ardyna separee en
feuilles HPLC/POC-PON, plutot la feuille "Samples" du carnet de labo CASCADE
(CASCADE_LogBook_lab.xlsx, remplie via `cascade-samples-log` dans le projet CASCADE) qui
marque chaque echantillon d'une croix ("X") dans les colonnes HPLC/CHN.

Cette feuille n'a pas de start_rosette_time_utc/end_rosette_time_utc explicites --
contrairement au Leg 1, la fenetre temporelle et la position de chaque cast sont donc
reconstruites a partir du .btl local du Leg 2 (~/Data/Amundsen_2026/L1/CTD_rosette/Btl/,
synchronise par scripts/sync_ctd_rosette.sh dans CASCADE) : debut = heure NMEA UTC de
l'entete, fin = heure de la derniere bouteille fermee, position = fix NMEA de l'entete.

Reutilise directement compute_all_matches / gather_candidate_casts / haversine_km /
station_folder_name / sync_pysas_files / remove_station_pysas / plot_rrs_mean_std de
match_pysas_leg1_stations.py (memes seuils convenus avec Simon : 1 km / 3h, conflits
resolus par proximite temporelle) -- seule la lecture des casts differe (read_leg2_casts
au lieu de read_leg_casts).

Usage:
    conda activate hypercp
    python match_pysas_leg2_stations.py                 # rapport seulement
    python match_pysas_leg2_stations.py --apply          # cree les dossiers + copie
"""
import os
import re
import sys
import argparse

import pandas as pd

MY_DIR = os.path.dirname(os.path.abspath(__file__))
if MY_DIR not in sys.path:
    sys.path.insert(0, MY_DIR)

# match_pysas_leg1_stations importe rrs_explorer_app (qui insere la racine HyperCP dans
# sys.path pour Source.*) avant Source.utils.dating -- donc l'importer en premier ici
# aussi garantit le meme ordre.
import match_pysas_leg1_stations as m1  # noqa: E402
import compare_pysas_cops as cpc  # noqa: E402

LOGBOOK_PATH = os.path.expanduser("~/Data/Amundsen_2026/Log/CASCADE_LogBook_lab.xlsx")
BTL_DIR = os.path.expanduser("~/Data/Amundsen_2026/L1/CTD_rosette/Btl")
MISSION_CODE = "2026_02"

_LAT_RE = re.compile(r"NMEA Latitude\s*=\s*(\d+)\s+([\d.]+)\s+([NS])")
_LON_RE = re.compile(r"NMEA Longitude\s*=\s*(\d+)\s+([\d.]+)\s+([EW])")
_TIME_RE = re.compile(r"NMEA UTC \(Time\)\s*=\s*(.+)")


def read_btl_window(filename):
    """Fenetre temporelle UTC et position d'un cast, a partir du .btl local. Pas de
    colonne start/end explicite comme dans le MasterSheet Leg 1 : le debut vient de
    l'entete (fix NMEA UTC, pres du debut de la descente), la fin de l'heure de la
    derniere bouteille fermee (pres de la surface, fin de la remontee)."""
    with open(filename, encoding="latin-1") as f:
        lines = f.readlines()

    lat = lon = header_dt = None
    for line in lines:
        if not line.startswith(("*", "#")):
            break
        if m := _LAT_RE.search(line):
            deg, minute, hemi = m.groups()
            lat = float(deg) + float(minute) / 60
            if hemi == "S":
                lat = -lat
        elif m := _LON_RE.search(line):
            deg, minute, hemi = m.groups()
            lon = float(deg) + float(minute) / 60
            if hemi == "W":
                lon = -lon
        elif m := _TIME_RE.search(line):
            header_dt = pd.to_datetime(m.group(1).strip(), format="%b %d %Y %H:%M:%S")

    header_idx = next(i for i, line in enumerate(lines) if line.split()[:2] == ["Bottle", "Date"])
    data_lines = [line for line in lines[header_idx + 2:] if line.strip()]
    bottle_times = []
    for i in range(0, len(data_lines), 2):
        avg = data_lines[i].split()
        month, day, year, time_str = avg[1], avg[2], avg[3], data_lines[i + 1].split()[0]
        bottle_times.append(pd.to_datetime(f"{year}-{month}-{day} {time_str}", format="%Y-%b-%d %H:%M:%S"))

    window_start = header_dt.tz_localize("UTC") if header_dt is not None else None
    window_end = max(bottle_times).tz_localize("UTC") if bottle_times else None
    if window_start is not None and window_end is not None and window_end < window_start:
        window_start, window_end = window_end, window_start  # garde-fou improbable

    return {"lat": lat, "lon": lon, "window_start": window_start, "window_end": window_end}


def read_leg2_casts(logbook_path, btl_dir):
    """Casts (Station, CTD cast) de la feuille Samples avec au moins un echantillon
    HPLC ET un CHN, avec fenetre temporelle/position issues du .btl local correspondant."""
    df = pd.read_excel(logbook_path, sheet_name="Samples", header=4)
    df = df.dropna(subset=["Station", "CTD cast"])  # exclut glace/echantillons hors-rosette

    matched = df.groupby(["Station", "CTD cast"]).filter(
        lambda g: (g["HPLC"] == "X").any() and (g["CHN"] == "X").any()
    )
    casts_meta = matched.drop_duplicates(subset=["Station", "CTD cast"])[["Station", "CTD cast"]]
    casts_meta = casts_meta.rename(columns={"Station": "station", "CTD cast": "cast"})

    rows, missing = [], []
    for r in casts_meta.itertuples(index=False):
        cast_num = int(r.cast)
        btl_path = os.path.join(btl_dir, f"CTD_{MISSION_CODE}_{cast_num:03d}.btl")
        if not os.path.exists(btl_path):
            missing.append(btl_path)
            continue
        win = read_btl_window(btl_path)
        rows.append({
            "station": r.station, "cast": cast_num, "date": win["window_start"],
            "lat": win["lat"], "lon": win["lon"],
            "window_start": win["window_start"], "window_end": win["window_end"],
        })

    if missing:
        print(f"⚠️  {len(missing)} fichier(s) .btl introuvable(s) localement (resynchroniser "
              f"scripts/sync_ctd_rosette.sh ?) :")
        for p in missing:
            print(f"   {p}")

    return pd.DataFrame(rows).reset_index(drop=True)


def main(apply_changes, max_dist_km, max_offset_hours, use_symlink):
    max_offset = pd.Timedelta(hours=max_offset_hours)
    casts = read_leg2_casts(LOGBOOK_PATH, BTL_DIR)
    matches = m1.compute_all_matches(casts, max_dist_km, max_offset)

    rows = []
    for idx, row in enumerate(casts.itertuples()):
        status, df_match, diag = matches[idx]

        dest = None
        if apply_changes:
            station_path = os.path.join(m1.L2_ROOT, m1.station_folder_name(row.date, row.station))
            if status in ("temporal", "spatial"):
                os.makedirs(station_path, exist_ok=True)
                dest = m1.sync_pysas_files(df_match, station_path, use_symlink=use_symlink)

                pysas_wl, pysas_specs = cpc.gather_pysas_spectra(df_match)
                if pysas_wl is not None:
                    station_label = m1.station_folder_name(row.date, row.station)
                    fig_path = os.path.join(dest, f"Rrs_mean_std_{station_label}.png")
                    m1.plot_rrs_mean_std(fig_path, pysas_wl, pysas_specs, station_label)
            else:
                m1.remove_station_pysas(station_path)

        rows.append({
            "station": row.station,
            "cast": row.cast,
            "date": row.date.date() if row.date is not None else None,
            "fenêtre_UTC": (f"{row.window_start:%H:%M}-{row.window_end:%H:%M}"
                             if row.window_start is not None else "?"),
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
        print(f"{n_no_data} station(s) sans aucune donnée pySAS traitée pour leur date.")

    return df_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                         help="Creer les dossiers station et copier les fichiers pySAS (defaut: rapport seulement)")
    parser.add_argument("--symlink", action="store_true",
                         help="Lier symboliquement au lieu de copier (voir compare_pysas_cops.py)")
    parser.add_argument("--max-distance-km", type=float, default=1.0,
                         help="Distance max pour un match spatial de repli (defaut: 1.0 km)")
    parser.add_argument("--max-offset-hours", type=float, default=3.0,
                         help="Ecart temporel max tolere pour un match spatial de repli (defaut: 3h)")
    args = parser.parse_args()
    main(args.apply, args.max_distance_km, args.max_offset_hours, args.symlink)
