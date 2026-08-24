"""
generate_hypersas_L3.py - Genere une base de donnees "L3" HyperSAS, une ligne par
station, dans l'esprit de la DB produite par pyCOPS (GreenEdge2016.csv:
station_id/directory/date/lat/lon/SZA/Rrs_<wl>_mean/_sd/...) -- mais a la resolution
hyperspectrale NATIVE HyperSAS (~137 bandes) plutot que rehantillonnee sur les 19
bandes discretes COPS, pour ne pas perdre l'information spectrale propre a HyperSAS.

La methode de correction (sky-glint + NIR/SimSpec, ex. M99NIR) doit avoir ete choisie
au prealable via compare_hypersas_cops.py / compare_hypersas_L3.py -- ce script ne
choisit rien, il agrege la methode donnee en --method.

Perimetre (decision Simon 2026-08-23) : TOUTES les stations
<GREENEDGE_ROOT>/<date>_Station<code>/ ayant un fichier
HyperSAS/<methode>/L2/<station>_L2.hdf pour la methode choisie -- match COPS ou non
(contrairement a compare_hypersas_L3.py, qui se limite aux stations avec matchup COPS
et exclut G107/G507).

Par station, moyenne/ecart-type sur tous les ensembles L2 valides du fichier (le champ
n_casts joue le meme role que le n_casts de pyCOPS : combien de mesures ont ete
moyennees) : Rrs_HYPER, nLw_HYPER, QWIP, WEI_QA, plus SZA/lat/lon/date moyens
(ANCILLARY).

Usage:
    conda activate hypercp
    python generate_hypersas_L3.py --method M99NIR
    python generate_hypersas_L3.py --method Z17NIR --out /path/to/out.csv
"""
import os
import re
import sys
import argparse

import h5py
import numpy as np
import pandas as pd

MY_DIR = os.path.dirname(os.path.abspath(__file__))
if MY_DIR not in sys.path:
    sys.path.insert(0, MY_DIR)

import compare_hypersas_cops as chc  # noqa: E402 -- reuse GREENEDGE_ROOT

STATION_RE = re.compile(r"^\d{8}_Station(.+)$")


def _wl_columns(dtype_names):
    return sorted([n for n in dtype_names if re.match(r'^[\d.]+$', n)], key=float)


def _mean_datetime(datetag, timetag2):
    times = []
    for d, t in zip(datetag.astype(int), timetag2.astype(int)):
        year, doy = d // 1000, d % 1000
        base = pd.Timestamp(year=year, month=1, day=1, tz="UTC") + pd.Timedelta(days=doy - 1)
        h, m, s, ms = t // 10000000, (t // 100000) % 100, (t // 1000) % 100, t % 1000
        times.append(base + pd.Timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms))
    return pd.DatetimeIndex(times).mean()


def summarize_station(station_dir, station_label, method):
    """Une ligne de DB pour cette station -- None si pas de L2 pour cette methode."""
    l2_path = os.path.join(station_dir, "HyperSAS", method, "L2", f"{station_label}_L2.hdf")
    if not os.path.exists(l2_path):
        return None

    row = {}
    with h5py.File(l2_path, "r") as f:
        rrs_raw = f["/REFLECTANCE/Rrs_HYPER"][...]
        wl_names = _wl_columns(rrs_raw.dtype.names)
        rrs = np.vstack([rrs_raw[w] for w in wl_names]).T.astype(float)  # (n_ens, nWL)
        row["n_casts"] = rrs.shape[0]
        row["date"] = _mean_datetime(rrs_raw["Datetag"], rrs_raw["Timetag2"])

        nlw, nlw_wl_names = None, []
        if "/REFLECTANCE/nLw_HYPER" in f:
            nlw_raw = f["/REFLECTANCE/nLw_HYPER"][...]
            nlw_wl_names = _wl_columns(nlw_raw.dtype.names)
            nlw = np.vstack([nlw_raw[w] for w in nlw_wl_names]).T.astype(float)

        anc = f.get("ANCILLARY")
        if anc is not None:
            if "SZA" in anc:
                row["sun_zenith_deg"] = float(np.nanmean(anc["SZA"]["SZA"][...]))
            if "LATITUDE" in anc:
                row["latitude"] = float(np.nanmean(anc["LATITUDE"]["LATITUDE"][...]))
            if "LONGITUDE" in anc:
                row["longitude"] = float(np.nanmean(anc["LONGITUDE"]["LONGITUDE"][...]))

        derived = f.get("DERIVED_PRODUCTS")
        if derived is not None:
            for dset_name, col_prefix in [("qwip", "QWIP"), ("wei_QA", "WEI_QA")]:
                if dset_name in derived:
                    raw = derived[dset_name][...]
                    cols = raw.dtype.names
                    vals = raw[cols[2]] if cols and len(cols) > 2 else raw[cols[-1]] if cols else raw
                    row[f"{col_prefix}_mean"] = float(np.nanmean(vals))
                    row[f"{col_prefix}_sd"] = float(np.nanstd(vals))

    for i, wl in enumerate(wl_names):
        row[f"Rrs_{wl}_mean"] = float(np.nanmean(rrs[:, i]))
        row[f"Rrs_{wl}_sd"] = float(np.nanstd(rrs[:, i]))
    for i, wl in enumerate(nlw_wl_names):
        row[f"nLw_{wl}_mean"] = float(np.nanmean(nlw[:, i]))
        row[f"nLw_{wl}_sd"] = float(np.nanstd(nlw[:, i]))

    return row


def discover_stations(root):
    return sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and STATION_RE.match(d)
    )


def write_seabass_l3(df, out_path, method, cruise="GreenEdge2016", experiment="GreenEdge"):
    """Ecrit la meme DB au format SeaBASS "rigide" deja utilise dans ce projet pour les
    fichiers ancillaires (voir generate_ancillary_file.py::generate_ancillary_sb) --
    /begin_header...​/end_header puis donnees separees par virgules, une ligne par
    station. /missing=-9999 (convention SeaBASS) remplace les NaN.

    Note : les champs '<param>_sd' (ecart-type inter-ensembles a la station) ne sont PAS
    le champ d'incertitude officiel SeaBASS (habituellement '_unc', un budget
    d'incertitude propage) -- nomme explicitement '_sd' pour ne pas laisser croire a une
    incertitude FRM/PIU alors que c'est une simple dispersion des ensembles moyennes.
    """
    sb_filename = os.path.splitext(os.path.basename(out_path))[0] + ".sb"
    sb_path = os.path.join(os.path.dirname(out_path), sb_filename)

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["year"] = d["date"].dt.strftime("%Y")
    d["month"] = d["date"].dt.strftime("%m")
    d["day"] = d["date"].dt.strftime("%d")
    d["hour"] = d["date"].dt.strftime("%H")
    d["minute"] = d["date"].dt.strftime("%M")
    d["second"] = d["date"].dt.strftime("%S")

    # Rrs<wl_entier>[_sd] -- SeaBASS n'utilise pas de point decimal dans les noms de
    # champs ; l'espacement natif HyperOCR (~3.3 nm) est assez large pour qu'un
    # arrondi a l'entier ne cree pas de collision entre bandes voisines.
    rrs_mean_cols = [c for c in d.columns if re.match(r"^Rrs_[\d.]+_mean$", c)]
    wl_map = {c: int(round(float(c.split("_")[1]))) for c in rrs_mean_cols}

    fields = ["station_id", "year", "month", "day", "hour", "minute", "second",
             "lat", "lon", "SZA", "n_casts"]
    units = ["none", "yyyy", "mo", "dd", "hh", "mn", "ss",
            "degrees", "degrees", "degrees", "none"]
    sb_cols = ["station_id", "year", "month", "day", "hour", "minute", "second",
              "latitude", "longitude", "sun_zenith_deg", "n_casts"]

    for mean_col, wl in sorted(wl_map.items(), key=lambda kv: kv[1]):
        sd_col = mean_col.replace("_mean", "_sd")
        fields += [f"Rrs{wl}", f"Rrs{wl}_sd"]
        units += ["sr^-1", "sr^-1"]
        sb_cols += [mean_col, sd_col if sd_col in d.columns else mean_col]

    body = d[sb_cols].copy()
    for col in body.columns:
        if col != "station_id":
            body[col] = pd.to_numeric(body[col], errors="coerce")
    body = body.fillna(-9999)

    header = [
        "/begin_header",
        f"/data_file_name={sb_filename}",
        "/affiliations=UQAR,Takuvik",
        "/investigators=Simon_Belanger",
        "/contact=simon_belanger@uqar.ca",
        "/data_status=preliminary",
        f"/experiment={experiment}",
        f"/cruise={cruise}",
        f"/rho_correction_method={method}",
        "/calibration_files=NA",
        "/missing=-9999",
        "/delimiter=comma",
        f"/start_date={d['date'].min().strftime('%Y%m%d')}",
        f"/end_date={d['date'].max().strftime('%Y%m%d')}",
        f"/start_time={d['date'].min().strftime('%H:%M:%S')}[GMT]",
        f"/end_time={d['date'].max().strftime('%H:%M:%S')}[GMT]",
        f"/north_latitude={d['latitude'].max():.4f}[DEG]",
        f"/south_latitude={d['latitude'].min():.4f}[DEG]",
        f"/east_longitude={d['longitude'].max():.4f}[DEG]",
        f"/west_longitude={d['longitude'].min():.4f}[DEG]",
        "/water_depth=NA",
        f"! Base de donnees HyperSAS L3 par station ({len(d)} stations), methode "
        f"{method}, moyenne/ecart-type (_sd) sur les n_casts ensembles L2 valides de "
        f"chaque station. Genere par generate_hypersas_L3.py.",
        f"/fields={','.join(fields)}",
        f"/units={','.join(units)}",
        "/end_header",
    ]

    with open(sb_path, "w", encoding="utf-8") as f_out:
        for line in header:
            f_out.write(f"{line}\n")
        body.to_csv(f_out, index=False, header=False, float_format="%.6g")

    print(f"Fichier SeaBASS : {sb_path}")
    return sb_path


def main(method, out_path):
    root = chc.GREENEDGE_ROOT
    stations = discover_stations(root)
    print(f"{len(stations)} dossier(s) station trouve(s) sous {root}")

    rows = []
    for label in stations:
        code = STATION_RE.match(label).group(1)
        station_dir = os.path.join(root, label)
        row = summarize_station(station_dir, label, method)
        if row is None:
            print(f"[SKIP] {label} : pas de L2 HyperSAS {method}")
            continue
        row = {
            "station_id": code,
            "directory": os.path.join(station_dir, "HyperSAS", method),
            "method": method,
            **row,
        }
        rows.append(row)
        print(f"[OK]   {label} -> {code} ({row['n_casts']} ensemble(s))")

    if not rows:
        print("Aucune station avec L2 HyperSAS pour cette methode -- rien a ecrire.")
        return

    df = pd.DataFrame(rows)
    front_cols = [c for c in ["station_id", "directory", "method", "n_casts", "date",
                              "sun_zenith_deg", "longitude", "latitude"] if c in df.columns]
    other_cols = sorted(c for c in df.columns if c not in front_cols)
    df = df[front_cols + other_cols].sort_values("station_id").reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"\nBase de donnees HyperSAS ({method}) : {len(df)} station(s) -> {out_path}")

    write_seabass_l3(df, out_path, method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True,
                        help="Methode de correction L2 a agreger (ex: M99NIR, Z17NIR, M99NN, ...)")
    parser.add_argument("--out", default=None,
                        help="Chemin du CSV de sortie (defaut: <GreenEdge>/L3/hypersas/GreenEdge2016_<methode>.csv)")
    args = parser.parse_args()

    if args.out:
        out_path = args.out
    else:
        greenedge_root = os.path.dirname(chc.GREENEDGE_ROOT.rstrip("/"))  # .../GreenEdge/L2 -> .../GreenEdge
        out_path = os.path.join(greenedge_root, "L3", "hypersas", f"GreenEdge2016_{args.method}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    main(args.method, out_path)
