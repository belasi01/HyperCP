"""
sync_allsky_camera.py - Copie sélectivement les images de la caméra all-sky qui
correspondent à des ensembles Rrs pySAS valides (ayant atteint le niveau L2 pour le
modèle M99NIR), plutôt que de rapatrier toute l'archive caméra (trop volumineuse).

Pour chaque date :
1. Lit /REFLECTANCE/Rrs_HYPER dans M99NIR/L2/*_L2.hdf pour obtenir l'horodatage UTC
   (Datetag + Timetag2) de chaque ensemble ayant passé le QC jusqu'à L2.
2. Pour chaque horodatage, trouve l'image la plus proche sur le montage caméra
   (cadence ~1 image/minute : <SMB_MOUNT_POINT>/<SMB_CAMERA_SUBPATH>/<YYYYMMDD>/
   <YYYYMMDDHHMMSS>_*.jpg), avec une tolérance de quelques minutes si l'image exacte
   manque (trous possibles dans l'archive caméra).
3. Copie (ou lie symboliquement) l'image trouvée dans
   <MAIN_DATA_PATH>/pySAS/AS_Camera/<YYYYMMDD>/.

Étape suivante (pas encore faite) : lier ces images aux points de mesure L2, par ex.
un champ "photo_url" dans le GeoJSON (pySAS_L2_Track_QC_<date>.geojson) pour un clic
QGIS, ou un affichage directement dans rrs_explorer_app.py.

Usage :
    conda activate hypercp
    python sync_allsky_camera.py --date 20260808
    python sync_allsky_camera.py --date 20260808 --date 20260809
    python sync_allsky_camera.py --all             # toutes les dates avec sortie M99NIR L2
    python sync_allsky_camera.py --date 20260808 --copy   # copie réelle plutôt qu'un lien
"""
import os
import sys
import re
import glob
import shutil
import argparse
from datetime import timedelta

import h5py

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
SMB_MOUNT_POINT = ENV.get("SMB_MOUNT_POINT", "/Volumes/data")
SMB_CAMERA_SUBPATH = ENV.get("SMB_CAMERA_SUBPATH", "Camera_All_Sky/2026")
CAMERA_SRC_ROOT = os.path.join(SMB_MOUNT_POINT, SMB_CAMERA_SUBPATH)
DEST_ROOT = os.path.join(PYSAS_PATH, "AS_Camera")

PATH_HCP = ENV["PATH_HCP"]
if PATH_HCP not in sys.path:
    sys.path.insert(0, PATH_HCP)
import Source.utils.dating as dating  # noqa: E402


def get_m99nir_cast_times(date_str):
    """Datetag/Timetag2 -> datetime UTC pour chaque ensemble présent dans la sortie
    M99NIR L2 de cette date (donc ayant passé le QC jusqu'à L2)."""
    l2_dir = os.path.join(PYSAS_PATH, "M99NIR", "L2")
    times = []
    for fpath in sorted(glob.glob(os.path.join(l2_dir, f"*{date_str}*_L2.hdf"))):
        with h5py.File(fpath, "r") as h5f:
            rrs_path = "/REFLECTANCE/Rrs_HYPER"
            if rrs_path not in h5f:
                continue
            raw = h5f[rrs_path][...]
            colnames = raw.dtype.names
            for datetag, timetag2 in zip(raw[colnames[0]], raw[colnames[1]]):
                dt = dating.timeTag2ToDateTime(dating.dateTagToDateTime(int(datetag)), int(timetag2))
                times.append(dt)
    return sorted(set(times))


def _find_day_dir(root_dir, date_str):
    """Le dossier du jour peut être directement sous `root_dir` (cache local
    AS_Camera/<date>/, plat) ou sous <root_dir>/<leg>/asi_16335/<date>/ (montage SMB
    source -- le partage a été réorganisé par leg en cours de campagne, 2026-08-12,
    et une date peut se trouver sous n'importe quel leg, ex. Camera_360 a un
    2026_LEG_01 en plus du 2026_LEG_02 attendu). On ne suppose pas lequel : on cherche
    dans les deux formes."""
    direct = os.path.join(root_dir, date_str)
    if os.path.isdir(direct):
        return direct
    for candidate in glob.glob(os.path.join(root_dir, "*", "asi_16335", date_str)):
        if os.path.isdir(candidate):
            return candidate
    return None


def find_nearest_camera_image(root_dir, date_str, dt, tolerance_s=150):
    """Cadence ~1 image/minute -- arrondit à la minute la plus proche, avec une petite
    tolérance (+/- quelques minutes) si l'image exacte manque. `root_dir` est soit le
    montage SMB source (CAMERA_SRC_ROOT, utilisé ici pour la synchro), soit le cache
    local déjà synchronisé (AS_Camera/, réutilisé par rrs_explorer_app.py pour l'affichage
    sans dépendre du montage réseau)."""
    day_dir = _find_day_dir(root_dir, date_str)
    if day_dir is None:
        return None
    rounded = dt.replace(second=0, microsecond=0)
    if dt.second >= 30:
        rounded += timedelta(minutes=1)
    for delta_min in (0, -1, 1, -2, 2):
        candidate_dt = rounded + timedelta(minutes=delta_min)
        if abs((candidate_dt - dt).total_seconds()) > tolerance_s:
            continue
        prefix = candidate_dt.strftime("%Y%m%d%H%M%S")
        matches = glob.glob(os.path.join(day_dir, f"{prefix}_*.jpg"))
        if matches:
            return matches[0]
    return None


def sync_date(date_str, use_symlink=True):
    cast_times = get_m99nir_cast_times(date_str)
    if not cast_times:
        print(f"⚠️  Aucun cast M99NIR L2 trouvé pour {date_str}.")
        return

    dest_dir = os.path.join(DEST_ROOT, date_str)
    os.makedirs(dest_dir, exist_ok=True)

    n_copied, n_missing = 0, 0
    for dt in cast_times:
        src = find_nearest_camera_image(CAMERA_SRC_ROOT, date_str, dt)
        if src is None:
            n_missing += 1
            continue
        dst = os.path.join(dest_dir, os.path.basename(src))
        if os.path.islink(dst) and not os.path.exists(dst):
            # Lien orphelin -- la source a été déplacée/renommée depuis (ex: réorg du
            # partage SMB), le recréer plutôt que le garder cassé indéfiniment.
            os.unlink(dst)
        if os.path.exists(dst):
            continue
        if use_symlink:
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        n_copied += 1

    verb = "liée(s)" if use_symlink else "copiée(s)"
    print(f"📷 {date_str}: {n_copied} image(s) {verb}, "
          f"{n_missing}/{len(cast_times)} cast(s) sans image correspondante -> {dest_dir}")


def available_m99nir_dates():
    l2_dir = os.path.join(PYSAS_PATH, "M99NIR", "L2")
    dates = set()
    for fpath in glob.glob(os.path.join(l2_dir, "*_L2.hdf")):
        m = re.search(r"(\d{8})", os.path.basename(fpath))
        if m:
            dates.add(m.group(1))
    return sorted(dates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", action="append", default=[], help="Date AAAAMMJJ (répétable)")
    parser.add_argument("--all", action="store_true", help="Toutes les dates avec sortie M99NIR L2")
    parser.add_argument("--copy", action="store_true", help="Copier au lieu de lier symboliquement")
    args = parser.parse_args()

    dates = available_m99nir_dates() if args.all else args.date
    if not dates:
        parser.error("Spécifier au moins une --date ou --all")

    if not os.path.isdir(CAMERA_SRC_ROOT):
        print(f"❌ Montage caméra introuvable : {CAMERA_SRC_ROOT} -- le partage SMB est-il monté ?")
        sys.exit(1)

    for date_str in dates:
        sync_date(date_str, use_symlink=not args.copy)
