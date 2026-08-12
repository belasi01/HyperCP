"""
sync_360_camera.py - Copie sélectivement les mosaïques de la caméra 360 qui
correspondent à des ensembles Rrs pySAS valides (ayant atteint le niveau L2 pour le
modèle M99NIR), même logique que sync_allsky_camera.py mais pour l'archive 360.

Structure source (cadence ~2 min, irrégulière) :
    <SMB_MOUNT_POINT>/<SMB_360_SUBPATH>/<YYYYMMDD>/<HHMMSS>/Camera360_<YYYYMMDD><HHMMSS>_mosaic.jpg
    (chaque dossier HHMMSS contient aussi cam_1/2/3.jpg bruts et un header.txt --
    seule la mosaïque déjà assemblée est récupérée ici)

Copie (ou lien symbolique) vers :
    <MAIN_DATA_PATH>/pySAS/Mosaic360/<YYYYMMDD>/Camera360_<YYYYMMDD><HHMMSS>_mosaic.jpg

Usage :
    conda activate hypercp
    python sync_360_camera.py --date 20260808
    python sync_360_camera.py --all
    python sync_360_camera.py --date 20260808 --copy
"""
import os
import sys
import re
import glob
import shutil
import argparse
from datetime import datetime, timezone

MY_DIR = os.path.dirname(os.path.abspath(__file__))
if MY_DIR not in sys.path:
    sys.path.insert(0, MY_DIR)

# Réutilise la lecture des horodatages de casts M99NIR L2 déjà écrite pour la caméra
# all-sky -- même définition de "ensemble valide" pour les deux archives caméra.
from sync_allsky_camera import ENV, PYSAS_PATH, SMB_MOUNT_POINT, get_m99nir_cast_times, available_m99nir_dates  # noqa: E402

SMB_360_SUBPATH = ENV.get("SMB_360_SUBPATH", "360")
CAMERA360_SRC_ROOT = os.path.join(SMB_MOUNT_POINT, SMB_360_SUBPATH)
DEST_ROOT = os.path.join(PYSAS_PATH, "Mosaic360")


def _index_mosaics(root_dir, date_str):
    """Liste le dossier du jour UNE SEULE fois et retourne [(datetime, chemin), ...].
    `root_dir` est soit le montage SMB source (structure imbriquée
    <HHMMSS>/Camera360_..._mosaic.jpg), soit le cache local Mosaic360/ déjà synchronisé
    par sync_date() ci-dessous (fichiers plats Camera360_<date><HHMMSS>_mosaic.jpg,
    réutilisé pour l'affichage dans rrs_explorer_app.py) -- les deux structures sont
    gérées ici. Séparé de find_nearest_mosaic() pour que sync_date() puisse construire
    l'index une seule fois par date plutôt qu'une fois par cast (un dossier journalier
    peut contenir des centaines d'entrées, coûteux à relister sur un montage SMB)."""
    day_dir = os.path.join(root_dir, date_str)
    if not os.path.isdir(day_dir):
        return []

    flat_pattern = re.compile(rf"Camera360_{date_str}(\d{{6}})_mosaic\.jpg$")
    index = []
    for entry in os.listdir(day_dir):
        entry_path = os.path.join(day_dir, entry)
        if os.path.isdir(entry_path) and re.fullmatch(r"\d{6}", entry):
            time_str = entry
            mosaic_matches = glob.glob(os.path.join(entry_path, "*_mosaic.jpg"))
        else:
            m = flat_pattern.match(entry)
            if not m:
                continue
            time_str = m.group(1)
            mosaic_matches = [entry_path]

        if not mosaic_matches:
            continue
        entry_dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        index.append((entry_dt, mosaic_matches[0]))
    return index


def _nearest_in_index(index, dt, tolerance_s):
    best_path, best_diff = None, None
    for entry_dt, path in index:
        diff = abs((entry_dt - dt).total_seconds())
        if diff > tolerance_s:
            continue
        if best_diff is None or diff < best_diff:
            best_path, best_diff = path, diff
    return best_path


def find_nearest_mosaic(root_dir, date_str, dt, tolerance_s=180):
    """Cadence ~2 min et irrégulière (pas de nom de fichier prévisible comme pour
    l'all-sky) -- cherche la mosaïque la plus proche de `dt` dans la tolérance.
    Pratique pour un lookup ponctuel (ex: rrs_explorer_app.py sur clic d'un cast) ;
    pour synchroniser une journée entière (beaucoup de casts), sync_date() construit
    l'index une seule fois via _index_mosaics() plutôt que d'appeler cette fonction en
    boucle."""
    return _nearest_in_index(_index_mosaics(root_dir, date_str), dt, tolerance_s)


def sync_date(date_str, use_symlink=True):
    cast_times = get_m99nir_cast_times(date_str)
    if not cast_times:
        print(f"⚠️  Aucun cast M99NIR L2 trouvé pour {date_str}.")
        return

    index = _index_mosaics(CAMERA360_SRC_ROOT, date_str)
    if not index:
        print(f"⚠️  Aucune mosaïque 360 disponible pour {date_str} (dossier absent ou vide sur le montage source).")
        return

    dest_dir = os.path.join(DEST_ROOT, date_str)
    os.makedirs(dest_dir, exist_ok=True)

    n_copied, n_missing = 0, 0
    for dt in cast_times:
        src = _nearest_in_index(index, dt, tolerance_s=180)
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
    print(f"🌐 {date_str}: {n_copied} mosaïque(s) {verb}, "
          f"{n_missing}/{len(cast_times)} cast(s) sans mosaïque correspondante -> {dest_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", action="append", default=[], help="Date AAAAMMJJ (répétable)")
    parser.add_argument("--all", action="store_true", help="Toutes les dates avec sortie M99NIR L2")
    parser.add_argument("--copy", action="store_true", help="Copier au lieu de lier symboliquement")
    args = parser.parse_args()

    dates = available_m99nir_dates() if args.all else args.date
    if not dates:
        parser.error("Spécifier au moins une --date ou --all")

    if not os.path.isdir(CAMERA360_SRC_ROOT):
        print(f"❌ Montage 360 introuvable : {CAMERA360_SRC_ROOT} -- le partage SMB est-il monté ?")
        sys.exit(1)

    for date_str in dates:
        sync_date(date_str, use_symlink=not args.copy)
