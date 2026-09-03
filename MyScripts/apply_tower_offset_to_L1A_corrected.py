"""
apply_tower_offset_to_L1A_corrected.py - Applique UNIQUEMENT la correction d'orientation
de la tour pySAS (+11 deg, voir pipeline_config.env TOWER_OFFSET/TOWER_OFFSET_CUTOFF_UTC)
directement sur les fichiers L1A_corrected DEJA EXISTANTS, sans toucher au reste
(heading-interp, ROLL, PITCH, derive d'horloge) -- ceux-ci ont deja ete corriges
correctement au moment ou chaque date a ete traitee la premiere fois, avec quelle que
soit la valeur de ROLL_OFFSET/PITCH_OFFSET alors en vigueur (voir
IMU_Roll_Pitch_Offsets.csv / README_pipeline.md "IMU roll/pitch bias" -- le biais a
saute plusieurs fois sur la mission, un unique ROLL_OFFSET/PITCH_OFFSET global ne peut
pas etre correct pour toute la mission a la fois). Simon (2026-09-02): repartir des
L1A_corrected existants plutot que de RAW/L1A pour le retraitement de la tour, afin de
ne PAS ecraser ce roll/pitch deja correct par la valeur globale actuelle.

Modifie les fichiers /L1A_corrected/*.hdf EN PLACE (mutation deja dans l'esprit du
pipeline existant -- correct_L1A_files.py mute deja L1A_corrected de la meme facon,
toujours regenerable depuis RAW/L1A si besoin). A utiliser AVANT
`run_pySAS006_processing.py --level L1AQC --skip-l1a-correction`, qui reutilise ensuite
L1A_corrected tel quel comme entree (voir reprocess_mission.sh).

Usage:
    conda activate hypercp
    python apply_tower_offset_to_L1A_corrected.py --date 20260822
    python apply_tower_offset_to_L1A_corrected.py --all   # toutes les dates de L1A_corrected/
"""
import os
import re
import sys
import glob
import argparse
from datetime import datetime, timezone

MY_DIR = os.path.dirname(os.path.abspath(__file__))
if MY_DIR not in sys.path:
    sys.path.insert(0, MY_DIR)

from correct_L1A_files import process_heading_offset  # noqa: E402


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


ENV = load_pipeline_config(os.path.join(MY_DIR, "pipeline_config.env"))
L1A_CORRECTED_DIR = os.path.join(ENV["MAIN_DATA_PATH"], "pySAS", "L1A_corrected")
TOWER_OFFSET = float(ENV.get("TOWER_OFFSET", 0.0))
TOWER_OFFSET_CUTOFF_UTC = None
if ENV.get("TOWER_OFFSET_CUTOFF_UTC"):
    TOWER_OFFSET_CUTOFF_UTC = datetime.strptime(
        ENV["TOWER_OFFSET_CUTOFF_UTC"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def resolve_tower_offset(filename):
    """Meme logique que run_pySAS006_processing.py::resolve_tower_offset (dupliquee ici
    plutot qu'importee -- ce module a des effets de bord au chargement, argparse compris,
    donc pas importable proprement)."""
    if TOWER_OFFSET_CUTOFF_UTC is None:
        return TOWER_OFFSET
    m = re.search(r"(\d{8})_(\d{6})", filename)
    if not m:
        return TOWER_OFFSET
    file_dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return TOWER_OFFSET if file_dt < TOWER_OFFSET_CUTOFF_UTC else 0.0


def main(dates):
    files = sorted(glob.glob(os.path.join(L1A_CORRECTED_DIR, "*_L1A.hdf")))
    if dates:
        files = [f for f in files if any(d in os.path.basename(f) for d in dates)]

    n_patched, n_skipped = 0, 0
    for f in files:
        filename = os.path.basename(f)
        offset = resolve_tower_offset(filename)
        if not offset:
            n_skipped += 1
            continue
        process_heading_offset(f, offset_value=offset)
        n_patched += 1

    print(f"✅ {n_patched} fichier(s) patché(s) (offset appliqué), {n_skipped} inchangé(s) "
          f"(après TOWER_OFFSET_CUTOFF_UTC ou offset nul) sur {len(files)} au total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", action="append", default=None,
                         help="Date AAAAMMJJ à patcher (répétable) -- défaut: --all requis explicitement.")
    parser.add_argument("--all", action="store_true", help="Toutes les dates trouvées dans L1A_corrected/.")
    args = parser.parse_args()
    if not (args.date or args.all):
        parser.error("Spécifier --date (répétable) ou --all")
    main(args.date)
