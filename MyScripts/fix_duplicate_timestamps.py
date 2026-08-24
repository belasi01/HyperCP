"""
fix_duplicate_timestamps.py - Detecte (et corrige optionnellement) les horodatages
TIMETAG2 a egalite (ties) dans les groupes *_DARK_L1AQC/*_LIGHT_L1AQC d'un fichier
L1B SeaBird -- PAS un probleme des donnees brutes : HyperCP lui-meme les introduit en
L1B. Apres la correction dark, `Source/ProcessL1b.py:934-945` reecrit le TIMETAG2 de
chaque trame dark par celui de la trame light la plus proche ("matching nearest
lights", pour faciliter le filtrage plus tard) sans garantir l'unicite -- avec un
echantillonnage dark eparse (ex: acquisition par cycles avec coupures), deux trames
dark distinctes peuvent s'associer a la MEME trame light la plus proche. Le fichier
L1A/L1AQC en amont n'a lui aucune egalite (verifie).

Une seule egalite suffit a faire echouer la verification "strictement croissant" de
HyperOCRUtils.LightDarkInterp (Source/PIU/HyperOCR.py:369, comparing.isIncreasing
exige x<y strict) lors du traitement L2 suivant, qui peut alors planter (ensembles
vides) -- independamment de fL2TimeInterval.

Le correctif decale le(s) horodatage(s) en trop d'1 ms (plus proche resolution du
champ TIMETAG2, format HHMMSSmmm) -- les deux mesures reelles sont preservees, seule
l'egalite est cassee pour permettre la spline d'interpolation. A appliquer sur le
fichier L1B (pas L1A: le bug n'existe pas encore a ce niveau, et de toute facon L1B
regenere TIMETAG2_ADJUSTED a chaque execution, donc corriger L1A ne survivrait pas).

Usage :
    conda activate hypercp
    python fix_duplicate_timestamps.py /path/to/20160609_StationG100_L1B.hdf
        # rapport seul (aucune modification)
    python fix_duplicate_timestamps.py /path/to/20160609_StationG100_L1B.hdf --fix
        # corrige en place ; l'original est sauvegarde en <nom>.orig.hdf a cote
        # -- poursuivre le traitement (L1BQC puis L2) a partir de ce fichier corrige,
        # ne pas regenerer ce L1B depuis L1AQC (la correction serait perdue).
"""
import os
import sys
import shutil
import argparse

import numpy as np
import h5py

MY_DIR = os.path.dirname(os.path.abspath(__file__))
if MY_DIR not in sys.path:
    sys.path.insert(0, MY_DIR)


def _tt2_to_sec(tt2):
    tt2 = np.asarray(tt2, dtype=np.int64)
    h = tt2 // 10000000
    m = (tt2 // 100000) % 100
    s = (tt2 // 1000) % 100
    ms = tt2 % 1000
    return h * 3600 + m * 60 + s + ms / 1000.0


def _sec_to_tt2(sec):
    sec = np.asarray(sec, dtype=np.float64)
    total_ms = np.round(sec * 1000).astype(np.int64)
    h, rem = np.divmod(total_ms, 3600 * 1000)
    m, rem = np.divmod(rem, 60 * 1000)
    s, ms = np.divmod(rem, 1000)
    return (h * 10000000 + m * 100000 + s * 1000 + ms).astype(np.float64)


def find_ties(h5path):
    """Retourne {group_name: [(index, tt2_value), ...]} pour chaque groupe
    ShutterDark/ShutterLight ayant au moins une paire consecutive de TIMETAG2 egaux."""
    results = {}
    with h5py.File(h5path, "r") as h5f:
        for gname in h5f.keys():
            g = h5f[gname]
            frame_type = g.attrs.get("FrameType", b"").decode() if hasattr(g.attrs.get("FrameType", b""), "decode") else g.attrs.get("FrameType", "")
            if frame_type not in ("ShutterDark", "ShutterLight"):
                continue
            if "TIMETAG2" not in g:
                continue
            tt2 = g["TIMETAG2"][...]["NONE"]
            sec = _tt2_to_sec(tt2)
            dup_idx = np.where(np.diff(sec) == 0)[0]  # index i means sec[i] == sec[i+1]
            if len(dup_idx):
                results[gname] = [(int(i), float(tt2[i])) for i in dup_idx]
    return results


def fix_ties(h5path, ties):
    """Decale d'1 ms chaque horodatage en trop dans une sequence de doublons
    consecutifs, en place dans le fichier HDF5 (h5path doit deja etre la copie a
    corriger, pas l'original)."""
    with h5py.File(h5path, "r+") as h5f:
        for gname, occurrences in ties.items():
            g = h5f[gname]
            tt2_ds = g["TIMETAG2"][...]
            tt2 = tt2_ds["NONE"].copy()
            sec = _tt2_to_sec(tt2)

            # Regroupe les indices consecutifs (ex: doublon sur 3 trames -> 2 occurrences
            # adjacentes) pour decaler chaque trame suivante d'un multiple de 1 ms.
            offset_ms = 0
            fixed_positions = []
            for i, _ in occurrences:
                offset_ms += 1
                sec[i + 1] += offset_ms / 1000.0
                fixed_positions.append(i + 1)

            new_tt2 = _sec_to_tt2(sec)
            tt2_ds["NONE"] = new_tt2
            g["TIMETAG2"][...] = tt2_ds
            print(f"    🕒 {gname}: {len(occurrences)} égalité(s) corrigée(s) "
                 f"(indices {fixed_positions}, décalage +1ms chacune)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("l1a_path", help="Chemin du fichier L1A HDF5 à analyser")
    parser.add_argument("--fix", action="store_true",
                        help="Corrige en place (l'original est sauvegardé en <nom>.orig.hdf). "
                             "Sans ce flag: rapport seul, aucune modification.")
    args = parser.parse_args()

    ties = find_ties(args.l1a_path)
    if not ties:
        print(f"✅ Aucune égalité de TIMETAG2 trouvée dans {os.path.basename(args.l1a_path)}.")
        sys.exit(0)

    total = sum(len(v) for v in ties.values())
    print(f"⚠️  {total} égalité(s) de TIMETAG2 trouvée(s) dans {os.path.basename(args.l1a_path)} :")
    for gname, occurrences in ties.items():
        for i, val in occurrences:
            print(f"    {gname}: trames aux indices {i}/{i+1}, TIMETAG2={val:.0f}")

    if args.fix:
        backup_path = os.path.splitext(args.l1a_path)[0] + ".orig.hdf"
        if not os.path.exists(backup_path):
            shutil.copy2(args.l1a_path, backup_path)
            print(f"💾 Original sauvegardé : {backup_path}")
        else:
            print(f"ℹ️  Sauvegarde déjà présente ({backup_path}), non écrasée.")
        fix_ties(args.l1a_path, ties)
        print(f"✅ Correction appliquée en place sur {args.l1a_path}.")
        print("⚠️  Ne PAS régénérer ce fichier depuis le niveau précédent (ex: L1B depuis "
             "L1AQC) -- HyperCP réécrit TIMETAG2_ADJUSTED à chaque exécution "
             "(Source/ProcessL1b.py:934-945) et effacerait ce correctif. Poursuivre "
             "directement le traitement à partir de CE fichier corrigé (ex: L1BQC puis L2).")
    else:
        print("ℹ️  Rapport seul (aucune modification). Relancer avec --fix pour corriger.")
