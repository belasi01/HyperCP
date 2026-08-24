"""
compare_es_ed0_experiment.py - Comparaison dédiée Es (pySAS) vs Ed0 (COPS) pour
l'expérience du 2026-08-21 (acquisition pySAS continue + COPS Ed0 à 6 Hz pendant
plusieurs heures, .../L1/cops/Es_experiment/), indépendante des stations CTD/COPS
routinières traitées par compare_pysas_cops.py --ed0-vs-es.

Réutilise directement la logique déjà écrite pour la comparaison Ed0/Es systématique
par station (compare_pysas_cops.py::match_ed0_es, binnage 5s, stats par longueur
d'onde/SZA/nébulosité, mêmes figures) -- seule la source des données diffère (dossier
cops/ autonome hors arborescence L2, pas de dossier station L2/YYYYMMDD_StationID/).

1. (optionnel, --process) Traite les fichiers RAW pySAS de l'expérience
   (pySAS006_20260821_HHMMSS.raw, plage donnée) jusqu'à L1BQC via
   run_pySAS006_processing.py (L1A -> L1AQC -> L1B -> L1BQC), un appel par fichier/
   niveau pour ne (re)traiter QUE cette plage horaire -- CLOBBER=true dans
   pipeline_config.env, donc un appel journée entière repartirait de zéro sur TOUTE
   la journée du 21 août, pas seulement l'expérience.
2. Lit Ed0 (COPS, brut calibré _URC, .../cops/Es_experiment/) et Es (pySAS, L1BQC par
   scan) sur la fenêtre commune, binnés à 5s, appariés par longueur d'onde COPS --
   même méthode que --ed0-vs-es dans compare_pysas_cops.py.
3. Sauvegarde les mêmes sorties (matched_bins.csv, stats.csv, 2 figures) dans un
   sous-dossier dédié (Ed0_vs_Es/Es_experiment_20260821/) pour ne pas écraser la
   comparaison toutes-stations déjà existante dans Ed0_vs_Es/.

Usage:
    conda activate hypercp
    python compare_es_ed0_experiment.py                  # comparaison seule (L1BQC doit déjà exister)
    python compare_es_ed0_experiment.py --process         # traite RAW->L1BQC d'abord, puis compare
    python compare_es_ed0_experiment.py --process --time 181943 --time 184943  # sous-ensemble
"""
import os
import sys
import argparse
import subprocess

MY_DIR = os.path.dirname(os.path.abspath(__file__))
if MY_DIR not in sys.path:
    sys.path.insert(0, MY_DIR)

# rrs_explorer_app insere la racine HyperCP dans sys.path (necessaire pour Source.*) --
# doit donc etre importe avant compare_pysas_cops (qui importe Source.utils.dating).
import rrs_explorer_app as rea  # noqa: E402
import compare_pysas_cops as cpc  # noqa: E402

DATE_STR = "20260821"
# Fichiers RAW dedies a l'experience (voir ls .../L1/pySAS/RAW_noheaders/), cadence
# ~30 min a partir de 18:19:43 -- distincts des acquisitions routinieres du reste de
# la journee (00:00-17:19, deja traitees).
RAW_TIMES = [
    "181943", "184943", "191943", "194944",
    "201944", "204944", "211944", "214944", "221944",
]
COPS_DIR = os.path.expanduser("~/Data/Amundsen_2026/L1/cops/Es_experiment")
OUT_DIR = os.path.join(rea.MAIN_DATA_PATH, "pySAS", "Ed0_vs_Es", f"Es_experiment_{DATE_STR}")
RUN_SCRIPT = os.path.join(MY_DIR, "run_pySAS006_processing.py")
LEVELS = ["L1A", "L1AQC", "L1B", "L1BQC"]


def process_raw(times, date_str):
    """RAW -> L1BQC pour chaque horodatage donne, un appel par fichier/niveau (pas par
    journee entiere, voir docstring du module)."""
    for level in LEVELS:
        for t in times:
            cmd = [sys.executable, RUN_SCRIPT, "--date", date_str, "--time", t, "--level", level]
            print(f"▶️  {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=MY_DIR)
            if result.returncode != 0:
                raise RuntimeError(f"Echec du traitement {level} pour {date_str}_{t} (code {result.returncode})")


def main(do_process, times, date_str):
    if do_process:
        process_raw(times, date_str)

    label = f"Es_experiment_{date_str}"
    df = cpc.match_ed0_es(COPS_DIR, date_str, label)
    if df.empty:
        print("⚠️  Aucun bin Ed0/Es apparié -- vérifier que le traitement L1BQC a été fait "
              "(relancer avec --process) et que la fenêtre COPS chevauche des données pySAS.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "Ed0_vs_Es_matched_bins.csv")
    df.to_csv(csv_path, index=False)
    print(f"📋 Bins appariés : {csv_path}")

    cpc.plot_ed0_es_per_wavelength(df, OUT_DIR, label=label)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--process", action="store_true",
                   help="Traiter d'abord les RAW pySAS de l'expérience jusqu'à L1BQC "
                        "(sinon, suppose que c'est déjà fait).")
    p.add_argument("--date", default=DATE_STR, help=f"Date de l'expérience (défaut: {DATE_STR})")
    p.add_argument("--time", action="append", default=None,
                   help="Horodatage RAW à traiter (HHMMSS, répétable) -- défaut: la plage "
                        "de l'expérience du 21 août listée dans RAW_TIMES.")
    args = p.parse_args()
    main(args.process, args.time or RAW_TIMES, args.date)
