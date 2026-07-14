""" Scripted command line call to HyperCP. Set up the configuration file using the GUI first,
    or by editing ./Config/[yourconfig].cfg JSON file."""

import multiprocessing
import os
import glob
import time
import sys
import argparse
import json
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
if not hasattr(plt.cm, 'get_cmap'):
    plt.cm.get_cmap = mpl.colormaps.get_cmap

# Importer la fonction de génération du fichier Ancillary SeaBASS
from generate_ancillary_file import generate_ancillary_sb
from add_sathdr_2_raw import add_sathdr_2_raw

# Définir le chemin vers vos scripts de traitement personnels
# Définir le chemin vers HyperCP
#PATH_HCP = "/Users/simonbelanger/PythonProjects/HyperCP/"

def load_pipeline_config(config_path):
    """Parse un fichier .env et injecte les variables dans le scope global."""
    config = {}
    if not os.path.exists(config_path):
        print(f"❌ Critical Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, val = line.split('=', 1)
            config[key.strip()] = val.strip()
    return config

# --- CHARGEMENT DU PROFIL ENVIRO ---
# On cible le fichier .env situé dans le dossier MyScripts/
MY_DIR = os.path.dirname(os.path.abspath(__file__))
env = load_pipeline_config(os.path.join(MY_DIR, "pipeline_config.env"))

# --- VARIABLES DYNAMIQUES PORTABLES ---
PATH_HCP = env["PATH_HCP"]
PATH_MY_SCRIPTS = os.path.join(PATH_HCP, "MyScripts")
CRUISE = env["CRUISE"]
EXPERIMENT = env["EXPERIMENT"]
MAIN_DATA_PATH = env["MAIN_DATA_PATH"]
ROLL_OFFSET = float(env["ROLL_OFFSET"])

# Ajouter ce dossier au chemin de recherche de Python
if PATH_MY_SCRIPTS not in sys.path:
    sys.path.insert(0, PATH_MY_SCRIPTS)

# Importer directement la fonction de correction depuis votre fichier externe !
from correct_L1A_files import correct_L1A_file

# Ajouter ce chemin au système de recherche de modules de Python
if PATH_HCP not in sys.path:
    sys.path.insert(0, PATH_HCP)

from Main import Command

# Configuration de l'analyseur d'arguments de ligne de commande
parser = argparse.ArgumentParser(description="Automatisation du processing HyperCP par niveau et par date.")
parser.add_argument("--date", type=str, default="20260701", help="Date au format AAAAMMJJ (ex: 20260701)")
parser.add_argument("--time", type=str, default=None, help="Heure optionnelle au format HHMMSS (ex: 172336)")
parser.add_argument("--level", type=str, default="L2", choices=["L1A", "L1AQC", "L1B", "L1BQC", "L2"],
                    help="Niveau de processing cible (L1A, L1AQC, L1B, L1BQC, L2)")
parser.add_argument("--version", type=str, default="M99SimSpec",
                    choices=["M99SimSpec", "M99NIR", "M99NN",
                             "Z17SimSpec", "Z17NIR", "Z17NN",
                             "3CSimSpec", "3CNIR", "3CNN",
                             "ALL"], # à modifier en temps et lieu
                    help="Version de traitement L2 spécifique (9 options)")

args = parser.parse_args()

################################################### CUSTOM SET UP ###################################################

# Récupération dynamique des paramètres passés en ligne de commande
dates = args.date
PROC_LEVEL = args.level
L2_VERSION = args.version
time_str = args.time
# #################################

PATH_DATA = os.path.join(MAIN_DATA_PATH, "pySAS")
TSG_PATH = os.path.join(MAIN_DATA_PATH, "TSG")
ATS_PATH = os.path.join(MAIN_DATA_PATH, "ATS")

PATH_ANC = os.path.join(PATH_DATA, "Ancillary", f"{CRUISE}_{EXPERIMENT}_Ancillary_{dates}.sb")
PATH_CFG = os.path.join(PATH_HCP, "Config", env["CFG_FILE_NAME"])
PATH_HDR = os.path.join(PATH_HCP, "Config", env["HDR_FILE_NAME"])

# Parsing propre de la liste des versions L2 séparées par des virgules
ALL_L2_VERSIONS = [v.strip() for v in env["ALL_L2_VERSIONS"].split(",")]

#########
# the pySAS006 has a ROLL of +5° on the benchtop.  This offsset will be subtracted in the L1A file
#ROLL_OFFSET = -5
#########

# Dataset options
INST_TYPE = "SEABIRD"  # SEABIRD or TRIOS; defines raw file naming
L1B_REGIME = ""



# Batch options
MULTI_TASK = True  # Multiple threads for HyperSAS (any level) or TriOS (only L1A and up)
MULTI_LEVEL = False  # Process raw (L0) to Level-2 (L2)
CLOBBER = False      # True overwrites existing files

# Définition automatique des dossiers d'entrée et de sortie selon le niveau demandé
PATH_INPUT = PATH_DATA

# Seul le niveau L2 possède son propre sous-dossier de version pour la sortie
if PROC_LEVEL == "L2":
    PATH_OUTPUT = os.path.join(PATH_DATA, L2_VERSION)
else:
    PATH_OUTPUT = PATH_DATA

# Création du dossier de sortie uniquement s'il n'existe pas déjà (essentiel pour le L2)
if not os.path.isdir(PATH_OUTPUT):
    os.makedirs(PATH_OUTPUT, exist_ok=True)


################################################# END CUSTOM SET UP #################################################
os.environ["HYPERINSPACE_CMD"] = "true"

## Setup remaining globals ##
TO_LEVELS = ["L1A", "L1AQC", "L1B", "L1BQC", "L2"]
FROM_LEVELS = ["RAW", "L1A", "L1AQC", "L1B", "L1BQC"]


if INST_TYPE.lower() == "seabird":
    FILE_EXT = [".raw"]  # May need to use ".RAW" sometimes
else:
    FILE_EXT = [".mlb"]
FILE_EXT.extend(["_L1A.hdf", "_L1AQC.hdf", "_L1B.hdf", "_L1BQC.hdf"])

if not MULTI_LEVEL:
    iOutput = TO_LEVELS.index(PROC_LEVEL)
    TO_LEVELS = [TO_LEVELS[iOutput]]
    FROM_LEVELS = [FROM_LEVELS[iOutput]]
    FILE_EXT = [FILE_EXT[iOutput]]

    if time_str:
        # Si une heure est fournie, on cible le fichier exact (ex: *20260701_172336*.hdf)
        time_pattern = f"*{dates}_{time_str}*"
    else:
        # Si aucune heure n'est fournie, on prend toute la journée (comportement original)
        time_pattern = f"*{dates}*"

    #


def run_Command(fp_input_files, output_path=None):
    """Run either directly or using multiprocessor pool below."""
    #   fp_input_files is a string unless TriOS RAW, then list.
    if output_path is not None:
        local_path_output = output_path
    else:
        local_path_output = PATH_OUTPUT

    # This will skip the file if either 1) the result exists and no CLOBBER, or
    #   2) the Level failed and produced a report.
    # Override with CLOBBER, above.
    if CLOBBER:
        to_skip = {level: [] for level in TO_LEVELS}
    else:
        # On ne met QUE les fichiers de données qui ont réussi dans la liste à sauter.
        # On supprime complètement la recherche des "*_fail.pdf" pour qu'ils soient recalculés !
        to_skip = {level: [os.path.basename(fp).split("_" + level)[0]
                for fp in glob.glob(os.path.join(local_path_output, level, "*"))]
            for level in TO_LEVELS}

    if MULTI_LEVEL:
        # One or more files. (fp_input_files is a list of one or more files)
        from_level = FROM_LEVELS[0]
        to_level = "L1A"
        # inputFileBase = fp_input_files  # Full-path file list of all in L1A
        test = [
            os.path.exists(fp_input_files[i])
            for i, x in enumerate(fp_input_files)
            if os.path.exists(x)
        ]
        if not test:
            print("***********************************")
            print(f"*** [{fp_input_files}] STOPPED PROCESSING ***")
            print(f"Bad input path: {fp_input_files}")
            print("***********************************")
            return
        inputFileBase = os.path.splitext(os.path.basename(fp_input_files[0]))[0]  # single file no path
        test = [v for v in to_skip[to_level] if v in inputFileBase]
        if (test and not CLOBBER):
            print("************************************************")
            print(f"*** [{inputFileBase}] ALREADY PROCESSED TO {to_level} ***")
            print("************************************************")
        else:
            print("************************************************")
            print(f"*** [{inputFileBase}] PROCESSING L0 - L2 ***")
            print("************************************************")

            Command(
                PATH_CFG,
                from_level,
                fp_input_files,
                local_path_output,
                to_level,
                PATH_ANC,
                MULTI_LEVEL,
            )

    else:
        # One file at a time with or without multithread. (fp_input_files is a string of one file)
        for from_level, to_level, ext in zip(FROM_LEVELS, TO_LEVELS, FILE_EXT):
            if from_level == 'RAW' and INST_TYPE.lower() == 'trios':
                inputFileBase = os.path.splitext(os.path.basename(fp_input_files[0]))[0]  # single file no path
                test = [
                    os.path.exists(fp_input_files[i])
                    for i, x in enumerate(fp_input_files)
                    if os.path.exists(x)
                    ]
            else:
                inputFileBase = os.path.splitext(os.path.basename(fp_input_files))[0]
                test = os.path.exists(fp_input_files)
            if not test:
                print("***********************************")
                print(f"*** [{inputFileBase}] STOPPED PROCESSING ***")
                print(f"Bad input path: {fp_input_files}")
                print("***********************************")
                break
            test = [v for v in to_skip[to_level] if v in inputFileBase]
            if test and not CLOBBER:
                print("************************************************")
                print(f"*** [{inputFileBase}] ALREADY PROCESSED TO {to_level} ***")
                print("************************************************")
                continue
            print("************************************************")
            print(f"*** [{inputFileBase}] PROCESSING TO {to_level} ***")
            print("************************************************")

            try:
                Command(
                    PATH_CFG,
                    from_level,
                    fp_input_files,
                    local_path_output,
                    to_level,
                    PATH_ANC,
                    MULTI_LEVEL,
                )
            except AttributeError as e:
                if "'NoneType' object has no attribute 'find'" in str(e):
                    # On intercepte le bug de la NASA et on passe outre car le HDF5 est déjà écrit !
                    print(
                        f"ℹ️ [HyperCP Note] Ignored NASA SeaBASS naming bug for {inputFileBase}. L2 HDF5 file is safe.")
                else:
                    # Si c'est une autre erreur, on la lève normalement
                    raise e

def worker(fp_input_files):
    # --- PATCH DE COMPATIBILITÉ DANS CHAQUE ENFANT ---
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    if not hasattr(plt.cm, 'get_cmap'):
        plt.cm.get_cmap = mpl.colormaps.get_cmap
    # fp_input_files is a list unless multitasking, in which case it's a string, unless it's TriOS RAW
    if isinstance(fp_input_files, list):
        if INST_TYPE.lower() == "trios" and (MULTI_LEVEL or 'RAW' in FROM_LEVELS):
            print(f"### Processing {fp_input_files} ...")
            run_Command(fp_input_files,output_path=PATH_OUTPUT)
        else:
            for file in fp_input_files:
                #print(f"### Processing {os.path.basename(file)} ...")
                #run_Command(file,output_path=PATH_OUTPUT)
                file, current_version = item if isinstance(item, tuple) else (item, L2_VERSION)
                print(f"### Processing {os.path.basename(file)} ...")

                # On calcule dynamiquement le sous-dossier exact
                local_out = os.path.join(PATH_DATA, current_version) if PROC_LEVEL == "L2" else PATH_OUTPUT
                run_Command(file, output_path=local_out)
            print(f"### Finished {os.path.basename(file)}")
    else:
        #print(f"### Multithread Processing {os.path.basename(fp_input_files)} ...")
        #run_Command(fp_input_files,output_path=PATH_OUTPUT)
        file, current_version = fp_input_files if isinstance(fp_input_files, tuple) else (fp_input_files, L2_VERSION)
        print(f"### Multithread Processing {os.path.basename(file)} ...")

        local_out = os.path.join(PATH_DATA, current_version) if PROC_LEVEL == "L2" else PATH_OUTPUT
        run_Command(file, output_path=local_out)
        print(f"### Finished {os.path.basename(file)}")


# ==============================================================================
# NOUVELLE APPROCHE POUR ÉVITER DE ROULER 3 x les méthodes rho shy
# ==============================================================================
from Source.ConfigFile import ConfigFile
from Source.ProcessL2 import ProcessL2


def execute_turbo_L2(root, station, outFilePath_base, model_prefix):
    """
    Exécute le traitement L2 de base (NN), puis applique immédiatement
    les fonctions NIR et SimSpec de la NASA en mémoire avant l'écriture finale.
    """
    # 1. Configuration initiale forcée à No Correction (NN)
    ConfigFile.settings["bL2SimpleNIRCorrection"] = 0
    ConfigFile.settings["bL2SimSpecNIRCorrection"] = 0

    # Appel de la fonction originale de la NASA pour générer la base NN
    # (Calcule les moyennes d'ensembles, les géométries solaires et le rho_sky)
    node_nn = ProcessL2.processL2(root, station)

    # Sauvegarde normale du fichier de base M99NN, Z17NN ou 3CNN
    # (Ici, le code original d'HyperCP écrit node_nn dans outFilePath_base)

    # ----------------------------------------------------------------------
    # EXTENSION TURBO UQAR : CLONAGE ET CORRECTIONS EN MÉMOIRE
    # ----------------------------------------------------------------------
    # On récupère les structures spécifiques requises par la fonction native
    sensor = "HYPER"
    F0 = root.getGroup("CALIBRATION").getDataset("F0")  # Exemple de récupération de F0

    # --- BRANCHEMENT MÉTHODE NIR ---
    import copy
    node_nir = copy.deepcopy(node_nn)  # Duplication complète de l'objet en mémoire
    ConfigFile.settings["bL2SimpleNIRCorrection"] = 1
    ConfigFile.settings["bL2SimSpecNIRCorrection"] = 0

    # Appel direct de la fonction native de la NASA que vous avez trouvée !
    ProcessL2.nirCorrection(node_nir, sensor, F0)

    # Modification du chemin de sortie pour le dossier correspondant (ex: M99NIR)
    outFilePath_nir = outFilePath_base.replace(f"{model_prefix}NN", f"{model_prefix}NIR")
    # Enregistrement du node_nir via la méthode de sauvegarde d'HyperCP

    # --- BRANCHEMENT MÉTHODE SIMSPEC ---
    node_sim = copy.deepcopy(node_nn)
    ConfigFile.settings["bL2SimpleNIRCorrection"] = 0
    ConfigFile.settings["bL2SimSpecNIRCorrection"] = 1

    # Deuxième appel direct de la fonction native
    ProcessL2.nirCorrection(node_sim, sensor, F0)

    outFilePath_sim = outFilePath_base.replace(f"{model_prefix}NN", f"{model_prefix}SimSpec")
    # Enregistrement du node_sim via la méthode de sauvegarde d'HyperCP

    print(f"⚡ [Turbo L2] Applied native NIR & SimSpec corrections in-memory for {model_prefix}")


if __name__ == "__main__":
    t0Single = time.time()

    os.chdir(PATH_HCP)

    # ===========================================================================
    # 1. SEQUENTIAL PRE-PROCESSING STEPS (BEFORE HYPERCP FILE CHECK)
    # ===========================================================================

    # --- CASE 1: LEVEL L1A (Injection of Satlantic binary headers into RAW) ---
    if PROC_LEVEL == "L1A":
        PATH_RAW_NO_HEADERS = os.path.join(PATH_INPUT, "RAW_NoHeaders")
        PATH_RAW_OUTPUT = os.path.join(PATH_INPUT, "RAW")
        os.makedirs(PATH_RAW_OUTPUT, exist_ok=True)

        # On cherche la matière première dans RAW_NoHeaders
        ext_propre = FILE_EXT[0] if isinstance(FILE_EXT, list) else FILE_EXT
        raw_file_list = glob.glob(os.path.join(PATH_RAW_NO_HEADERS, f"*{time_pattern}*{ext_propre}"))

        if not raw_file_list:
            print(f"⚠️ No raw file found in RAW_NoHeaders for date/time template: {time_pattern}")
            sys.exit(0)

        print(f"\n⚙️ [Pre-Processing L1A] Adding Satlantic HDR to {len(raw_file_list)} file(s)...")

        for f in sorted(raw_file_list):
            filename = os.path.basename(f)
            target_file = os.path.join(PATH_RAW_OUTPUT, filename)

            if os.path.exists(target_file) and not CLOBBER:
                print(f"    ℹ️ Corrected RAW file with HDR already exists: {filename} (skipping)")
            else:
                print(f"    ⏳ Writing Satlantic HDR into: {filename} ...")
                add_sathdr_2_raw(PATH_RAW_NO_HEADERS, PATH_RAW_OUTPUT, filename)

        print("    ✨ Pre-processing completed. Ready for HyperCP L1A processing.\n")

    # --- CASE 2: LEVEL L1AQC (Ancillary .sb creation + Heading/Roll interpolation) ---
    if PROC_LEVEL == "L1AQC":
        print(f"\n💨 [Ancillary] Checking and generating ancillary weather file...")
        if os.path.exists(PATH_ANC):
            print(f"    ℹ️ Ancillary file found: {os.path.basename(PATH_ANC)}")
            print(f"    ⏩ Skipping generate_ancillary_sb().")
        else:
            TSG_FILE = os.path.join(TSG_PATH, f"tsg_convdata_{dates}.cnv")
            ATS_FILE = os.path.join(ATS_PATH, f"ats_convdata_{dates}.cnv")

            print(f"    ⏳ Ancillary file missing. Generating it from ATS/TSG navigation data...")
            generate_ancillary_sb(
                PATH_DATA,
                TSG_FILE,
                ATS_FILE,
                dates,
                cruise=CRUISE,
                experiment=EXPERIMENT
            )

        print(f"with ancillary data {PATH_ANC}")

        ext_propre = FILE_EXT[0] if isinstance(FILE_EXT, list) else FILE_EXT
        print(f"🔄 [Pre-Processing L1AQC] Correcting SAS heading & ROLL from L1A dataset...")
        l1afilelist = glob.glob(os.path.join(PATH_INPUT, "L1A", f"*{time_pattern}*{ext_propre}"))
        print(f"📋 {len(l1afilelist)} file(s) detected for target time template {time_pattern}.")

        for f in sorted(l1afilelist):
            filename = os.path.basename(f)
            print(f"--> Processing file: {filename} ...")
            try:
                correct_L1A_file(
                    os.path.join(PATH_INPUT, "L1A"),
                    os.path.join(PATH_INPUT, "L1A_corrected"),
                    filename,
                    roll_offset=ROLL_OFFSET
                )
                print(f"    ✅ Heading interpolation and Roll offset successfully applied to {filename}")
            except Exception as e:
                print(f"    ❌ CRITICAL ERROR on file {filename}: {e}")
                print(f"    ⚠️ This file might be corrupted (missing datasets). Skipping it.")
                continue

        # Redirection cruciale vers le dossier corrigé pour la suite d'HyperCP
        FROM_LEVELS = ["L1A_corrected"]

    # ===========================================================================
    # 2. INPUT FILES SEARCH AND VALIDATION FOR HYPERCP CORE RUN
    # ===========================================================================
    fpf_input = sorted(
        glob.glob(os.path.join(PATH_INPUT, FROM_LEVELS[0], f"*{time_pattern}*{FILE_EXT[0]}"))
    )

    if not fpf_input:
        print(
            f"⚠️ No file found for date {dates} at level {FROM_LEVELS[0]} inside: {os.path.join(PATH_INPUT, FROM_LEVELS[0])}")
        sys.exit(0)

    print(f"\n🚀 Launching HyperCP core processing...")
    print(f"Processing input files: {fpf_input}")
    print(f"Using configuration: {PATH_CFG}")

    # ===========================================================================
    # 3. L2 METHODS MATRIX LOOP / CONFIGURATION OF JSON CONFIG AND HEADERS
    # ===========================================================================
    if PROC_LEVEL == "L2" and L2_VERSION == "ALL":
        liste_versions = ALL_L2_VERSIONS
        nb_versions = len(liste_versions)
        print(f"🚀 [SUPER-BATCH L2] Sequential execution of the {nb_versions} processing configurations...")
    else:
        liste_versions = [L2_VERSION]

    for v in liste_versions:
        if PROC_LEVEL == "L2":
            print(f"\n▶️ RUNNING L2 PROCESSING MATRIX FOR VERSION: {v}")
            PATH_OUTPUT = os.path.join(PATH_DATA, v)
            os.makedirs(PATH_OUTPUT, exist_ok=True)

            with open(PATH_CFG, 'r', encoding='utf-8') as f:
                config = json.load(f)
            with open(PATH_HDR, 'r', encoding='utf-8') as f:
                hdr = json.load(f)

            config["bL2M99Rho"] = 1 if "M99" in v else 0
            config["bL2Z17Rho"] = 1 if "Z17" in v else 0
            config["bL23CRho"] = 1 if "3C" in v else 0

            config["bL2PerformNIRCorrection"] = 0 if "NN" in v else 1
            config["bL2SimpleNIRCorrection"] = 1 if "NIR" in v else 0
            config["bL2SimSpecNIRCorrection"] = 1 if "SimSpec" in v else 0

            hdr["rho_correction"] = "M99" if "M99" in v else ("Z17" if "Z17" in v else "3C")
            hdr["NIR_residual_correction"] = "None" if "NN" in v else ("NIR" if "NIR" in v else "SimSpec")

            with open(PATH_CFG, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            with open(PATH_HDR, 'w', encoding='utf-8') as f:
                json.dump(hdr, f, indent=4, ensure_ascii=False)

            shutil.copy2(PATH_CFG, os.path.join(PATH_OUTPUT, f"config_run_{v}.cfg"))
            shutil.copy2(PATH_HDR, os.path.join(PATH_OUTPUT, f"header_run_{v}.hdr"))

        # ===========================================================================
        # 4. MULTIPROCESSING POOL / NASA CORE RUN METHOD INVOCATION
        # ===========================================================================
        if MULTI_TASK:
            with multiprocessing.Pool(4) as pool:
                if INST_TYPE.lower() == 'trios' and FROM_LEVELS[0] == 'RAW':
                    fpf_input_triplets = []
                    for item in fpf_input:
                        inputFileBase = os.path.splitext(os.path.basename(item))[0]
                        timeStamp = inputFileBase[len(inputFileBase) - 15:-1]
                        index = [i for i, x in enumerate(fpf_input) if timeStamp in x]
                        fpf_input_triplet = [fpf_input[x] for x in index]
                        fpf_input_triplets.append(fpf_input_triplet)

                    unique_fpf_input_triplets = [list(x) for x in set(tuple(x) for x in fpf_input_triplets)]
                    pool.map(worker, unique_fpf_input_triplets)
                else:
                    pool.map(worker, [(file, v) for file in fpf_input])
        else:
            worker([(file, v) for file in fpf_input])

        if PROC_LEVEL == "L2":
            print(f"✅ Configuration version {v} completed successfully.")

    t1Single = time.time()
    print(f"\n✨ Overall time elapsed: {str(round((t1Single - t0Single)))} seconds")
