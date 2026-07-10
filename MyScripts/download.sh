#!/bin/bash

# ==============================================================================
# PIPELINE AUTOMATISÉ DE TERRE (RIMOUSKI - UQAR)
# Téléchargement via WebDAV (Amundsen Science) et traitement pySAS quotidien
# ==============================================================================

TEST_MODE="true"

# --- CONFIGURATION DES ACCÈS WEBDAV ---
WEBDAV_HOST="https://webdisk.data.amundsen.ulaval.ca"
WEBDAV_USER="donnees03@data.amundsen.ulaval.ca"
WEBDAV_PASS="KePXV!cb-RyqFwpbj6p-f9"

# --- CONFIGURATION DES CHEMINS LOCAUX À RIMOUSKI ---
MAIN_DATA_PATH="/Users/simonbelanger/Data/Amundsen_2026/L1/"
PATH_PYTHON_SCRIPTS="/Users/simonbelanger/PythonProjects/HyperCP/MyScripts"

# Si l'utilisateur passe un argument (ex: ./download.sh 20260701), on prend cette date.
# Sinon (mode CRON), le script calcule automatiquement la date de la veille.
if [ ! -z "$1" ]; then
    DATE_TARGET="$1"
    echo "📅 Target date provided as argument: ${DATE_TARGET}"
else
    DATE_TARGET=$(date -v-1d +%Y%m%d)
    echo "🤖 No argument provided. CRON mode active: processing yesterday's date (${DATE_TARGET})"
fi

echo "=========================================================================="
echo "🌍 [RIMOUSKI] Starting remote processing for date: ${DATE_TARGET}"
echo "=========================================================================="

# On crée les dossiers locaux s'ils n'existent pas
mkdir -p "${MAIN_DATA_PATH}/TSG"
mkdir -p "${MAIN_DATA_PATH}/ATS"
mkdir -p "${MAIN_DATA_PATH}/pySAS/RAW_NoHeaders"

# ------------------------------------------------------------------------------
# ÉTAPE 1 : TÉLÉCHARGEMENT MIROIR VIA WEBDAV (lftp)
# ------------------------------------------------------------------------------

if [ "$TEST_MODE" = "true" ]; then
    echo "⏩ [TEST MODE ACTIVE] Step 1 (WebDAV download) has been skipped."
else
    echo "📡 Connecting to Amundsen WebDAV server and syncing files..."

# Exécution de la commande lftp avec les paramètres corrigés
# --only-newer assure le mode incrémental
# --use-pget-n=4 force le découpage multi-flux pour passer les fichiers > 12 MB
    lftp -c "
    set ssl:verify-certificate no;
    set net:timeout 20;
    set net:max-retries 10;
    open -u ${WEBDAV_USER},${WEBDAV_PASS} ${WEBDAV_HOST};

    echo '   -> Syncing TSG data (flattening LEGs structural paths)...';
    mirror --flat --only-newer --parallel=2 --use-pget-n=4 --include=\".*${DATE_TARGET}.*\" TSG_CDOM/ ${MAIN_DATA_PATH}/TSG/;

    echo '   -> Syncing ATS data (flattening LEGs structural paths)...';
    mirror --flat --only-newer --parallel=2 --use-pget-n=4 --include=\".*${DATE_TARGET}.*\" ATS/ ${MAIN_DATA_PATH}/ATS/;
    "
    #echo '   -> Syncing pySAS RAW data...';
    #mirror --only-newer --parallel=2 --use-pget-n=4 --include=\".*${DATE_TARGET}.*\" pySAS/RAW_NoHeaders/ ${MAIN_DATA_PATH}/pySAS/RAW_NoHeaders/;
    #"
    echo "✅ WebDAV synchronization completed."
fi




# ------------------------------------------------------------------------------
# ÉTAPE 2 : EXÉCUTION DE LA CHAÎNE DE TRAITEMENT PYTHON
# ------------------------------------------------------------------------------
echo "🐍 Activating environment and launching pipeline..."
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hypercp
cd "${PATH_PYTHON_SCRIPTS}"

# On définit le nom du fichier log daté (ex: pySAS_processing_20260707.log)
# Il sera enregistré proprement dans votre dossier MyScripts/
LOG_FILE="${MAIN_DATA_PATH}/pySAS/Automated_Pipeline_Log/pySAS_processing_${DATE_TARGET}.log"

echo "📝 All terminal outputs (verbose) are being redirected to: ${LOG_FILE}"

# Le symbole '=>' redirige à la fois le flux standard (stdout) et les erreurs (stderr)
# de toutes les étapes de traitement vers le fichier log en temps réel.
{
    echo "=========================================================================="
    echo "⚓ STARTING HYPERCP PROCESSING RUN FOR DATE: ${DATE_TARGET}"
    echo "=========================================================================="

    python run_pySAS006_processing.py --date "${DATE_TARGET}" --level "L1A"
    python run_pySAS006_processing.py --date "${DATE_TARGET}" --level "L1AQC"
    python run_pySAS006_processing.py --date "${DATE_TARGET}" --level "L1B"
    python run_pySAS006_processing.py --date "${DATE_TARGET}" --level "L1BQC"
    python run_pySAS006_processing.py --date "${DATE_TARGET}" --level "L2" --version "ALL"

    python extract_l2_qc_tables.py

    echo "=========================================================================="
    echo "🏁 PROCESSING COMPLETED FOR DATE: ${DATE_TARGET}"
    echo "=========================================================================="
} > "${LOG_FILE}" 2>&1

echo "🏁 [SUCCESS] Processing finished for ${DATE_TARGET}. PDF Report generated."


