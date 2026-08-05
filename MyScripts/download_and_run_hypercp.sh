#!/bin/bash

# ==============================================================================
# PIPELINE AUTOMATISÉ DE TERRE (RIMOUSKI - UQAR)
# Téléchargement via WebDAV (Amundsen Science) et traitement pySAS quotidien
# ==============================================================================

# Déterminer le dossier où se trouve le script actuel
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/pipeline_config.env"

# Vérification de sécurité
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "❌ Error: Configuration file missing at ${CONFIG_FILE}"
    echo "👉 Please copy pipeline_config.env.template to pipeline_config.env and adjust paths."
    exit 1
fi

# IMPORTATION DES VARIABLES DANS LE SHELL BASH
source "${CONFIG_FILE}"

LFTP_BIN=${LFTP_BIN:-lftp}
source "${CONDA_SH_PATH}"
conda activate hypercp

LOG_DIR="${MAIN_DATA_PATH}pySAS/Automated_Pipeline_Log"
# -p s'assure de le créer s'il manque, ou ne fait rien sans planter s'il existe déjà !
mkdir -p "${LOG_DIR}"

# Si l'utilisateur passe un argument (ex: ./download_and_run_hypercp.sh 20260701), on prend cette date.
# Sinon (mode CRON), le script calcule automatiquement la date de la veille.
if [ ! -z "$1" ]; then
    DATE_TARGET="$1"
    echo "📅 Target date provided as argument: ${DATE_TARGET}"
else
    #DATE_TARGET=$(date -v-1d +%Y%m%d)
    DATE_TARGET=$(date +%Y%m%d)
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

if [ "$RUN_WebDAV" = "false" ]; then
    echo "⏩ [TEST MODE ACTIVE] Step 1 (WebDAV download) has been skipped."
else

  if [ "${VERBOSE_TERMINAL}" = "false" ]; then
    LOG_FILE="${LOG_DIR}/pySAS_processing_${DATE_TARGET}.log"

    echo "📝 All terminal outputs for WebDAV (verbose) are being redirected to: ${LOG_FILE}"
  {
    echo "📡 Connecting to Amundsen WebDAV server and syncing files..."

# Exécution de la commande lftp
# --only-newer assure le mode incrémental
# --use-pget-n=4 force le découpage multi-flux
    $LFTP_BIN -c "
    set ssl:verify-certificate no;
    set net:timeout 20;
    set net:max-retries 20;
    open -u ${WEBDAV_USER},${WEBDAV_PASS} ${WEBDAV_HOST};

    echo '   -> Syncing TSG data (flattening LEGs structural paths)...';
    mirror --flat -c -n --ignore-time --parallel=2 --use-pget-n=4 --include=\".*convdata_${DATE_TARGET}.*\" TSG_CDOM/ ${MAIN_DATA_PATH}/TSG/;

    echo '   -> Syncing ATS data (flattening LEGs structural paths)...';
    mirror --flat -c -n --ignore-time --parallel=2 --use-pget-n=4 --include=\".*${DATE_TARGET}.*\" ATS/ ${MAIN_DATA_PATH}/ATS/;

    echo '   -> Syncing pySAS RAW data...';
    mirror -c -n --ignore-time --parallel=2 --use-pget-n=4 --include=\".*${DATE_TARGET}.*\" PySAS/ ${MAIN_DATA_PATH}/pySAS/RAW_NoHeaders/;
    "
    echo "✅ WebDAV synchronization completed."
  } > "${LOG_FILE}" 2>&1
  else
    echo "📡 Connecting to Amundsen WebDAV server and syncing files..."

    $LFTP_BIN -c "
    set ssl:verify-certificate no;
    set net:timeout 20;
    set net:max-retries 20;
    set net:connection-limit 1;
    set hftp:cache no;
    set http:use-propfind true;
    open -u ${WEBDAV_USER},${WEBDAV_PASS} ${WEBDAV_HOST};

    echo '   -> Syncing TSG data (flattening LEGs structural paths)...';
    mirror --flat -c -n --ignore-time --parallel=2 --use-pget-n=4 --include=\".*convdata_${DATE_TARGET}.*\" TSG_CDOM/ ${MAIN_DATA_PATH}/TSG/;

    echo '   -> Syncing ATS data (flattening LEGs structural paths)...';
    mirror --flat -c -n --ignore-time --parallel=2 --use-pget-n=4 --include=\".*${DATE_TARGET}.*\" ATS/ ${MAIN_DATA_PATH}/ATS/;

    echo '   -> Syncing pySAS RAW data...';
    mirror -c -n --ignore-time --parallel=2 --use-pget-n=4 --include=\".*${DATE_TARGET}.*\" PySAS/ ${MAIN_DATA_PATH}/pySAS/RAW_NoHeaders/;
    "
    echo "✅ WebDAV synchronization completed."
  fi
fi



# ------------------------------------------------------------------------------
# GARDE-FOU : PAS DE DONNÉES BRUTES PYSAS = PAS DE TRAITEMENT
# ------------------------------------------------------------------------------
# Sans ça, tout le pipeline (L1A -> L2 -> NIR -> extract_l2_qc_tables.py) tourne
# à vide sur une journée sans données (instrument down, panne réseau, etc.) et finit
# par planter à la toute fin dans extract_l2_qc_tables.py (KeyError sur 'Filename',
# faute de lignes à pivoter) au lieu de s'arrêter proprement dès le départ.
LOG_FILE="${LOG_DIR}/pySAS_processing_${DATE_TARGET}.log"
RAW_COUNT=$(find "${MAIN_DATA_PATH}pySAS/RAW_NoHeaders/" -maxdepth 1 -type f -name "*${DATE_TARGET}*.raw" 2>/dev/null | wc -l | tr -d ' ')

if [ "${RAW_COUNT}" -eq 0 ]; then
  MSG="⚠️  No pySAS RAW data found for ${DATE_TARGET} in ${MAIN_DATA_PATH}pySAS/RAW_NoHeaders/ -- skipping HyperCP processing (instrument down or not yet synced)."
  echo "${MSG}"
  echo "${MSG}" >> "${LOG_FILE}"
  exit 0
fi

# ------------------------------------------------------------------------------
# ÉTAPE 2 : EXÉCUTION DE LA CHAÎNE DE TRAITEMENT PYTHON
# ------------------------------------------------------------------------------
if [ "$RUN_HCP" = "false" ]; then
  echo "⏩ [TEST MODE ACTIVE] Step 2 (HyperCP processing) has been skipped."
else
  echo "🐍 Activating environment and launching pipeline..."
  cd "${PATH_HCP}MyScripts/"

  if [ "${VERBOSE_TERMINAL}" = "true" ]; then
    echo "📺 [INTERACTIVE MODE] Displaying live processing outputs directly in the terminal..."
    echo "=========================================================================="
    echo "⚓ STARTING HYPERCP PROCESSING RUN FOR DATE: ${DATE_TARGET}"
    echo "=========================================================================="

    python run_pySAS006_processing.py --date "${DATE_TARGET}" --level "L1A"
    python run_pySAS006_processing.py --date "${DATE_TARGET}" --level "L1AQC"
    python run_pySAS006_processing.py --date "${DATE_TARGET}" --level "L1B"
    python run_pySAS006_processing.py --date "${DATE_TARGET}" --level "L1BQC"

    # SKY_MODEL: "ALL" (M99+Z17+3C), or a comma-separated subset (e.g. "M99,Z17"), or a single model.
    # Each requested model runs its NN version through the full L2 pipeline, then
    # apply_nir_corrections.py derives its NIR/SimSpec variants (fast, no full rerun).
    if [ "${SKY_MODEL}" = "ALL" ]; then
      MODELS_LIST="M99 Z17 3C"
    else
      MODELS_LIST=$(echo "${SKY_MODEL}" | tr ',' ' ')
    fi
    MODELS_CSV=$(echo "${MODELS_LIST}" | tr ' ' ',')

    for model in ${MODELS_LIST}; do
      python run_pySAS006_processing.py --date "${DATE_TARGET}" --level "L2" --version "${model}NN"
      # NIR/SimSpec offsets are not relevant on top of 3C (decided 2026-07-17) -- only
      # 3CNN is kept, skip the derivation step entirely to avoid wasting time on it.
      if [ "${model}" != "3C" ]; then
        python apply_nir_corrections.py --date "${DATE_TARGET}" --model "${model}"
      fi
    done

    python extract_l2_qc_tables.py "${DATE_TARGET}" "${MODELS_CSV}"
  else
    # On définit le nom du fichier log daté (ex: pySAS_processing_20260707.log)
    LOG_FILE="${LOG_DIR}/pySAS_processing_${DATE_TARGET}.log"
    echo "📝 All terminal outputs for HYPERCP (verbose) are being redirected to: ${LOG_FILE}"

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

    if [ "${SKY_MODEL}" = "ALL" ]; then
      MODELS_LIST="M99 Z17 3C"
    else
      MODELS_LIST=$(echo "${SKY_MODEL}" | tr ',' ' ')
    fi
    MODELS_CSV=$(echo "${MODELS_LIST}" | tr ' ' ',')

    for model in ${MODELS_LIST}; do
      python run_pySAS006_processing.py --date "${DATE_TARGET}" --level "L2" --version "${model}NN"
      # NIR/SimSpec offsets are not relevant on top of 3C (decided 2026-07-17) -- only
      # 3CNN is kept, skip the derivation step entirely to avoid wasting time on it.
      if [ "${model}" != "3C" ]; then
        python apply_nir_corrections.py --date "${DATE_TARGET}" --model "${model}"
      fi
    done

    python extract_l2_qc_tables.py "${DATE_TARGET}" "${MODELS_CSV}"

    echo "=========================================================================="
    echo "🏁 PROCESSING COMPLETED FOR DATE: ${DATE_TARGET}"
    echo "=========================================================================="
  } >> "${LOG_FILE}" 2>&1
  fi

echo "🏁 [SUCCESS] Processing finished for ${DATE_TARGET}. PDF Report generated."

fi
