#!/bin/bash
# ==============================================================================
# RETRAITEMENT COMPLET DE LA MISSION CASCADE 2026 -- correction d'orientation de la
# tour pySAS (+11 deg avant TOWER_OFFSET_CUTOFF_UTC, voir pipeline_config.env et
# correct_L1A_files.py::process_heading_offset).
#
# Boucle sur toutes les dates AAAAMMJJ trouvees dans RAW_NoHeaders/ et reprend, pour
# chacune, la sequence de traitement de download_and_run_hypercp.sh (L1AQC -> L1B ->
# L1BQC -> L2 -> apply_nir_corrections.py -> extract_l2_qc_tables.py) -- SANS son etape
# de synchronisation (donnees deja locales) et SANS toucher a pipeline_config.env
# (download_and_run_hypercp.sh source ce fichier directement, donc RUN_SYNC/SKY_MODEL
# n'y sont pas surchargeables depuis l'environnement appelant -- voir discussion
# 2026-09-02). Le modele 3C est exclu ici (MODELS ci-dessous), independamment de
# SKY_MODEL dans pipeline_config.env qui reste "ALL" pour le cron quotidien.
#
# IMPORTANT (Simon, 2026-09-02) : ne PAS regenerer L1A_corrected depuis RAW/L1A -- le
# ROLL_OFFSET/PITCH_OFFSET actuel dans pipeline_config.env n'est correct que pour la
# periode recente (voir IMU_Roll_Pitch_Offsets.csv / README_pipeline.md, le biais IMU a
# saute plusieurs fois sur la mission) et l'appliquer partout ecraserait le roll/pitch
# deja correct des periodes anterieures. On repart donc des L1A_corrected EXISTANTS
# (deja corriges correctement roll/pitch a l'epoque), on y patche seulement le cap de
# tour via apply_tower_offset_to_L1A_corrected.py, puis on appelle
# --level L1AQC --skip-l1a-correction pour reutiliser ce L1A_corrected patche tel quel
# (pas de re-regeneration). Ni "L1A" ni "L1A brut" ne sont donc touches.
#
# Usage:
#   ./reprocess_mission.sh                 # toutes les dates de RAW_NoHeaders/
#   ./reprocess_mission.sh 20260701 20260715  # une seule date, ou une plage explicite
# ==============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/pipeline_config.env"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "❌ Error: Configuration file missing at ${CONFIG_FILE}"
    exit 1
fi
source "${CONFIG_FILE}"

source "${CONDA_SH_PATH}"
conda activate hypercp

LOG_DIR="${MAIN_DATA_PATH}pySAS/Automated_Pipeline_Log"
mkdir -p "${LOG_DIR}"
MISSION_LOG="${LOG_DIR}/reprocess_mission_$(date +%Y%m%d_%H%M%S).log"

MODELS="M99 Z17"   # 3C exclu (Simon, 2026-09-02) -- voir en-tete du script
MODELS_CSV=$(echo "${MODELS}" | tr ' ' ',')

if [ "$#" -ge 1 ]; then
    DATES="$*"
else
    DATES=$(ls "${MAIN_DATA_PATH}pySAS/RAW_NoHeaders/" | grep -oE '[0-9]{8}' | sort -u)
fi
N_DATES=$(echo "${DATES}" | wc -w | tr -d ' ')

echo "🌍 Retraitement complet -- ${N_DATES} date(s), modeles: ${MODELS}" | tee -a "${MISSION_LOG}"
echo "📝 Log de mission : ${MISSION_LOG}" | tee -a "${MISSION_LOG}"

cd "${PATH_HCP}MyScripts/"

i=0
for DATE_TARGET in ${DATES}; do
    i=$((i+1))
    DATE_LOG="${LOG_DIR}/pySAS_reprocess_${DATE_TARGET}.log"
    echo "[$i/${N_DATES}] ${DATE_TARGET} -- log: ${DATE_LOG}" | tee -a "${MISSION_LOG}"

    RAW_COUNT=$(find "${MAIN_DATA_PATH}pySAS/RAW_NoHeaders/" -maxdepth 1 -type f -name "*${DATE_TARGET}*.raw" 2>/dev/null | wc -l | tr -d ' ')
    if [ "${RAW_COUNT}" -eq 0 ]; then
        echo "   ⚠️  Aucune donnee RAW pour ${DATE_TARGET}, ignoree." | tee -a "${MISSION_LOG}"
        continue
    fi

    {
        echo "=========================================================================="
        echo "⚓ RETRAITEMENT ${DATE_TARGET}"
        echo "=========================================================================="

        python apply_tower_offset_to_L1A_corrected.py --date "${DATE_TARGET}"
        python run_pySAS006_processing.py --date "${DATE_TARGET}" --level "L1AQC" --skip-l1a-correction
        python run_pySAS006_processing.py --date "${DATE_TARGET}" --level "L1B"
        python run_pySAS006_processing.py --date "${DATE_TARGET}" --level "L1BQC"

        for model in ${MODELS}; do
            python run_pySAS006_processing.py --date "${DATE_TARGET}" --level "L2" --version "${model}NN"
            python apply_nir_corrections.py --date "${DATE_TARGET}" --model "${model}"
        done

        python extract_l2_qc_tables.py "${DATE_TARGET}" "${MODELS_CSV}"

        echo "🏁 TERMINE ${DATE_TARGET}"
    } > "${DATE_LOG}" 2>&1

    if grep -qi "Traceback\|CRITICAL ERROR" "${DATE_LOG}"; then
        echo "   ❌ Erreur(s) detectee(s) pour ${DATE_TARGET} -- voir ${DATE_LOG}" | tee -a "${MISSION_LOG}"
    else
        echo "   ✅ ${DATE_TARGET} termine sans erreur." | tee -a "${MISSION_LOG}"
    fi
done

echo "🏁 Retraitement de mission termine (${N_DATES} date(s))." | tee -a "${MISSION_LOG}"
