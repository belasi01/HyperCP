import os
import re
import glob
from datetime import datetime


def add_sathdr_2_raw(inpath, outpath, fn):
    infn = os.path.join(inpath, fn)
    outfn = os.path.join(outpath, fn)

    os.makedirs(outpath, exist_ok=True)

    # 1. Clés du header (ordre strict de 25 éléments)
    sathdr_keys = [
        'CRUISE-ID', 'OPERATOR', 'INVESTIGATOR', 'AFFILIATION', 'CONTACT', 'EXPERIMENT',
        'LATITUDE', 'LONGITUDE', 'ZONE', 'CLOUD_PERCENT', 'WAVE_HEIGHT', 'WIND_SPEED', 'COMMENT',
        'DOCUMENT', 'STATION-ID', 'CAST', 'TIME-STAMP', 'MODE', 'TIMETAG', 'DATETAG', 'TIMETAG2',
        'PROFILER', 'REFERENCE', 'PRO-DARK', 'REF-DARK'
    ]

    # 2. Extraction dynamique de la date et l'heure depuis le nom du fichier
    match = re.search(r'_(\d{8})_(\d{6})', os.path.basename(fn))
    if not match:
        print(f"⚠️ [add_sathdr_2_raw] : Unable to extract timestamp from: {fn}")
        return

    date_str, time_str = match.group(1), match.group(2)

    # Création de l'objet datetime (UTC)
    dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")

    # Formatage identique au "C" de R : "Tue Jul 01 14:51:26 2026"
    # Note: %a et %b génèrent l'anglais par défaut sur la plupart des systèmes
    stamp = dt.strftime("%a %b %d %H:%M:%S %Y")

    # Dictionnaire des valeurs (seul TIME-STAMP est rempli pour le moment)
    values = {'TIME-STAMP': stamp}

    # 3. Construction de l'en-tête binaire (25 blocs de 128 octets = 3200 octets au total)
    header_bytes = bytearray()
    for k in sathdr_keys:
        v = values.get(k, "")
        sentence = f"SATHDR {v} ({k})\r\n"
        sentence_bytes = sentence.encode('ascii')

        # Sécurité de taille
        if len(sentence_bytes) > 128:
            print(f"⚠️ Warning: SATHDR {k} too long ({len(sentence_bytes)} bytes)")
            sentence_bytes = sentence_bytes[:128]

        # Remplissage automatique avec des zéros (NUL bytes 0x00) pour atteindre exactement 128 octets
        padding_len = 128 - len(sentence_bytes)
        padded_sentence = sentence_bytes + b'\x00' * padding_len
        header_bytes.extend(padded_sentence)

    # 4. Écriture du header puis copie par bloc du contenu binaire original
    with open(outfn, "wb") as f_out:
        f_out.write(header_bytes)  # Écrit les 3200 octets d'en-tête

        # Lecture/Écriture par blocs de 1 Mo pour ne pas surcharger la mémoire vive
        with open(infn, "rb") as f_in:
            while True:
                chunk = f_in.read(1024 * 1024)  # 1 MiB
                if not chunk:
                    break
                f_out.write(chunk)

    #print(f"✅ Fichier créé -> {outfn}")


# ---------------------------------------------------------------------------
# BLOC DE TEST INDÉPENDANT
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    inpath = "/Users/simonbelanger/Data/Amundsen/2026_LEG_00/L1/pySAS/RAW_NoHeaders/"
    outpath = "/Users/simonbelanger/Data/Amundsen/2026_LEG_00/L1/pySAS/RAW/"

    # Option : cibler un filtre de date précis (ex: "20260702")
    date_filter = "20260702"

    search_pattern = os.path.join(inpath, f"*{date_filter}*.raw")
    fn_list = [os.path.basename(f) for f in glob.glob(search_pattern)]

    print(f"🔍 Liste des fichiers trouvés dans {inpath} :")
    print(fn_list)

    for fn in sorted(fn_list):
        add_sathdr_2_raw(inpath, outpath, fn)
