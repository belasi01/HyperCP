import h5py
import numpy as np
import pandas as pd
import shutil
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
from datetime import datetime, timedelta


def interpolate_heading_timeaware(heading_deg, time_s):
    """
    Interpolation circulaire (sin/cos) prenant en compte le temps.
    Gère les NaN et extrapole aux bords (équivalent à rule=2 de R).
    """
    # Conversion en array numpy pour faciliter les calculs
    h = np.array(heading_deg, dtype=float)
    t = np.array(time_s, dtype=float)

    # Masque des valeurs valides (non NaN)
    mask_valid = ~np.isnan(h)

    # S'il y a moins de 2 valeurs valides, on ne peut pas interpoler
    if np.sum(mask_valid) < 2:
        return h

    # Passage en radians pour les valeurs valides
    rad = np.radians(h[mask_valid])
    t_valid = t[mask_valid]
    sin_valid = np.sin(rad)
    cos_valid = np.cos(rad)

    # Interpolation 1D (np.interp extrapole par défaut aux bords avec les valeurs limites)
    sin_interp = np.interp(t, t_valid, sin_valid)
    cos_interp = np.interp(t, t_valid, cos_valid)

    # Retour en degrés (0..360) via atan2
    rad_interp = np.arctan2(sin_interp, cos_interp)
    h_interp = np.degrees(rad_interp) % 360.0

    # On préserve les données valides d'origine pour éviter les micro-arrondis
    h_interp[mask_valid] = h[mask_valid]
    return h_interp



def plot_verification(time_raw, heading_orig, heading_interp, filename, plot_dir):
    """
    Génère et sauvegarde une figure de contrôle pour comparer le heading original (points)
    et le heading interpolé (ligne rouge), avec un axe X temporel propre.
    - Points gris : Données originales
    - Points verts : Données interpolées sur des points valides d'origine
    - Points rouges : Données interpolées remplaçant les NaN
    """

    base_name = os.path.basename(filename)  # Récupère juste "pySAS006_..._L1A.hdf"
    output_png = os.path.join(plot_dir, base_name.replace('.hdf', '_verification_heading.png'))

    # Extraction de la date (8 chiffres consécutifs) depuis le nom du fichier
    match = re.search(r'\d{8}', base_name)
    if match:
        date_str = match.group(0)
        annee = int(date_str[0:4])
        mois = int(date_str[4:6])
        jour = int(date_str[6:8])
    else:
        # Valeurs de secours par défaut si le nom du fichier est atypique
        annee, mois, jour = 2026, 1, 1

    # 1. Conversion de TIMETAG2 (HHMMSSms) en objets datetime pour l'axe X
    time_pts = []
    for t in time_raw:
        h = int(t // 10000000)
        m = int((t % 10000000) // 100000)
        s = int((t % 100000) // 1000)
        ms = int(t % 1000)
        # Création d'un datetime de référence fictif pour la journée
        time_pts.append(datetime(annee, mois, jour, h, m, s, ms * 1000))

    time_pts = np.array(time_pts)

    # Création des masques de sélection
    mask_nan = np.isnan(heading_orig) | (heading_orig <= -999.0)
    mask_valid = ~mask_nan

    # 2. Création de la figure
    plt.figure(figsize=(10, 5))

    # 1. Valeurs originales en gros points gris
    plt.scatter(time_pts[mask_valid], heading_orig[mask_valid],
                color='darkgray', s=25, alpha=0.7, label='Original', zorder=1)

    # 2. Valeurs interpolées (valides) en vert plus petit par-dessus
    plt.scatter(time_pts[mask_valid], heading_interp[mask_valid],
                color='forestgreen', s=8, alpha=0.9, label='Validé (Python)', zorder=2)

    # 3. Valeurs interpolées (qui remplacent les NaN) en rouge
    plt.scatter(time_pts[mask_nan], heading_interp[mask_nan],
                color='crimson', s=12, alpha=0.9, label='Interpolé (Remplacement NaN)', zorder=3)

    # 3. Formatage de l'axe X (Temporel)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()  # Incline les étiquettes de temps pour éviter les chevauchements

    # 4. Habillage du graphique
    plt.title(f"Vérification de l'interpolation du SAS Heading\nFichier : {filename}", fontsize=12, fontweight='bold')
    plt.xlabel("Heure (UTC)", fontweight='bold')
    plt.ylabel("Heading (degrés 0..360)", fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(-10, 370)  # Un peu de marge pour voir les valeurs proches de 0 et 360
    plt.legend(loc='upper right')
    plt.tight_layout()

    # 5. Sauvegarde de l'image à côté du fichier de données
    #output_png = filename.replace('.hdf', '_verification_heading.png')
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"📊 Figure de contrôle sauvegardée sous : {output_png}")


# --- FONCTION 1 : INTERPOLATION DU HEADING ---
def process_heading_interpolation(outfn, plot_dir):
    """Gère l'interpolation du SAS Heading en place."""
    with h5py.File(outfn, "r+") as h5f:
        dataset_path = "/UMTWR_v0.tdf/HEADING"
        time_path = "/UMTWR_v0.tdf/TIMETAG2"
        if dataset_path in h5f and time_path in h5f:
            heading_data = h5f[dataset_path][...]
            time_data = h5f[time_path][...]
            heading_sas_original = np.array(heading_data['SAS'], dtype=np.float64).copy()
            sas_corrected = interpolate_heading_timeaware(heading_sas_original, time_data['NONE'])
            plot_verification(time_data['NONE'], heading_sas_original, sas_corrected, outfn, plot_dir)
            heading_data['SAS'] = sas_corrected
            h5f[dataset_path][...] = heading_data


# --- FONCTION 2 : CORRECTION DE L'OFFSET DE ROLL ---
def process_roll_offset(outfn, offset_value=-5.0):
    """Applique un offset statique sur le Roll en place."""
    with h5py.File(outfn, "r+") as h5f:
        roll_path = "/SATTHS1500A.tdf/ROLL"
        if roll_path in h5f:
            roll_data = h5f[roll_path][...]
            roll_data['NONE'] = roll_data['NONE'] + offset_value
            h5f[roll_path][...] = roll_data
            print(f"    📐 Offset de Roll appliqué ({offset_value}°) sur {os.path.basename(outfn)}")


# --- FONCTION 3 : CORRECTION DE L'OFFSET DE PITCH ---
def process_pitch_offset(outfn, offset_value=0.0):
    """Applique un offset statique sur le Pitch en place."""
    with h5py.File(outfn, "r+") as h5f:
        pitch_path = "/SATTHS1500A.tdf/PITCH"
        if pitch_path in h5f:
            pitch_data = h5f[pitch_path][...]
            pitch_data['NONE'] = pitch_data['NONE'] + offset_value
            h5f[pitch_path][...] = pitch_data
            print(f"    📐 Offset de Pitch appliqué ({offset_value}°) sur {os.path.basename(outfn)}")


# --- FONCTION 4 : CORRECTION DE LA DÉRIVE D'HORLOGE DU DATALOGGER ---
def _tags_to_datetime(datetag, timetag2):
    """Convertit des tableaux DATETAG (YYYYDDD) + TIMETAG2 (HHMMSSmmm) -- l'horloge
    interne du datalogger SeaBird -- en un DatetimeIndex UTC (vectorisé, via pandas)."""
    datetag = np.asarray(datetag, dtype=np.int64)
    timetag2 = np.asarray(timetag2, dtype=np.int64)

    year = datetag // 1000
    doy = datetag % 1000
    base = pd.to_datetime(year.astype(str), format='%Y') + pd.to_timedelta(doy - 1, unit='D')

    hh = timetag2 // 10000000
    mm = (timetag2 // 100000) % 100
    ss = (timetag2 // 1000) % 100
    ms = timetag2 % 1000
    tod = (pd.to_timedelta(hh, unit='h') + pd.to_timedelta(mm, unit='m')
           + pd.to_timedelta(ss, unit='s') + pd.to_timedelta(ms, unit='ms'))
    return pd.DatetimeIndex(base + tod)


def _gps_to_datetime(date, utcpos):
    """Convertit les champs GPS $GPRMC DATE (DDMMYY) + UTCPOS (HHMMSS.ss) -- l'heure
    UTC vraie du GPS -- en un DatetimeIndex UTC (vectorisé, via pandas)."""
    date = np.asarray(date, dtype=np.int64)
    day = date // 10000
    mon = (date // 100) % 100
    year = 2000 + (date % 100)
    base = pd.to_datetime({'year': year, 'month': mon, 'day': day})

    utcpos = np.asarray(utcpos, dtype=np.float64)
    hh = (utcpos // 10000).astype(np.int64)
    mm = ((utcpos // 100) % 100).astype(np.int64)
    ss = utcpos % 100
    tod = pd.to_timedelta(hh, unit='h') + pd.to_timedelta(mm, unit='m') + pd.to_timedelta(ss, unit='s')
    return pd.DatetimeIndex(base + tod)


def compute_clock_drift_offset(outfn):
    """Détecte automatiquement la dérive entre l'horloge interne du datalogger
    (DATETAG/TIMETAG2, utilisée partout dans le pipeline y compris pour la géométrie
    solaire) et l'heure GPS vraie (DATE/UTCPOS de la sentence $GPRMC), en comparant
    les deux sur les mêmes enregistrements. Retourne le décalage en secondes à
    AJOUTER à DATETAG/TIMETAG2 pour les aligner sur le GPS, ou None si indisponible."""
    gp_path = "/GPRMC_NMEA0183v3.01.tdf"
    with h5py.File(outfn, "r") as h5f:
        if gp_path not in h5f:
            return None
        g = h5f[gp_path]
        status = g["STATUS"][...]
        valid = np.array([row[0] == b'A' for row in status])
        if not valid.any():
            return None
        datetag = g["DATETAG"][...]['NONE'][valid]
        timetag2 = g["TIMETAG2"][...]['NONE'][valid]
        date = g["DATE"][...]['NONE'][valid]
        utcpos = g["UTCPOS"][...]['NONE'][valid]

    instr_dt = _tags_to_datetime(datetag, timetag2)
    true_dt = _gps_to_datetime(date, utcpos)

    offset_s = (true_dt - instr_dt) / pd.Timedelta(seconds=1)
    median_offset = float(np.median(offset_s))
    spread = float(np.std(offset_s))
    if spread > 5:
        print(f"    ⚠️  Dérive d'horloge incohérente sur {os.path.basename(outfn)} "
              f"(médiane={median_offset:.1f}s, écart-type={spread:.1f}s) -- à vérifier manuellement.")
    return median_offset


def process_time_offset(outfn, offset_seconds):
    """Décale DATETAG/TIMETAG2 d'un nombre fixe de secondes dans TOUS les groupes du
    fichier HDF5 (y compris $GPRMC lui-même), pour corriger une dérive de l'horloge
    interne du datalogger par rapport au temps GPS vrai. N'affecte pas DATE/UTCPOS
    (déjà corrects, dérivés du GPS)."""
    if offset_seconds is None or abs(offset_seconds) < 0.5:
        return
    delta = pd.to_timedelta(offset_seconds, unit='s')
    with h5py.File(outfn, "r+") as h5f:
        for gname in h5f.keys():
            g = h5f[gname]
            if "DATETAG" not in g or "TIMETAG2" not in g:
                continue
            datetag = g["DATETAG"][...]
            timetag2 = g["TIMETAG2"][...]
            shifted = _tags_to_datetime(datetag['NONE'], timetag2['NONE']) + delta

            new_datetag = shifted.strftime('%Y%j').astype(np.int64).astype(np.float64)
            new_timetag2 = (shifted.hour * 10**7 + shifted.minute * 10**5
                             + shifted.second * 10**3 + (shifted.microsecond // 1000)).astype(np.float64)

            datetag['NONE'] = new_datetag
            timetag2['NONE'] = new_timetag2
            h5f[gname]["DATETAG"][...] = datetag
            h5f[gname]["TIMETAG2"][...] = timetag2
    print(f"    🕒 Correction de dérive d'horloge appliquée ({offset_seconds:+.1f}s) sur {os.path.basename(outfn)}")


# --- FONCTION MAÎTRESSE APPELÉE PAR LE BATCH ---
def correct_L1A_file(inpath, outpath, fn, roll_offset, pitch_offset=0.0, fix_clock_drift=False):
    """Copie le fichier et applique séquentiellement les corrections requises."""
    infn = os.path.join(inpath, fn)
    outfn = os.path.join(outpath, fn)

    os.makedirs(outpath, exist_ok=True)
    plot_dir = os.path.join(outpath, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    shutil.copy2(infn, outfn)

    # Exécution des corrections modulaires
    if fix_clock_drift:
        offset_s = compute_clock_drift_offset(outfn)
        if offset_s is not None:
            process_time_offset(outfn, offset_s)
        else:
            print(f"    ⚠️  Impossible de calculer la dérive d'horloge pour {fn} (pas de $GPRMC valide).")
    process_heading_interpolation(outfn, plot_dir)
    process_roll_offset(outfn, offset_value=roll_offset)
    process_pitch_offset(outfn, offset_value=pitch_offset)


# ---------------------------------------------------------------------------
# BLOC D'EXÉCUTION (Équivalent du if/else en R)
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # 1. Définition des chemins d'accès (Modifiez si nécessaire)
    inpath = "/Users/simonbelanger/Data/Amundsen/2026_LEG_00/L1/pySAS/L1A/"
    outpath = "/Users/simonbelanger/Data/Amundsen/2026_LEG_00/L1/pySAS/L1A_corrected/"

    # 2. Choix du mode de test (0 = Un seul fichier, 1 = Tout le dossier en batch)
    RUN_BATCH = 0  # Changez par 1 pour traiter tous les fichiers d'un coup

    if RUN_BATCH == 1:
        print(f"🔍 Recherche de fichiers dans : {inpath}")
        files = [f for f in os.listdir(inpath) if f.endswith('.hdf')]
        print(f"📋 {len(files)} fichier(s) trouvé(s) : {files}")

        for fn in files:
            correct_L1A_file(inpath, outpath, fn, roll_offset=0)

    else:
        fn = "pySAS006_20260701_180621_L1A.hdf"
        print(f"🚀 Traitement du fichier unique : {fn}")
        correct_L1A_file(inpath, outpath, fn, roll_offset=0)
