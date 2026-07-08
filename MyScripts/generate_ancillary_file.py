import os
import sys
import pandas as pd
import numpy as np


def read_ats(filen):
    """Lit les données météo de l'Amundsen et calcule le vent réel (True Wind)."""
    df = pd.read_csv(filen, sep=";", header=0, low_memory=False,
                     skipinitialspace=True, na_values=["NaN"])

    # Nettoyage des lignes textuelles résiduelles (si doublons d'en-têtes dans le fichier)
    for col in ['ATS_wind', 'ATS_wind_dir', 'speed', 'ATS_temp_surface']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # Diagnostic en cas d'absence
            print(f"Les colonnes disponibles sont : {list(df.columns)}")
            raise KeyError(f"La colonne indispensable '{col}' est introuvable. Vérifiez les espaces.")

    # Élimination des lignes blanches ou corrompues
    df = df.dropna(subset=['ATS_wind', 'speed']).reset_index(drop=True)

    # Conversion de la colonne date en objets datetime UTC
    df['datetime'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    df = df.dropna(subset=['datetime']).reset_index(drop=True)

    AWS = df['ATS_wind']
    AWA = df['ATS_wind_dir']
    BS = df['speed'] * 0.514444
    awa_rad = np.radians(AWA)

    df['TWS'] = np.sqrt(AWS ** 2 + BS ** 2 - 2 * AWS * BS * np.cos(awa_rad))
    df['TWA'] = np.degrees(np.arctan2(AWS * np.sin(awa_rad), AWS * np.cos(awa_rad) - BS))
    df['SST'] = df['ATS_temp_surface']
    return df


def read_tsg(filen):
    """
    Lit les données du TSG de l'Amundsen (sans en-tête),
    nettoie les espaces cachés et gère les NaN textuels.
    """
    # skipinitialspace=True élimine les espaces invisibles après les ";"
    # na_values=["NaN"] gère les chaînes de texte vides
    df = pd.read_csv(filen, sep=";", header=None, low_memory=False,
                     skipinitialspace=True, na_values=["NaN"])

    df.columns = ["datetime", "time", "lat", "lon", "t1", "c1",
                  "s", "sv", "t2", "fluo", "debi", "sv2"]

    # Conversion de la colonne temporelle en datetime UTC
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
    df = df.dropna(subset=['datetime']).reset_index(drop=True)
    return df


def generate_ancillary_sb(base_pysas_path, tsg_path, ats_path, date_str, cruise="AMD", experiment="LEG_00"):
    """
    Lit, fusionne les données ATS/TSG et écrit le fichier final .sb
    au format SeaBASS rigide de la NASA.
    """
    sb_filename = f"{cruise}_{experiment}_Ancillary_{date_str}.sb"
    output_dir = os.path.join(base_pysas_path, "Ancillary")
    os.makedirs(output_dir, exist_ok=True)
    out_filepath = os.path.join(output_dir, sb_filename)

    # 1. Lecture des fichiers d'origine
    try:
        df_ats = read_ats(ats_path)
        df_tsg = read_tsg(tsg_path)
    except Exception as e:
        print(f"❌ Erreur lors de la lecture des fichiers sources ATS/TSG : {e}")
        return False

    # 2. Fusion des jeux de données (merge by datetime)
    df_merge = pd.merge(df_ats, df_tsg, on="datetime", how="inner")

    if df_merge.empty:
        print(f"⚠️ Attention : Aucune ligne synchrone trouvée entre ATS et TSG pour la date {date_str}.")
        return False

    # 3. Extraction des composantes temporelles et formatage
    df_merge['year'] = df_merge['datetime'].dt.strftime('%Y')
    df_merge['month'] = df_merge['datetime'].dt.strftime('%m')
    df_merge['day'] = df_merge['datetime'].dt.strftime('%d')
    df_merge['hour'] = df_merge['datetime'].dt.strftime('%H')
    df_merge['minute'] = df_merge['datetime'].dt.strftime('%M')
    df_merge['second'] = df_merge['datetime'].dt.strftime('%S')
    df_merge['station'] = 9999

    # 4. Sélection des colonnes et sous-échantillonnage (1 ligne sur 10)
    df_anc = df_merge[['station', 'year', 'month', 'day', 'hour', 'minute', 'second',
                       'lat_x', 'lon_x', 'SST', 'TWS', 'TWA', 's']].copy()

    df_anc.columns = ['station', 'year', 'month', 'day', 'hour', 'minute', 'second',
                      'lat', 'lon', 'Wt', 'wind', 'wdir', 'sal']

    # Nettoyage des valeurs manquantes et sous-échantillonnage [::10]
    df_anc_subsampled = df_anc.dropna().iloc[::10].reset_index(drop=True)

    if df_anc_subsampled.empty:
        print("⚠️ Le tableau final sous-échantillonné est vide après nettoyage des NaN.")
        return False

    # 5. Extraction des métadonnées géographiques pour le bloc d'en-tête SeaBASS
    df_anc_subsampled['lat'] = pd.to_numeric(df_anc_subsampled['lat'], errors='coerce')
    df_anc_subsampled['lon'] = pd.to_numeric(df_anc_subsampled['lon'], errors='coerce')
    df_anc_subsampled = df_anc_subsampled.dropna(subset=['lat', 'lon']).reset_index(drop=True)

    north = df_anc_subsampled['lat'].max()
    south = df_anc_subsampled['lat'].min()
    east = df_anc_subsampled['lon'].max()
    west = df_anc_subsampled['lon'].min()

    start_time = f"{df_anc_subsampled.iloc[0]['hour']}{df_anc_subsampled.iloc[0]['minute']}{df_anc_subsampled.iloc[0]['second']}"
    end_time = f"{df_anc_subsampled.iloc[-1]['hour']}{df_anc_subsampled.iloc[-1]['minute']}{df_anc_subsampled.iloc[-1]['second']}"

    # Construction de l'en-tête officiel SeaBASS
    header = [
        "/begin_header",
        f"/data_file_name={sb_filename}",
        "/affiliations=UQAR",
        "/investigators=Simon_Belanger",
        "/contact=simon_belanger@uqar.ca",
        "/data_status=final",
        f"/experiment={experiment}",
        f"/cruise={cruise}",
        "/calibration_files=doesntapply.txt",
        "/missing=-9999.0",
        "/delimiter=comma",
        f"/start_date={date_str}",
        f"/end_date={date_str}",
        f"/north_latitude={north:.4f}",
        f"/south_latitude={south:.4f}",
        f"/east_longitude={east:.4f}",
        f"/west_longitude={west:.4f}",
        f"/start_time={start_time}",
        f"/end_time={end_time}",
        "/measurement_depth=0",
        "/water_depth=",
        "/fields=station,year,month,day,hour,minute,second,lat,lon,Wt,wind,wdir,sal",
        "/units=none,yyyy,mo,dd,hh,mn,ss,degrees,degrees,degreesC,m/s,degrees,psu",
        "/end_header"
    ]

    # 6. Écriture physique sur disque
    with open(out_filepath, 'w', encoding='utf-8') as f_out:
        for line in header:
            f_out.write(f"{line}\n")
        df_anc_subsampled.to_csv(f_out, index=False, header=False, sep=',')

    print(f"✅ SeaBASS Ancillary File created from ATS and TSG : {out_filepath} ({len(df_anc_subsampled)} lignes)")
    return True
