import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from datetime import datetime
from fpdf import FPDF
import matplotlib as mpl  # S'assurer que matplotlib est importé en haut
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def extract_cast_metadata_and_qc(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        with h5py.File(file_path, 'r') as h5f:
            rrs_path = "/REFLECTANCE/Rrs_HYPER"
            if rrs_path not in h5f:
                return None

            rrs_raw = h5f[rrs_path][...]
            colnames = rrs_raw.dtype.names
            datetag = rrs_raw[colnames[0]]
            timetag2 = rrs_raw[colnames[1]]
            n_records = len(timetag2)

            # 1. Extraction robuste du QWIP (3e colonne de la structure)
            if "/DERIVED_PRODUCTS/qwip" in h5f:
                qwip_raw = h5f["/DERIVED_PRODUCTS/qwip"][...]
                qwip_cols = qwip_raw.dtype.names
                qwip_values = qwip_raw[qwip_cols[2]] if len(qwip_cols) > 2 else qwip_raw[qwip_cols[-1]]
            else:
                qwip_values = np.zeros(n_records)

            # 2. Extraction robuste du WEI_QA (votre chemin /wei_QA, 3e colonne)
            if "/DERIVED_PRODUCTS/wei_QA" in h5f:
                wei_raw = h5f["/DERIVED_PRODUCTS/wei_QA"][...]
                wei_cols = wei_raw.dtype.names
                wei_values = wei_raw[wei_cols[2]] if len(wei_cols) > 2 else wei_raw[wei_cols[-1]]
            else:
                wei_values = np.zeros(n_records)

            # 3. Extraction de la Navigation (votre chemin /ANCILLARY, 3e colonne)
            if "/ANCILLARY/LATITUDE" in h5f:
                lat_raw = h5f["/ANCILLARY/LATITUDE"][...]
                lat_cols = lat_raw.dtype.names
                lat_values = lat_raw[lat_cols[2]] if lat_cols and len(lat_cols) > 2 else lat_raw
            else:
                lat_values = np.zeros(n_records)

            if "/ANCILLARY/LONGITUDE" in h5f:
                lon_raw = h5f["/ANCILLARY/LONGITUDE"][...]
                lon_cols = lon_raw.dtype.names
                lon_values = lon_raw[lon_cols[2]] if lon_cols and len(lon_cols) > 2 else lon_raw
            else:
                lon_values = np.zeros(n_records)

            records = []
            for i in range(n_records):
                records.append({
                    "Datetag": int(datetag[i]),
                    "Timetag2": int(timetag2[i]),
                    "Latitude": float(lat_values[i]) if i < len(lat_values) else 0.0,
                    "Longitude": float(lon_values[i]) if i < len(lon_values) else 0.0,
                    "QWIP": float(qwip_values[i]) if i < len(qwip_values) else 0.0,
                    "WEI_QA": float(wei_values[i]) if i < len(wei_values) else 0.0
                })
            return records

    except Exception as e:
        print(f"⚠️ Erreur d'extraction sur {os.path.basename(file_path)} : {e}")
        return None


def plot_top2_methods_comparison(base_path, date_str, top1_method, top2_method, df_ranked):
    """
    Génère une figure comparative (2 panels) pour les deux meilleures méthodes de la journée.
    Tous les spectres de la journée sont superposés avec un dégradé de couleur (viridis) selon l'heure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # 1. Préparation du dégradé de couleur basé sur l'heure (Timetag2)
    # On récupère toutes les heures uniques de la journée pour caler le dégradé
    all_times = sorted(df_ranked["Timetag2"].unique())
    norm = mcolors.Normalize(vmin=0, vmax=len(all_times) - 1)
    cmap = cm.get_cmap("viridis")  # Utilisation sécurisée de viridis

    # Dictionnaire pour associer chaque timestamp à sa couleur viridis
    time_color_map = {t: cmap(norm(i)) for i, t in enumerate(all_times)}

    def plot_single_method(ax, method_name):
        # On filtre les données pour cette méthode
        df_method = df_ranked[df_ranked["Method"] == method_name]

        # On va ouvrir les fichiers HDF5 correspondants pour tracer les vrais spectres
        for idx, row in df_method.iterrows():
            fpath = os.path.join(base_path, method_name, "L2", row["Filename"])
            if os.path.exists(fpath):
                with h5py.File(fpath, 'r') as h5f:
                    rrs_raw = h5f["/REFLECTANCE/Rrs_HYPER"][...]
                    colnames = rrs_raw.dtype.names
                    wavelengths = np.array([float(w) for w in colnames[2:]])

                    # On trouve la ligne qui correspond au Timetag2 actuel
                    timetag2_vector = rrs_raw[colnames[1]]
                    line_idx = np.where(timetag2_vector == row["Timetag2"])[0]

                    if len(line_idx) > 0:
                        # Extraction du spectre Rrs
                        rrs_spectrum = np.array([rrs_raw[w][line_idx[0]] for w in colnames[2:]])
                        color = time_color_map[row["Timetag2"]]
                        ax.plot(wavelengths, rrs_spectrum, color=color, alpha=0.6, linewidth=1)

        ax.set_title(f"Méthode : {method_name} (Top)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Longueur d'onde (nm)")
        ax.set_xlim(350, 900)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)

    # Tracé pour les deux méthodes
    plot_single_method(ax1, top1_method)
    plot_single_method(ax2, top2_method)
    ax1.set_ylabel("$R_{rs}$ ($sr^{-1}$)", fontweight='bold')

    # Ajouter une barre de couleur commune (Colorbar) pour l'heure
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical', pad=0.03, aspect=30)

    # Ajuster les étiquettes de la colorbar pour afficher les heures de début et de fin
    t_start, t_end = all_times[0], all_times[-1]
    cbar.set_ticks([0, len(all_times) - 1])
    cbar.set_ticklabels([
        f"{int(t_start // 10000000):02d}:{int((t_start % 10000000) // 100000):02d} UTC",
        f"{int(t_end // 10000000):02d}:{int((t_end % 10000000) // 100000):02d} UTC"
    ])
    cbar.set_label("Progression temporelle (Heure de la journée)", fontweight='bold')

    fig.subplots_adjust(top=0.85)
    fig.suptitle(f"Comparaison Spectrale Rrs des 2 Meilleures Méthodes ({date_str})", fontsize=13, fontweight='bold')

    out_png = os.path.join(base_path, f"L2_Top2_Methods_Comparison_{date_str}.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Figure comparative Top 2 exportée : {out_png}")
    return out_png


def plot_top4_methods_with_map(base_path, date_str, top_methods, df_ranked, time_color_map):
    """
    Génère une figure diagnostic avancée :
    - 4 panels (2x2) isolés en haut pour les spectres (en anglais).
    - 1 grand panel indépendant en bas prenant 100% de la largeur pour la carte.
    - 1 insert dézoomé (InsetValue) au 1:1 000 000 pour voir la côte élargie.
    """
    # On agrandit la figure pour donner de l'espace vertical
    fig = plt.figure(figsize=(18, 14))

    # --- LES SPECTRES EN HAUT (2x2) ---
    ax1 = plt.subplot(3, 2, 1)
    ax2 = plt.subplot(3, 2, 2)
    ax3 = plt.subplot(3, 2, 3)
    ax4 = plt.subplot(3, 2, 4)

    axes_spectra = [ax1, ax2, ax3, ax4]

    # Préparation de la colormap pour la Colorbar du bas
    all_times = sorted(df_ranked["Timetag2"].unique())
    norm = mcolors.Normalize(vmin=0, vmax=len(all_times) - 1)
    cmap = mpl.colormaps["viridis"]

    # 1. TRACÉ DES SPECTRES (2x2)
    for idx, method_name in enumerate(top_methods[:4]):
        ax = axes_spectra[idx]
        df_method = df_ranked[df_ranked["Method"] == method_name]

        for _, row in df_method.iterrows():
            fpath = os.path.join(base_path, method_name, "L2", row["Filename"])
            if os.path.exists(fpath):
                with h5py.File(fpath, 'r') as h5f:
                    rrs_raw = h5f["/REFLECTANCE/Rrs_HYPER"][...]
                    colnames = rrs_raw.dtype.names
                    wavelengths = np.array([float(w) for w in colnames[2:]])
                    timetag2_vector = rrs_raw[colnames[1]]
                    line_indices = np.where(timetag2_vector == row["Timetag2"])[0]

                    if len(line_indices) > 0:
                        line_idx = line_indices[0]
                        rrs_spectrum = np.array([rrs_raw[w][line_idx] for w in colnames[2:]])
                        color = time_color_map[row["Timetag2"]]
                        ax.plot(wavelengths, rrs_spectrum, color=color, alpha=0.5, linewidth=1)

        ax.set_title(f"Rank #{idx + 1}: {method_name}", fontsize=16, fontweight='bold')
        ax.set_xlim(380, 800)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)

        if idx >= 2:
            ax.set_xlabel("Wavelength (nm)", fontweight='bold', fontsize=14)
        if idx % 2 == 0:
            ax.set_ylabel("$R_{rs}$ ($sr^{-1}$)", fontweight='bold',fontsize=14)

    # ---------------------------------------------------------------------------
    # 2. TRACÉ DE LA CARTE PRINCIPALE AVEC LAND MASK (CARTOPY)
    # ---------------------------------------------------------------------------
    ax_map = plt.subplot(3, 2, (5, 6), projection=ccrs.PlateCarree())

    df_unique_casts = df_ranked[df_ranked["Method"] == top_methods[0]].sort_values(by="Timetag2")

    # Détermination des frontières de votre zone d'étude (Zoomé serré sur le trajet)
    lon_min, lon_max = df_unique_casts["Longitude"].min() - 0.15, df_unique_casts["Longitude"].max() + 0.15
    lat_min, lat_max = df_unique_casts["Latitude"].min() - 0.1, df_unique_casts["Latitude"].max() + 0.1
    ax_map.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Éléments cartographiques haute résolution (10m)
    ax_map.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#e0f3ff')
    ax_map.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#f4f3ef', edgecolor='#bdc3c7', linewidth=0.5)
    ax_map.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='#e0f3ff')
    ax_map.add_feature(cfeature.RIVERS.with_scale('10m'), edgecolor='#e0f3ff', linewidth=0.5)

    # Grille géographique SIG
    gl = ax_map.gridlines(draw_labels=True, linestyle=':', alpha=0.5, color='gray')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12, 'weight': 'bold'}
    gl.ylabel_style = {'size': 12, 'weight': 'bold'}

    # Tracé du trajet du navire et des points de stations
    ax_map.plot(df_unique_casts["Longitude"], df_unique_casts["Latitude"],
                color="#7f8c8d", linestyle="-", linewidth=2.5, transform=ccrs.PlateCarree(), zorder=4)

    # Tracé des stations de mesure (Points Viridis)
    for idx, (pandas_idx, row) in enumerate(df_unique_casts.iterrows()):
        color = time_color_map[row["Timetag2"]]
        ax_map.scatter(row["Longitude"], row["Latitude"],
                       color=color, edgecolors='black', s=160, zorder=5, linewidth=1.2,
                       transform=ccrs.PlateCarree())

        # --- FILTRE D'ANNOTATION INTELLIGENT ---
        # Affiche l'heure uniquement pour le premier point, le dernier point, et un point sur 5
        if idx == 0 or idx == len(df_unique_casts) - 1 or idx % 5 == 0:
            t_raw = row["Timetag2"]
            time_lbl = f"{int(t_raw // 10000000):02d}:{int((t_raw % 10000000) // 100000):02d}"

            ax_map.text(row["Longitude"], row["Latitude"], f"  {time_lbl}",
                        fontsize=12, fontweight='bold', alpha=0.85, va='center', zorder=6,
                        transform=ccrs.PlateCarree())

    ax_map.set_title("🌐 Campaign Mapping & Vessel Track", fontsize=15,
                     fontweight='bold', pad=12)
    ax_map.set_aspect('auto')

    # ---------------------------------------------------------------------------
    # 3. CRÉATION DE L'INSERT RÉGIONAL DÉZOOMÉ (MINI-MAP AU 1:1 000 000)
    # ---------------------------------------------------------------------------
    # On ajoute un axe flottant en haut à droite de l'axe de la carte principale
    # [X_gauche, Y_bas, Largeur, Hauteur] relatifs à la figure
    #ax_inset = fig.add_axes([0.76, 0.15, 0.13, 0.13], projection=ccrs.PlateCarree())
    # [X_gauche, Y_bas, Largeur, Hauteur]
    ax_inset = fig.add_axes([0.15, -0.05, 0.14, 0.14], projection=ccrs.PlateCarree())

    # Élargissement significatif des frontières géographiques (Dézoom à large échelle)
    inset_lon_min, inset_lon_max = df_unique_casts["Longitude"].mean() - 5, df_unique_casts["Longitude"].mean() + 5
    inset_lat_min, inset_lat_max = df_unique_casts["Latitude"].mean() - 3, df_unique_casts["Latitude"].mean() + 3
    ax_inset.set_extent([inset_lon_min, inset_lon_max, inset_lat_min, inset_lat_max], crs=ccrs.PlateCarree())

    # Couches cartographiques simplifiées (50m) pour l'insert régional
    ax_inset.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#e0f3ff')
    ax_inset.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#f4f3ef', edgecolor='#bdc3c7', linewidth=0.4)
    ax_inset.add_feature(cfeature.LAKES.with_scale('50m'), facecolor='#e0f3ff')

    # Dessiner un rectangle rouge représentant la zone d'étude zoomée
    import matplotlib.patches as mpatches
    rect = mpatches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                              fill=False, edgecolor='red', linewidth=1., zorder=10,
                              transform=ccrs.PlateCarree())
    ax_inset.add_patch(rect)

    # Tracer la ligne de route en miniature pour bien la situer dans le Golfe/Estuaire
    ax_inset.plot(df_unique_casts["Longitude"], df_unique_casts["Latitude"],
                  color="green", linestyle="-", linewidth=1., transform=ccrs.PlateCarree(), zorder=11)

    ax_inset.set_title("Regional Context", fontsize=12, fontweight='bold')
    ax_inset.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
                         labelleft=False)

    # ---------------------------------------------------------------------------
    # 4. POSITIONNEMENT DE LA COLORBAR (TOUT EN BAS)
    # ---------------------------------------------------------------------------
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.35, 0.02, 0.55, 0.015])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')

    t_start, t_end = all_times[0], all_times[-1]
    cbar.set_ticks([0, len(all_times) - 1])
    cbar.set_ticklabels([
        f"Start: {int(t_start // 10000000):02d}:{int((t_start % 10000000) // 100000):02d} UTC",
        f"End: {int(t_end // 10000000):02d}:{int((t_end % 10000000) // 100000):02d} UTC"
    ])
    cbar.set_label("Time progression along the trackline", fontweight='bold', fontsize=16)

    fig.suptitle(f"Multi-Method L2 Diagnostic Analysis — pySAS ({date_str})", fontsize=20, fontweight='bold', y=0.95)

    # Sauvegarde
    analysis_dir = os.path.join(base_path, "AnalysisComparison")
    os.makedirs(analysis_dir, exist_ok=True)
    out_png = os.path.join(analysis_dir, f"L2_Top4_Methods_With_Map_{date_str}.png")

    plt.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"🚀 Figure and Context Map exported: {out_png}")
    return out_png


def generate_pdf_report(base_path, date_str, global_ranking, fig_path):
    """
    Génère un rapport de diagnostic PDF propre et compatible avec fpdf2 v2.5+.
    Élimine les émojis non-Unicode et les paramètres dépréciés (ln=True).
    """
    pdf = FPDF()
    pdf.add_page()

    # Titre du rapport (Syntaxe moderne : new_x et new_y remplacent ln=True)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "pySAS L2 Diagnostic Report - Amundsen Cruise",
             new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Data Date: {date_str} | Report Generated On: {datetime.now().strftime('%Y-%m-%d')}",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)

    # Section 1 : Classement
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "1. GLOBAL METHOD RANKING (Based on QWIP Mean Score)",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)

    # En-tête du tableau dans le PDF
    pdf.set_fill_color(230, 230, 230)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(40, 8, "Rank", border=1, align="C", fill=True)
    pdf.cell(60, 8, "Processing Method", border=1, align="C", fill=True)
    pdf.cell(60, 8, "Mean Score (Rank)", border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    for rank, (method, avg_quote) in enumerate(global_ranking.items(), 1):
        # Mettre en évidence la ligne du vainqueur en vert clair
        if rank == 1:
            pdf.set_fill_color(220, 245, 220)
            fill_bool = True
        else:
            fill_bool = False

        pdf.cell(40, 8, f"#{rank}", border=1, align="C", fill=fill_bool)
        pdf.cell(60, 8, f"{method}", border=1, align="C", fill=fill_bool)
        pdf.cell(60, 8, f"{avg_quote:.2f} / {len(global_ranking)}", border=1, align="C", fill=fill_bool, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 11)

    # --- CORRECTION DE L'ÉMOJI ICI (Remplacé par [TOP]) ---
    pdf.cell(0, 10, f"[BEST] Recommended method for this day: {global_ranking.index[0]}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # Section 2 : Figure
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "2. SPECTRAL COMPARISON OF THE TOP 2 METHODS",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Insertion de la figure Matplotlib
    if os.path.exists(fig_path):
        pdf.image(fig_path, x=10, w=190)

    out_pdf = os.path.join(base_path, f"L2_Processing_Report_{date_str}.pdf")
    pdf.output(out_pdf)
    print(f"📄 Final PDF Report successfully generated: {out_pdf}")

# ===========================================================================
# 🛠️ PARSING DU FICHIER DE CONFIGURATION DYNAMIQUE (.ENV)
# ===========================================================================
# Fonction de lecture rapide
def load_pipeline_config(config_path):
    config = {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '#' in line:
                line = line.split('#')[0].strip()
            if '=' in line:
                key, val = line.split('=', 1)
                config[key.strip()] = val.strip()
    return config



if __name__ == "__main__":

    try:
        import geojson
        from geojson import Feature, Point, FeatureCollection
        HAS_GEOJSON = True
    except ImportError:
        HAS_GEOJSON = False

        #DATE_STR = "20260705"

        #BASE_PATH = "/Users/simonbelanger/Data/Amundsen/2026_LEG_00/L1/pySAS/"

    # Détermination du dossier actuel du script pour cibler le .env
    MY_DIR = os.path.dirname(os.path.abspath(__file__))
    env = load_pipeline_config(os.path.join(MY_DIR, "pipeline_config.env"))

    import sys

    if len(sys.argv) > 1:
        DATE_STR = sys.argv[1]
    else:
        # Si aucun argument n'est passé au script python de diagnostic,
        # on prend par défaut la date d'hier calculée par le système
        from datetime import datetime, timedelta

        DATE_STR = (datetime.utcnow() - timedelta(days=1)).strftime('%Y%m%d')

    # Assignation des chemins et des versions
    MAIN_DATA_PATH = env["MAIN_DATA_PATH"]
    BASE_PATH = os.path.join(MAIN_DATA_PATH, "pySAS")

    ANALYSIS_DIR = os.path.join(BASE_PATH, "AnalysisComparison")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # Sky models to compare (matches SKY_MODEL passed by download_and_run_hypercp.sh):
    # optional 2nd CLI arg, comma-separated (e.g. "M99,Z17"); defaults to all 3.
    if len(sys.argv) > 2:
        SKY_MODELS = [m.strip() for m in sys.argv[2].split(",")]
    else:
        SKY_MODELS = ["M99", "Z17", "3C"]
    METHODS = [f"{model}{correction}" for model in SKY_MODELS for correction in ("NN", "NIR", "SimSpec")]

    print(f"📊 [Diagnostics] Analyzing metrics for date: {DATE_STR}")
    print(f"📂 Source directory: {BASE_PATH}")
    print(f"🎯 Methods matrix to evaluate: {METHODS}")

    nb_methods = len(METHODS)

    # Trouver tous les fichiers HDF5 L2 de référence via le premier modèle demandé
    ref_dir = os.path.join(BASE_PATH, f"{SKY_MODELS[0]}NN", "L2")
    files = sorted([f for f in os.listdir(ref_dir) if f.endswith('.hdf') and DATE_STR in f])

    all_data = []

    for fname in files:
        for method in METHODS:
            fpath = os.path.join(BASE_PATH, method, "L2", fname)
            records = extract_cast_metadata_and_qc(fpath)
            if records:
                for r in records:
                    r["Filename"] = fname
                    r["Method"] = method
                    all_data.append(r)

    df = pd.DataFrame(all_data)

    # Export 1 : Tableau croisé global (méthodes en colonnes pour QWIP et WEI_QC)
    df_pivot = df.pivot(index=["Filename", "Datetag", "Timetag2", "Latitude", "Longitude"],
                        columns="Method", values=["QWIP", "WEI_QA"]).reset_index()
    # Aplatir les multi-index de colonnes (ex: QWIP_M99SimSpec)
    df_pivot.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_pivot.columns]

    out_csv_global = os.path.join(BASE_PATH, f"L2_QC_Global_Summary_{DATE_STR}.csv")
    df_pivot.to_csv(out_csv_global, index=False)
    print(f"✅ Tableau global exporté : {out_csv_global}")

    # Export 2 : Tableau d'ordonnancement (Trouver le classement de la meilleure méthode par ligne)
    best_records = []
    for (fname, ttag), group in df.groupby(["Filename", "Timetag2"]):
        # Tri par QWIP croissant pour isoler le plus faible
        sorted_group = group.sort_values(by="QWIP")
        best_row = sorted_group.iloc[0]
        best_records.append({
            "Filename": fname, "Datetag": best_row["Datetag"], "Timetag2": ttag,
            "Latitude": best_row["Latitude"], "Longitude": best_row["Longitude"],
            "Best_Method": best_row["Method"], "Min_QWIP": best_row["QWIP"], "Associated_WEI": best_row["WEI_QA"]
        })

    df_best = pd.DataFrame(best_records)
    out_csv_best = os.path.join(BASE_PATH, f"L2_Best_Methods_Ranking_{DATE_STR}.csv")
    df_best.to_csv(out_csv_best, index=False)
    print(f"✅ Tableau du classement optimal exporté : {out_csv_best}")

    # ---------------------------------------------------------------------------
    # Export 2 : Attribution des notes (Rangs 1 à 6) et Classement Global
    # ---------------------------------------------------------------------------
    ranked_records = []

    # On boucle sur chaque cast individuel (chaque tranche de chaque fichier)
    for (fname, ttag), group in df.groupby(["Filename", "Timetag2"]):
        # On trie le groupe par QWIP croissant
        sorted_group = group.sort_values(by="QWIP").reset_index(drop=True)

        # On attribue un rang de 1 (meilleur) à N (moins bon) selon la position dans le tri
        for rank_idx, row in sorted_group.iterrows():
            ranked_records.append({
                "Filename": fname,
                "Datetag": row["Datetag"],
                "Timetag2": ttag,
                "Latitude": row["Latitude"],
                "Longitude": row["Longitude"],
                "Method": row["Method"],
                "QWIP": row["QWIP"],
                "WEI_QA": row["WEI_QA"],
                "Rank_Quote": rank_idx + 1  # Donne une note de 1 à 6
            })

    df_ranked = pd.DataFrame(ranked_records)

    # Sauvegarde du tableau détaillé des notes par cast
    out_csv_ranks = os.path.join(ANALYSIS_DIR, f"L2_Methods_Quotes_{DATE_STR}.csv")
    df_ranked.to_csv(out_csv_ranks, index=False)
    print(f"✅ Tableau des notes (1 à 6) exporté : {out_csv_ranks}")

    # ---------------------------------------------------------------------------
    # EXPORT 3 : Calcul du score global de la journée au terminal
    # ---------------------------------------------------------------------------
    print("\n📊 --- CLASSEMENT GLOBAL DES MÉTHODES POUR LA JOURNÉE ---")
    global_ranking = df_ranked.groupby("Method")["Rank_Quote"].mean().sort_values()

    for rank, (method, avg_quote) in enumerate(global_ranking.items(), 1):
        print(f"Position {rank} : {method:<12} | Note moyenne (Rang) : {avg_quote:.2f}/{nb_methods}")

    out_csv_top = os.path.join(BASE_PATH, f"L2_Global_Leaderboard_{DATE_STR}.csv")
    global_ranking.to_frame(name="Average_Rank_Quote").to_csv(out_csv_top)

    # --- AJOUT AUTOMATIQUE DE LA FIGURE TOP 2 ET DU PDF RAPPORT ---
    top1 = global_ranking.index[0]  # La méthode gagnante (ex: M99SimSpec ou M99NN)
    top2 = global_ranking.index[1]  # La deuxième meilleure méthode

    # Récupération de la liste ordonnée complète des méthodes
    top_methods_ordered = list(global_ranking.index)

    # GÉNÉRATION DE LA PALETTE DE COULEURS AVANT L'APPEL
    all_times = sorted(df_ranked["Timetag2"].unique())
    norm = mcolors.Normalize(vmin=0, vmax=len(all_times) - 1)
    cmap = mpl.colormaps["viridis"]
    time_color_map = {t: cmap(norm(i)) for i, t in enumerate(all_times)}

    print(f"\n📈 Génération de la figure 2x2 + Carte pour les 4 meilleures méthodes...")
    # On passe le time_color_map généré à la fonction

    chemin_figure = plot_top4_methods_with_map(BASE_PATH, DATE_STR, top_methods_ordered, df_ranked, time_color_map)

    print(f"📝 Compilation du rapport d'analyse PDF final...")
    ANALYSIS_DIR = os.path.join(BASE_PATH, "AnalysisComparison")
    generate_pdf_report(ANALYSIS_DIR, DATE_STR, global_ranking, chemin_figure)

# ---------------------------------------------------------------------------
# EXPORT 4 : Génération automatique du fichier SIG GeoJSON (Votre code actuel)
# ---------------------------------------------------------------------------


if HAS_GEOJSON:
    features = []
    for idx, row in df_best.iterrows():
        lon = float(row["Longitude"])
        lat = float(row["Latitude"])

        # Exclusion des données sans coordonnées GPS valides (0,0)
        if lon == 0.0 and lat == 0.0:
            continue

        t_raw = int(row["Timetag2"])
        time_str = f"{int(t_raw // 10000000):02d}:{int((t_raw % 10000000) // 100000):02d}:{int((t_raw % 100000) // 1000):02d} UTC"

        properties = {
            "filename": str(row["Filename"]),
            "time_utc": time_str,
            "best_meth": str(row["Best_Method"]),
            "min_qwip": float(row["Min_QWIP"]),
            "wei_qa": float(row["Associated_WEI"])
        }

        features.append(Feature(geometry=Point((lon, lat)), properties=properties))

    out_geojson = os.path.join(ANALYSIS_DIR, f"pySAS_L2_Track_QC_{DATE_STR}.geojson")
    with open(out_geojson, 'w', encoding='utf-8') as f:
        geojson.dump(FeatureCollection(features), f, indent=4)
    print(f"🌍 Fichier SIG GeoJSON généré avec succès : {out_geojson}")
else:
    print("\n⚠️ [Note] Le fichier GeoJSON n'a pas pu être créé car la bibliothèque 'geojson' n'est pas installée.")
    print("💡 Pour l'activer, tapez simplement : pip install geojson")
