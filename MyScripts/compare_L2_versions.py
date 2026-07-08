import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


def extract_l2_data(file_path):
    """
    Extrait les longueurs d'onde, la matrice Rrs, les timestampt2 et le vecteur QWIP
    depuis la structure réelle d'un fichier L2 d'HyperCP.
    """
    if not os.path.exists(file_path):
        return None, None, None, None

    with h5py.File(file_path, 'r') as h5f:
        try:
            # 1. Extraction de Rrs_HYPER et des longueurs d'onde
            rrs_path = "/REFLECTANCE/Rrs_HYPER"
            rrs_raw = h5f[rrs_path][...]
            colnames = rrs_raw.dtype.names

            timetag2 = rrs_raw[colnames[1]]  # Deuxième colonne = Timetag2
            wavelengths = np.array([float(w) for w in colnames[2:]])
            rrs_matrix = np.column_stack([rrs_raw[w] for w in colnames[2:]])

            # 2. Extraction du QWIP (Quality Water Index Parameter) d'après votre structure
            # HyperCP stocke souvent le QWIP sous forme de structure avec Datetag/Timetag en colonnes 0 et 1
            qwip_path = "/DERIVED_PRODUCTS/qwip"
            if qwip_path in h5f:
                qwip_raw = h5f[qwip_path][...]
                qwip_cols = qwip_raw.dtype.names
                # La valeur du QWIP est généralement dans la 3e colonne (index 2) ou nommée spécifiquement
                qwip_values = qwip_raw[qwip_cols[2]]
            else:
                qwip_values = np.zeros(len(timetag2))  # Secours si absent

            return wavelengths, rrs_matrix, timetag2, qwip_values

        except Exception as e:
            print(f"❌ Erreur de lecture sur {os.path.basename(file_path)} : {e}")
            return None, None, None, None


def plot_all_casts_grid(base_path, filename, date_str, methods_to_compare):
    """
    Génère une grille de figures (une par cast/tranche) avec classement dynamique
    des légendes par score QWIP croissant.
    """
    # 1. Première passe pour déterminer le nombre total de lignes (casts) à partir d'une méthode valide
    num_casts = 7  # Valeur fixée par vos soins pour ce fichier
    wavelengths_ref = None

    for method in methods_to_compare:
        test_path = os.path.join(base_path, method, "L2", filename)
        w, r, _, _ = extract_l2_data(test_path)
        if r is not None:
            num_casts = r.shape[0]
            wavelengths_ref = w
            break

    if wavelengths_ref is None:
        print("❌ Impossible de lire les fichiers HDF5 L2 spécifiés.")
        return

    # 2. Configuration dynamique de la grille matplotlib (ex: 3 colonnes, lignes adaptées)
    n_cols = 3
    n_rows = int(np.ceil(num_casts / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True)
    axes = axes.flatten()  # Rend la matrice d'axes linéaire pour la boucler facilement

    # Palette de couleurs fixe pour la cohérence visuelle d'un graphique à l'autre
    colors = {
        "M99SimSpec": "#1f77b4", "M99NIR": "#aec7e8", "M99NN": "#ff7f0e",
        "Z17SimSpec": "#2ca02c", "Z17NIR": "#98df8a", "Z17NN": "#d62728",
        "3CSimSpec": "#9467bd", "3CNIR": "#c5b0d5", "3CNN": "#8c564b"
    }

    # 3. Boucle principale sur chaque ligne/tranche temporelle du fichier (les 7 casts)
    for cast_idx in range(num_casts):
        ax = axes[cast_idx]
        time_label = ""

        # Liste temporaire pour stocker les éléments à tracer afin de pouvoir les trier par QWIP
        cast_lines_data = []

        for method in methods_to_compare:
            file_path = os.path.join(base_path, method, "L2", filename)
            wavelengths, rrs_matrix, timetag2, qwip_values = extract_l2_data(file_path)

            if rrs_matrix is not None and cast_idx < rrs_matrix.shape[0]:
                spectrum = rrs_matrix[cast_idx, :]
                qwip_score = qwip_values[cast_idx]

                # Décodage de l'horodatage pour le titre (HHMMSSms)
                t_raw = timetag2[cast_idx]
                h, m, s = int(t_raw // 10000000), int((t_raw % 10000000) // 100000), int((t_raw % 100000) // 1000)
                time_label = f"{h:02d}:{m:02d}:{s:02d} UTC"

                # Enregistrement des données de la méthode pour ce cast précis
                cast_lines_data.append({
                    "method": method,
                    "spectrum": spectrum,
                    "qwip": qwip_score,
                    "color": colors.get(method, "#7f7f7f")
                })

        # --- LE TRI MAGIQUE PAR QWIP CROISSANT ---
        # On trie la liste en fonction du dictionnaire sur la clé 'qwip'
        cast_lines_data = sorted(cast_lines_data, key=lambda x: x["qwip"])

        # 4. Tracé effectif sur le sous-graphique (Subplot) actuel
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)

        for line in cast_lines_data:
            # Affichage de la courbe
            ax.plot(wavelengths_ref, line["spectrum"],
                    color=line["color"], linewidth=1.3,
                    label=f"{line['method']} ({line['qwip']:.3f})")

        # Mise en forme de chaque boîte individuelle de la matrice
        ax.set_title(f"Tranche #{cast_idx + 1} — {time_label}", fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlim(380, 820)

        # Configuration de la légende ordonnée par QWIP croissant
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8, facecolor='white', edgecolor='none')

    # Nettoyage des sous-graphiques vides si le nombre de casts n'est pas un multiple de 3
    for i in range(num_casts, len(axes)):
        fig.delaxes(axes[i])

    # Titre général de la figure et axes maîtres
    fig.suptitle(f"L2 Methods comparison (QWIP scores)\nFichier source : {filename}", fontsize=13,
                 fontweight='bold')

    # Ajout d'étiquettes globales communes pour alléger l'affichage
    fig.supxlabel("Longueur d'onde (nm)", fontweight='bold', fontsize=11)
    fig.supylabel("$R_{rs}$ ($sr^{-1}$)", fontweight='bold', fontsize=11)

    plt.tight_layout()

    # Enregistrement final du poster de comparaison
    out_dir = os.path.join(base_path, "Comparaisons_L2_Plots")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"Grid_Matrix_Rrs_QWIP_Sorted_{date_str}.png")
    plt.savefig(out_png, dpi=180)  # Augmentation du DPI pour la lisibilité des petites légendes
    plt.close()
    print(f"🚀 Poster matriciel de comparaison enregistré avec succès sous :\n👉 {out_png}")


def extract_rrs_hyper(file_path):
    """
    Extrait les longueurs d'onde, la matrice Rrs_HYPER et les timestamps
    depuis la structure réelle d'un fichier L2 d'HyperCP.
    """
    if not os.path.exists(file_path):
        return None, None, None, None

    with h5py.File(file_path, 'r') as h5f:
        try:
            # Le chemin d'accès d'après votre structure
            dataset_path = "/REFLECTANCE/Rrs_HYPER"
            if dataset_path not in h5f:
                print(f"⚠️ Dataset {dataset_path} introuvable dans {os.path.basename(file_path)}")
                return None, None, None, None

            data_raw = h5f[dataset_path][...]

            # Gestion des tableaux structurés HDF5 (Compound arrays)
            colnames = data_raw.dtype.names

            # 1. Extraction des deux premières colonnes temporelles
            datetag = data_raw[colnames[0]]
            timetag2 = data_raw[colnames[1]]

            # 2. Extraction des longueurs d'onde (à partir de la 3e colonne)
            # On convertit les noms de colonnes en valeurs numériques (ex: "412.3" -> 412.3)
            wavelengths = np.array([float(w) for w in colnames[2:]])

            # 3. Reconstruction de la matrice Rrs (N_lignes x N_longueurs_d'onde)
            # On empile les données de chaque colonne de longueur d'onde
            rrs_matrix = np.column_stack([data_raw[w] for w in colnames[2:]])

            return wavelengths, rrs_matrix, datetag, timetag2

        except Exception as e:
            print(f"❌ Erreur lors de la lecture de {os.path.basename(file_path)} : {e}")
            return None, None, None, None


def plot_l2_matrix_comparison(base_path, filename, date_str, methods_to_compare, row_index=0):
    """
    Superpose les spectres Rrs des différentes méthodes pour une ligne (tranche) donnée.
    """
    plt.figure(figsize=(11, 6))

    # Couleurs distinctes pour l'affichage
    colors = {
        "M99SimSpec": "#1f77b4", "M99NIR": "#aec7e8", "M99NN": "#ff7f0e",
        "Z17SimSpec": "#2ca02c", "Z17NIR": "#98df8a", "Z17NN": "#d62728",
        "3CSimSpec": "#9467bd", "3CNIR": "#c5b0d5", "3CNN": "#8c564b"
    }

    success = False
    time_label = "Inconnu"

    for method in methods_to_compare:
        # Reconstitution du chemin d'accès exact : /pySAS/[METHODE]/L2/pySAS006_[DATE]_[HEURE]_L2.hdf
        file_path = os.path.join(base_path, method, "L2", filename)

        wavelengths, rrs_matrix, datetag, timetag2 = extract_rrs_hyper(file_path)

        if rrs_matrix is not None:
            if row_index < rrs_matrix.shape[0]:
                success = True
                # Récupération du spectre de la ligne demandée (toutes les colonnes sauf les 2 premières)
                spectrum = rrs_matrix[row_index, :]

                # Décodage de l'heure pour la légende (Timetag2 HHMMSSms)
                t_raw = timetag2[row_index]
                h = int(t_raw // 10000000)
                m = int((t_raw % 10000000) // 100000)
                s = int((t_raw % 100000) // 1000)
                time_label = f"{h:02d}:{m:02d}:{s:02d} UTC"

                plt.plot(wavelengths, spectrum, label=method, color=colors.get(method, "#7f7f7f"), linewidth=1.5)
            else:
                print(f"⚠️ Ligne {row_index} hors plage pour la méthode {method} (Max: {rrs_matrix.shape[0] - 1})")

    if not success:
        print("❌ Aucun spectre n'a pu être extrait pour l'index demandé.")
        plt.close()
        return

    # Cosmétique du graphique
    plt.title(
        f"Comparaison spectrale $R_{{rs}}$ ($Rrs\_HYPER$)\nFichier : {filename} | Tranche : n°{row_index} ({time_label})",
        fontsize=11, fontweight='bold')
    plt.xlabel("Longueur d'onde (nm)", fontweight='bold')
    plt.ylabel("$R_{rs}$ ($sr^{-1}$)", fontweight='bold')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='-')
    plt.xlim(350, 900)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', frameon=True, facecolor='white')
    plt.tight_layout()

    # Dossier de sortie des figures
    out_dir = os.path.join(base_path, "Comparaisons_L2_Plots")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"Compare_Rrs_{date_str}_ligne_{row_index}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"✅ Graphique enregistré avec succès sous : {out_png}")


# ---------------------------------------------------------------------------
# ZONE DE CONFIGURATION ET DE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    BASE_PATH = "/Users/simonbelanger/Data/Amundsen/2026_LEG_00/L1/pySAS/"
    DATE_STR = "20260701"

    # Modifiez le nom du fichier selon la station spécifique à analyser
    FILENAME = "pySAS006_20260701_172336_L2.hdf"

    # Sélectionnez les méthodes présentes dans vos dossiers à comparer
    METHODS_TO_PLOT = ["M99NN", "M99SimSpec", "M99NIR", "3CSimSpec", "3CNIR", "3CNN"]

    # Ligne à tracer (0 = première tranche de 5 minutes, 1 = suivante...)
    LIGNE_INDEX = 0

    #plot_l2_matrix_comparison(BASE_PATH, FILENAME, DATE_STR, METHODS_TO_PLOT, row_index=LIGNE_INDEX)

    plot_all_casts_grid(BASE_PATH, FILENAME, DATE_STR, METHODS_TO_PLOT)
