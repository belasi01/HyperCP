import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


def extract_l2_data_safe(file_path):
    """
    Extrait les longueurs d'onde, la matrice Rrs et le QWIP.
    Renvoie (None, None, None, None) si le fichier est manquant (ex: rejeté par le filtre NIR).
    """
    if not os.path.exists(file_path):
        return None, None, None, None

    try:
        with h5py.File(file_path, 'r') as h5f:
            rrs_path = "/REFLECTANCE/Rrs_HYPER"
            if rrs_path not in h5f:
                return None, None, None, None

            rrs_raw = h5f[rrs_path][...]
            colnames = rrs_raw.dtype.names

            timetag2 = rrs_raw[colnames[1]]  # Deuxième colonne = Timetag2
            wavelengths = np.array([float(w) for w in colnames[2:]])
            rrs_matrix = np.column_stack([rrs_raw[w] for w in colnames[2:]])

            qwip_path = "/DERIVED_PRODUCTS/qwip"
            if qwip_path in h5f:
                qwip_raw = h5f[qwip_path][...]
                qwip_cols = qwip_raw.dtype.names
                qwip_values = qwip_raw[qwip_cols[2]]  # Valeur numérique du score QWIP
            else:
                qwip_values = np.zeros(len(timetag2))

            return wavelengths, rrs_matrix, timetag2, qwip_values
    except Exception:
        # En cas de fichier corrompu ou illisible, on le saute gentiment
        return None, None, None, None


def process_single_file_grid(base_path, filename, date_str, methods_to_compare, output_dir):
    """
    Génère la grille de comparaison pour UN fichier HDF5 donné, en ignorant les méthodes absentes.
    """
    # 1. Détermination du nombre maximal de lignes (casts) en cherchant une méthode qui a fonctionné
    num_casts = 0
    wavelengths_ref = None

    for method in methods_to_compare:
        # Format d'arborescence : .../pySAS/[METHODE]/L2/pySAS006_..._L2.hdf
        test_path = os.path.join(base_path, method, "L2", filename)
        w, r, _, _ = extract_l2_data_safe(test_path)
        if r is not None:
            num_casts = r.shape[0]
            wavelengths_ref = w
            break

    if num_casts == 0 or wavelengths_ref is None:
        print(f"⚠️ [Skipped] Aucun produit L2 disponible pour le fichier {filename} (toutes les méthodes ont échoué).")
        return

    # 2. Configuration de la grille de graphiques (3 colonnes)
    n_cols = 3
    n_rows = int(np.ceil(num_casts / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True)
    # Gestion du cas où il n'y a qu'une seule ligne/tranche
    #axes = np.array([axes]) if num_casts == 1 else axes.flatten()
    axes = np.atleast_1d(axes).ravel()

    # Couleurs fixes pour les 6 méthodes
    colors = {
        "M99SimSpec": "#1f77b4", "M99NIR": "#aec7e8", "M99NN": "#ff7f0e",
        "Z17SimSpec": "#2ca02c", "Z17NIR": "#98df8a", "Z17NN": "#d62728",
        "3CSimSpec": "#9467bd", "3CNIR": "#c5b0d5", "3CNN": "#8c564b"
    }

    # 3. Boucle sur chaque tranche (cast) détectée
    for cast_idx in range(num_casts):
        ax = axes[cast_idx]
        time_label = "Inconnu"
        cast_lines_data = []

        for method in methods_to_compare:
            file_path = os.path.join(base_path, method, "L2", filename)
            wavelengths, rrs_matrix, timetag2, qwip_values = extract_l2_data_safe(file_path)

            # Si la méthode a échoué (fichier absent), rrs_matrix est None : on passe au suivant sans planter !
            if rrs_matrix is not None and cast_idx < rrs_matrix.shape[0]:
                spectrum = rrs_matrix[cast_idx, :]
                qwip_score = qwip_values[cast_idx]

                t_raw = timetag2[cast_idx]
                h, m, s = int(t_raw // 10000000), int((t_raw % 10000000) // 100000), int((t_raw % 100000) // 1000)
                time_label = f"{h:02d}:{m:02d}:{s:02d} UTC"

                cast_lines_data.append({
                    "method": method, "spectrum": spectrum, "qwip": qwip_score, "color": colors.get(method, "#7f7f7f")
                })

        # Tri des légendes par QWIP croissant (le meilleur en premier)
        cast_lines_data = sorted(cast_lines_data, key=lambda x: x["qwip"])

        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)

        for line in cast_lines_data:
            ax.plot(wavelengths_ref, line["spectrum"], color=line["color"], linewidth=1.3,
                    label=f"{line['method']} ({line['qwip']:.4f})")

        ax.set_title(f"Tranche #{cast_idx + 1} — {time_label}", fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlim(380, 820)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8, facecolor='white', edgecolor='none')

    # Nettoyage des sous-graphiques vides
    for i in range(num_casts, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(f"L2 Methods comparison (QWIP scores)\nFichier source : {filename}", fontsize=13,
                 fontweight='bold')
    fig.supxlabel("Longueur d'onde (nm)", fontweight='bold', fontsize=11)
    fig.supylabel("$R_{rs}$ ($sr^{-1}$)", fontweight='bold', fontsize=11)

    plt.tight_layout()
    out_png = os.path.join(output_dir, f"Grid_Matrix_{filename.replace('.hdf', '')}.png")
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"✅ Poster matriciel généré avec succès pour : {filename}")


# ---------------------------------------------------------------------------
# BOUCLE PRINCIPALE DE BATCH SUR LES 10 FICHIERS
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    BASE_PATH = "/Users/simonbelanger/Data/Amundsen/2026_LEG_00/L1/pySAS/"
    DATE_STR = "20260701"

    # 1. Vos 6 méthodes actuellement traitées
    METHODS_TREATEES = ["M99NN", "M99SimSpec",
                        "3CSimSpec", "3CNIR",
                        "Z17NN", "Z17SimSpec"]

    # Dossier centralisé pour stocker les 10 affiches de comparaison
    OUTPUT_COMP_DIR = os.path.join(BASE_PATH, "Comparaisons_Globales_L2")
    os.makedirs(OUTPUT_COMP_DIR, exist_ok=True)

    # 2. Détection dynamique des 10 fichiers L2 à partir d'une méthode de référence (ex: M99NN)
    reference_dir = os.path.join(BASE_PATH, "M99NN", "L2")
    if os.path.exists(reference_dir):
        tous_les_fichiers_l2 = [f for f in os.listdir(reference_dir) if f.endswith('.hdf') and DATE_STR in f]
        print(
            f"📋 {len(tous_les_fichiers_l2)} fichier(s) L2 détecté(s) pour la date {DATE_STR}. Lancement du batch graphique...\n")

        # Boucle sur vos 10 fichiers
        for hdf_file in sorted(tous_les_fichiers_l2):
            process_single_file_grid(BASE_PATH, hdf_file, DATE_STR, METHODS_TREATEES, OUTPUT_COMP_DIR)

        print(f"\n✨ Opération terminée ! Les 10 posters comparatifs sont disponibles dans :\n👉 {OUTPUT_COMP_DIR}")
    else:
        print(f"❌ Dossier de référence introuvable à l'adresse : {reference_dir}")

