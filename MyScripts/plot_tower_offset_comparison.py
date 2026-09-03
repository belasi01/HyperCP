"""
plot_tower_offset_comparison.py - Compare les stats Rrs-vs-COPS (biais, RMSD, R²) AVANT
et APRES la correction d'orientation de la tour pySAS (+11 deg, voir pipeline_config.env
TOWER_OFFSET / correct_L1A_files.py::process_heading_offset), sur les 5 stations pilote
ciel dégagé retraitées cette session (NUS1/NUS2/CS-MEL/CS4-5/CS4-3b).

Lit les CSV "avant" sauvegardés dans /tmp/before_tower_offset/<station>.csv (capturés
juste avant le retraitement) et les CSV "après" déjà régénérés dans
L2/<station>/pySAS/Stats_vs_COPS_<station>.csv -- produit un scatter R²/|biais| avant vs
après (référence COPS "linear" seulement), un point par (station, méthode), coloré par
famille de méthode (M99/Z17/3C, mêmes teintes que compare_pysas_cops.py) et formé par
variante (NN/NIR/SimSpec) -- au-dessus de la diagonale y=x = amélioration.

Usage:
    conda activate hypercp
    python plot_tower_offset_comparison.py
"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

BEFORE_DIR = "/tmp/before_tower_offset"
L2_ROOT = "/Users/simonbelanger/Data/Amundsen_2026/L2"
OUT_DIR = "/Users/simonbelanger/Data/Amundsen_2026/L1/pySAS/AnalysisComparison"

STATIONS = [
    "20260822_StationNUS1", "20260822_StationNUS2", "20260824_StationCS-MEL",
    "20260825_StationCS4-5", "20260825_StationCS4-3b",
]

# Palette catégorielle (dataviz skill) -- même teintes que compare_pysas_cops.py,
# réutilisées pour l'identité de famille de méthode (3 familles fixes, jamais recyclées).
_CAT_BLUE = "#2a78d6"    # M99
_CAT_ORANGE = "#eb6834"  # Z17
_CAT_AQUA = "#1baf7a"    # 3C
_INK_SECONDARY = "#52514e"
_GRIDLINE = "#e1e0d9"
_CHART_SURFACE = "#fcfcfb"

FAMILY_COLOR = {"M99": _CAT_BLUE, "Z17": _CAT_ORANGE, "3C": _CAT_AQUA}
VARIANT_MARKER = {"NN": "o", "NIR": "^", "SimSpec": "s"}


def method_family_variant(method):
    for fam in ("M99", "Z17", "3C"):
        if method.startswith(fam):
            variant = method[len(fam):]
            return fam, variant
    return method, ""


def load_stats(path):
    df = pd.read_csv(path)
    return df[df["Référence COPS"] == "linear"].copy()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for station in STATIONS:
        before_path = os.path.join(BEFORE_DIR, f"{station}.csv")
        after_path = os.path.join(L2_ROOT, station, "pySAS", f"Stats_vs_COPS_{station}.csv")
        if not (os.path.exists(before_path) and os.path.exists(after_path)):
            print(f"⚠️  Fichier manquant pour {station}, ignorée.")
            continue
        before = load_stats(before_path).set_index("Méthode")
        after = load_stats(after_path).set_index("Méthode")
        for method in after.index:
            if method not in before.index:
                continue
            fam, variant = method_family_variant(method)
            rows.append({
                "Station": station.split("_", 1)[1], "Méthode": method,
                "Famille": fam, "Variante": variant,
                "R2_avant": before.loc[method, "R²"], "R2_apres": after.loc[method, "R²"],
                "Biais_avant": abs(before.loc[method, "Biais (pySAS-COPS)"]),
                "Biais_apres": abs(after.loc[method, "Biais (pySAS-COPS)"]),
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "TowerOffset_comparison_pilote.csv")
    df.to_csv(csv_path, index=False)
    print(f"📋 Table : {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor=_CHART_SURFACE)

    ax = axes[0]
    ax.set_facecolor(_CHART_SURFACE)
    lims = [df[["R2_avant", "R2_apres"]].min().min() - 0.01, 1.0]
    ax.plot(lims, lims, color=_INK_SECONDARY, linewidth=1, linestyle="--", zorder=1)
    for fam, color in FAMILY_COLOR.items():
        sub = df[df["Famille"] == fam]
        for variant, marker in VARIANT_MARKER.items():
            s = sub[sub["Variante"] == variant]
            if s.empty:
                continue
            ax.scatter(s["R2_avant"], s["R2_apres"], color=color, marker=marker, s=70,
                       alpha=0.85, edgecolor="white", linewidth=0.5,
                       label=f"{fam}{variant}")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("R² avant correction (tower_offset=0)")
    ax.set_ylabel("R² après correction (+11°)")
    ax.set_title("R² vs COPS -- avant/après", fontweight="bold")
    ax.grid(True, linestyle="--", color=_GRIDLINE)
    ax.text(0.02, 0.98, "au-dessus de la diagonale = amélioration", transform=ax.transAxes,
            fontsize=8, color=_INK_SECONDARY, va="top")

    ax = axes[1]
    ax.set_facecolor(_CHART_SURFACE)
    lims2 = [0, df[["Biais_avant", "Biais_apres"]].max().max() * 1.1]
    ax.plot(lims2, lims2, color=_INK_SECONDARY, linewidth=1, linestyle="--", zorder=1)
    for fam, color in FAMILY_COLOR.items():
        sub = df[df["Famille"] == fam]
        for variant, marker in VARIANT_MARKER.items():
            s = sub[sub["Variante"] == variant]
            if s.empty:
                continue
            ax.scatter(s["Biais_avant"], s["Biais_apres"], color=color, marker=marker, s=70,
                       alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xlim(lims2)
    ax.set_ylim(lims2)
    ax.set_xlabel("|Biais| avant (sr⁻¹)")
    ax.set_ylabel("|Biais| après (sr⁻¹)")
    ax.set_title("|Biais pySAS-COPS| -- avant/après", fontweight="bold")
    ax.grid(True, linestyle="--", color=_GRIDLINE)
    ax.text(0.02, 0.98, "sous la diagonale = amélioration", transform=ax.transAxes,
            fontsize=8, color=_INK_SECONDARY, va="top")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Correction d'orientation de la tour pySAS (+11°) -- {df['Station'].nunique()} "
                 f"station(s) pilote, ciel dégagé", fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(OUT_DIR, "TowerOffset_comparison_pilote.png")
    fig.savefig(out_path, dpi=150, facecolor=_CHART_SURFACE)
    plt.close(fig)
    print(f"📊 Figure : {out_path}")


if __name__ == "__main__":
    main()
