"""
Chargement des donnees freMTPL2 depuis OpenML.

freMTPL2freq (data_id 41214) : une ligne par police,
    nombre de sinistres (ClaimNb) et exposition (Exposure).
freMTPL2sev  (data_id 41215) : une ligne par sinistre,
    montant (ClaimAmount), rattachee a la police via IDpol.

On telecharge les deux tables, on agrege les montants par police,
puis on les joint pour obtenir un portefeuille complet.
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

# Racine du projet : deux niveaux au-dessus de ce fichier (src/data/..)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_fremtpl2() -> pd.DataFrame:
    """Telecharge et assemble le portefeuille freMTPL2 complet."""
    # Table de frequence : une ligne par police
    freq = fetch_openml(data_id=41214, as_frame=True).data
    freq["IDpol"] = freq["IDpol"].astype(int)
    freq = freq.set_index("IDpol")

    # Table de severite : une ligne par sinistre.
    # On agrege en conservant DEUX informations par police :
    #  - ClaimAmount : la somme des montants des sinistres de la police
    #  - ClaimNbSev  : le NOMBRE de sinistres effectivement observes en montant
    # ClaimNbSev est indispensable comme denominateur du cout moyen. ClaimNb
    # (table freq) compte des sinistres declares dont certains n'ont aucun
    # montant dans la table sev : l'utiliser diviserait mecaniquement le cout
    # moyen par un facteur faux pour les polices multi-sinistres partielles.
    sev = fetch_openml(data_id=41215, as_frame=True).data
    sev = sev.groupby("IDpol").agg(
        ClaimAmount=("ClaimAmount", "sum"),
        ClaimNbSev=("ClaimAmount", "size"),
    )

    # Jointure a gauche : on garde toutes les polices, y compris
    # celles sans sinistre. Un montant absent = police sans sinistre = 0.
    df = freq.join(sev, how="left")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0.0)
    df["ClaimNbSev"] = df["ClaimNbSev"].fillna(0).astype(int)

    # Certaines colonnes texte arrivent entourees d'apostrophes ('B12'),
    # heritage du format ARFF d'OpenML. On les nettoie.
    for col in df.columns[[t == object for t in df.dtypes.values]]:
        df[col] = df[col].str.strip("'")

    return df.reset_index()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Telechargement depuis OpenML (peut prendre une minute)...")
    df = load_fremtpl2()

    output_path = RAW_DIR / "fremtpl2.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Portefeuille sauvegarde : {output_path}")
    print(f"Dimensions : {df.shape[0]} polices, {df.shape[1]} colonnes")
    print(f"Colonnes : {list(df.columns)}")


if __name__ == "__main__":
    main()