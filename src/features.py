"""
Variables tarifaires : construction a partir des seules variables brutes.

Toutes les definitions de regroupement du projet sont centralisees ici, avec
leur origine (notebook et decision). Le notebook 03 et l'appli Streamlit
importent les memes fonctions : un assure est donc tarife avec exactement les
memes bornes partout.

Convention des tranches : pd.cut avec right=True, chaque intervalle est ferme
a droite. Exemple : les bornes [17, 23, 28] donnent ]17, 23] et ]23, 28],
soit les ages 18 a 23 puis 24 a 28.
"""
import numpy as np
import pandas as pd

# --- Age du conducteur (notebook 01, section 7.1) ---
# Courbe en "L" : surrisque jeune marque puis plateau. La bande haute absorbe
# la sentinelle a 99 ans sans imputation.
BORNES_DRIVAGE = [17, 23, 28, 45, 60, 75, 101]
LABELS_DRIVAGE = ["18-23", "24-28", "29-45", "46-60", "61-75", "76+"]

# --- Age du vehicule (notebook 01, section 7.2) ---
# Decroissance monotone. La bande "20+" absorbe les sentinelles 99-100.
BORNES_VEHAGE = [-1, 1, 5, 11, 15, 19, 101]
LABELS_VEHAGE = ["0-1", "2-5", "6-11", "12-15", "16-19", "20+"]

# --- Puissance du vehicule (notebook 02, section 2.1) ---
# Effet non monotone : categoriel. Puissances 12 a 15 fusionnees (chacune
# sous 1 % de l'exposition).
SEUIL_POWER = 12
CATS_POWER = [str(p) for p in range(4, SEUIL_POWER)] + [f"{SEUIL_POWER}+"]

# --- Bonus-malus (notebook 02, section 2.2) ---
# Plancher 50 isole, seuil 100 isole, paliers par dizaines entre les deux.
BORNES_BM = [49, 50, 59, 69, 79, 89, 99, 100, 125, np.inf]
LABELS_BM = ["50", "51-59", "60-69", "70-79", "80-89", "90-99",
             "100", "101-125", "126+"]

# --- Region (notebook 01, section 7.5) ---
# Regions d'au moins 1 % de l'exposition du portefeuille. Les autres sont
# fusionnees dans "Autres". Liste figee : la recalculer sur un autre jeu de
# donnees changerait la definition de la variable.
REGIONS_GARDEES = ["R11", "R24", "R25", "R26", "R31", "R41", "R52", "R53",
                   "R54", "R72", "R73", "R82", "R91", "R93"]


def _verifie_bornes(resultat: pd.Series, source: pd.Series, nom: str) -> None:
    """Leve une erreur si une valeur brute tombe hors des bornes prevues.

    pd.cut renvoie NaN hors bornes, sans message. Une valeur manquante
    deviendrait ensuite silencieusement la modalite de reference au scoring.
    """
    hors = source[resultat.isna() & source.notna()]
    if not hors.empty:
        raise ValueError(f"{nom} : valeurs hors bornes {sorted(hors.unique())[:10]}")


def tranches_drivage(serie: pd.Series) -> pd.Series:
    res = pd.cut(serie, bins=BORNES_DRIVAGE, labels=LABELS_DRIVAGE, right=True)
    _verifie_bornes(res, serie, "DrivAge")
    return res


def tranches_vehage(serie: pd.Series) -> pd.Series:
    res = pd.cut(serie, bins=BORNES_VEHAGE, labels=LABELS_VEHAGE, right=True)
    _verifie_bornes(res, serie, "VehAge")
    return res


def tranches_bm(serie: pd.Series) -> pd.Series:
    res = pd.cut(serie, bins=BORNES_BM, labels=LABELS_BM, right=True)
    _verifie_bornes(res, serie, "BonusMalus")
    return res


def regroupe_power(serie: pd.Series) -> pd.Series:
    """VehPower en categoriel ordonne, puissances >= 12 fusionnees en '12+'."""
    lib = serie.astype(int).astype(str).where(serie < SEUIL_POWER, f"{SEUIL_POWER}+")
    res = pd.Series(pd.Categorical(lib, categories=CATS_POWER, ordered=True),
                    index=serie.index)
    _verifie_bornes(res, serie, "VehPower")
    return res


def regroupe_region(serie: pd.Series) -> pd.Series:
    """Region d'origine, ou 'Autres' si elle pese moins de 1 % de l'exposition."""
    region = serie.astype(str)
    return region.where(region.isin(REGIONS_GARDEES), "Autres")


def nettoie_texte(serie: pd.Series) -> pd.Series:
    """Retire les apostrophes heritees du format ARFF d'OpenML ('Regular').

    Idempotente : sans effet sur une valeur deja propre.
    """
    return serie.astype(str).str.strip("'")


def prepare_variables(data: pd.DataFrame) -> pd.DataFrame:
    """Ajoute toutes les variables tarifaires a partir des variables brutes.

    Entree : DrivAge, VehAge, VehPower, BonusMalus, Density, Region,
    VehBrand, VehGas. Les colonnes deja presentes sont recalculees, ce qui
    garantit qu'elles suivent les definitions de ce module.
    """
    d = data.copy()
    d["VehGas"] = nettoie_texte(d["VehGas"])
    d["VehBrand"] = nettoie_texte(d["VehBrand"])
    d["DrivAgeBand"] = tranches_drivage(d["DrivAge"])
    d["VehAgeBand"] = tranches_vehage(d["VehAge"])
    d["VehPowerGrp"] = regroupe_power(d["VehPower"])
    d["BonusMalusBand"] = tranches_bm(d["BonusMalus"])
    d["RegionGrouped"] = regroupe_region(d["Region"])
    d["log_Density"] = np.log(d["Density"])
    return d


def echantillon_severite(data: pd.DataFrame, seuil: float) -> pd.DataFrame:
    """Polices dont le cout moyen est observable, cible ecretee au seuil.

    Cout moyen = ClaimAmount / ClaimNbSev : on divise par le nombre de
    sinistres dotes d'un montant, pas par ClaimNb (notebook 01). Le seuil
    d'ecretement est celui du train (99e percentile), passe en argument pour
    ne jamais etre recalcule sur l'echantillon evalue.
    """
    d = data[(data["ClaimNbSev"] > 0) & (data["ClaimAmount"] > 0)].copy()
    d["cout_moyen"] = d["ClaimAmount"] / d["ClaimNbSev"]
    d["cout_ecrete"] = d["cout_moyen"].clip(upper=seuil)
    return d