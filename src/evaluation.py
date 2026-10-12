"""
Metriques d'evaluation d'un tarif, communes au GLM et au GBM.

Toutes les fonctions prennent des tableaux (numpy ou pandas) alignes police par
police, et jamais un modele : un tarif est evalue uniquement a travers ses
predictions. C'est ce qui garantit que GLM et GBM sont mesures par le meme code.

Notations (police i) :
    e_i      exposition en annees
    N_i      nombre de sinistres declares
    S_i      charge observee (EUR)
    pi_i     prime pure annuelle predite (EUR par annee-police)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_gamma_deviance, mean_poisson_deviance,
                             mean_tweedie_deviance)


def _np(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


# --- Indicateurs de rang ---------------------------------------------------

def lorenz(prime, expo, charge) -> tuple:
    """Courbe de Lorenz et indice de Gini.

    Polices triees par prime predite croissante. Abscisse : part cumulee
    d'exposition. Ordonnee : part cumulee de charge observee.
    G = 1 - 2 x aire sous la courbe.
    Le tri est stable : a primes egales, l'ordre des lignes est conserve, ce
    qui rend le resultat independant de l'algorithme de tri de la plateforme.
    """
    ordre = np.argsort(_np(prime), kind="stable")
    e, s = _np(expo)[ordre], _np(charge)[ordre]
    x = np.concatenate([[0.0], np.cumsum(e) / e.sum()])
    y = np.concatenate([[0.0], np.cumsum(s) / s.sum()])
    return x, y, 1 - 2 * np.trapezoid(y, x)


def groupes_risque(prime, expo, n_groupes: int = 10) -> np.ndarray:
    """Numero de groupe (1 = moins risque) par tranches d'exposition egale."""
    prime, expo = _np(prime), _np(expo)
    ordre = np.argsort(prime, kind="stable")
    # L'arrondi flottant peut porter la derniere part cumulee a 1.0000000000002 :
    # hors de l'intervalle ]0.9, 1], pd.cut renverrait NaN et la police la plus
    # chere sortirait du dernier groupe. On borne donc a 1.
    part_cum = np.minimum(np.cumsum(expo[ordre]) / expo.sum(), 1.0)
    g_trie = pd.cut(part_cum, bins=np.linspace(0, 1, n_groupes + 1),
                    labels=False, include_lowest=True) + 1
    groupes = np.empty(len(prime), dtype=int)
    groupes[ordre] = g_trie
    return groupes


def tableau_lift(prime, expo, nb, charge, n_groupes: int = 10) -> pd.DataFrame:
    """Frequence et charge observees, prime moyenne predite, par groupe de risque."""
    d = pd.DataFrame({"groupe": groupes_risque(prime, expo, n_groupes),
                      "prime": _np(prime), "expo": _np(expo),
                      "nb": _np(nb), "charge": _np(charge)})
    g = d.groupby("groupe").agg(expo=("expo", "sum"), nb=("nb", "sum"),
                                charge=("charge", "sum"), prime_moy=("prime", "mean"))
    g["frequence"] = g["nb"] / g["expo"]
    g["charge_obs"] = g["charge"] / g["expo"]
    g.index = [f"G{i}" for i in g.index]
    return g


# --- Deviances et D2 ---------------------------------------------------------
# D2 = 1 - D(modele) / D(modele nul) : part de la deviance du modele nul
# expliquee par le tarif. Le modele nul est precise pour chaque composante,
# car il conditionne la valeur du D2 (conventions de la Phase 2).

def d2_poisson(nb, expo, freq_pred) -> float:
    """D2 de la frequence. Deviance de Poisson sur les frequences N_i / e_i,
    ponderee par e_i. Modele nul : frequence constante de l'echantillon evalue."""
    nb, expo, freq_pred = _np(nb), _np(expo), _np(freq_pred)
    dev = mean_poisson_deviance(nb / expo, freq_pred, sample_weight=expo)
    nul = np.full_like(freq_pred, nb.sum() / expo.sum())
    dev_nul = mean_poisson_deviance(nb / expo, nul, sample_weight=expo)
    return 1 - dev / dev_nul


def d2_gamma(cout, poids, cout_pred) -> float:
    """D2 de la severite. Deviance Gamma ponderee par le nombre de sinistres.
    Modele nul : cout moyen pondere de l'echantillon evalue."""
    cout, poids, cout_pred = _np(cout), _np(poids), _np(cout_pred)
    dev = mean_gamma_deviance(cout, cout_pred, sample_weight=poids)
    nul = np.full_like(cout, np.average(cout, weights=poids))
    dev_nul = mean_gamma_deviance(cout, nul, sample_weight=poids)
    return 1 - dev / dev_nul


def d2_tweedie(charge, expo, prime, p: float, prime_nulle: float) -> float:
    """D2 de la prime pure. Deviance de Tweedie d'indice p sur S_i / e_i,
    ponderee par e_i. Modele nul : prime uniforme prime_nulle, calibree sur le
    TRAIN (convention de la Phase 2), donc identique pour tous les tarifs."""
    charge, expo, prime = _np(charge), _np(expo), _np(prime)
    y = charge / expo
    dev = mean_tweedie_deviance(y, prime, sample_weight=expo, power=p)
    nul = np.full_like(prime, prime_nulle)
    dev_nul = mean_tweedie_deviance(y, nul, sample_weight=expo, power=p)
    return 1 - dev / dev_nul


# --- Calibration et segmentation --------------------------------------------

def ratio_ae(charge, expo, prime) -> float:
    """A/E = charge observee / charge predite = somme S_i / somme e_i pi_i."""
    return _np(charge).sum() / (_np(prime) * _np(expo)).sum()


def amplitude(prime) -> float:
    """Ratio 99e / 1er percentile des primes predites (non pondere)."""
    p1, p99 = np.percentile(_np(prime), [1, 99])
    return p99 / p1