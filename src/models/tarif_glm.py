"""
Scoring du tarif GLM a partir de son export JSON (models/tarif_glm_phase2.json).

Le tarif est recalcule a partir des coefficients, sans statsmodels : pas de
dependance a la version de la librairie, et le meme code sert au notebook et
a l'appli.

Pour une composante (frequence ou severite) a lien log, le predicteur lineaire
d'une police vaut

    eta = intercept + somme des coefficients de ses modalites
                    + coefficient de log_Density x log_Density

une modalite de reference ayant un coefficient nul. La prediction vaut exp(eta).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Nom produit par patsy : C(Var, Treatment(reference='ref'))[T.modalite]
_MOTIF = re.compile(r"^C\((\w+), Treatment\(reference='([^']*)'\)\)\[T\.(.+)\]$")


def _decode_coefficients(coefs: dict) -> dict:
    """Range les coefficients par nature : intercept, continus, categoriels."""
    structure = {"intercept": 0.0, "continus": {}, "categoriels": {}}
    for nom, valeur in coefs.items():
        m = _MOTIF.match(nom)
        if nom == "Intercept":
            structure["intercept"] = valeur
        elif m:
            var, ref, modalite = m.groups()
            bloc = structure["categoriels"].setdefault(var, {"reference": ref, "coefs": {}})
            bloc["coefs"][modalite] = valeur
        else:
            structure["continus"][nom] = valeur
    return structure


def charge_tarif(chemin: Path) -> dict:
    """Lit le JSON du tarif et decode les deux composantes."""
    brut = json.loads(Path(chemin).read_text(encoding="utf-8"))
    return {
        "brut": brut,
        "constantes": brut["constantes"],
        "frequence": _decode_coefficients(brut["frequence"]["coefficients"]),
        "severite": _decode_coefficients(brut["severite"]["coefficients"]),
    }


def predicteur_lineaire(composante: dict, data: pd.DataFrame) -> np.ndarray:
    """eta = intercept + effets categoriels + effets continus, police par police."""
    eta = np.full(len(data), composante["intercept"])
    for var, bloc in composante["categoriels"].items():
        valeurs = data[var].astype(str)
        connues = set(bloc["coefs"]) | {bloc["reference"]}
        inconnues = set(valeurs.unique()) - connues
        # Garde-fou : une modalite inconnue recevrait sinon un effet nul,
        # c'est-a-dire le tarif de la reference, sans aucun message.
        if inconnues:
            raise ValueError(f"{var} : modalites absentes du tarif {sorted(inconnues)}")
        eta += valeurs.map(bloc["coefs"]).fillna(0.0).to_numpy()
    for var, coef in composante["continus"].items():
        eta += coef * data[var].to_numpy()
    return eta


def frequence_annuelle(tarif: dict, data: pd.DataFrame) -> np.ndarray:
    """lambda_i : nombre attendu de sinistres pour une annee d'exposition."""
    return np.exp(predicteur_lineaire(tarif["frequence"], data))


def cout_moyen_ecrete(tarif: dict, data: pd.DataFrame) -> np.ndarray:
    """mu_i : cout moyen attendu d'un sinistre indemnise, ecrete au seuil."""
    return np.exp(predicteur_lineaire(tarif["severite"], data))


def prime_pure(tarif: dict, data: pd.DataFrame) -> np.ndarray:
    """PP_i = k x lambda_i x tau_obs x mu_i x (1 + tau_graves), en EUR par an."""
    c = tarif["constantes"]
    return (c["facteur_recalage"]
            * frequence_annuelle(tarif, data)
            * c["tau_obs"]
            * cout_moyen_ecrete(tarif, data)
            * c["facteur_rechargement"])