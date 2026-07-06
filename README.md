# Moteur de tarification IARD explicable et equitable

Projet portfolio : tarification automobile (donnees freMTPL2) fondee sur
un GLM actuariel, benchmarkee contre du machine learning (GBM), rendue
explicable (SHAP) et auditee sous l'angle de la non-discrimination (AI Act),
puis deployee dans une appli en ligne.

## Donnees

Portefeuille auto francais freMTPL2 (~678 000 polices), charge depuis OpenML.
- freMTPL2freq (data_id 41214) : frequence des sinistres et exposition.
- freMTPL2sev  (data_id 41215) : montants des sinistres.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chargement des donnees

```bash
python -m src.data.load_data
```

Le portefeuille assemble est ecrit dans data/raw/fremtpl2.parquet.

## Structure du projet

- data/        donnees brutes (raw) et nettoyees (processed)
- notebooks/   exploration et analyses (EDA, diagnostics)
- src/         code reutilisable (chargement, modeles, visualisations)
- app/         appli Streamlit
- docs/        brief et etude de cas
- tests/       tests pytest

## Feuille de route

Projet mene en 6 phases : cadrage, EDA, GLM, benchmark ML,
explicabilite et equite, deploiement, valorisation.
