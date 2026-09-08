# Brief : Moteur de tarification IARD explicable et équitable

## Problème
Tarifer un risque automobile de façon actuariellement fondée, tout en
rendant la prime explicable et en auditant l'absence de discrimination.
Enjeu métier : concilier performance prédictive, interprétabilité
réglementaire (AI Act) et acceptabilité de la tarification.

## Données
Portefeuille auto français freMTPL2 (~680 000 polices).
- freMTPL2freq : fréquence des sinistres + exposition.
- freMTPL2sev  : montants des sinistres.
Source : OpenML (data_id 41214 et 41215).

## Approche
1. EDA et cadrage actuariel (fréquence x coût, exposition en offset).
2. GLM comme socle interprétable (Poisson, Gamma, Tweedie).
3. Benchmark machine learning (GBM) et comparaison rigoureuse.
4. Explicabilité (SHAP) et audit de non-discrimination.
5. Appli en ligne : saisie d'un profil, prime justifiée.

## Livrables
Repo GitHub structuré, notebooks documentés, modèles évalués,
rapport d'explicabilite et d'équité, appli Streamlit déployée.

## Critères de succès
- GLM et GBM comparés sur les mêmes métriques (Gini, lift, calibration).
- Prime décomposée et expliquée pour un profil donné.
- Appli accessible via une URL publique.