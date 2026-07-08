# Phase 0 : notes techniques

Memo des gestes de la Phase 0 (cadrage et socle technique), avec le
raisonnement derriere chaque etape. Sert d'aide-memoire et de trace.

## 1. Initialiser le depot

`git init` cree le sous-dossier cache `.git` qui fait du dossier un depot
versionne. Git suit alors trois zones : les fichiers sur le disque (working
directory), une zone d'attente (staging), et l'historique des commits. Un
fichier voyage de gauche a droite via `git add` puis `git commit`.

## 2. .gitignore

Liste les fichiers que Git ignore completement. Choix retenus :
- `.venv/` : environnement lourd et propre a la machine, reconstruit via
  requirements. On versionne la recette, pas le resultat.
- `__pycache__/`, `*.pyc` : fichiers compiles Python, regenerables.
- `data/raw/`, `data/processed/` : donnees lourdes et regenerables via le
  script de chargement. Principe cle de reproductibilite.
- `.vscode/` : configuration d'editeur personnelle.

## 3. Structure du projet

`mkdir -p` cree les dossiers (le `-p` cree les parents manquants et ne
rale pas si le dossier existe). Chaque dossier a un role unique : data
(donnees), notebooks (exploration racontee), src (code reutilisable), app
(appli), docs (documentation), tests. Regle : le notebook explore, le src
code pour de bon.

## 4. Packages Python et __init__.py

Un `__init__.py`, meme vide, signale a Python qu'un dossier est un package
importable. Sans lui, `python -m src.data.load_data` echoue avec
`No module named src.data`. Le `-m` execute un module en le cherchant dans
les packages ; les points remplacent les slashs (src.data.load_data =
src/data/load_data.py). A lancer depuis la racine du projet.

## 5. Environnement virtuel

Un venv est un Python isole propre au projet, avec ses propres paquets.
Evite les conflits de versions entre projets.
- `conda deactivate` : sortir de Conda base pour ne pas melanger deux
  gestionnaires concurrents.
- `python3 -m venv .venv` : creer l'environnement.
- `source .venv/bin/activate` : l'activer (le prompt affiche (.venv)).
- `which python` : verifier quel Python est reellement utilise. Reflexe
  a avoir avant tout pip install.

## 6. Dependances

- `pip install -r requirements.txt` installe les paquets listes. Sur un
  fichier vide : ni installation ni erreur, juste du silence. D'ou le
  reflexe `pip list` pour verifier l'etat reel.
- `requirements.txt` : ce que je veux (lisible, sans versions).
- `requirements.lock.txt` (via `pip freeze`) : versions exactes installees,
  pour reconstruire un environnement identique. Distinction intention vs
  etat reel.

## 7. Identite Git

Chaque commit porte nom et email de l'auteur. Par defaut Git devine une
fausse adresse a partir du nom de machine, ce qui detache les commits du
compte GitHub.
- `git config --global user.name / user.email` : fixer l'identite pour
  tous les projets. Email @users.noreply.github.com : relie les commits au
  compte sans exposer l'adresse perso.
- `git commit --amend --reset-author --no-edit` : reecrire l'auteur du
  dernier commit. Sans danger tant que le commit n'est pas pousse ; risque
  de conflits une fois partage.

## 8. Chargement des donnees

- `fetch_openml(data_id=...)` telecharge depuis OpenML (41214 frequence,
  41215 severite).
- `groupby("IDpol").sum()` agrege les montants par police (cout total).
- `join(..., how="left")` : jointure a gauche depuis la table de frequence
  pour conserver toutes les polices, y compris sans sinistre. `fillna(0)`
  met ces montants a zero. Les zeros sont l'information centrale de la
  frequence : les perdre biaiserait la tarification.
- Sauvegarde en Parquet : colonnaire, type, rapide, conserve les types.

## 9. Cycle Git complet

- `git add .` : mettre en attente les fichiers modifies (le .gitignore
  filtre).
- `git commit -m "..."` : graver la photo dans l'historique.
- `git remote add origin <url>` : declarer le depot distant (nom origin).
- `git branch -M main` : renommer la branche courante en main.
- `git push -u origin main` : envoyer les commits ; le -u etablit le
  suivi, un simple `git push` suffit ensuite.

## Fil rouge

Toute la Phase 0 sert la reproductibilite et la tracabilite : venv et lock
pour l'environnement, .gitignore et script de chargement pour la donnee,
Git pour tracer chaque etape. Le socle qui distingue un projet defendable
d'un notebook jetable.
