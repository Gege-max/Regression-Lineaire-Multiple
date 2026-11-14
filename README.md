#Prédiction de la durée des livraisons

#Présentation

Ce projet consiste à développer un modèle de régression linéaire multivariée capable de prédire la durée d'une livraison en minutes à partir de plusieurs caractéristiques.

L'objectif est de mettre en pratique les différentes étapes d'un projet de Machine Learning, de l'exploration des données jusqu'à la prédiction sur de nouvelles observations.

#Objectif

Prédire la colonne `duree_reelle_min` à partir des informations suivantes :

- Distance de livraison
- Nombre d'arrêts
- Poids transporté
- Heure de départ
- Jour de la semaine
- Conditions météorologiques
- Type de véhicule
- Indice de trafic

#Structure du projet

```text
Regression_lineaire/
│
├── README.md
├── requirements.txt
│
├── notebook/
│   └── regression_lineaire.ipynb
│
├── data/
│   ├── livraisons_train_propre.csv
│   └── livraisons_a_predire_500_propre.csv
│
└── outputs/
    └── predictions_livraisons_500.csv
```

#Étapes réalisées

1. Exploration et visualisation des données (EDA).
2. Identification et traitement des valeurs invalides.
3. Nettoyage des données.
4. Conversion et tri des dates.
5. Séparation des données en ensembles d'entraînement, de validation et de test.
6. Standardisation des variables numériques.
7. Encodage des variables catégorielles avec OneHotEncoder.
8. Implémentation manuelle de la régression linéaire avec NumPy.
9. Entraînement du modèle avec la descente de gradient.
10. Évaluation des performances.
11. Analyse des résidus et des erreurs sur les livraisons longues.
12. Expérimentation de nouvelles caractéristiques (Feature Engineering).
13. Prédiction sur 500 nouvelles livraisons.
14. Exportation des résultats au format CSV.

#Modèle

J'ai implémenté la régression linéaire multivariée avec NumPy, sans utiliser le modèle LinearRegression de Scikit-learn.

La prédiction repose sur :

\[
\hat{Y}=XW+b
\]

Les paramètres sont appris à l'aide de la descente de gradient.

Scikit-learn est utilisé pour la standardisation et l'encodage des variables catégorielles.

#Résultats

Performances obtenues sur l'ensemble de test :

| Métrique | Résultat |
|---|---:|
| MAE | 3.27 minutes |
| RMSE | 5.63 minutes |
| R² | 0.855 |

Le modèle obtient de bons résultats globaux. Cependant, l'analyse des erreurs montre qu'il sous-estime certaines livraisons longues.

Pour les 37 livraisons de test dont la durée réelle dépasse 77 minutes :

- MAE : 14.76 minutes.
- Erreur moyenne signée : -13.12 minutes.

Cette faiblesse montre que les performances globales ne suffisent pas à évaluer correctement un modèle.

#Prédictions sur de nouvelles données

J'ai également utilisé le modèle pour effectuer des prédictions sur un fichier contenant 500 nouvelles livraisons.

Le fichier ne contient pas les durées réelles. Il s'agit donc de prédictions sans évaluation possible à ce stade.

Résultats :

| Indicateur | Valeur |
|---|---:|
| Nombre de prédictions | 500 |
| Durée minimale prédite | 13.97 min |
| Durée maximale prédite | 98.23 min |
| Durée moyenne prédite | 41.74 min |

Les résultats sont disponibles dans :

`outputs/predictions_livraisons_500.csv`

#Technologies utilisées

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

#Installation

Cloner le dépôt :

```bash
git clone URL_DU_DEPOT
cd Regression_lineaire
```

Créer un environnement virtuel :

```bash
python -m venv .venv
source .venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer Jupyter :

```bash
jupyter notebook
```

Ouvrir le notebook dans le dossier `notebook/` et exécuter les cellules dans l'ordre.

#Limites

Le modèle présente encore des difficultés sur les livraisons longues. Les prochaines expérimentations pourront porter sur des modèles non linéaires, l'amélioration des variables et l'analyse des erreurs.

#Auteur

Projet personnel réalisé dans le cadre de mon apprentissage du Machine Learning Engineering.
