# Projet d'apprentissage automatique - DEFT Classification Multiclasses

Ce projet s'inscrit dans le cadre du cours d’apprentissage automatique et la tâche correspond à la tâche 3 de l’édition 2009 du DÉfi Fouille de Texte (DEFT) (Grouin et al.): Apprentissage de classification par parti politique d’interventions au parlement européen.


## Description

**L’objectif du projet :**
Proposer différents classifieurs, les évaluer et comparer les résultats avec les informations présentes dans les actes. 

De la même manière que la seule équipe ayant envoyé un fichier de résultats pour cette tâche (Forest, 2009), nous avons décidé de nous concentrer uniquement sur les interventions du français. Néanmoins, puisque nos modèles ne contiennent pas d’informations linguistiques, les classifieurs présentés sont facilement applicables aux autres langues (italien et anglais).

## Getting Started

### Préparation

* Récupérez et téléchargez le dossier du projet via ce lien (demandez les autorisations)
```
https://github.com/EstelleSalmon/DEFT_2009_classifier.git
```

* Afin de pouvoir exécuter le script, vous avez besoin de:
 
* 1- Créer un environnement virtuel nommé .venv
```
python3 -m virtualenv .venv
source .venv/bin/activate
```
* 2- Installer également les dépendances
```
pip install -U -r requirements.txt
```

* Vous êtes prêt(e) à exécuter

### Le programme

* Le programme prend un entrée les fichiers suivant:
deft09_parlement_appr_fr.xml
deft09_parlement_ref_fr.txt
deft09_parlement_test_fr.xml

* Exécutez le script avec la commande ci-dessous:
```
python3 src/deft_2009_classifier.py
```
* Ce script produit les tableaux de performance de tous les modèles proposés pour la tâche ainsi que leur matrices de confusion et le score de leur validation croisée.

## Auteurs

Anna COLLI
Lingyun GAO 
Estelle SALMON


## License

This project is licensed under the [M2 TAL] License 


