# Assignement 1 - Cadrage du projet

## 1. Description du sujet

Le projet porte sur la prediction de la consommation electrique a partir des donnees publiees par RTE.
L'objectif est de construire un modele capable d'estimer la consommation future (a l'heure, a la demi-heure ou au jour) en s'appuyant sur l'historique de consommation et des variables explicatives pertinentes.

Ce projet s'inscrit dans un contexte operationnel fort:
- anticipation de la demande,
- aide au pilotage du reseau,
- meilleure planification des ressources energetiques.

## 2. Premiere idee de problematique

Problematique proposee:

Comment predire au mieux la consommation electrique en France a court terme a partir des donnees RTE, en tenant compte des effets calendrier, meteo et des tendances historiques?

Premiere hypothese de travail:
- la consommation depend fortement des saisons,
- elle varie selon les jours (semaine/week-end, jours feries),
- la temperature et d'autres variables meteo influencent fortement la demande.

Formulation ML (premiere version):
- type de tache: regression,
- variable cible: consommation electrique a t + h,
- horizon possible: 1h, 24h, voire plusieurs horizons compares.

## 3. A quoi ressemblerait un dataset ideal

Un dataset ideal devrait etre:
- temporel, regulier, propre et historise sur plusieurs annees,
- aligne sur le meme pas de temps pour toutes les sources,
- enrichi par des variables exogenes explicatives.

### Structure minimale attendue

Chaque ligne represente un timestamp (ex: toutes les 30 minutes ou toutes les heures), avec:
- `timestamp`
- `consommation_mw` (cible)
- variables calendrier: heure, jour_semaine, mois, week_end, jour_ferie, vacances_scolaires
- variables meteo: temperature, humidite, nebulosite, vent (national ou par zone agregee)
- variables contextuelles eventuelles: prix de l'electricite, production ENR, indicateurs economiques simples

### Qualite et couverture attendues

- faible taux de valeurs manquantes,
- gestion claire des changements d'heure (heure d'ete / heure d'hiver),
- donnees coherentes et sans doublons,
- historique suffisamment long (idealement >= 3 a 5 ans),
- decoupage train/validation/test respectant l'ordre temporel.

### Features derivees utiles

- lags de consommation (t-1, t-24, t-48, t-168),
- moyennes glissantes (24h, 7j),
- indicateurs de tendance et saisonnalite,
- interactions meteo x calendrier.

## Conclusion (version initiale)

Le coeur du projet est une prediction de consommation electrique a court terme avec une approche de regression supervisee.
La prochaine etape sera de confirmer la granularite temporelle, selectionner les sources exactes RTE/meteo et preparer un premier jeu de donnees propre pour l'entrainement.