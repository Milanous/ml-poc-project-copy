# Assignement 1 - Cadrage du projet

## 1. Description du dataset

Le dataset utilisé est constitué de **8 384 certifications musicales** officielles publiées par le **SNEP** (Syndicat National de l'Édition Phonographique), scraped depuis [snepmusique.com/les-certifications](https://snepmusique.com/les-certifications/).

### Structure du dataset

Chaque ligne représente une certification obtenue par un titre musical en France :

| Colonne | Description |
|---|---|
| `Interprete` | Nom de l'artiste ou du groupe (peut contenir plusieurs artistes : "FEAT.", "&") |
| `Titre` | Titre du single ou de l'album certifié |
| `Editeur / Distributeur` | Label(s) et distributeur(s) associés |
| `Categorie` | Type de produit : Singles, Albums, Vidéos |
| `Certification` | Niveau de certification : Or, Platine, Diamant, Double Platine, etc. |
| `Date de sortie` | Date de publication du titre (format DD/MM/YYYY) |
| `Date de constat` | Date à laquelle la certification a été officiellement constatée |
| `Duree obtention` | Durée entre la sortie et l'obtention de la certification |

### Statistiques clés

- **8 384 certifications** au total (après déduplication)
- **Certifications** : Or (57%), Platine (20%), Diamant (12%), Double Or (5%), etc.
- **Catégories** : Singles (~47%), Albums (~45%), Vidéos (~6%)
- **~1 078 entrées** impliquant plusieurs artistes (collaborations "FEAT." ou "&")
- **Valeurs manquantes** : faibles (72 dates de sortie manquantes, 531 durées manquantes)
- **Artistes prolifiques** : JUL (98 certifications), NINHO (65), JOHNNY HALLYDAY (61), DAMSO (58), CÉLINE DION (53)

---

## 2. Méthode de collecte

Les données ont été collectées par **web scraping** du site officiel du SNEP.

### Approche technique

Le site organise les certifications en **378 pages** paginées. La stratégie de scraping adoptée tient compte de la structure cumulative du site :

- **Pages 1–189** (certifications récentes) : chaque page affiche les 30 dernières certifications ajoutées ; on extrait les 30 derniers éléments de chaque page pour éviter les doublons entre pages consécutives.
- **Pages 190–378** (certifications plus anciennes) : les items ne se chevauchent pas entre pages ; on extrait la totalité des certifications de chaque page.
- **Déduplication globale** par clé composite `(Interprete, Titre, Categorie, Certification)`.

### Outils utilisés

- `requests` pour les requêtes HTTP avec gestion des retry (3 tentatives par page)
- `BeautifulSoup` (lxml) pour le parsing HTML
- `re` (regex) pour l'extraction rapide des blocs `<div class="certification">`
- `ThreadPoolExecutor` (6 workers) pour le scraping parallèle des 378 pages
- Export en CSV encodé UTF-8 BOM pour compatibilité

Le code de scraping est disponible dans `src/scraping.py`.

---

## 3. Justification du choix du dataset

### Adéquation avec l'objectif business

Une plateforme de streaming musical a besoin de recommander des artistes à ses utilisateurs. Les certifications SNEP constituent un **proxy de popularité officiel et fiable** : elles traduisent le volume de streams, ventes et téléchargements d'un titre en France, validé par l'industrie.

### Richesse des relations entre artistes

Le champ `Interprete` contient de nombreuses **collaborations explicites** ("FEAT.", "&", featuring) qui permettent de construire un **graphe de co-artistes** : deux artistes sont connectés s'ils ont collaboré sur au moins un titre certifié. Ce graphe est directement exploitable pour un algorithme de détection de communautés.

### Qualité et accessibilité

- Données **officielles** (source industrielle reconnue), sans biais de plateforme
- Données **publiques** et librement accessibles
- Couvre une large période temporelle (certifications des années 1970 à aujourd'hui)
- Granularité suffisante pour construire un graphe dense avec des artistes francophones et internationaux

---

## 4. Objectif business et ML

### Objectif business

Construire un moteur de **recommandation d'artistes** pour une plateforme de streaming musicale. L'idée : si un utilisateur apprécie un artiste donné, lui recommander des artistes issus de la même **communauté musicale** (même genre, mêmes collaborations, même réseau de labels).

### Approche ML

**Étape 1 – Construction du graphe de co-artistes**  
- Nœuds : artistes uniques  
- Arêtes : une arête entre deux artistes s'ils apparaissent ensemble sur un titre certifié (feat, duo, etc.)  
- Poids possibles : nombre de collaborations, niveau de certification moyen  

**Étape 2 – Détection de communautés (Louvain)**  
- Application de l'**algorithme de Louvain** (maximisation de la modularité) pour identifier des clusters d'artistes partageant un réseau dense de collaborations  
- Ces communautés correspondent intuitivement à des genres ou scènes musicales (rap FR, variété, électro, etc.)  

**Étape 3 – Recommandation**  
- Étant donné un artiste cible, recommander les artistes du même cluster, triés par centralité ou nombre de certifications  

---

## 5. Métriques et objectifs d'évaluation envisagés

### Métriques de qualité du clustering

| Métrique | Description |
|---|---|
| **Modularité (Q)** | Mesure la densité des connexions intra-cluster vs inter-cluster (Louvain optimise directement cette mesure). Valeur cible : Q > 0.3 |
| **Silhouette Score** | Mesure la cohésion intra-cluster et la séparation inter-cluster. Calculé sur une représentation vectorielle des nœuds (embeddings Node2Vec ou features agrégées). Valeur cible : > 0.2 |
| **Davies-Bouldin Index** | Rapport entre dispersion intra-cluster et distance inter-cluster. Plus faible = meilleur. |
| **Conductance** | Fraction d'arêtes sortant d'un cluster (bas = cluster bien isolé) |

### Évaluation qualitative

- **Cohérence sémantique** des clusters : vérification manuelle que les artistes regroupés partagent le même genre musical
- **Couverture** : proportion d'artistes avec au moins un voisin dans le graphe

### Baseline simple

Un recommandeur naïf basé uniquement sur le label/distributeur servira de baseline pour comparer la qualité des recommandations issues des communautés Louvain.