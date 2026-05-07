# 🌙 Moon & Baby Sex — Debunker

Analyse statistique pour démystifier la croyance populaire selon laquelle la phase lunaire (montante ou descendante) influencerait le sexe du bébé à la conception.

**Résultat :** sur 39 ans de naissances américaines (CDC, 1969–2008), l'écart de ratio garçons/filles entre toutes les phases lunaires est de **0.054 pour 100**, soit du bruit statistique pur.

---

## Prérequis

- [uv](https://docs.astral.sh/uv/getting-started/installation/) installé
- Python 3.12 (téléchargé automatiquement par uv)

---

## Installation

```bash
git clone https://github.com/Ecrevisse/moon-debunk.git
cd moon-debunk
uv sync
```

---

## Utilisation

### Interface web interactive

```bash
uv run streamlit run app.py
```

Ouvre **http://localhost:8501** dans ton navigateur.

L'interface affiche :
- Ratio garçons/100 filles pour **lune montante vs descendante** (avec intervalles de confiance à 95%)
- Ratio pour les **8 phases lunaires** (nouvelle lune, croissant, quartiers, gibbeuse, pleine lune)
- Tableau détaillé + explorateur de données brutes filtrable

### Script console

```bash
uv run python moon_sex_ratio.py
```

Affiche les ratios dans le terminal et génère `result_chart.png`.

---

## Données

`births.csv` — CDC National Vital Statistics, naissances quotidiennes aux États-Unis de 1969 à 2008.

Colonnes : `year`, `month`, `day`, `gender` (`M`/`F`), `births`.

Note : les lignes avec `day = 99` sont des agrégats mensuels (sexe non ventilé par jour) et sont exclues de l'analyse.

---

## Méthodologie

1. **Distribution de gestation** générée via `generate_gestation_distribution.py` (skew-normal calée sur Jukic et al. 2013 + CDC) → `daily_gestation_probabilities.csv` (jours 200–300 depuis la conception)
2. **Phase lunaire** pré-calculée avec [`ephem`](https://rhodesmill.org/pyephem/) pour toutes les dates de conception possibles (7 400+ dates uniques)
3. **Pondération** : chaque naissance est distribuée sur les 101 jours de conception possibles, chaque fraction pondérée par la probabilité de gestation du jour correspondant
4. **Lune montante** = illumination croissante jour J → jour J+1
5. **Ratio** = (naissances pondérées garçons / naissances pondérées filles) × 100
6. **Intervalle de confiance** à 95% via approximation de Wilson

## Fichiers

| Fichier | Rôle |
|---|---|
| `births.csv` | Données CDC brutes |
| `generate_gestation_distribution.py` | Génère `daily_gestation_probabilities.csv` |
| `daily_gestation_probabilities.csv` | Distribution de gestation (j200–j300) |
| `moon_sex_ratio.py` | Script console complet |
| `app.py` | Interface Streamlit |

## Stack

- `pandas` — manipulation des données
- `ephem` — calculs astronomiques
- `scipy` — distribution skew-normal pour la gestation
- `matplotlib` — graphiques
- `streamlit` — interface web
