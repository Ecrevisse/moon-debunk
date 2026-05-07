# 🌙 Moon & Baby Sex — Debunker

Analyse statistique pour démystifier la croyance populaire selon laquelle la phase lunaire (montante ou descendante) influencerait le sexe du bébé à la conception.

**Résultat :** sur 39 ans de naissances américaines (CDC, 1969–2008), la différence de ratio garçons/filles entre lune montante et descendante est de **−0.04 pour 100**, soit du bruit statistique pur.

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

1. **Date de conception** = date de naissance − 38 semaines (durée moyenne de grossesse)
2. **Phase lunaire** calculée avec la bibliothèque [`ephem`](https://rhodesmill.org/pyephem/) à la date de conception
3. **Lune montante** = illumination croissante jour J → jour J+1
4. **Ratio** = (naissances garçons / naissances filles) × 100
5. **Intervalle de confiance** à 95% via approximation de Wilson

## Stack

- `pandas` — manipulation des données
- `ephem` — calculs astronomiques
- `matplotlib` — graphiques
- `streamlit` — interface web
