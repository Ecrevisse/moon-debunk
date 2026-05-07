import pandas as pd
from scipy.stats import skewnorm
import numpy as np

# 1. Génération des jours (de 200 à 300 jours après conception)
jours = np.arange(200, 301)

# 2. Paramètres calqués sur l'étude Jukic (2013) et les données du CDC
# a = asymétrie vers la gauche (-3.5) pour la traîne des prématurés
# loc = décalage pour que le pic tombe au jour 268
# scale = étalement (écart-type d'environ 12 jours)
probabilites = skewnorm.pdf(jours, a=-3.5, loc=272, scale=12)

# 3. Ajustement pour les déclenchements médicaux (post-terme)
# Après le jour 278, la probabilité chute drastiquement dans la vraie vie
for i in range(len(jours)):
    if jours[i] > 278:
        probabilites[i] = probabilites[i] * (0.6 ** (jours[i] - 278))

# 4. Normalisation (pour que la somme de toutes les probabilités soit égale à 100%)
probabilites = probabilites / np.sum(probabilites)

# 5. Création du tableau de données (DataFrame)
df = pd.DataFrame({
    'jours_depuis_conception': jours,
    'probabilite': probabilites
})

# 6. Sauvegarde en fichier CSV pour votre projet principal
df.to_csv('daily_gestation_probabilities.csv', index=False)

print("✅ Fichier 'daily_gestation_probabilities.csv' généré avec succès !")
print(f"La somme des probabilités est de : {df['probabilite'].sum() * 100:.1f} %")
print(f"Le jour du pic est le jour {df.loc[df['probabilite'].idxmax(), 'jours_depuis_conception']} avec {df['probabilite'].max()*100:.2f}% de probabilité.")