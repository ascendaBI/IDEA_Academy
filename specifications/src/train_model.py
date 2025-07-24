

import pandas as pd
from pycaret.classification import *

# Charger les données
df = pd.read_excel("data/bronze/GRH-DATA.xlsx")

# Initialiser PyCaret (clean, simple, reproductible)
clf = setup(
    data=df,
    target='Attrition',
    session_id=123,
    normalize=True,
    use_gpu=False,
    fold_shuffle=True,
    verbose=False  # réduit l'affichage mais ne bloque pas
)

# Comparer les modèles et sélectionner le meilleur
best_model = compare_models()

# Sauvegarder dans le dossier models/
save_model(best_model, "models/model_attrition_fixed")

print("✅ Nouveau modèle entraîné et sauvegardé : model_attrition_fixed.pkl")
