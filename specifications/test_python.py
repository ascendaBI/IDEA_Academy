import pandas as pd
from pycaret.classification import load_model, predict_model

# Charger les données
df = pd.read_excel("data/GRH-DATA.xlsx")

# Charger le modèle
model = load_model("models/model_attrition_fixed")


# Appliquer la prédiction
result = predict_model(model, data=df)

# 🔍 Afficher les colonnes disponibles
print("\n📌 Colonnes disponibles après prédiction :")
print(result.columns.tolist())

# 🔍 Afficher un aperçu
print("\n🧪 Aperçu du résultat :")
print(result.head())

# 🔁 Si la colonne prediction_score existe, l'afficher
if 'prediction_score' in result.columns:
    print("\n✅ prediction_score détecté :")
    print(result[['prediction_label', 'prediction_score']].head())
else:
    print("\n❌ La colonne 'prediction_score' est absente !")
