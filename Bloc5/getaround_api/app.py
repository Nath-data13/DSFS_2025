# from fastapi import FastAPI
# from pydantic import BaseModel
# import pandas as pd
# import joblib
# import os

# app = FastAPI()

# MODEL_PATH = os.path.join("model", "modele_xgb_getaround.pkl")
# model = joblib.load(MODEL_PATH)

# # donnée d'entrée
# class InputData(BaseModel):
#     model_key: str
#     mileage: int
#     engine_power: int
#     fuel: str
#     paint_color: str
#     car_type: str
#     private_parking_available: bool
#     has_gps: bool
#     has_air_conditioning: bool
#     automatic_car: bool
#     has_getaround_connect: bool
#     has_speed_regulator: bool
#     winter_tires: bool

# @app.post("/predict")
# def predict(data: InputData):
#     # Convertir l'entrée en DataFrame
#     df = pd.DataFrame([data.dict()])
    
#     # Faire la prédiction
#     prediction = model.predict(df)
    
#     # Retourner la prédiction sous forme JSON
#     return {"prediction": prediction.tolist()}

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(
    title="API Getaround",
    description="""

L'API GetAround estime le prix journalier de location d'un véhicule. Le modèle prédictif a été entraîné sur les données réelles de GetAround.

---
### Instructions pour remplir les champs :

| Champ | Type | Description | Valeurs possibles |
|-------|------|-------------|------------------|
| `model_key` | str | Marque du véhicule | Citroën, Peugeot, PGO, Renault, Audi, BMW, Ford, Mercedes, Opel, Porsche, Volkswagen, KIA Motors, Alfa Romeo, Ferrari, Fiat, Lamborghini, Maserati, Lexus, Honda, Mazda, Mini, Mitsubishi, Nissan, SEAT, Subaru, Suzuki, Toyota, Yamaha |
| `mileage` | int | Kilométrage du véhicule | Nombre entier |
| `engine_power` | int | Puissance du moteur (en chevaux) | Nombre entier |
| `fuel` | str | Type de carburant | diesel, petrol, hybrid_petrol, electro |
| `paint_color` | str | Couleur de la voiture | black, grey, white, red, silver, blue, orange, beige, brown, green |
| `car_type` | str | Type de véhicule | convertible, coupe, estate, hatchback, sedan, subcompact, suv, van |
| `private_parking_available` | bool | Parking privé disponible | true / false |
| `has_gps` | bool | GPS intégré | true / false |
| `has_air_conditioning` | bool | Climatisation | true / false |
| `automatic_car` | bool | Transmission automatique | true / false |
| `has_getaround_connect` | bool | Connectivité GetAround | true / false |
| `has_speed_regulator` | bool | Régulateur de vitesse | true / false |
| `winter_tires` | bool | Pneus hiver | true / false |

---

### Exemple JSON d'entrée :

```json
{
  "model_key": "Renault",
  "mileage": 50000,
  "engine_power": 120,
  "fuel": "diesel",
  "paint_color": "white",
  "car_type": "estate",
  "private_parking_available": false,
  "has_gps": true,
  "has_air_conditioning": false,
  "automatic_car": false,
  "has_getaround_connect": false,
  "has_speed_regulator": false,
  "winter_tires": true
}

""",
    version="1.0"
)

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

# Chemin vers le modèle
MODEL_PATH = os.path.join("model", "modele_xgb_getaround.pkl")

# Charger le modèle
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print(f"Attention : modèle non trouvé à {MODEL_PATH}")

# Classe de données d'entrée
class InputData(BaseModel):
    model_key: str
    mileage: int
    engine_power: int
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

# Route racine pour test
@app.get("/")
def root():
    return {"message": "API FastAPI Getaround en ligne !"}

# Route de prédiction
@app.post("/predict")
def predict(data: InputData):
    if model is None:
        return {"error": "Modèle non chargé"}
    
    # Convertir l'entrée en DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Faire la prédiction
    prediction = model.predict(df)
    
    # Retourner la prédiction sous forme JSON
    return {"prediction": prediction.tolist()}
