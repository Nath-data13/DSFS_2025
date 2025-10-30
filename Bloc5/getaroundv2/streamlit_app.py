import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

import joblib
import os

st.set_page_config(
    page_title="Tableau de bord",
    layout="wide"
)
# Chemin vers le dossier de ton script app.py
BASE_DIR = os.path.dirname(__file__)

# Chemin vers le CSV des prix
DATA_URL_PRICE = os.path.join(BASE_DIR, "data", "get_around_pricing_project.csv")

# Chemin vers le CSV des retards
DATA_URL_DELAY = os.path.join(BASE_DIR, "data", "get_around_delay_analysis.csv")

# Charger les CSVs
df_price = pd.read_csv(DATA_URL_PRICE)
df_delay = pd.read_csv(DATA_URL_DELAY)


st.title("TABLEAU DE BORD")

# logo getaround
st.image("https://upload.wikimedia.org/wikipedia/commons/8/8e/Getaround_%28Europe%29.png", width=300)

st.text("Getaround, est une plateforme de location de voitures entre particuliers")

# Paramétre des titres
def st_titre(text):
    st.markdown(f"""
    <div style="
        background-color:#1f77b4;  /* bleu */
        color:white;                /* texte blanc */
        padding:8px;
        font-weight:bold;
        font-size:20px;             /* un peu plus grand */
        border-radius:5px;
        width:100%;
    ">{text}</div>
    """, unsafe_allow_html=True)

# Informations generale delay
st_titre("Informations générales Jeu de données 'Retards'")
total_locations = df_delay.shape[0]  # Toutes les locations
ended = df_delay[df_delay['state'] == 'ended']
canceled = df_delay[df_delay['state'] == 'canceled']
ended_count = ended.shape[0]  # Locations terminées
canceled_count = canceled.shape[0] # Locations annulées

type=df_delay['checkin_type'].value_counts().reset_index()
type.columns=['Type','Count']

# Initialiser à 0
mobile_count = 0
connect_count = 0

# Parcourir le DataFrame 'type'
for i, row in type.iterrows():
    if row['Type'] == 'mobile':
        mobile_count = row['Count']
    elif row['Type'] == 'connect':
        connect_count = row['Count']

# Affichage des KPI côte à côte
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total locations", total_locations)
col2.metric("Locations terminées", ended_count)
col3.metric("Locations annulées", canceled_count)
col4.metric("Locations sur mobile", mobile_count)
col5.metric("Locations connectées", connect_count )

# Informations generale price
st_titre("Informations générales Jeu de données 'Prix'")
# Calcul des KPI pour le jeu de données Prix
voiture_plus_louee = df_price['model_key'].mode()[0]         
carburant_plus_loue = df_price['fuel'].mode()[0]         
puissance_mediane = int(df_price['engine_power'].median())         
prix_median = int(df_price['rental_price_per_day'].median())           
modele_plus_loue = df_price['car_type'].mode()[0]           

# Traduction des types si nécessaire
translation = {
    'estate': 'Break',
    'sedan': 'Berline',
    'suv': 'SUV',
    'hatchback': 'Compacte',
    'subcompact': 'Sous-compacte',
    'coupe': 'Coupé',
    'convertible': 'Cabriolet',
    'van': 'Van'
}
modele_plus_loue_fr = translation.get(modele_plus_loue, modele_plus_loue)

# Affichage des KPI côte à côte
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Voiture la plus louée", voiture_plus_louee)
col2.metric("Carburant le plus utilisé", carburant_plus_loue.capitalize())
col3.metric("Puissance médiane (ch)", puissance_mediane)
col4.metric("Prix médian (€)", prix_median)
col5.metric("Type de véhicule le plus loué", modele_plus_loue_fr)


# Analyse des retards
st_titre("Analyse des retards en minutes")

delay_real = ended[ended["delay_at_checkout_in_minutes"] > 0]

q95 = delay_real["delay_at_checkout_in_minutes"].quantile(0.95)
filtered = delay_real[delay_real["delay_at_checkout_in_minutes"] <= q95]

num_delay= delay_real.shape[0]
mean_delay = filtered["delay_at_checkout_in_minutes"].mean().round(0)
median_delay = filtered["delay_at_checkout_in_minutes"].median()
std_delay = round(filtered["delay_at_checkout_in_minutes"].std(),0)

# Affichage des KPI côte à côte
col1, col2, col3 = st.columns(3)

col1.metric("Locations avec retards", num_delay)
col2.metric("Retards moy", mean_delay)
col3.metric("Retards médian", median_delay)

# Analyse des seuils
st_titre("Analyse interactive du seuil")
seuil=[30, 60, 90, 120, 150, 180, 210, 240]
resultat=[]
nbr_car_delay_2_loc = len(ended.dropna(subset=['time_delta_with_previous_rental_in_minutes']))

for i in seuil:
    nbr_loc_bloquee = (ended['time_delta_with_previous_rental_in_minutes']<i).sum()
    pct_nbr_loc_bloquee = ((nbr_loc_bloquee / nbr_car_delay_2_loc) * 100).round(2)
    nbr_loc_ok = (ended['time_delta_with_previous_rental_in_minutes']>=i).sum()
    pct_nbr_loc_ok = ((nbr_loc_ok / nbr_car_delay_2_loc) * 100).round(2)
 
    # Retards réels couverts (seulement pour les retards > 0)
    retards_reels = ended[ended['delay_at_checkout_in_minutes'] > 0]
    pct_retards_couverts = ((retards_reels['delay_at_checkout_in_minutes'] <= i).mean() * 100).round(2)

    # Ajouter les résultats à la liste
    resultat.append({
        'Seuil (min)': i,
        'Locations bloquées': nbr_loc_bloquee,
        '% Locations bloquées': pct_nbr_loc_bloquee,
        'Locations possibles' : nbr_loc_ok,
        '% Locations possibles': pct_nbr_loc_ok,
        '% Retards couverts': pct_retards_couverts,
    })
df_resultat = pd.DataFrame(resultat)

# Calcul des seuils
nbr_car_delay_2_loc = len(ended.dropna(subset=['time_delta_with_previous_rental_in_minutes']))
seuils = [30, 60, 90, 120, 150, 180, 210, 240]
resultat = []

for s in seuils:
    nbr_loc_bloquee = (ended['time_delta_with_previous_rental_in_minutes'] < s).sum()
    nbr_loc_ok = (ended['time_delta_with_previous_rental_in_minutes'] >= s).sum()
    retards_reels = ended[ended['delay_at_checkout_in_minutes'] > 0]
    pct_retards_couverts = ((retards_reels['delay_at_checkout_in_minutes'] <= s).mean() * 100).round(2)
    pct_loc_bloquees = ((nbr_loc_bloquee / nbr_car_delay_2_loc) * 100).round(2)

    resultat.append({
        'Seuil (min)': s,
        'Locations bloquées (%)': pct_loc_bloquees,
        'Locations possibles': nbr_loc_ok,
        'Retards couverts (%)': pct_retards_couverts
    })

df_resultat = pd.DataFrame(resultat)

# Slider pour seuil
seuil = st.slider(
    "Sélectionnez le seuil en minutes :",
    min_value=int(df_resultat['Seuil (min)'].min()),
    max_value=int(df_resultat['Seuil (min)'].max()),
    value=60,
    step=30
)

# Récupérer la ligne correspondant au seuil
row = df_resultat[df_resultat['Seuil (min)'] == seuil].iloc[0]

# Calcul du nombre réel de retards couverts
num_retards = len(ended[ended["delay_at_checkout_in_minutes"] > 0])
retards_couverts_nb = int((row['Retards couverts (%)'] / 100) * num_retards)
loc_bloquees_nb = int((row['Locations bloquées (%)'] / 100) * nbr_car_delay_2_loc)

# KPI avec style
kpi_style = """
<div style="background-color:#cce6ff; padding:4px; border-radius:15px; text-align:center; width:200px;">
    <h6 style="color:black; margin:1px; font-size:12px;">{title}</h6>
    <p style="color:black; margin:1px; font-size:14px; font-weight:bold;">{value}</p>
</div>
"""
col1, col2, col3, col4, col5 = st.columns(5)
col1.markdown(kpi_style.format(title="Seuil sélectionné (min)", value=seuil), unsafe_allow_html=True)
col2.markdown(kpi_style.format(title="Retards couverts (%)", value=f"{row['Retards couverts (%)']:.2f}"), unsafe_allow_html=True)
col3.markdown(kpi_style.format(title="Retards couverts (nb)", value=retards_couverts_nb), unsafe_allow_html=True)
col4.markdown(kpi_style.format(title="Locations bloquées (%)", value=f"{row['Locations bloquées (%)']:.2f}"), unsafe_allow_html=True)
col5.markdown(kpi_style.format(title="Locations bloquées (nb)", value=loc_bloquees_nb), unsafe_allow_html=True)

# Graphique interactif
fig = px.line(
    df_resultat,
    x='Seuil (min)',
    y=['Retards couverts (%)', 'Locations bloquées (%)'],
    title='Compromis entre retards couverts et locations bloquées'
)
fig.add_vline(x=seuil, line_dash="dash", line_color="green", annotation_text=f"Seuil {seuil} min")
st.plotly_chart(fig, use_container_width=True)

# Prédiction du prix
# Chemin vers le dossier de ton script app.py
BASE_DIR = os.path.dirname(__file__)

# Chemin vers le dossier du modèle
MODEL_PATH = os.path.join(BASE_DIR, "model", "modele_xgb_getaround.pkl")

# Charger le modèle
model = joblib.load(MODEL_PATH)
st_titre("Estimation du prix de location")

# Créer 2 colonnes pour les sliders numériques
col1, col2 = st.columns(2)

with col1:
    mileage = st.slider("Kilométrage (km)", min_value=0, max_value=300_000, step=1000, value=50_000)

with col2:
    engine_power = st.slider("Puissance moteur (ch)", min_value=40, max_value=300, step=5, value=100)

# Créer 4 colonnes pour selectbox
col1, col2, col3, col4 = st.columns(4)

with col1:
    model_key = st.selectbox("Modèle", sorted(df_price["model_key"].unique()))

with col2:
    fuel = st.selectbox("Carburant", sorted(df_price["fuel"].unique()))

with col3:
    paint_color = st.selectbox("Couleur", sorted(df_price["paint_color"].unique()))

#traduction en fr
car_type_options = [translation[ct] for ct in sorted(df_price["car_type"].unique())]
# Selectbox en français
with col4:
    car_type_fr = st.selectbox("Type de voiture", car_type_options)


# Créer 4 colonnes pour la première ligne de checkbox
col1, col2, col3, col4 = st.columns(4)

with col1:
    private_parking_available = st.checkbox("Parking privé disponible")
with col2:
    has_gps = st.checkbox("GPS")
with col3:
    has_air_conditioning = st.checkbox("Climatisation")
with col4:
    automatic_car = st.checkbox("Boîte automatique")

# Créer 4 colonnes pour la deuxième ligne
col5, col6, col7, col8 = st.columns(4)

with col5:
    has_getaround_connect = st.checkbox("Getaround Connect")
with col6:
    has_speed_regulator = st.checkbox("Régulateur de vitesse")
with col7:
    winter_tires = st.checkbox("Pneus hiver")
# col8 peut rester vide si tu n'as pas de 4ème checkbox
with col8:
    st.write("")  # juste pour laisser la colonne vide

# Création du DataFrame d'entrée avec toutes les colonnes
input_data = pd.DataFrame([{
    "mileage": mileage,
    "engine_power": engine_power,
    "fuel": fuel,
    "car_type": car_type_fr,
    "model_key": model_key,
    "paint_color": paint_color,
    "private_parking_available": int(private_parking_available),
    "has_gps": int(has_gps),
    "has_air_conditioning": int(has_air_conditioning),
    "automatic_car": int(automatic_car),
    "has_getaround_connect": int(has_getaround_connect),
    "has_speed_regulator": int(has_speed_regulator),
    "winter_tires": int(winter_tires)
}])

# Prédiction 
price_pred = model.predict(input_data)[0]

# Affichage
st.write(f"### Estimation du prix par jour : {price_pred:.2f} $ ")