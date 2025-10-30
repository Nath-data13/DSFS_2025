
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import os
import boto3
import statsmodels.api as sm
import plotly.express as px

from dotenv import load_dotenv
from io import BytesIO

# Nettoyage du cache au démarrage
st.cache_data.clear()
st.cache_resource.clear()

# Configuration page pleine largeur 
st.set_page_config(
    page_title="Simulation Geodechets",
    page_icon="♻️",
    layout="wide"
)

# Titre de l'app
st.title("Application Geodechets")
st.write("### Simulation de production de déchets par département")

# Chargement des secrets AWS
load_dotenv("secrets.env")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

# Connexion S3
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET,
    region_name="eu-west-3"
)

# Fonction pour charger CSV depuis S3 
@st.cache_data
def load_data_from_s3(bucket_name, key):
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    df = pd.read_csv(BytesIO(obj['Body'].read()))
    return df

# Fonction pour charger Pickle depuis S3
@st.cache_resource
def load_pickle_from_s3(bucket_name, key):
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    return pickle.load(BytesIO(obj['Body'].read()))

# Chargement des données CSV 
try:
    df = load_data_from_s3("mygeodechet", "df_reduced.csv")
except Exception as e:
    st.error(f"Erreur lors du chargement des données : {e}")
    df = None

# Chargement des modèles Pickle
model_files = {
    "Déblais et gravats": "models/model_ols_Déblais_gravats.pkl",
    "Déchets verts": "models/model_ols_Déchets_verts.pkl",
    "Encombrants": "models/model_ols_Encombrants.pkl",
    "Matériaux recyclables": "models/model_ols_Matériaux_recyclables.pkl",
    "Total autres déchets": "models/model_ols_Total_autres_dechets.pkl"
}

models = {}
for label, path in model_files.items():
    try:
        models[label] = load_pickle_from_s3("mygeodechet", path)
        # st.success(f"✅ Modèle '{label}' chargé depuis S3 !")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle '{label}' : {e}")
        models[label] = None

# Sélection du département
if df is not None:
    departement = st.selectbox("Choix du département :", sorted(df["Département"].unique()))
    df_dep = df[df["Département"] == departement].copy()
    st.dataframe(df_dep.head(7), use_container_width=True)

# -------------------------------------------------------------------------------------------------
# Ajustement des paramètres sur 4 colonnes, avant simultaion, pour l'année 2021 
# selection année 2021
df_2021 = df_dep[df_dep["année"] == 2021]

# liste des variables à slider
numeric_cols = [
    'pop_globale', 'densité',
    'tranche_age_0-24', 'tranche_age_25-59', 'tranche_age_60+',
    'csp1_agriculteurs', 'csp2_artisans_commerçant_chef_entreprises',
    'csp3_cadres_professions_intellectuelles', 'csp4_professions_intermédiaires',
    'csp5_employés', 'csp6_ouvriers', 'csp7_retraités', 'csp8_sans_activité',
    'nbre_entreprises', 'nbre_entreprises_agricole', 'nbre_entreprises_industrie',
    'nbre_entreprises_service', 'nb_salaries_secteur_agricole',
    'nb_salaries_secteur_industrie', 'nb_salaries_secteur_service'
    ]
  
# titre
st.write("### Paramètres à ajuster")

col1, col2, col3, col4 = st.columns(4)
cols = [col1, col2, col3, col4]
user_inputs = {}

for i, feature in enumerate(numeric_cols):
    min_val = float(df_dep[feature].min())*0.5 # 50% de moins
    max_val = float(df_dep[feature].max()) *1.5 # 50% de plus
    mean_val = float(df_dep[feature].mean())

    target_col = cols[i % 4]   # distribue à travers les 4 colonnes
    with target_col:
            # slider avec step raisonnable
            step = max((max_val - min_val) / 50.0, 1.0)
            user_inputs[feature] = st.slider(feature, min_val, max_val, float(mean_val), step=step, key=f"sl_{feature}")

# -------------------------------------------------------------------------------------------------------
# Partie simulation 
st.markdown("---")
# st.markdown("### Lancez la simulation")
if st.button("Lancer la simulation"):
    # Préparer X_user
    X_user = pd.DataFrame([user_inputs])

    # Nettoyage / conversion : remplace virgules (si jamais), force numérique et fillna
    X_user = X_user.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(",", "."), errors="coerce"))
    X_user = X_user.fillna(0.0)

    results = []

    for label, pipeline in models.items():
        if pipeline is None:
            st.error(f"Modèle manquant pour {label}")
            results.append({"Type de déchet": label, "Prédiction (tonnes)": "Erreur"})
            continue

        try:
            preprocessor = pipeline["preprocessor"]
            ols_model = pipeline["ols_model"]

            # Ajouter toutes les colonnes attendues par le preprocessor avec 0.0 si manquantes
            for feat in preprocessor.feature_names_in_:
                if feat not in X_user.columns:
                    X_user[feat] = 0.0

            # Réordonner et forcer float uniquement pour les colonnes du preprocessor
            X_sim = X_user[preprocessor.feature_names_in_].astype(float)

            # Transformation + constante
            X_transformed = preprocessor.transform(X_sim)
            X_const = sm.add_constant(X_transformed, has_constant='add')

            # Prédiction
            pred = ols_model.predict(X_const)[0]
            results.append({"Type de déchet": label, "Prédiction (tonnes)": round(float(pred), 2)})

        except Exception as e:
            st.error(f"Erreur modèle {label}: {e}")
            results.append({"Type de déchet": label, "Prédiction (tonnes)": "Erreur"})

# -----------------------------------------------------------------------------------------------------------------
# 2éme Partie

    # Affichage résultats en deux colonnes (résultats / graph) 
    res_col, graph_col = st.columns([1, 2])
    with res_col:
        df_results = pd.DataFrame(results)
        st.write("#### Résultats Prédictions")
        st.dataframe(df_results, use_container_width=True)

    with graph_col:
        st.write("#### Analyse comparative des déchets : 2021 (réel) vs modèle prédictif")

        # Filtrer uniquement les résultats numériques
        good = df_results[df_results["Prédiction (tonnes)"].apply(lambda v: isinstance(v, (int, float)))].copy()

        if not good.empty:
            # Créer un DataFrame avec les valeurs 2021
            df_2021_vals = pd.DataFrame({
                'Type de déchet': ['Déblais et gravats', 'Déchets verts', 'Encombrants',
                                'Matériaux recyclables', 'Total autres déchets'],
                '2021': [
                    float(df_2021['Déblais_gravats'].values[0]),
                    float(df_2021['Déchets_verts'].values[0]),
                    float(df_2021['Encombrants'].values[0]),
                    float(df_2021['Matériaux_recyclables'].values[0]),
                    float(df_2021['Total_autres_dechets'].values[0])
                ]
            })
            # Fusionner avec les prédictions
            df_plot = pd.merge(good, df_2021_vals, on='Type de déchet') # Le merge sert à mettre ces données côte à côte pour le même type de déchet.
            df_plot = df_plot.melt(id_vars='Type de déchet', value_vars=['2021', 'Prédiction (tonnes)'],
                                var_name='Année', value_name='Tonnes') # attend un format “long” pour faire des barres côte à côte par catégorie.

            # Graphique barres côte à côte
            fig = px.bar(df_plot, x='Type de déchet', y='Tonnes', color='Année', barmode='group', height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun résultat numérique à afficher")   