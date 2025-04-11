import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="Prédiction des Prix des Maisons en Californie",
    page_icon="🏠",
    layout="wide"
)

# Chargement des données et du modèle
@st.cache_data
def load_data():
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv')
    return X_train, y_train

@st.cache_resource
def load_model():
    return joblib.load('models/optimized_lightgbm.joblib')

# Chargement des visualisations
@st.cache_resource
def load_images():
    images = {
        'distribution': Image.open('images/price_distribution.png'),
        'importance': Image.open('images/feature_importance.png'),
        'comparison': Image.open('images/model_comparison.png'),
        'residuals': Image.open('images/residuals.png')
    }
    return images

def main():
    # Chargement des données nécessaires
    X_train, y_train = load_data()
    model = load_model()
    images = load_images()
    
    # Titre et introduction
    st.title("🏠 Prédiction des Prix des Maisons en Californie")
    st.markdown("""
    Cette application utilise un modèle d'apprentissage automatique pour prédire les prix des maisons en Californie.
    Le modèle a été entraîné sur des données historiques et optimisé pour une meilleure précision.
    """)
    
    # Création de deux colonnes pour l'interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Analyse des Données")
        
        # Distribution des prix
        st.write("### Distribution des Prix des Maisons")
        st.image(images['distribution'], caption="Distribution des prix des maisons dans le jeu de données")
        
        # Importance des caractéristiques
        st.write("### Importance des Caractéristiques")
        st.image(images['importance'], caption="Impact relatif de chaque caractéristique sur le prix")
        
        # Comparaison des modèles
        st.write("### Performance des Modèles")
        st.image(images['comparison'], caption="Comparaison des différents modèles testés")
    
    with col2:
        st.subheader("🎯 Prédiction de Prix")
        
        # Création des sliders pour les caractéristiques
        features = {}
        for col in X_train.columns:
            min_val = float(X_train[col].min())
            max_val = float(X_train[col].max())
            mean_val = float(X_train[col].mean())
            features[col] = st.slider(
                f"{col}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                format="%.2f"
            )
        
        # Bouton de prédiction
        if st.button("Prédire le Prix"):
            # Préparation des données pour la prédiction
            X_pred = pd.DataFrame([features])
            
            # Prédiction
            prediction = model.predict(X_pred)[0]
            
            # Affichage du résultat
            st.success(f"Prix prédit : {prediction:.2f} x 100,000 $")
            st.write(f"(soit {prediction * 100000:.2f} $)")
        
        # Informations sur le modèle
        st.write("### 📈 Analyse des Résidus")
        st.image(images['residuals'], caption="Distribution des erreurs de prédiction")

if __name__ == "__main__":
    main() 