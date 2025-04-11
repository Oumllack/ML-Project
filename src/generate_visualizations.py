import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Créer le dossier images s'il n'existe pas
os.makedirs('images', exist_ok=True)

def load_data():
    """Charge les données d'entraînement et de test"""
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')
    return X_train, y_train, X_test, y_test

def plot_price_distribution(y_train):
    """Visualise la distribution des prix des maisons"""
    plt.figure(figsize=(10, 6))
    sns.histplot(y_train['MedHouseVal'], kde=True)
    plt.title('Distribution des Prix des Maisons')
    plt.xlabel('Prix (en centaines de milliers de dollars)')
    plt.ylabel('Fréquence')
    plt.savefig('images/price_distribution.png')
    plt.close()

def plot_feature_importance(model, X_train):
    """Visualise l'importance des caractéristiques"""
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Importance des Caractéristiques')
    plt.tight_layout()
    plt.savefig('images/feature_importance.png')
    plt.close()

def plot_model_comparison(models, X_test, y_test):
    """Compare les performances des modèles"""
    metrics = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics.append({
            'Model': name,
            'MAE': mean_absolute_error(y_test['MedHouseVal'], y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test['MedHouseVal'], y_pred)),
            'R2': r2_score(y_test['MedHouseVal'], y_pred)
        })
    
    metrics_df = pd.DataFrame(metrics)
    plt.figure(figsize=(12, 6))
    metrics_df.set_index('Model').plot(kind='bar')
    plt.title('Comparaison des Performances des Modèles')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/model_comparison.png')
    plt.close()

def plot_residuals(model, X_test, y_test):
    """Visualise les résidus des prédictions"""
    y_pred = model.predict(X_test)
    residuals = y_test['MedHouseVal'] - y_pred
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Analyse des Résidus')
    plt.xlabel('Prédictions')
    plt.ylabel('Résidus')
    plt.savefig('images/residuals.png')
    plt.close()

def main():
    # Charger les données
    X_train, y_train, X_test, y_test = load_data()
    
    # Charger les modèles optimisés
    models = {
        'Random Forest': joblib.load('models/optimized_random_forest.joblib'),
        'XGBoost': joblib.load('models/optimized_xgboost.joblib'),
        'LightGBM': joblib.load('models/optimized_lightgbm.joblib')
    }
    
    # Générer les visualisations
    plot_price_distribution(y_train)
    plot_feature_importance(models['LightGBM'], X_train)  # Utiliser LightGBM comme référence
    plot_model_comparison(models, X_test, y_test)
    plot_residuals(models['LightGBM'], X_test, y_test)  # Utiliser LightGBM comme référence
    
    print("Visualisations générées avec succès dans le dossier 'images'")

if __name__ == "__main__":
    main() 