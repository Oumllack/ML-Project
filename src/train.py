import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_data():
    """Charge les données prétraitées"""
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv')
    y_test = pd.read_csv('data/y_test.csv')
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """Entraîne différents modèles de régression"""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Entraînement du modèle {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Évalue les performances des modèles"""
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        print(f"\n=== {name} ===")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")
    
    return results

def plot_predictions(models, X_test, y_test):
    """Visualise les prédictions vs valeurs réelles"""
    plt.figure(figsize=(15, 10))
    
    for idx, (name, model) in enumerate(models.items(), 1):
        y_pred = model.predict(X_test)
        
        plt.subplot(2, 2, idx)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Valeurs réelles')
        plt.ylabel('Prédictions')
        plt.title(f'{name}')
    
    plt.tight_layout()
    plt.savefig('data/predictions_vs_actual.png')
    plt.close()

def save_models(models):
    """Sauvegarde les modèles entraînés"""
    for name, model in models.items():
        joblib.dump(model, f'models/{name.lower().replace(" ", "_")}.joblib')

def main():
    # Charger les données
    X_train, X_test, y_train, y_test = load_data()
    
    # Entraîner les modèles
    models = train_models(X_train, y_train)
    
    # Évaluer les modèles
    results = evaluate_models(models, X_test, y_test)
    
    # Visualiser les prédictions
    plot_predictions(models, X_test, y_test)
    
    # Sauvegarder les modèles
    save_models(models)
    
    print("\nEntraînement et évaluation terminés avec succès !")

if __name__ == "__main__":
    main() 