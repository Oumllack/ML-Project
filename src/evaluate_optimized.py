import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Charge les données de test"""
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')
    return X_test, y_test

def load_models():
    """Charge les modèles optimisés"""
    models = {
        'Random Forest': joblib.load('models/optimized_random_forest.joblib'),
        'XGBoost': joblib.load('models/optimized_xgboost.joblib'),
        'LightGBM': joblib.load('models/optimized_lightgbm.joblib')
    }
    return models

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

def plot_feature_importance(models, X_test):
    """Visualise l'importance des features pour chaque modèle"""
    plt.figure(figsize=(15, 5))
    
    for idx, (name, model) in enumerate(models.items(), 1):
        plt.subplot(1, 3, idx)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            continue
        
        feature_importance = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title(f'Importance des Features - {name}')
    
    plt.tight_layout()
    plt.savefig('data/feature_importance.png')
    plt.close()

def plot_residuals(models, X_test, y_test):
    """Visualise les résidus des prédictions"""
    plt.figure(figsize=(15, 5))
    
    for idx, (name, model) in enumerate(models.items(), 1):
        plt.subplot(1, 3, idx)
        
        y_pred = model.predict(X_test)
        residuals = y_test.values.ravel() - y_pred
        
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Prédictions')
        plt.ylabel('Résidus')
        plt.title(f'Résidus - {name}')
    
    plt.tight_layout()
    plt.savefig('data/residuals.png')
    plt.close()

def main():
    # Charger les données
    X_test, y_test = load_data()
    
    # Charger les modèles
    models = load_models()
    
    # Évaluer les modèles
    results = evaluate_models(models, X_test, y_test)
    
    # Visualiser l'importance des features
    plot_feature_importance(models, X_test)
    
    # Visualiser les résidus
    plot_residuals(models, X_test, y_test)
    
    print("\nÉvaluation des modèles optimisés terminée avec succès !")

if __name__ == "__main__":
    main() 