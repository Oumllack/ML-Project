import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib

def load_data():
    """Charge les données prétraitées"""
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv')
    return X_train, y_train

def optimize_random_forest(X_train, y_train):
    """Optimise les hyperparamètres du Random Forest"""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    
    print("Optimisation du Random Forest...")
    grid_search.fit(X_train, y_train.values.ravel())
    
    print("Meilleurs paramètres :", grid_search.best_params_)
    print("Meilleur score :", grid_search.best_score_)
    
    return grid_search.best_estimator_

def optimize_xgboost(X_train, y_train):
    """Optimise les hyperparamètres de XGBoost"""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.2]
    }
    
    xgb = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    
    print("Optimisation de XGBoost...")
    grid_search.fit(X_train, y_train.values.ravel())
    
    print("Meilleurs paramètres :", grid_search.best_params_)
    print("Meilleur score :", grid_search.best_score_)
    
    return grid_search.best_estimator_

def optimize_lightgbm(X_train, y_train):
    """Optimise les hyperparamètres de LightGBM"""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.2]
    }
    
    lgbm = LGBMRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    
    print("Optimisation de LightGBM...")
    grid_search.fit(X_train, y_train.values.ravel())
    
    print("Meilleurs paramètres :", grid_search.best_params_)
    print("Meilleur score :", grid_search.best_score_)
    
    return grid_search.best_estimator_

def save_optimized_models(models):
    """Sauvegarde les modèles optimisés"""
    for name, model in models.items():
        joblib.dump(model, f'models/optimized_{name.lower().replace(" ", "_")}.joblib')

def main():
    # Charger les données
    X_train, y_train = load_data()
    
    # Optimiser les modèles
    optimized_models = {
        'Random Forest': optimize_random_forest(X_train, y_train),
        'XGBoost': optimize_xgboost(X_train, y_train),
        'LightGBM': optimize_lightgbm(X_train, y_train)
    }
    
    # Sauvegarder les modèles optimisés
    save_optimized_models(optimized_models)
    
    print("\nOptimisation terminée avec succès !")

if __name__ == "__main__":
    main() 