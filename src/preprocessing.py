import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_preprocessed_data():
    """Charge les données prétraitées"""
    X = pd.read_csv('data/features.csv')
    y = pd.read_csv('data/target.csv')
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Divise les données en ensembles d'entraînement et de test"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """Standardise les features"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir en DataFrame pour conserver les noms des colonnes
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Sauvegarder le scaler
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return X_train_scaled, X_test_scaled, scaler

def save_preprocessed_data(X_train, X_test, y_train, y_test):
    """Sauvegarde les données prétraitées"""
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)

def main():
    # Charger les données
    X, y = load_preprocessed_data()
    
    # Diviser les données
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Standardiser les features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Sauvegarder les données prétraitées
    save_preprocessed_data(X_train_scaled, X_test_scaled, y_train, y_test)
    
    print("Prétraitement terminé avec succès !")
    print(f"Taille de l'ensemble d'entraînement : {X_train_scaled.shape}")
    print(f"Taille de l'ensemble de test : {X_test_scaled.shape}")

if __name__ == "__main__":
    main() 