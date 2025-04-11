import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def load_data():
    """Charge le dataset California Housing"""
    california = fetch_california_housing()
    X = pd.DataFrame(california.data, columns=california.feature_names)
    y = pd.Series(california.target, name='MedHouseVal')
    return X, y

def basic_analysis(X, y):
    """Effectue une analyse basique des données"""
    print("\n=== Informations sur le dataset ===")
    print(f"Nombre d'observations : {X.shape[0]}")
    print(f"Nombre de features : {X.shape[1]}")
    
    print("\n=== Statistiques descriptives ===")
    print(X.describe())
    
    print("\n=== Vérification des valeurs manquantes ===")
    print(X.isnull().sum())
    
    print("\n=== Vérification des doublons ===")
    print(f"Nombre de doublons : {X.duplicated().sum()}")

def plot_correlations(X, y):
    """Visualise les corrélations entre les variables"""
    # Ajouter la variable cible au DataFrame
    df = X.copy()
    df['MedHouseVal'] = y
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Matrice de corrélation')
    plt.tight_layout()
    plt.savefig('data/correlation_matrix.png')
    plt.close()

def plot_distributions(X, y):
    """Visualise les distributions des variables"""
    # Distribution de la variable cible
    plt.figure(figsize=(10, 6))
    sns.histplot(y, kde=True)
    plt.title('Distribution des prix des logements')
    plt.xlabel('Prix médian des logements')
    plt.savefig('data/target_distribution.png')
    plt.close()
    
    # Distributions des features
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(X.columns):
        sns.histplot(X[col], kde=True, ax=axes[idx])
        axes[idx].set_title(f'Distribution de {col}')
    
    plt.tight_layout()
    plt.savefig('data/features_distributions.png')
    plt.close()

def main():
    # Charger les données
    X, y = load_data()
    
    # Effectuer l'analyse basique
    basic_analysis(X, y)
    
    # Visualiser les corrélations
    plot_correlations(X, y)
    
    # Visualiser les distributions
    plot_distributions(X, y)
    
    # Sauvegarder les données
    X.to_csv('data/features.csv', index=False)
    y.to_csv('data/target.csv', index=False)

if __name__ == "__main__":
    main() 