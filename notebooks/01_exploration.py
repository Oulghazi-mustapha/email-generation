"""
Script 1 : Exploration du dataset Enron
Objectif : Comprendre la structure et le contenu des données
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

print("="*60)
print("📊 EXPLORATION DU DATASET ENRON")
print("="*60)

# Déterminer le chemin du fichier automatiquement
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, 'data', 'raw', 'emails.csv')

print(f"\n🔍 Chemin du fichier : {data_path}")
print(f"📂 Le fichier existe : {os.path.exists(data_path)}")

# 1. CHARGEMENT DES DONNÉES
print("\n1️⃣ Chargement des données...")
try:
    df = pd.read_csv(data_path)    #Charge le fichier CSV dans un DataFrame 'tableu'
    print(f"✅ Dataset chargé : {len(df):,} emails")
except FileNotFoundError:
    print(f"❌ ERREUR : Le fichier n'existe pas à : {data_path}")
    print("\n📍 Vérifiez que emails.csv est dans : data/raw/")
    exit(1)

# 2. STRUCTURE DES DONNÉES
print("\n2️⃣ Structure des données :")
print(f"   - Colonnes : {df.columns.tolist()}")
print(f"   - Taille mémoire : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 3. APERÇU DES DONNÉES
print("\n3️⃣ Aperçu des 3 premiers emails :")
print(df.head(3))

# 4. VALEURS MANQUANTES
print("\n4️⃣ Valeurs manquantes :")
print(df.isnull().sum())

# 5. STATISTIQUES SUR LA LONGUEUR
print("\n5️⃣ Analyse de la longueur des messages :")
df['message_length'] = df['message'].str.len()    #Crée une nouvelle colonne avec la longueur de chaque email
print(df['message_length'].describe())

# 6. VISUALISATION
print("\n6️⃣ Création des graphiques...")

# Créer le dossier results/figures s'il n'existe pas
figures_dir = os.path.join(project_root, 'results', 'figures')
os.makedirs(figures_dir, exist_ok=True)

plt.figure(figsize=(14, 5))

# Histogramme
plt.subplot(1, 3, 1)
plt.hist(df['message_length'], bins=100, edgecolor='black', color='skyblue', alpha=0.7)
plt.xlabel('Longueur (caractères)', fontsize=10)
plt.ylabel('Nombre d\'emails', fontsize=10)
plt.title('Distribution des longueurs', fontsize=12, fontweight='bold')
plt.xlim(0, 10000)
plt.grid(axis='y', alpha=0.3)

# Boxplot
plt.subplot(1, 3, 2)
plt.boxplot(df['message_length'], vert=True)
plt.ylabel('Longueur (caractères)', fontsize=10)
plt.title('Répartition des longueurs', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# Répartition par quantiles
plt.subplot(1, 3, 3)
quantiles = df['message_length'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
plt.barh(range(len(quantiles)), quantiles.values, color='coral', edgecolor='black')
plt.yticks(range(len(quantiles)), [f'{int(q*100)}%' for q in quantiles.index])
plt.xlabel('Longueur (caractères)', fontsize=10)
plt.title('Quantiles de longueur', fontsize=12, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()

# Sauvegarder
output_path = os.path.join(figures_dir, '01_data_distribution.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Graphique sauvegardé : {output_path}")
plt.close()

# 7. EXEMPLE D'EMAIL
print("\n7️⃣ Exemple d'email (email #100) :")
print("="*60)
print(df['message'].iloc[100])
print("="*60)

# 8. STATISTIQUES FINALES
print("\n8️⃣ Statistiques résumées :")
print(f"   - Nombre total d'emails : {len(df):,}")
print(f"   - Longueur moyenne : {df['message_length'].mean():.0f} caractères")
print(f"   - Longueur médiane : {df['message_length'].median():.0f} caractères")
print(f"   - Email le plus court : {df['message_length'].min()} caractères")
print(f"   - Email le plus long : {df['message_length'].max():,} caractères")

print("\n✅ Exploration terminée !")
print("="*60)