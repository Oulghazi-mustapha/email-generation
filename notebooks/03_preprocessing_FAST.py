"""
Script 3 RAPIDE : Prétraitement optimisé pour CPU
Utilise MOINS de données pour un entraînement rapide
"""

import pandas as pd
import numpy as np
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("="*60)
print("🔧 PRÉTRAITEMENT RAPIDE (Optimisé CPU)")
print("="*60)

# PARAMÈTRES RÉDUITS
MAX_VOCAB_SIZE = 10000  # Réduit de 20k à 10k
CONTEXT_LENGTH = 30     # Réduit de 50 à 30
SAMPLE_SIZE = 5000      # Réduit de 50k à 5k !!!

print("\n⚙️ Configuration RAPIDE :")
print(f"   - Taille vocabulaire : {MAX_VOCAB_SIZE:,}")
print(f"   - Longueur contexte : {CONTEXT_LENGTH}")
print(f"   - Échantillon : {SAMPLE_SIZE:,} emails (au lieu de 50k)")
print("   💡 Entraînement estimé : 20-30 minutes au lieu de 2h+")

# Chemins
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
input_path = os.path.join(project_root, 'data', 'processed', 'emails_cleaned.csv')
output_dir = os.path.join(project_root, 'data', 'processed')

# 1. CHARGEMENT
print("\n1️⃣ Chargement des données...")
df = pd.read_csv(input_path)
print(f"✅ {len(df):,} emails disponibles")

# Échantillonnage RÉDUIT
df_sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
print(f"📊 Échantillon sélectionné : {len(df_sample):,} emails")

# 2. TOKENIZATION
print("\n2️⃣ Tokenization...")
texts = df_sample['cleaned_full'].tolist()

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)

word_index = tokenizer.word_index
vocab_size_used = min(MAX_VOCAB_SIZE, len(word_index))

print(f"✅ Vocabulaire : {vocab_size_used:,} mots")

# 3. SÉQUENCES
print("\n3️⃣ Conversion en séquences...")
sequences = tokenizer.texts_to_sequences(texts)

# 4. PAIRES D'ENTRAÎNEMENT
print("\n4️⃣ Création des paires...")

X = []
y = []

for seq in tqdm(sequences, desc="🎯 Création"):
    for i in range(1, min(len(seq), 100)):  # Limiter à 100 mots par email
        context = seq[:i]
        
        if len(context) < CONTEXT_LENGTH:
            context = [0] * (CONTEXT_LENGTH - len(context)) + context
        else:
            context = context[-CONTEXT_LENGTH:]
        
        target = seq[i]
        
        X.append(context)
        y.append(target)

X = np.array(X)
y = np.array(y)

print(f"✅ {len(X):,} exemples créés")

# 5. DIVISION
print("\n5️⃣ Division des données...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"   - Train : {X_train.shape[0]:,}")
print(f"   - Val   : {X_val.shape[0]:,}")
print(f"   - Test  : {X_test.shape[0]:,}")

# 6. SAUVEGARDE
print("\n6️⃣ Sauvegarde...")
np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

with open(os.path.join(output_dir, 'tokenizer.pickle'), 'wb') as f:
    pickle.dump(tokenizer, f)

params = {
    'MAX_VOCAB_SIZE': MAX_VOCAB_SIZE,
    'CONTEXT_LENGTH': CONTEXT_LENGTH,
    'vocab_size_actual': len(word_index)
}
with open(os.path.join(output_dir, 'params.pickle'), 'wb') as f:
    pickle.dump(params, f)

print("✅ Sauvegarde terminée !")

# RÉSUMÉ
print("\n" + "="*60)
print("📊 RÉSUMÉ")
print("="*60)
print(f"Emails utilisés : {SAMPLE_SIZE:,} (au lieu de 50k)")
print(f"Exemples créés  : {len(X):,} (au lieu de 7M)")
print(f"Vocabulaire     : {vocab_size_used:,} mots")
print(f"Contexte        : {CONTEXT_LENGTH} mots")
print(f"\n💡 Temps d'entraînement estimé : 20-30 minutes")
print("="*60)