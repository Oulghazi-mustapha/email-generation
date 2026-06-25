"""
Script 3 : Prétraitement pour l'entraînement (VERSION OPTIMISÉE)
Objectif : Tokenization et création des séquences SANS one-hot encoding
"""

import pandas as pd
import numpy as np
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("="*60)
print("🔧 PRÉTRAITEMENT POUR L'ENTRAÎNEMENT")
print("="*60)

# PARAMÈTRES
MAX_VOCAB_SIZE = 20000
CONTEXT_LENGTH = 50
SAMPLE_SIZE = 50000  # Utiliser 50k emails

print("\n⚙️ Configuration :")
print(f"   - Taille vocabulaire : {MAX_VOCAB_SIZE:,}")
print(f"   - Longueur contexte : {CONTEXT_LENGTH}")
print(f"   - Échantillon : {SAMPLE_SIZE:,} emails")

# Déterminer les chemins
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
input_path = os.path.join(project_root, 'data', 'processed', 'emails_cleaned.csv')
output_dir = os.path.join(project_root, 'data', 'processed')

# 1. CHARGEMENT
print("\n1️⃣ Chargement des données nettoyées...")
df = pd.read_csv(input_path)
print(f"✅ {len(df):,} emails chargés")

# Échantillonnage
df_sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
print(f"📊 Échantillon sélectionné : {len(df_sample):,} emails")

# 2. TOKENIZATION
print("\n2️⃣ Tokenization...")
texts = df_sample['cleaned_full'].tolist()

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')  #Crée un "dictionnaire" de 20,000 mots les plus fréquents
tokenizer.fit_on_texts(texts)

word_index = tokenizer.word_index
vocab_size_actual = len(word_index)
vocab_size_used = min(MAX_VOCAB_SIZE, vocab_size_actual)

print(f"✅ Tokenizer créé :")
print(f"   - Vocabulaire total : {vocab_size_actual:,} mots")
print(f"   - Vocabulaire utilisé : {vocab_size_used:,} mots")

# Exemple
print(f"\n📝 Exemple de tokenization :")
sample_text = texts[0][:80]
print(f"Texte : \"{sample_text}...\"")
sample_tokens = tokenizer.texts_to_sequences([sample_text])[0]
print(f"Tokens : {sample_tokens[:15]}...")

# 3. CONVERSION EN SÉQUENCES
print("\n3️⃣ Conversion en séquences...")
sequences = tokenizer.texts_to_sequences(texts)
print(f"✅ {len(sequences):,} séquences créées")

# 4. CRÉATION DES PAIRES (X, y)
print("\n4️⃣ Création des paires d'entraînement...")

def create_training_pairs(sequences, context_length): #Crée des paires (entrée, sortie) pour l'entraînement
    """Crée les paires (contexte, mot_suivant)"""
    X = []
    y = []
    
    for seq in tqdm(sequences, desc="🎯 Création paires"):
        for i in range(1, len(seq)):
            # Contexte
            context = seq[:i]
            
            # Padding si trop court
            if len(context) < context_length:
                context = [0] * (context_length - len(context)) + context
            else:
                context = context[-context_length:]
            
            # Mot cible
            target = seq[i]
            
            X.append(context)
            y.append(target)
    
    return np.array(X), np.array(y)

X, y = create_training_pairs(sequences, CONTEXT_LENGTH)

print(f"✅ Paires créées :")
print(f"   - X shape : {X.shape}")
print(f"   - y shape : {y.shape}")
print(f"   - Total exemples : {len(X):,}")

# 5. DIVISION TRAIN/VAL/TEST (SANS ONE-HOT)
print("\n5️⃣ Division des données...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"✅ Division effectuée :")
print(f"   - Train      : {X_train.shape[0]:,} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   - Validation : {X_val.shape[0]:,} ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"   - Test       : {X_test.shape[0]:,} ({X_test.shape[0]/len(X)*100:.1f}%)")

# 6. SAUVEGARDE
print("\n6️⃣ Sauvegarde des données...")

np.save(os.path.join(output_dir, 'X_train.npy'), X_train)  #Sauvegarde en format numpy (binaire, rapide)
np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

with open(os.path.join(output_dir, 'tokenizer.pickle'), 'wb') as f:
    pickle.dump(tokenizer, f)                 #Sauvegarde l'objet tokenizer complet

params = {
    'MAX_VOCAB_SIZE': MAX_VOCAB_SIZE,
    'CONTEXT_LENGTH': CONTEXT_LENGTH,
    'vocab_size_actual': vocab_size_actual
}
with open(os.path.join(output_dir, 'params.pickle'), 'wb') as f:
    pickle.dump(params, f)

print("✅ Fichiers sauvegardés :")
print("   - X_train.npy, X_val.npy, X_test.npy")
print("   - y_train.npy, y_val.npy, y_test.npy")
print("   - tokenizer.pickle")
print("   - params.pickle")

# 7. CALCUL DE LA TAILLE MÉMOIRE
train_size_mb = (X_train.nbytes + y_train.nbytes) / 1024**2
val_size_mb = (X_val.nbytes + y_val.nbytes) / 1024**2
test_size_mb = (X_test.nbytes + y_test.nbytes) / 1024**2
total_size_mb = train_size_mb + val_size_mb + test_size_mb

# 8. RÉSUMÉ
print("\n" + "="*60)
print("📊 RÉSUMÉ DU PRÉTRAITEMENT")
print("="*60)
print(f"Emails traités      : {len(df_sample):,}")
print(f"Vocabulaire         : {vocab_size_used:,} mots")
print(f"Exemples créés      : {len(X):,}")
print(f"Taille train        : {X_train.shape[0]:,}")
print(f"Taille validation   : {X_val.shape[0]:,}")
print(f"Taille test         : {X_test.shape[0]:,}")
print(f"Contexte            : {CONTEXT_LENGTH} mots")
print(f"Taille totale       : ~{total_size_mb:.1f} MB")
print("="*60)

print("\n✅ Prétraitement terminé !")
print("\n💡 Prochaine étape : Construction et entraînement du modèle")
print("💡 Note : Le modèle utilisera 'sparse_categorical_crossentropy'")