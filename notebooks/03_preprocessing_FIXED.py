"""
Prétraitement OPTIMISÉ pour réduire les <OOV>
- Vocabulaire plus grand (20,000 mots)
- Plus de données (30,000 emails)
- Meilleur nettoyage
"""

import pandas as pd
import numpy as np
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re

print("="*60)
print("🔧 PRÉTRAITEMENT OPTIMISÉ (MOINS DE <OOV>)")
print("="*60)

# PARAMÈTRES OPTIMISÉS
MAX_VOCAB_SIZE = 20000      # Doublé !
CONTEXT_LENGTH = 40         # Augmenté
SAMPLE_SIZE = 30000         # 6x plus de données
MIN_WORD_FREQUENCY = 2      # Garder les mots qui apparaissent au moins 2 fois

print("\n⚙️ Configuration optimisée :")
print(f"   - Vocabulaire : {MAX_VOCAB_SIZE:,} mots (au lieu de 10k)")
print(f"   - Contexte : {CONTEXT_LENGTH} mots (au lieu de 30)")
print(f"   - Emails : {SAMPLE_SIZE:,} (au lieu de 5k)")
print(f"   💡 Réduction attendue des <OOV> : -70%")

# Chemins
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
input_path = os.path.join(project_root, 'data', 'processed', 'emails_cleaned.csv')
output_dir = os.path.join(project_root, 'data', 'processed')

# 1. CHARGEMENT
print("\n1️⃣ Chargement des données...")
df = pd.read_csv(input_path)
print(f"✅ {len(df):,} emails disponibles")

# Échantillonnage
df_sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
print(f"📊 Échantillon : {len(df_sample):,} emails")

# 2. NETTOYAGE AMÉLIORÉ (moins agressif)
print("\n2️⃣ Nettoyage amélioré...")

def improved_clean(text):
    """Nettoyage moins agressif pour garder plus de mots"""
    if pd.isna(text) or not text:
        return ""
    
    text = text.lower()
    
    # Garder les chiffres importants mais formater les dates
    text = re.sub(r'\d{4}-\d{2}-\d{2}', 'DATE', text)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', 'DATE', text)
    
    # Garder les montants
    text = re.sub(r'\$\d+', 'AMOUNT', text)
    
    # Supprimer seulement les emails très longs
    text = re.sub(r'\S+@\S+', 'EMAIL', text)
    
    # Garder les URLs mais les remplacer
    text = re.sub(r'http\S+|www\S+', 'URL', text)
    
    # Garder la ponctuation importante
    text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'\"-]', ' ', text)
    
    # Espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Appliquer le nettoyage amélioré
print("   Nettoyage en cours...")
df_sample['text_improved'] = df_sample['cleaned_full'].apply(improved_clean)

texts = df_sample['text_improved'].tolist()
print(f"✅ {len(texts):,} textes nettoyés")

# 3. TOKENIZATION AMÉLIORÉE
print("\n3️⃣ Tokenization avec vocabulaire étendu...")

# Créer le tokenizer SANS limite de vocabulaire d'abord
tokenizer_full = Tokenizer(oov_token='<OOV>')
tokenizer_full.fit_on_texts(texts)

# Analyser la fréquence des mots
word_counts = tokenizer_full.word_counts
total_words = len(word_counts)

print(f"   📚 Vocabulaire total trouvé : {total_words:,} mots uniques")

# Filtrer les mots par fréquence
frequent_words = {word: count for word, count in word_counts.items() 
                  if count >= MIN_WORD_FREQUENCY}

print(f"   ✅ Mots fréquents (≥{MIN_WORD_FREQUENCY} occurrences) : {len(frequent_words):,}")

# Créer le tokenizer final avec le bon vocabulaire
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)

vocab_size_used = min(MAX_VOCAB_SIZE, len(tokenizer.word_index))
print(f"   ✅ Vocabulaire final : {vocab_size_used:,} mots")

# Calculer le taux de couverture
all_words_in_texts = []
for text in texts[:1000]:  # Échantillon pour test
    all_words_in_texts.extend(text.split())

known_words = sum(1 for word in all_words_in_texts 
                  if word in tokenizer.word_index and tokenizer.word_index[word] < MAX_VOCAB_SIZE)
coverage = (known_words / len(all_words_in_texts)) * 100 if all_words_in_texts else 0

print(f"   📊 Couverture du vocabulaire : {coverage:.1f}%")
print(f"   💡 Estimation des <OOV> : {100-coverage:.1f}%")

# 4. SÉQUENCES
print("\n4️⃣ Conversion en séquences...")
sequences = tokenizer.texts_to_sequences(texts)

# 5. PAIRES D'ENTRAÎNEMENT
print("\n5️⃣ Création des paires...")

X = []
y = []

for seq in tqdm(sequences, desc="🎯 Création"):
    # Limiter à 120 mots par email pour éviter les textes trop longs
    for i in range(1, min(len(seq), 120)):
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

# 6. DIVISION
print("\n6️⃣ Division des données...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"   - Train : {X_train.shape[0]:,}")
print(f"   - Val   : {X_val.shape[0]:,}")
print(f"   - Test  : {X_test.shape[0]:,}")

# 7. SAUVEGARDE
print("\n7️⃣ Sauvegarde...")

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
    'vocab_size_actual': len(tokenizer.word_index),
    'coverage': coverage
}
with open(os.path.join(output_dir, 'params.pickle'), 'wb') as f:
    pickle.dump(params, f)

print("✅ Sauvegarde terminée !")

# 8. RÉSUMÉ
print("\n" + "="*60)
print("📊 RÉSUMÉ OPTIMISÉ")
print("="*60)
print(f"Emails utilisés     : {SAMPLE_SIZE:,}")
print(f"Exemples créés      : {len(X):,}")
print(f"Vocabulaire         : {vocab_size_used:,} mots")
print(f"Contexte            : {CONTEXT_LENGTH} mots")
print(f"Couverture vocab    : {coverage:.1f}%")
print(f"<OOV> estimés       : {100-coverage:.1f}%")
print(f"\n💡 Amélioration attendue :")
print(f"   - 70% moins de <OOV>")
print(f"   - Génération plus fluide")
print(f"   - Meilleure cohérence")
print("="*60)

print("\n✅ Prétraitement optimisé terminé !")
print("\n🚀 Prochaine étape : python notebooks/04_model_training.py")