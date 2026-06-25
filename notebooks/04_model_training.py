"""
Script 4 : Construction et entraînement du modèle LSTM
Objectif : Créer un modèle de génération d'emails
"""

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

print("="*60)
print("🏗️ CONSTRUCTION ET ENTRAÎNEMENT DU MODÈLE")
print("="*60)

# Déterminer les chemins
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, 'data', 'processed')
models_dir = os.path.join(project_root, 'models', 'final')
figures_dir = os.path.join(project_root, 'results', 'figures')
metrics_dir = os.path.join(project_root, 'results', 'metrics')

# Créer les dossiers
os.makedirs(models_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# 1. CHARGEMENT DES DONNÉES
print("\n1️⃣ Chargement des données...")
X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

with open(os.path.join(data_dir, 'params.pickle'), 'rb') as f:
    params = pickle.load(f)

MAX_VOCAB_SIZE = params['MAX_VOCAB_SIZE']
CONTEXT_LENGTH = params['CONTEXT_LENGTH']

print(f"✅ Données chargées :")
print(f"   - X_train : {X_train.shape}")
print(f"   - X_val   : {X_val.shape}")
print(f"   - X_test  : {X_test.shape}")
print(f"   - Vocabulaire : {MAX_VOCAB_SIZE:,} mots")
print(f"   - Contexte : {CONTEXT_LENGTH} mots")

# 2. CONSTRUCTION DU MODÈLE
print("\n2️⃣ Construction du modèle LSTM...")

EMBEDDING_DIM = 128
LSTM_UNITS = 256

model = Sequential([
    # Couche d'embedding
    Embedding(                               #Convertit les tokens(nombres 1-20000) 
        input_dim=MAX_VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        input_length=CONTEXT_LENGTH,
        name='embedding_layer'
    ),
    
    # Première couche LSTM bidirectionnelle
    Bidirectional(LSTM(         #Réseau de neurones qui "mémorise" les patterns
        units=LSTM_UNITS,
        return_sequences=True,       #Retourne une sortie pour chaque token
        dropout=0.2,
        recurrent_dropout=0.2,
        name='lstm_1'
    ), name='bidirectional_lstm'),
    
    # Dropout
    Dropout(0.3, name='dropout_1'),
    
    # Deuxième couche LSTM
    LSTM(
        units=LSTM_UNITS,
        dropout=0.2,
        recurrent_dropout=0.2,
        name='lstm_2'
    ),
    
    # Dropout
    Dropout(0.3, name='dropout_2'),
    
    # Couche dense intermédiaire
    Dense(units=LSTM_UNITS//2, activation='relu', name='dense_intermediate'),
    
    Dropout(0.2, name='dropout_3'),
    
    # Couche de sortie    Couche complètement connectée
    Dense(units=MAX_VOCAB_SIZE, activation='softmax', name='output_layer')
])

print("✅ Modèle construit")

# 3. AFFICHER L'ARCHITECTURE
print("\n3️⃣ Architecture du modèle :")
model.summary()

# Sauvegarder le schéma
try:
    plot_model(
        model, 
        to_file=os.path.join(figures_dir, 'model_architecture.png'),
        show_shapes=True,
        show_layer_names=True,
        dpi=150
    )
    print(f"✅ Schéma sauvegardé : results/figures/model_architecture.png")
except:
    print("⚠️ Installation de graphviz nécessaire pour générer le schéma")

# 4. COMPILATION
print("\n4️⃣ Compilation du modèle...")
model.compile(                               
    optimizer=Adam(learning_rate=0.001),     #Algorithme d'optimisation (ajuste les poids du réseau)
    loss='sparse_categorical_crossentropy',  # IMPORTANT : sparse pour éviter one-hot
    metrics=['accuracy']
)
print("✅ Modèle compilé")

# 5. CALLBACKS
print("\n5️⃣ Configuration des callbacks...")
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    
    ModelCheckpoint(
        filepath=os.path.join(models_dir, 'best_email_model.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
]
print("✅ Callbacks configurés")

# 6. ENTRAÎNEMENT
print("\n" + "="*60)
print("🚀 DÉBUT DE L'ENTRAÎNEMENT")
print("="*60)
print("\n⚠️ ATTENTION : Cela peut prendre 2-4 HEURES !")
print("💡 Conseil : Laissez l'ordinateur allumé et branché\n")

history = model.fit(   #Lance l'entraînement
    X_train, y_train,  #Données d'entraînement
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=256,  # Augmenté pour être plus rapide
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*60)
print("✅ ENTRAÎNEMENT TERMINÉ")
print("="*60)

# 7. SAUVEGARDE DE L'HISTORIQUE
print("\n7️⃣ Sauvegarde de l'historique...")
with open(os.path.join(metrics_dir, 'training_history.pickle'), 'wb') as f:
    pickle.dump(history.history, f)
print("✅ Historique sauvegardé")

# 8. VISUALISATION
print("\n8️⃣ Création des graphiques...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2, color='blue')         #Crée un graphique montrant l'évolution de l'erreur
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Évolution de la Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, color='blue')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='red')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Évolution de l\'Accuracy', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
print(f"✅ Graphiques sauvegardés : results/figures/training_curves.png")
plt.close()

# 9. DIAGNOSTIC
print("\n9️⃣ Diagnostic du modèle :")
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
gap = abs(final_val_loss - final_train_loss)

print(f"\n📊 Résultats finaux :")
print(f"   - Loss train      : {final_train_loss:.4f}")
print(f"   - Loss validation : {final_val_loss:.4f}")
print(f"   - Acc train       : {final_train_acc:.4f}")
print(f"   - Acc validation  : {final_val_acc:.4f}")
print(f"   - Écart loss      : {gap:.4f}")

if gap > 0.5:
    print("\n⚠️ ATTENTION : Possible sur-apprentissage (overfitting)")
    print("   Solutions : augmenter dropout, plus de données")
elif final_train_loss > 3.0:
    print("\n⚠️ ATTENTION : Possible sous-apprentissage (underfitting)")
    print("   Solutions : modèle plus complexe, plus d'epochs")
else:
    print("\n✅ Le modèle semble bien entraîné !")

# 10. RÉSUMÉ FINAL
print("\n" + "="*60)
print("📊 RÉSUMÉ DE L'ENTRAÎNEMENT")
print("="*60)
print(f"Nombre d'epochs     : {len(history.history['loss'])}")
print(f"Meilleure val_loss  : {min(history.history['val_loss']):.4f}")
print(f"Meilleure val_acc   : {max(history.history['val_accuracy']):.4f}")
print(f"Modèle sauvegardé   : models/final/best_email_model.keras")
print("="*60)

print("\n✅ Entraînement complet terminé !")
print("\n💡 Prochaine étape : Évaluation et génération d'emails")