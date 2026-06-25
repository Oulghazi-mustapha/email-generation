"""
Script 5 : Génération d'emails avec le modèle entraîné
"""

import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model

print("="*60)
print("📧 GÉNÉRATION D'EMAILS")
print("="*60)

# Chemins
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
model_path = os.path.join(project_root, 'models', 'final', 'best_email_model.keras')
tokenizer_path = os.path.join(project_root, 'data', 'processed', 'tokenizer.pickle')
params_path = os.path.join(project_root, 'data', 'processed', 'params.pickle')

# 1. CHARGEMENT
print("\n1️⃣ Chargement du modèle et du tokenizer...")
model = load_model(model_path)      #Charge le modèle entraîné depuis le disque
print("✅ Modèle chargé")

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
print("✅ Tokenizer chargé")

with open(params_path, 'rb') as f:
    params = pickle.load(f)

CONTEXT_LENGTH = params['CONTEXT_LENGTH']
print(f"✅ Contexte : {CONTEXT_LENGTH} mots")

# 2. FONCTION DE GÉNÉRATION
def generate_email(seed_text, num_words=50, temperature=1.0):    #Commence avec le texte de départ (seed)  Boucle 50 fois pour générer 50 mots
    """                                                          
    Génère un email à partir d'un texte de départ
    
    Args:
        seed_text: Texte de départ (ex: "meeting tomorrow")
        num_words: Nombre de mots à générer
        temperature: Créativité (0.5=conservateur, 1.0=normal, 1.5=créatif)
    """
    generated_text = seed_text.lower()
    
    for _ in range(num_words):
        # Tokenizer le texte actuel
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        
        # Prendre les derniers CONTEXT_LENGTH tokens
        if len(token_list) > CONTEXT_LENGTH:    #Le modèle attend exactement 40 tokens
            token_list = token_list[-CONTEXT_LENGTH:]
        else:
            # Padding si trop court
            token_list = [0] * (CONTEXT_LENGTH - len(token_list)) + token_list
        
        # Prédire le mot suivant
        token_list = np.array([token_list])
        predictions = model.predict(token_list, verbose=0)[0]
        
        # Appliquer la température
        predictions = np.log(predictions + 1e-10) / temperature   # Contrôle la "créativité"
        predictions = np.exp(predictions)   # Le modèle donne les probabilités pour les 20,000 mots
        predictions = predictions / np.sum(predictions)
        
        # Choisir le mot
        predicted_token = np.random.choice(len(predictions), p=predictions)   #Choisit un token avec les probabilités
        
        # Convertir en mot
        output_word = ""
        for word, index in tokenizer.word_index.items(): #Cherche le mot correspondant dans le dictionnaire
            if index == predicted_token:
                output_word = word
                break
        
        if output_word:
            generated_text += " " + output_word
        else:
            break
    
    return generated_text

# 3. EXEMPLES DE GÉNÉRATION
print("\n2️⃣ Génération d'exemples d'emails...")
print("="*60)

examples = [
    ("meeting tomorrow", 40),
    ("please send", 30),
    ("thank you", 35),
    ("urgent request", 40),
    ("project update", 45),
]

for i, (seed, length) in enumerate(examples, 1):
    print(f"\n📧 Email #{i}")
    print(f"Prompt : \"{seed}\"")
    print("-"*60)
    
    # Générer 3 variations avec différentes températures
    for temp, label in [(0.7, "Conservative"), (1.0, "Normal"), (1.3, "Creative")]:
        email = generate_email(seed, num_words=length, temperature=temp)
        print(f"\n{label} (temp={temp}) :")
        print(email)
    
    print("="*60)

# 4. GÉNÉRATION INTERACTIVE
print("\n3️⃣ Mode interactif")
print("="*60)
print("Entrez un début d'email pour générer la suite.")
print("(Tapez 'quit' pour quitter)\n")

while True:
    user_input = input("💬 Votre prompt : ")
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Au revoir !")
        break
    
    if not user_input.strip():
        continue
    
    print("\n📧 Email généré :")
    print("-"*60)
    generated = generate_email(user_input, num_words=50, temperature=1.0)
    print(generated)
    print("-"*60)
    print()

print("\n✅ Génération terminée !")