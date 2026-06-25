from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
CORS(app)  # Permettre les requêtes depuis le navigateur

# Chargement du modèle
print("Chargement du modèle...")
model = load_model('../../models/final/best_email_model.keras')
with open('../../data/processed/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
with open('../../data/processed/params.pickle', 'rb') as f:
    params = pickle.load(f)

CONTEXT_LENGTH = params['CONTEXT_LENGTH']
print("✅ Modèle chargé")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    seed_text = data.get('prompt', '')
    num_words = data.get('length', 50)
    temperature = data.get('temperature', 1.0)
    
    # Génération
    generated = seed_text.lower()
    
    for _ in range(num_words):
        tokens = tokenizer.texts_to_sequences([generated])[0]
        
        if len(tokens) > CONTEXT_LENGTH:
            tokens = tokens[-CONTEXT_LENGTH:]
        else:
            tokens = [0] * (CONTEXT_LENGTH - len(tokens)) + tokens
        
        tokens = np.array([tokens])
        preds = model.predict(tokens, verbose=0)[0]
        
        # Température
        preds = np.log(preds + 1e-10) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))
        
        predicted = np.random.choice(len(preds), p=preds)
        
        word = ""
        for w, idx in tokenizer.word_index.items():
            if idx == predicted:
                word = w
                break
        
        if word and word not in ['<OOV>', '']:
            generated += " " + word
        else:
            break

    
    return jsonify({'generated': generated})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'loaded'})

if __name__ == '__main__':
    app.run(debug=False, port=5000)
