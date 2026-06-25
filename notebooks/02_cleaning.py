"""
Script 2 : Nettoyage des emails
Objectif : Nettoyer et filtrer les emails pour l'entraînement
"""

import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm

print("="*60)
print("🧹 NETTOYAGE DES DONNÉES")
print("="*60)

# Déterminer les chemins
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
input_path = os.path.join(project_root, 'data', 'raw', 'emails.csv')
output_dir = os.path.join(project_root, 'data', 'processed')
output_path = os.path.join(output_dir, 'emails_cleaned.csv')

# Créer le dossier de sortie
os.makedirs(output_dir, exist_ok=True)

# 1. CHARGEMENT
print("\n1️⃣ Chargement du dataset...")
df = pd.read_csv(input_path)
print(f"✅ {len(df):,} emails chargés")

# 2. FONCTIONS DE TRAITEMENT
print("\n2️⃣ Création des fonctions de traitement...")

def parse_email(text):
    """Extrait le sujet et le corps de l'email"""
    if pd.isna(text):
        return {'subject': '', 'body': '', 'full': ''}
    
    # Extraire le sujet
    subject_match = re.search(r'Subject:\s*(.*?)(?:\n|$)', text, re.IGNORECASE)
    subject = subject_match.group(1).strip() if subject_match else ""
    
    # Extraire le corps
    body_match = re.search(r'\n\n(.*)', text, re.DOTALL)
    body = body_match.group(1).strip() if body_match else text
    
    return {'subject': subject, 'body': body, 'full': text}

def clean_email_text(text):
    """Nettoie le texte"""
    if not text or pd.isna(text):
        return ""
    
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)  # Supprimer emails
    text = re.sub(r'http\S+|www\S+', '', text)  # Supprimer URLs
    text = re.sub(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', '', text)  # Supprimer tél
    text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'\"-]', ' ', text)  # Garder alphanumérique
    text = re.sub(r'\s+', ' ', text)  # Supprimer espaces multiples
    text = re.sub(r'-----original message-----.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL|re.IGNORECASE)
    
    return text.strip()

def is_quality_email(text, min_length=50, max_length=5000):    #Vérifie que l'email a entre 50 et 5000 caractères
    """Vérifie la qualité"""
    if not text or len(text) < min_length or len(text) > max_length:
        return False
    if len(text.split()) < 10:
        return False
    return True

print("✅ Fonctions créées")

# 3. PARSING
print("\n3️⃣ Parsing des emails...")
tqdm.pandas(desc="📧 Parsing")
df['parsed'] = df['message'].progress_apply(parse_email)
df['subject'] = df['parsed'].apply(lambda x: x['subject'])
df['body'] = df['parsed'].apply(lambda x: x['body'])
print("✅ Parsing terminé")

# 4. NETTOYAGE
print("\n4️⃣ Nettoyage des textes...")
tqdm.pandas(desc="🧼 Sujet")
df['cleaned_subject'] = df['subject'].progress_apply(clean_email_text)

tqdm.pandas(desc="🧼 Corps")
df['cleaned_body'] = df['body'].progress_apply(clean_email_text)

df['cleaned_full'] = df['cleaned_subject'] + " . " + df['cleaned_body']
print("✅ Nettoyage terminé")

# 5. FILTRAGE
print("\n5️⃣ Filtrage des emails de qualité...")
df['is_quality'] = df['cleaned_full'].apply(is_quality_email)
df_clean = df[df['is_quality']].copy()       #Garde juste les emails de bonne qualité

emails_supprimes = len(df) - len(df_clean)
pourcentage_supprime = emails_supprimes / len(df) * 100

print(f"✅ Filtrage terminé :")
print(f"   - Emails conservés : {len(df_clean):,}")
print(f"   - Emails supprimés : {emails_supprimes:,} ({pourcentage_supprime:.1f}%)")

# 6. COMPARAISON AVANT/APRÈS
print("\n6️⃣ Exemple de transformation :")
print("\n🔴 AVANT (email brut) :")
print("-"*60)
print(df['message'].iloc[100][:400])
print("...\n")

print("🟢 APRÈS (email nettoyé) :")
print("-"*60)
if len(df_clean) > 0:
    print(df_clean['cleaned_full'].iloc[min(10, len(df_clean)-1)][:400])
    print("...")

# 7. STATISTIQUES POST-NETTOYAGE
print("\n7️⃣ Statistiques après nettoyage :")
df_clean['cleaned_length'] = df_clean['cleaned_full'].str.len()
print(df_clean['cleaned_length'].describe())

# 8. SAUVEGARDE
print("\n8️⃣ Sauvegarde des données nettoyées...")
df_clean[['cleaned_full', 'cleaned_subject', 'cleaned_body']].to_csv(output_path, index=False)
print(f"✅ Données sauvegardées : {output_path}")
print(f"📊 Fichier contient {len(df_clean):,} emails propres")

# 9. RÉSUMÉ FINAL
print("\n" + "="*60)
print("📊 RÉSUMÉ DU NETTOYAGE")
print("="*60)
print(f"Emails initiaux     : {len(df):,}")
print(f"Emails conservés    : {len(df_clean):,}")
print(f"Emails supprimés    : {emails_supprimes:,}")
print(f"Taux de conservation: {(len(df_clean)/len(df)*100):.1f}%")
print(f"Longueur moyenne    : {df_clean['cleaned_length'].mean():.0f} caractères")
print("="*60)

print("\n✅ Nettoyage complet terminé !")