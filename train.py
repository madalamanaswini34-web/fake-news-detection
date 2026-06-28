#!/usr/bin/env python3
"""
Fake News Detection - Model Training Pipeline
---------------------------------------------
This script loads fake and real news datasets, performs robust NLP text preprocessing,
vectorizes the text using TF-IDF, trains and compares Logistic Regression and
Passive-Aggressive Classifier, and exports the best-performing model using pickle.

Author: AI Coding Assistant
Suitable for College Resume & Portfolio
"""

import os
import re
import string
import pickle
import pandas as pd
import numpy as np

# Import machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import Natural Language Toolkit (NLTK) for NLP preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- SETUP AND DOWNLOAD NLTK RESOURCES ---
print("[*] Setting up NLTK resources...")
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
    print("[+] NLTK resources downloaded successfully.")
except Exception as e:
    print(f"[-] NLTK Download Warning: {e}. If offline, ensure resources are pre-installed.")

# Initialize lemmatizer and load stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Performs text cleaning and preprocessing:
    1. Converts text to lowercase
    2. Removes punctuation and special characters
    3. Removes stop words
    4. Applies lemmatization (reduces words to their root form)
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove URLs, HTML tags, and bracketed text
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    
    # 3. Remove punctuation
    # Translation table to replace all punctuation with empty string
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # 4. Tokenization and filtering (Remove stop words & Lemmatize)
    words = text.split()
    cleaned_words = []
    for word in words:
        if word not in stop_words:
            # Lemmatize nouns (default) and verbs
            lemma = lemmatizer.lemmatize(word)
            lemma = lemmatizer.lemmatize(lemma, pos='v')
            cleaned_words.append(lemma)
            
    return " ".join(cleaned_words)

def load_data():
    """
    Loads Fake.csv and True.csv from the dataset/ directory,
    labels them (0 for Fake, 1 for Real), and combines them.
    """
    print("[*] Loading datasets...")
    fake_path = os.path.join("dataset", "Fake.csv")
    true_path = os.path.join("dataset", "True.csv")
    
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError(
            "Datasets not found in dataset/ directory. "
            "Please ensure dataset/Fake.csv and dataset/True.csv exist."
        )
        
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    
    # Assign labels: 0 for FAKE news, 1 for REAL news
    df_fake['label'] = 0
    df_true['label'] = 1
    
    # Combine datasets
    df = pd.concat([df_fake, df_true], ignore_index=True)
    
    # Combine news title and body content for richer semantic representation
    print("[*] Combining Title and Text...")
    df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def main():
    # 1. Load Data
    try:
        df = load_data()
        print(f"[+] Loaded {len(df)} total articles.")
    except Exception as e:
        print(f"[-] Error loading data: {e}")
        return

    # 2. Preprocess Data
    print("[*] Preprocessing news content (lowercasing, cleaning, stopword removal, lemmatizing)...")
    print("    Note: This may take a minute on large datasets...")
    df['clean_content'] = df['content'].apply(preprocess_text)
    
    # Drop rows with empty content after preprocessing
    df = df[df['clean_content'].str.strip() != ''].reset_index(drop=True)
    
    X = df['clean_content']
    y = df['label']
    
    # 3. Train-Test Split (Avoid Data Leakage)
    # We split BEFORE vectorization so that the TF-IDF vocabulary is built ONLY on the training set.
    # This ensures that test set terms do not influence IDF scores (preventing target/data leakage).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[+] Data split complete: Training set size = {len(X_train)}, Testing set size = {len(X_test)}")
    
    # 4. Feature Extraction: TF-IDF Vectorizer
    print("[*] Initializing TF-IDF Vectorizer...")
    # max_df=0.7 ignores terms that appear in more than 70% of documents (removes corpus-specific stopwords)
    # min_df=2 ignores terms that appear in fewer than 2 documents (removes typos/rare words)
    vectorizer = TfidfVectorizer(max_df=0.7, min_df=2, ngram_range=(1, 2))
    
    print("[*] Fitting vectorizer on training data and transforming...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"[+] TF-IDF Matrix shape: {X_train_vec.shape}")
    
    # 5. Model Training & Comparison
    results = {}
    models = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Passive-Aggressive Classifier": PassiveAggressiveClassifier(max_iter=1000, random_state=42, C=0.5)
    }
    
    print("\n" + "="*50)
    print(" TRAINING AND COMPARISON METRICS ")
    print("="*50)
    
    for name, model in models.items():
        print(f"\n[*] Training {name}...")
        model.fit(X_train_vec, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test_vec)
        
        # Evaluate performance
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "cm": cm
        }
        
        print(f" {name} Performance Summary:")
        print(f"  - Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"  - Precision: {prec:.4f} ({prec*100:.2f}%)")
        print(f"  - Recall:    {rec:.4f} ({rec*100:.2f}%)")
        print(f"  - F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        print("  - Confusion Matrix:")
        print(f"    [[TN: {cm[0][0]}, FP: {cm[0][1]}],")
        print(f"     [FN: {cm[1][0]}, TP: {cm[1][1]}]]")
        
    print("\n" + "="*50)
    
    # 6. Select and Save the Best-Performing Model
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = results[best_name]['model']
    best_acc = results[best_name]['accuracy']
    
    print(f"[+] Best-performing model: {best_name} with {best_acc*100:.2f}% accuracy.")
    
    # Create model export directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    
    model_path = os.path.join("model", "best_model.pkl")
    vectorizer_path = os.path.join("model", "vectorizer.pkl")
    
    # Export model and vectorizer
    print(f"[*] Exporting best model to '{model_path}'...")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
        
    print(f"[*] Exporting fitted TF-IDF vectorizer to '{vectorizer_path}'...")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    # Also save metadata for the application
    meta_path = os.path.join("model", "model_metadata.pkl")
    metadata = {
        "model_name": best_name,
        "metrics": {
            "accuracy": results[best_name]['accuracy'],
            "precision": results[best_name]['precision'],
            "recall": results[best_name]['recall'],
            "f1_score": results[best_name]['f1'],
            "confusion_matrix": results[best_name]['cm'].tolist()
        }
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
        
    print("[+] Model exporting complete! Ready for Streamlit deployment.")

if __name__ == "__main__":
    main()
