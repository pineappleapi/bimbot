#!/usr/bin/env python3
"""
Save the best NLP model for production use.
This script trains and exports models in a format compatible with hazard_nlp_classifier.py

Usage:
    python save_best_nlp_model.py --input data/training/text_hazard_data.csv --model logistic
    python save_best_nlp_model.py --input data/training/text_hazard_data.csv --model naive_bayes
    python save_best_nlp_model.py --input data/training/text_hazard_data.csv --model random_forest
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


def load_text_dataset(csv_path):
    """Load text-based hazard dataset"""
    print(f"📂 Loading text dataset from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Map labels to numeric
    label_mapping = {'Low': 0, 'Moderate': 1, 'Severe': 2}
    df['label_numeric'] = df['label'].map(label_mapping)
    
    X = df['text'].values
    y = df['label_numeric'].values
    
    print(f"✅ Loaded {len(df)} text samples")
    return X, y


def train_and_save_model(X_train, X_test, y_train, y_test, model_type, output_path):
    """Train model and save with its vectorizer"""
    
    print(f"\n🔧 Training {model_type.upper()} model...")
    
    if model_type == 'logistic':
        # Logistic Regression with Bag-of-Words
        vectorizer = CountVectorizer(max_features=500, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vec, y_train)
    
    elif model_type == 'naive_bayes':
        # Naive Bayes with TF-IDF
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
    
    elif model_type == 'random_forest':
        # Random Forest with TF-IDF
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_vec, y_train)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"✅ Training complete!")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   F1 (Macro): {f1_macro:.3f}")
    
    # Classification report
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low', 'Moderate', 'Severe']))
    
    # Save model and vectorizer together
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'vectorizer': vectorizer,
        'model_type': model_type,
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'label_names': ['Low Risk', 'Moderate Risk', 'Severe Risk']
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n💾 Model saved to: {output_path}")
    
    # Save metadata
    metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
    metadata = {
        'model_type': model_type,
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'num_train_samples': len(X_train),
        'num_test_samples': len(X_test),
        'features': vectorizer.get_feature_names_out().tolist()[:50]  # First 50 features
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Metadata saved to: {metadata_path}")
    
    return model, vectorizer, accuracy, f1_macro


def main():
    parser = argparse.ArgumentParser(
        description='Train and save NLP model for production'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to text dataset CSV'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['logistic', 'naive_bayes', 'random_forest'],
        default='logistic',
        help='Model type to train (default: logistic)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path (default: models/nlp/{model_type}.pkl)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        args.output = f"models/nlp/{args.model}.pkl"
    
    # Banner
    print("\n" + "=" * 60)
    print(f"  Training {args.model.upper()} for Production")
    print("=" * 60 + "\n")
    
    # Load data
    X, y = load_text_dataset(args.input)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print(f"🔀 Split: {len(X_train)} train, {len(X_test)} test")
    
    # Train and save
    model, vectorizer, accuracy, f1 = train_and_save_model(
        X_train, X_test, y_train, y_test,
        args.model, args.output
    )
    
    # Test with sample inputs
    print("\n🧪 Testing with sample inputs:")
    test_samples = [
        "Temperature is 28°C, humidity is 65%, heat index is normal.",
        "Temperature is 34°C, humidity is 82%, heat index is high.",
        "Heat index exceeds threshold at 41°C."
    ]
    
    X_samples = vectorizer.transform(test_samples)
    predictions = model.predict(X_samples)
    probabilities = model.predict_proba(X_samples) if hasattr(model, 'predict_proba') else None
    
    labels = ['Low', 'Moderate', 'Severe']
    for i, text in enumerate(test_samples):
        pred_label = labels[predictions[i]]
        if probabilities is not None:
            conf = probabilities[i][predictions[i]]
            print(f"  [{pred_label}] (conf={conf:.2f}) {text}")
        else:
            print(f"  [{pred_label}] {text}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ MODEL READY FOR PRODUCTION!")
    print("=" * 60)
    print(f"Model: {args.model.upper()}")
    print(f"File: {args.output}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"F1 Score: {f1:.1%}")
    print("\nTo use in app.py, update:")
    print(f"  from hazard_nlp_classifier import get_classifier")
    print(f"  classifier = get_classifier('{args.output}', '{args.model}')")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()