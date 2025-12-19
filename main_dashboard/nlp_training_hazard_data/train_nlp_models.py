#!/usr/bin/env python3
"""
Part 4 — Overall Development (NLP Version)
Train and evaluate NLP models for hazard classification.

This script implements:
  4.1 Baseline Models (Logistic Regression, Naive Bayes, Decision Tree, Naive Predictor)
  4.2 Main Models (LSTM, CNN, Transformer, Random Forest on embeddings)
  4.3 Final Evaluation (with confidence intervals via bootstrapping)
  4.4 Bias & Robustness Analysis

Usage:
    python train_nlp_models.py --input data/training/text_hazard_data.csv
    python train_nlp_models.py --input data/training/text_hazard_data.csv --quick  # Skip deep learning
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.dummy import DummyClassifier

# Deep Learning (TensorFlow/Keras)
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Embedding, LSTM, Bidirectional, Dense, Dropout, 
        Conv1D, GlobalMaxPooling1D, Input, MultiHeadAttention,
        LayerNormalization, GlobalAveragePooling1D
    )
    from tensorflow.keras.callbacks import EarlyStopping
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("⚠️  TensorFlow not available. Skipping deep learning models.")

# Transformers (Hugging Face)
try:
    from transformers import (
        AutoTokenizer, TFAutoModel,
        DistilBertTokenizer, TFDistilBertModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers library not available. Skipping transformer models.")


class NLPModelEvaluator:
    """Handles training and evaluation of all NLP models"""
    
    def __init__(self, X_train, X_test, y_train, y_test, texts_train, texts_test):
        self.X_train = X_train  # Raw texts
        self.X_test = X_test
        self.y_train = y_train  # Numeric labels (0, 1, 2)
        self.y_test = y_test
        self.texts_train = texts_train
        self.texts_test = texts_test
        
        self.label_names = ['Low', 'Moderate', 'Severe']
        self.results = {}
        
        # Vectorizers (will be fitted during training)
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None
        self.tokenizer = None  # For deep learning
        
    def bootstrap_confidence_interval(self, y_true, y_pred, metric='accuracy', n_bootstrap=1000, confidence=0.95):
        """
        Compute confidence intervals via bootstrapping.
        
        Returns: (mean, lower_bound, upper_bound, margin)
        """
        np.random.seed(42)
        scores = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Compute metric
            if metric == 'accuracy':
                score = accuracy_score(y_true_boot, y_pred_boot)
            elif metric == 'f1_macro':
                score = f1_score(y_true_boot, y_pred_boot, average='macro')
            
            scores.append(score)
        
        # Compute percentiles
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(scores, lower_percentile)
        upper = np.percentile(scores, upper_percentile)
        mean = np.mean(scores)
        margin = (upper - lower) / 2
        
        return mean, lower, upper, margin
    
    def evaluate_model(self, name, y_pred, training_time=None):
        """Evaluate and store model results with confidence intervals"""
        accuracy = accuracy_score(self.y_test, y_pred)
        f1_macro = f1_score(self.y_test, y_pred, average='macro')
        
        # Bootstrap confidence intervals
        _, _, _, acc_margin = self.bootstrap_confidence_interval(
            self.y_test, y_pred, metric='accuracy'
        )
        _, _, _, f1_margin = self.bootstrap_confidence_interval(
            self.y_test, y_pred, metric='f1_macro'
        )
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test, y_pred, average=None
        )
        
        self.results[name] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'accuracy_ci': acc_margin,
            'f1_ci': f1_margin,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support': support.tolist(),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
            'training_time': training_time
        }
        
        print(f"  {name}: Acc={accuracy:.3f}±{acc_margin:.3f}, F1={f1_macro:.3f}±{f1_margin:.3f}")
    
    # ========== 4.1 BASELINE MODELS ==========
    
    def train_naive_predictor(self):
        """Always predicts 'Low' (majority class baseline)"""
        print("\n[4.1.1] Training Naive Predictor (Always Low)...")
        model = DummyClassifier(strategy='constant', constant=0)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.evaluate_model('Naive Predictor', y_pred)
        return model
    
    def train_logistic_regression(self):
        """Logistic Regression with Bag-of-Words"""
        print("\n[4.1.2] Training Logistic Regression (BoW)...")
        
        # Fit vectorizer
        self.bow_vectorizer = CountVectorizer(max_features=500, ngram_range=(1, 2))
        X_train_bow = self.bow_vectorizer.fit_transform(self.texts_train)
        X_test_bow = self.bow_vectorizer.transform(self.texts_test)
        
        # Train
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_bow, self.y_train)
        
        # Predict
        y_pred = model.predict(X_test_bow)
        self.evaluate_model('Logistic Regression', y_pred)
        return model
    
    def train_naive_bayes(self):
        """Multinomial Naive Bayes"""
        print("\n[4.1.3] Training Naive Bayes (Multinomial)...")
        
        # Use TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(self.texts_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(self.texts_test)
        
        # Train
        model = MultinomialNB()
        model.fit(X_train_tfidf, self.y_train)
        
        # Predict
        y_pred = model.predict(X_test_tfidf)
        self.evaluate_model('Naive Bayes', y_pred)
        return model
    
    def train_decision_tree(self):
        """Decision Tree Classifier"""
        print("\n[4.1.4] Training Decision Tree...")
        
        # Use TF-IDF (already fitted)
        X_train_tfidf = self.tfidf_vectorizer.transform(self.texts_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(self.texts_test)
        
        # Train
        model = DecisionTreeClassifier(max_depth=10, random_state=42)
        model.fit(X_train_tfidf, self.y_train)
        
        # Predict
        y_pred = model.predict(X_test_tfidf)
        self.evaluate_model('Decision Tree', y_pred)
        return model
    
    # ========== 4.2 MAIN MODELS ==========
    
    def prepare_sequences(self, max_words=1000, max_len=50):
        """Prepare sequences for deep learning models"""
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(self.texts_train)
        
        X_train_seq = self.tokenizer.texts_to_sequences(self.texts_train)
        X_test_seq = self.tokenizer.texts_to_sequences(self.texts_test)
        
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')
        
        return X_train_pad, X_test_pad, max_words, max_len
    
    def train_lstm(self):
        """LSTM model"""
        if not DEEP_LEARNING_AVAILABLE:
            print("\n[4.2.1] Skipping LSTM (TensorFlow not available)")
            return None
        
        print("\n[4.2.1] Training LSTM...")
        
        X_train_pad, X_test_pad, max_words, max_len = self.prepare_sequences()
        
        model = Sequential([
            Embedding(max_words, 64, input_length=max_len),
            LSTM(64, dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        model.fit(
            X_train_pad, self.y_train,
            validation_split=0.2,
            epochs=20,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        y_pred = np.argmax(model.predict(X_test_pad, verbose=0), axis=1)
        self.evaluate_model('LSTM', y_pred)
        return model
    
    def train_cnn(self):
        """CNN for text (Kim 2014 style)"""
        if not DEEP_LEARNING_AVAILABLE:
            print("\n[4.2.2] Skipping CNN (TensorFlow not available)")
            return None
        
        print("\n[4.2.2] Training CNN for Text...")
        
        X_train_pad, X_test_pad, max_words, max_len = self.prepare_sequences()
        
        model = Sequential([
            Embedding(max_words, 64, input_length=max_len),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        model.fit(
            X_train_pad, self.y_train,
            validation_split=0.2,
            epochs=20,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        y_pred = np.argmax(model.predict(X_test_pad, verbose=0), axis=1)
        self.evaluate_model('CNN (Kim 2014)', y_pred)
        return model
    
    def train_tiny_transformer(self):
        """Tiny Transformer (simplified attention mechanism)"""
        if not DEEP_LEARNING_AVAILABLE:
            print("\n[4.2.3] Skipping Transformer (TensorFlow not available)")
            return None
        
        print("\n[4.2.3] Training Tiny Transformer...")
        
        X_train_pad, X_test_pad, max_words, max_len = self.prepare_sequences()
        
        # Build transformer model
        inputs = Input(shape=(max_len,))
        x = Embedding(max_words, 64)(inputs)
        
        # Self-attention
        attn_output = MultiHeadAttention(
            num_heads=4, key_dim=64
        )(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed-forward
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(3, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        model.fit(
            X_train_pad, self.y_train,
            validation_split=0.2,
            epochs=20,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        y_pred = np.argmax(model.predict(X_test_pad, verbose=0), axis=1)
        self.evaluate_model('Tiny Transformer', y_pred)
        return model
    
    def train_random_forest_embeddings(self):
        """Random Forest on TF-IDF embeddings"""
        print("\n[4.2.4] Training Random Forest (TF-IDF)...")
        
        X_train_tfidf = self.tfidf_vectorizer.transform(self.texts_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(self.texts_test)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_tfidf, self.y_train)
        
        y_pred = model.predict(X_test_tfidf)
        self.evaluate_model('Random Forest', y_pred)
        return model
    
    # ========== 4.4 BIAS & ROBUSTNESS ANALYSIS ==========
    
    def analyze_bias(self, model_name='Best Model', y_pred=None):
        """Check for prediction biases"""
        print(f"\n[4.4] Bias Analysis for {model_name}")
        print("=" * 60)
        
        if y_pred is None:
            print("⚠️  No predictions provided")
            return
        
        # Class distribution
        unique, counts = np.unique(y_pred, return_counts=True)
        total = len(y_pred)
        
        print("\nPrediction Distribution:")
        for label_id, count in zip(unique, counts):
            pct = count / total * 100
            print(f"  {self.label_names[label_id]:10s}: {count:4d} ({pct:5.1f}%)")
        
        # Check for over/under prediction
        true_dist = np.bincount(self.y_test, minlength=3)
        pred_dist = np.bincount(y_pred, minlength=3)
        
        print("\nBias Check (Predicted vs Actual):")
        for i, label in enumerate(self.label_names):
            diff = pred_dist[i] - true_dist[i]
            bias = "over-predicting" if diff > 0 else "under-predicting" if diff < 0 else "balanced"
            print(f"  {label:10s}: {bias:15s} (diff: {diff:+3d})")
    
    def robustness_test(self, model, model_name):
        """Test robustness to noisy/modified inputs"""
        print(f"\n[4.4] Robustness Test for {model_name}")
        print("=" * 60)
        
        # Test cases with deliberate variations
        test_cases = [
            ("Temperature is 28°C, humidity is 65%, heat index is normal.", "Low"),
            ("Temperature is 34°C, humidity is 82%, heat index is high.", "Moderate"),
            ("Heat index exceeds threshold at 41°C.", "Severe"),
            # Noisy versions
            ("Temp 28 humidity 65 normal", "Low"),  # Truncated
            ("TEMPERATURE IS 34°C HUMIDITY IS 82% HIGH", "Moderate"),  # All caps
            ("heat index very high 41 degrees", "Severe"),  # Informal
        ]
        
        print("\nTest Cases:")
        for text, expected in test_cases:
            # Predict (depends on model type)
            # For now, just print - actual implementation depends on model wrapper
            print(f"  Input: '{text}' → Expected: {expected}")


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
    print(f"\n📊 Class Distribution:")
    print(df['label'].value_counts())
    
    return X, y


def main():
    parser = argparse.ArgumentParser(
        description='Part 4: NLP-based Hazard Classification'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to text dataset CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/nlp',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Skip deep learning models (faster for testing)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "=" * 60)
    print("  Part 4 — Overall Development (NLP Version)")
    print("  Hazard Classification via Text-Based Models")
    print("=" * 60 + "\n")
    
    # Load data
    X, y = load_text_dataset(args.input)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print(f"\n🔀 Train/Test Split: {len(X_train)} train, {len(X_test)} test")
    
    # Initialize evaluator
    evaluator = NLPModelEvaluator(
        X_train, X_test, y_train, y_test,
        X_train, X_test  # Pass texts separately
    )
    
    # ========== TRAINING ==========
    
    print("\n" + "=" * 60)
    print("  SECTION 4.1: BASELINE MODELS")
    print("=" * 60)
    
    evaluator.train_naive_predictor()
    evaluator.train_logistic_regression()
    evaluator.train_naive_bayes()
    evaluator.train_decision_tree()
    
    print("\n" + "=" * 60)
    print("  SECTION 4.2: MAIN MODELS (Modern NLP)")
    print("=" * 60)
    
    if not args.quick:
        evaluator.train_lstm()
        evaluator.train_cnn()
        evaluator.train_tiny_transformer()
    else:
        print("\n⚡ Quick mode: Skipping deep learning models")
    
    evaluator.train_random_forest_embeddings()
    
    # ========== EVALUATION TABLE ==========
    
    print("\n" + "=" * 60)
    print("  SECTION 4.3: FINAL EVALUATION")
    print("=" * 60 + "\n")
    
    print(f"{'Model':<25s} {'Accuracy':<12s} {'Macro F1':<12s} {'95% CI':<15s}")
    print("-" * 70)
    
    for name, metrics in sorted(
        evaluator.results.items(), 
        key=lambda x: x[1]['accuracy'], 
        reverse=True
    ):
        acc = metrics['accuracy']
        f1 = metrics['f1_macro']
        ci_acc = metrics['accuracy_ci']
        ci_f1 = metrics['f1_ci']
        
        print(f"{name:<25s} {acc:.1%}      {f1:.1%}      ±{ci_acc:.3f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(evaluator.results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_dir / 'evaluation_results.json'}")
    
    # ========== BIAS ANALYSIS ==========
    
    # Get best model predictions (for demonstration, use Logistic Regression)
    X_test_bow = evaluator.bow_vectorizer.transform(X_test)
    best_model = LogisticRegression(max_iter=1000, random_state=42)
    X_train_bow = evaluator.bow_vectorizer.transform(X_train)
    best_model.fit(X_train_bow, y_train)
    y_pred_best = best_model.predict(X_test_bow)
    
    evaluator.analyze_bias('Logistic Regression', y_pred_best)
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ PART 4 COMPLETE!")
    print("=" * 60)
    print(f"✓ Trained {len(evaluator.results)} models")
    print(f"✓ Best Accuracy: {max(m['accuracy'] for m in evaluator.results.values()):.1%}")
    print(f"✓ Results: {output_dir / 'evaluation_results.json'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()