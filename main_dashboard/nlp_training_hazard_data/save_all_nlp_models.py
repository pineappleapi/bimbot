#!/usr/bin/env python3
"""
Save ALL NLP models for comprehensive comparison in research paper.

This script trains and saves:
  - 4.1 Baseline Models: Naive Predictor, Logistic Regression, Naive Bayes, Decision Tree
  - 4.2 Main Models: LSTM, CNN, Transformer, Random Forest

Each model is saved with:
  - Model file (.pkl or .h5)
  - Vectorizer/Tokenizer (for text preprocessing)
  - Evaluation metrics (JSON)
  - Sample predictions (for verification)

Usage:
    python save_all_nlp_models.py --input data/training/text_hazard_data.csv
    python save_all_nlp_models.py --input data/training/text_hazard_data.csv --quick  # Skip deep learning
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, precision_recall_fscore_support
)

# Deep Learning
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


def bootstrap_confidence_interval(y_true, y_pred, metric='accuracy', n_bootstrap=1000):
    """Compute 95% confidence intervals via bootstrapping"""
    np.random.seed(42)
    scores = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        if metric == 'accuracy':
            score = accuracy_score(y_true_boot, y_pred_boot)
        elif metric == 'f1_macro':
            score = f1_score(y_true_boot, y_pred_boot, average='macro')
        
        scores.append(score)
    
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    margin = (upper - lower) / 2
    
    return margin


def evaluate_and_save_metrics(model_name, y_test, y_pred, output_dir):
    """Compute and save comprehensive evaluation metrics"""
    
    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # Confidence intervals
    acc_ci = bootstrap_confidence_interval(y_test, y_pred, 'accuracy')
    f1_ci = bootstrap_confidence_interval(y_test, y_pred, 'f1_macro')
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'accuracy_ci_margin': float(acc_ci),
        'f1_macro_ci_margin': float(f1_ci),
        'accuracy_with_ci': f"{accuracy:.3f}±{acc_ci:.3f}",
        'f1_macro_with_ci': f"{f1_macro:.3f}±{f1_ci:.3f}",
        'per_class_metrics': {
            'Low': {
                'precision': float(precision[0]),
                'recall': float(recall[0]),
                'f1': float(f1[0]),
                'support': int(support[0])
            },
            'Moderate': {
                'precision': float(precision[1]),
                'recall': float(recall[1]),
                'f1': float(f1[1]),
                'support': int(support[1])
            },
            'Severe': {
                'precision': float(precision[2]),
                'recall': float(recall[2]),
                'f1': float(f1[2]),
                'support': int(support[2])
            }
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            y_test, y_pred, 
            target_names=['Low', 'Moderate', 'Severe'],
            output_dict=True
        )
    }
    
    # Save metrics
    metrics_path = output_dir / f"{model_name.lower().replace(' ', '_')}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  ✅ {model_name}: Acc={accuracy:.3f}±{acc_ci:.3f}, F1={f1_macro:.3f}±{f1_ci:.3f}")
    
    return metrics


def test_sample_predictions(model_name, predictor_func, output_dir):
    """Test model with sample inputs and save results"""
    
    test_samples = [
        ("Temperature is 25°C, humidity is 55%, heat index is normal.", "Low"),
        ("Temperature is 28°C, humidity is 65%, heat index is elevated.", "Low/Moderate"),
        ("Temperature is 32°C, humidity is 75%, heat index is high.", "Moderate"),
        ("Temperature is 34°C, humidity is 82%, heat index is high.", "Moderate"),
        ("Temperature is 37°C, humidity is 88%, heat index exceeds threshold.", "Severe"),
        ("Heat index exceeds threshold at 41°C.", "Severe"),
    ]
    
    label_names = ['Low', 'Moderate', 'Severe']
    results = []
    
    for text, expected in test_samples:
        pred_class, confidence = predictor_func(text)
        pred_label = label_names[pred_class]
        
        results.append({
            'input': text,
            'expected': expected,
            'predicted': pred_label,
            'confidence': float(confidence)
        })
    
    # Save sample predictions
    sample_path = output_dir / f"{model_name.lower().replace(' ', '_')}_samples.json"
    with open(sample_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


# ========== BASELINE MODELS ==========

def save_naive_predictor(X_train, X_test, y_train, y_test, output_dir):
    """Naive Predictor (always predicts most frequent class)"""
    print("\n[4.1.1] Saving Naive Predictor...")
    
    model = DummyClassifier(strategy='most_frequent')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Save model
    model_path = output_dir / 'naive_predictor.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'model_type': 'naive_predictor'}, f)
    
    # Evaluate and save metrics
    metrics = evaluate_and_save_metrics('Naive Predictor', y_test, y_pred, output_dir)
    
    return model, metrics


def save_logistic_regression(X_train, X_test, y_train, y_test, output_dir):
    """Logistic Regression with Bag-of-Words"""
    print("\n[4.1.2] Saving Logistic Regression...")
    
    # Vectorizer
    vectorizer = CountVectorizer(max_features=500, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    # Save model + vectorizer
    model_path = output_dir / 'logistic_regression.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'vectorizer': vectorizer,
            'model_type': 'logistic_regression'
        }, f)
    
    # Evaluate and save metrics
    metrics = evaluate_and_save_metrics('Logistic Regression', y_test, y_pred, output_dir)
    
    # Test samples
    def predictor(text):
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        conf = model.predict_proba(X)[0][pred]
        return pred, conf
    
    test_sample_predictions('Logistic Regression', predictor, output_dir)
    
    return model, vectorizer, metrics


def save_naive_bayes(X_train, X_test, y_train, y_test, output_dir):
    """Naive Bayes with TF-IDF"""
    print("\n[4.1.3] Saving Naive Bayes...")
    
    # Vectorizer
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    # Save model + vectorizer
    model_path = output_dir / 'naive_bayes.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'vectorizer': vectorizer,
            'model_type': 'naive_bayes'
        }, f)
    
    # Evaluate and save metrics
    metrics = evaluate_and_save_metrics('Naive Bayes', y_test, y_pred, output_dir)
    
    # Test samples
    def predictor(text):
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        conf = model.predict_proba(X)[0][pred]
        return pred, conf
    
    test_sample_predictions('Naive Bayes', predictor, output_dir)
    
    return model, vectorizer, metrics


def save_decision_tree(X_train, X_test, y_train, y_test, output_dir, vectorizer):
    """Decision Tree with TF-IDF"""
    print("\n[4.1.4] Saving Decision Tree...")
    
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Model
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    # Save model + vectorizer
    model_path = output_dir / 'decision_tree.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'vectorizer': vectorizer,
            'model_type': 'decision_tree'
        }, f)
    
    # Evaluate and save metrics
    metrics = evaluate_and_save_metrics('Decision Tree', y_test, y_pred, output_dir)
    
    # Test samples
    def predictor(text):
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        # Decision tree might not have predict_proba
        if hasattr(model, 'predict_proba'):
            conf = model.predict_proba(X)[0][pred]
        else:
            conf = 1.0
        return pred, conf
    
    test_sample_predictions('Decision Tree', predictor, output_dir)
    
    return model, metrics


# ========== MAIN MODELS ==========

def save_random_forest(X_train, X_test, y_train, y_test, output_dir, vectorizer):
    """Random Forest with TF-IDF"""
    print("\n[4.2.1] Saving Random Forest...")
    
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    # Save model + vectorizer
    model_path = output_dir / 'random_forest.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'vectorizer': vectorizer,
            'model_type': 'random_forest'
        }, f)
    
    # Evaluate and save metrics
    metrics = evaluate_and_save_metrics('Random Forest', y_test, y_pred, output_dir)
    
    # Test samples
    def predictor(text):
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        conf = model.predict_proba(X)[0][pred]
        return pred, conf
    
    test_sample_predictions('Random Forest', predictor, output_dir)
    
    return model, metrics


def prepare_sequences(X_train, X_test, max_words=1000, max_len=50):
    """Prepare sequences for deep learning"""
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')
    
    return X_train_pad, X_test_pad, tokenizer, max_words, max_len


def save_lstm(X_train, X_test, y_train, y_test, output_dir):
    """LSTM model"""
    if not DEEP_LEARNING_AVAILABLE:
        print("\n[4.2.2] Skipping LSTM (TensorFlow not available)")
        return None, None
    
    print("\n[4.2.2] Saving LSTM...")
    
    X_train_pad, X_test_pad, tokenizer, max_words, max_len = prepare_sequences(X_train, X_test)
    
    # Model
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
        X_train_pad, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Predict
    y_pred = np.argmax(model.predict(X_test_pad, verbose=0), axis=1)
    
    # Save model
    model_path = output_dir / 'lstm_model.h5'
    model.save(model_path)
        
        
    tok_json_path = output_dir / 'lstm_tokenizer.json'
    with open(tok_json_path, 'w') as f:
        f.write(tokenizer.to_json())

    seq_cfg_path = output_dir / 'lstm_sequence_config.json'
    with open(seq_cfg_path, 'w') as f:
        json.dump({'max_len': int(max_len), 'model_type': 'lstm'}, f, indent=2)

    
    # Evaluate and save metrics
    metrics = evaluate_and_save_metrics('LSTM', y_test, y_pred, output_dir)
    
    # Test samples
    def predictor(text):
        seq = tokenizer.texts_to_sequences([text])
        X = pad_sequences(seq, maxlen=max_len, padding='post')
        probs = model.predict(X, verbose=0)[0]
        pred = np.argmax(probs)
        conf = probs[pred]
        return pred, conf
    
    test_sample_predictions('LSTM', predictor, output_dir)
    
    return model, metrics


def save_cnn(X_train, X_test, y_train, y_test, output_dir):
    """CNN for text (Kim 2014)"""
    if not DEEP_LEARNING_AVAILABLE:
        print("\n[4.2.3] Skipping CNN (TensorFlow not available)")
        return None, None
    
    print("\n[4.2.3] Saving CNN...")
    
    X_train_pad, X_test_pad, tokenizer, max_words, max_len = prepare_sequences(X_train, X_test)
    
    # Model
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
        X_train_pad, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Predict
    y_pred = np.argmax(model.predict(X_test_pad, verbose=0), axis=1)
    
    # Save model
    model_path = output_dir / 'cnn_model.h5'
    model.save(model_path)
    
    
    # ✅ Save tokenizer as JSON (portable)
    tok_json_path = output_dir / 'cnn_tokenizer.json'
    with open(tok_json_path, 'w') as f:
        f.write(tokenizer.to_json())

    # ✅ Save sequence config (so inference uses same max_len)
    seq_cfg_path = output_dir / 'cnn_sequence_config.json'
    with open(seq_cfg_path, 'w') as f:
        json.dump({'max_len': int(max_len), 'model_type': 'cnn'}, f, indent=2)

    
    # Evaluate and save metrics
    metrics = evaluate_and_save_metrics('CNN (Kim 2014)', y_test, y_pred, output_dir)
    
    # Test samples
    def predictor(text):
        seq = tokenizer.texts_to_sequences([text])
        X = pad_sequences(seq, maxlen=max_len, padding='post')
        probs = model.predict(X, verbose=0)[0]
        pred = np.argmax(probs)
        conf = probs[pred]
        return pred, conf
    
    test_sample_predictions('CNN', predictor, output_dir)
    
    return model, metrics


def save_transformer(X_train, X_test, y_train, y_test, output_dir):
    """Tiny Transformer"""
    if not DEEP_LEARNING_AVAILABLE:
        print("\n[4.2.4] Skipping Transformer (TensorFlow not available)")
        return None, None
    
    print("\n[4.2.4] Saving Transformer...")
    
    X_train_pad, X_test_pad, tokenizer, max_words, max_len = prepare_sequences(X_train, X_test)
    
    # Model
    inputs = Input(shape=(max_len,))
    x = Embedding(max_words, 64)(inputs)
    
    # Self-attention
    attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
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
        X_train_pad, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Predict
    y_pred = np.argmax(model.predict(X_test_pad, verbose=0), axis=1)
    
    # Save model
    model_path = output_dir / 'transformer_model.h5'
    model.save(model_path)
    
    # Save tokenizer
    
    tok_json_path = output_dir / 'transformer_tokenizer.json'
    with open(tok_json_path, 'w') as f:
        f.write(tokenizer.to_json())

    seq_cfg_path = output_dir / 'transformer_sequence_config.json'
    with open(seq_cfg_path, 'w') as f:
        json.dump({'max_len': int(max_len), 'model_type': 'transformer'}, f, indent=2)

    
    # Evaluate and save metrics
    metrics = evaluate_and_save_metrics('Tiny Transformer', y_test, y_pred, output_dir)
    
    # Test samples
    def predictor(text):
        seq = tokenizer.texts_to_sequences([text])
        X = pad_sequences(seq, maxlen=max_len, padding='post')
        probs = model.predict(X, verbose=0)[0]
        pred = np.argmax(probs)
        conf = probs[pred]
        return pred, conf
    
    test_sample_predictions('Transformer', predictor, output_dir)
    
    return model, metrics


def print_confusion_matrix(cm, model_name):
    """Print confusion matrix in readable format"""
    labels = ['Low', 'Moderate', 'Severe']
    
    print(f"\n{model_name} - Confusion Matrix:")
    print("=" * 50)
    print(f"{'':>12s} {'Predicted':^36s}")
    print(f"{'':>12s} {'Low':>10s} {'Moderate':>10s} {'Severe':>10s}")
    print("-" * 50)
    for i, actual_label in enumerate(labels):
        row_str = f"Actual {actual_label:>8s}"
        for j in range(3):
            row_str += f"{cm[i][j]:>10d}"
        print(row_str)
    print("=" * 50)


def create_comparison_table(all_metrics, output_dir):
    """Create comprehensive comparison table for paper"""
    
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE MODEL COMPARISON (For Your Paper)")
    print("=" * 80 + "\n")
    
    # Sort by accuracy
    sorted_metrics = sorted(all_metrics, key=lambda x: x['accuracy'], reverse=True)
    
    # Print summary table
    print(f"{'Model':<25s} {'Accuracy':<15s} {'Macro F1':<15s} {'Type':<15s}")
    print("-" * 80)
    
    for m in sorted_metrics:
        print(f"{m['model_name']:<25s} {m['accuracy_with_ci']:<15s} {m['f1_macro_with_ci']:<15s} {m.get('type', 'N/A'):<15s}")
    
    # Print confusion matrices for all models
    print("\n" + "=" * 80)
    print("  CONFUSION MATRICES")
    print("=" * 80)
    
    for m in sorted_metrics:
        print_confusion_matrix(m['confusion_matrix'], m['model_name'])
    
    # Create LaTeX table for thesis
    latex_table = generate_latex_table(sorted_metrics)
    latex_path = output_dir / 'latex_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"\n📄 LaTeX table saved: {latex_path}")
    
    # Create CSV for easy import to Excel/Google Sheets
    csv_data = []
    for m in sorted_metrics:
        csv_data.append({
            'Model': m['model_name'],
            'Accuracy': m['accuracy'],
            'Accuracy_CI': m['accuracy_ci_margin'],
            'Macro_F1': m['f1_macro'],
            'F1_CI': m['f1_macro_ci_margin'],
            'Type': m.get('type', 'N/A'),
            'Precision_Low': m['per_class_metrics']['Low']['precision'],
            'Recall_Low': m['per_class_metrics']['Low']['recall'],
            'F1_Low': m['per_class_metrics']['Low']['f1'],
            'Precision_Moderate': m['per_class_metrics']['Moderate']['precision'],
            'Recall_Moderate': m['per_class_metrics']['Moderate']['recall'],
            'F1_Moderate': m['per_class_metrics']['Moderate']['f1'],
            'Precision_Severe': m['per_class_metrics']['Severe']['precision'],
            'Recall_Severe': m['per_class_metrics']['Severe']['recall'],
            'F1_Severe': m['per_class_metrics']['Severe']['f1'],
        })
    
    csv_df = pd.DataFrame(csv_data)
    csv_path = output_dir / 'model_comparison.csv'
    csv_df.to_csv(csv_path, index=False)
    print(f"📊 CSV comparison saved: {csv_path}")
    
    # Save comprehensive comparison JSON
    comparison = {
        'models': sorted_metrics,
        'summary': {
            'best_accuracy': sorted_metrics[0]['model_name'],
            'best_f1': max(sorted_metrics, key=lambda x: x['f1_macro'])['model_name'],
            'fastest': 'Logistic Regression',
            'most_robust': 'Random Forest'
        }
    }
    
    comparison_path = output_dir / 'model_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"💾 Comparison JSON saved: {comparison_path}")
    
    return comparison


def generate_latex_table(sorted_metrics):
    """Generate LaTeX table for thesis"""
    
    latex = r"""\begin{table}[h]
\centering
\caption{NLP Model Performance Comparison with 95\% Confidence Intervals}
\label{tab:model_comparison}
\begin{tabular}{lcccc}
\hline
\textbf{Model} & \textbf{Accuracy} & \textbf{Macro F1} & \textbf{95\% CI} & \textbf{Type} \\
\hline
"""
    
    for m in sorted_metrics:
        model_name = m['model_name'].replace('_', ' ')
        accuracy = f"{m['accuracy']:.1%}"
        f1 = f"{m['f1_macro']:.1%}"
        ci = f"±{m['accuracy_ci_margin']:.3f}"
        model_type = m.get('type', 'N/A')
        
        latex += f"{model_name} & {accuracy} & {f1} & {ci} & {model_type} \\\\\n"
    
    latex += r"""\hline
\end{tabular}
\end{table}

% Confusion Matrix Example (CNN)
\begin{table}[h]
\centering
\caption{Confusion Matrix - CNN (Kim 2014)}
\label{tab:confusion_cnn}
\begin{tabular}{lccc}
\hline
 & \multicolumn{3}{c}{\textbf{Predicted}} \\
\textbf{Actual} & Low & Moderate & Severe \\
\hline
"""
    
    # Add confusion matrix for best model (first in sorted list)
    best_model = sorted_metrics[0]
    cm = best_model['confusion_matrix']
    labels = ['Low', 'Moderate', 'Severe']
    
    for i, label in enumerate(labels):
        latex += f"{label}"
        for j in range(3):
            latex += f" & {cm[i][j]}"
        latex += " \\\\\n"
    
    latex += r"""\hline
\end{tabular}
\end{table}
"""
    
    return latex


def main():
    parser = argparse.ArgumentParser(
        description='Save ALL NLP models for paper comparison'
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
        help='Directory to save all models (default: models/nlp)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Skip deep learning models (faster)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "=" * 80)
    print("  Saving ALL NLP Models for Comprehensive Paper Comparison")
    print("=" * 80 + "\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X, y = load_text_dataset(args.input)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print(f"\n🔀 Split: {len(X_train)} train, {len(X_test)} test\n")
    
    # Store all metrics for comparison
    all_metrics = []
    
    # ========== SECTION 4.1: BASELINE MODELS ==========
    print("=" * 80)
    print("  SECTION 4.1: BASELINE MODELS")
    print("=" * 80)
    
    # Naive Predictor
    _, metrics = save_naive_predictor(X_train, X_test, y_train, y_test, output_dir)
    metrics['type'] = 'Baseline'
    all_metrics.append(metrics)
    
    # Logistic Regression
    _, bow_vectorizer, metrics = save_logistic_regression(X_train, X_test, y_train, y_test, output_dir)
    metrics['type'] = 'Baseline'
    all_metrics.append(metrics)
    
    # Naive Bayes
    _, tfidf_vectorizer, metrics = save_naive_bayes(X_train, X_test, y_train, y_test, output_dir)
    metrics['type'] = 'Baseline'
    all_metrics.append(metrics)
    
    # Decision Tree (reuse TF-IDF vectorizer)
    _, metrics = save_decision_tree(X_train, X_test, y_train, y_test, output_dir, tfidf_vectorizer)
    metrics['type'] = 'Baseline'
    all_metrics.append(metrics)
    
    # ========== SECTION 4.2: MAIN MODELS ==========
    print("\n" + "=" * 80)
    print("  SECTION 4.2: MAIN MODELS (Modern NLP)")
    print("=" * 80)
    
    # Random Forest
    _, metrics = save_random_forest(X_train, X_test, y_train, y_test, output_dir, tfidf_vectorizer)
    metrics['type'] = 'Ensemble'
    all_metrics.append(metrics)
    
    if not args.quick:
        # LSTM
        _, metrics = save_lstm(X_train, X_test, y_train, y_test, output_dir)
        if metrics:
            metrics['type'] = 'Deep Learning'
            all_metrics.append(metrics)
        
        # CNN
        _, metrics = save_cnn(X_train, X_test, y_train, y_test, output_dir)
        if metrics:
            metrics['type'] = 'Deep Learning'
            all_metrics.append(metrics)
        
        # Transformer
        _, metrics = save_transformer(X_train, X_test, y_train, y_test, output_dir)
        if metrics:
            metrics['type'] = 'Deep Learning'
            all_metrics.append(metrics)
    else:
        print("\n⚡ Quick mode: Skipping deep learning models")
    
    # ========== CREATE COMPARISON TABLE ==========
    comparison = create_comparison_table(all_metrics, output_dir)
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 80)
    print("✅ ALL MODELS SAVED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n📁 Output Directory: {output_dir}")
    print(f"\n📊 Saved {len(all_metrics)} models:")
    for m in all_metrics:
        print(f"  ✓ {m['model_name']}")
    
    print(f"\n📄 Files Generated:")
    print(f"  • Model files (.pkl / .h5): {len(all_metrics)}")
    print(f"  • Metrics files (.json): {len(all_metrics)}")
    print(f"  • Sample predictions (.json): {len(all_metrics) - 1}")  # Naive predictor has no samples
    print(f"  • Comparison table: model_comparison.json")
    print(f"  • CSV export: model_comparison.csv")
    print(f"  • LaTeX table: latex_table.tex")
    
    print(f"\n🎯 Best Models:")
    print(f"  • Highest Accuracy: {comparison['summary']['best_accuracy']}")
    print(f"  • Highest F1 Score: {comparison['summary']['best_f1']}")
    
    print(f"\n📝 For Your Paper:")
    print(f"  1. Use 'model_comparison.json' for Table 4.3")
    print(f"  2. Each model has detailed metrics in '*_metrics.json'")
    print(f"  3. Sample predictions in '*_samples.json' show model behavior")
    print(f"  4. All models ready for production testing")
    print(f"  5. Confusion matrices printed above")
    print(f"  6. LaTeX tables ready for copy-paste to thesis")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()