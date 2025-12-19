# Set the number of classes for sentiment classification (neutral, positive, negative)
NUM_CLASSES = 3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            confusion_matrix, classification_report, roc_curve, auc)
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from itertools import cycle

# ==================================================
# DISABLE XLA (prevent ptxas errors)
# ==================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit="0"'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.config.run_functions_eagerly(True)

# Verify GPU
print("=" * 60)
print("GPU SETUP CHECK")
print("=" * 60)
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU Available: {gpus}")
print(f"GPU Count: {len(gpus)}")
if gpus:
    print(f"Using: {gpus[0].name}")
    # Data preprocessing parameters
    MAX_SEQUENCE_LENGTH = 100
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 200
    BATCH_SIZE = 200
    NUM_CLASSES = 3  # sentiment (neutral, positive, negative)

    # Prepare text data preprocessing
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE)


    def build_text_cnn_model():
        inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH,))
        
        # Embedding layer
        x = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)(inputs)
        
        # Parallel convolution layers with different kernel sizes
        conv1 = layers.Conv1D(128, 3, activation='relu')(x)
        conv2 = layers.Conv1D(128, 4, activation='relu')(x)
        conv3 = layers.Conv1D(128, 5, activation='relu')(x)
        
        # Max pooling
        pool1 = layers.GlobalMaxPooling1D()(conv1)
        pool2 = layers.GlobalMaxPooling1D()(conv2)
        pool3 = layers.GlobalMaxPooling1D()(conv3)
        
        # Concatenate pooled features
        concat = layers.Concatenate()([pool1, pool2, pool3])
        
        # Dense layers
        x = layers.Dropout(0.5)(concat)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    # Create and compile model
    model = build_text_cnn_model()
    model.compile(
        optimizer=AdamW(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # Data loading and preprocessing
    data_path = 'training/data/sentimental_chat/chat_dataset.csv' 
    
    # Read and preprocess text data from CSV
    # Load data
    df = pd.read_csv(data_path)
    texts = df['message'].values
    label_map = {'neutral': 0, 'positive': 1, 'negative': 2}
    labels = np.array([label_map[label] for label in df['sentiment'].values])

    # Tokenize texts
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad sequences to ensure uniform length
    x_data = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, 
        maxlen=MAX_SEQUENCE_LENGTH,
        padding='post'
    )

    # Convert labels to categorical
    y_data = tf.keras.utils.to_categorical(labels, NUM_CLASSES)

    # Split data into training, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        x_data, 
        y_data, 
        test_size=0.15, 
        random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, 
        y_temp, 
        test_size=0.176,  # 0.176 * 0.85 ≈ 0.15 of total
        random_state=42
    )

    print(f"\nDataset Split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Create train and validation generators
    train_generator = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
    validation_generator = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )

    # Train the model
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr]
    )

    # ==================================================
    # MODEL TESTING AND EVALUATION
    # ==================================================
    print("\n" + "=" * 60)
    print("MODEL TESTING AND EVALUATION")
    print("=" * 60)

    # Predict on test set
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\nTest Set Performance:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Detailed classification report
    class_names = ['Neutral', 'Positive', 'Negative']
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ==================================================
    # VISUALIZATION PLOTS
    # ==================================================
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Plot 1: Training History (Accuracy and Loss)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots/training_history.png")
    plt.close()

    # Plot 2: Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots/confusion_matrix.png")
    plt.close()

    # Plot 3: ROC Curve (One-vs-Rest for multiclass)
    y_test_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green'])
    
    for i, color, class_name in zip(range(NUM_CLASSES), colors, class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/roc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots/roc_curves.png")
    plt.close()

    # Plot 4: Per-Class Metrics Bar Chart
    metrics_per_class = []
    for i, class_name in enumerate(class_names):
        # Create binary labels: current class vs all others
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        if np.sum(y_true_binary) > 0:
            acc = accuracy_score(y_true_binary, y_pred_binary)
            prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1_class = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            metrics_per_class.append([acc, prec, rec, f1_class])
        else:
            metrics_per_class.append([0, 0, 0, 0])
    
    metrics_per_class = np.array(metrics_per_class)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(class_names))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, metrics_per_class[:, 0], width, label='Accuracy', color='skyblue')
    bars2 = ax.bar(x - 0.5*width, metrics_per_class[:, 1], width, label='Precision', color='lightcoral')
    bars3 = ax.bar(x + 0.5*width, metrics_per_class[:, 2], width, label='Recall', color='lightgreen')
    bars4 = ax.bar(x + 1.5*width, metrics_per_class[:, 3], width, label='F1-Score', color='plum')
    
    ax.set_xlabel('Sentiment Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/per_class_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots/per_class_metrics.png")
    plt.close()

    # ==================================================
    # SAVE MODEL AND TOKENIZER
    # ==================================================
    print("\n" + "=" * 60)
    print("SAVING MODEL AND TOKENIZER")
    print("=" * 60)
    
    model_save_path = 'sentiment_cnn_model.h5'
    model.save(model_save_path)
    print(f"✓ Model saved to {model_save_path}")

    tokenizer_save_path = 'tokenizer.json'
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_save_path, 'w') as f:
        f.write(tokenizer_json)
    print(f"✓ Tokenizer saved to {tokenizer_save_path}")

    # Save metrics summary
    metrics_summary = {
        'Test Accuracy': accuracy,
        'Test Precision': precision,
        'Test Recall': recall,
        'Test F1-Score': f1,
        'Training Samples': len(X_train),
        'Validation Samples': len(X_val),
        'Test Samples': len(X_test),
        'Total Epochs': len(history.history['loss'])
    }
    
    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv('plots/metrics_summary.csv', index=False)
    print(f"✓ Metrics summary saved to plots/metrics_summary.csv")

    print("\n" + "=" * 60)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • sentiment_cnn_model.h5 - Trained model")
    print("  • tokenizer.json - Text tokenizer")
    print("  • plots/training_history.png - Training curves")
    print("  • plots/confusion_matrix.png - Confusion matrix")
    print("  • plots/roc_curves.png - ROC curves")
    print("  • plots/per_class_metrics.png - Per-class metrics")
    print("  • plots/metrics_summary.csv - Performance metrics")