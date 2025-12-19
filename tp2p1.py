"""
PW 2 - Part 1: Data Preparation and Feature Extraction
Master 2: Applied Artificial Intelligence
Module: Advanced Computer Vision

This script implements:
1. Data loading and preprocessing
2. Train/Val/Test split (70/15/15)
3. Feature extraction using pretrained CNNs (VGG16, InceptionV3, ResNet50)
4. Training traditional ML classifiers
5. Comprehensive evaluation and visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create output directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/features', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)

print("=" * 80)
print("PW 2 - Part 1: Data Preparation and Feature Extraction")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n[1] Loading and Preprocessing Data...")

# Dataset paths (adjust these to your dataset location)
DATASET_PATH = 'training/ACV_CNN/intel-image-classification'  # Change this to your path
TRAIN_DIR = os.path.join(DATASET_PATH, 'seg_train', 'seg_train')
TEST_DIR = os.path.join(DATASET_PATH, 'seg_test', 'seg_test')

# Image parameters
IMG_SIZE = (224, 224)  # Using 224x224 for compatibility with pretrained models
BATCH_SIZE = 100

# Classes
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
NUM_CLASSES = len(CLASSES)

def load_images_from_directory(directory, img_size=IMG_SIZE, max_images=None):
    """Load images from directory structure"""
    images = []
    labels = []
    
    for class_name in CLASSES:
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue
            
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if max_images:
            image_files = image_files[:max_images // NUM_CLASSES]
        
        print(f"  Loading {len(image_files)} images from {class_name}...")
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                # Load and preprocess image
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img)
                img_array = img_array / 255.0  # Normalize to [0, 1]
                
                images.append(img_array)
                labels.append(class_name)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels)

# Load all data
print("\nLoading training data...")
X_all, y_all = load_images_from_directory(TRAIN_DIR, max_images=1000)

print(f"\nTotal images loaded: {len(X_all)}")
print(f"Image shape: {X_all[0].shape}")
print(f"Class distribution:\n{pd.Series(y_all).value_counts()}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_all)

# ============================================================================
# 2. TRAIN/VAL/TEST SPLIT (70/15/15)
# ============================================================================

print("\n[2] Splitting data into Train/Val/Test (70/15/15)...")

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
)

# Second split: split temp into 50/50 for val and test (15% each of total)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Training set: {len(X_train)} images")
print(f"Validation set: {len(X_val)} images")
print(f"Test set: {len(X_test)} images")

# Visualize sample images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Sample Images from Dataset', fontsize=16, fontweight='bold')
for i, ax in enumerate(axes.flat):
    if i < len(X_train):
        ax.imshow(X_train[i])
        ax.set_title(f"{label_encoder.inverse_transform([y_train[i]])[0]}")
        ax.axis('off')
plt.tight_layout()
plt.savefig('outputs/plots/sample_images.png', dpi=300, bbox_inches='tight')
print("\n✓ Sample images saved to: outputs/plots/sample_images.png")
plt.close()

# ============================================================================
# 3. FEATURE EXTRACTION
# ============================================================================

print("\n[3] Feature Extraction using Pretrained CNNs...")

def extract_features(model_name, X_train, X_val, X_test):
    """Extract features using a pretrained CNN"""
    print(f"\n  Extracting features using {model_name}...")
    
    # Load pretrained model
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
        preprocess_func = vgg_preprocess
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
        preprocess_func = inception_preprocess
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
        preprocess_func = resnet_preprocess
    
    # Add global average pooling
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    feature_extractor = Model(inputs=base_model.input, outputs=x)
    
    # Freeze all layers
    for layer in feature_extractor.layers:
        layer.trainable = False
    
    print(f"    Model loaded. Feature dimension: {feature_extractor.output_shape[1]}")
    
    # Preprocess and extract features
    def extract_batch(X, batch_size=BATCH_SIZE):
        features = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size].copy()
            batch = batch * 255.0  # Denormalize
            batch = preprocess_func(batch)
            batch_features = feature_extractor.predict(batch, verbose=0)
            features.append(batch_features)
        return np.vstack(features)
    
    train_features = extract_batch(X_train)
    val_features = extract_batch(X_val)
    test_features = extract_batch(X_test)
    
    print(f"    Train features shape: {train_features.shape}")
    print(f"    Val features shape: {val_features.shape}")
    print(f"    Test features shape: {test_features.shape}")
    
    return train_features, val_features, test_features

# Extract features from all models
models = ['VGG16', 'InceptionV3', 'ResNet50']
features_dict = {}

for model_name in models:
    train_feat, val_feat, test_feat = extract_features(
        model_name, X_train, X_val, X_test
    )
    features_dict[model_name] = {
        'train': train_feat,
        'val': val_feat,
        'test': test_feat
    }
    
    # Save features to CSV
    # Combine val and test for evaluation
    combined_features = np.vstack([val_feat, test_feat])
    combined_labels = np.concatenate([y_val, y_test])
    
    df = pd.DataFrame(combined_features)
    df['label'] = combined_labels
    csv_path = f'outputs/features/{model_name}_features.csv'
    df.to_csv(csv_path, index=False)
    print(f"    ✓ Features saved to: {csv_path}")

# ============================================================================
# 4. TRAIN ML CLASSIFIERS
# ============================================================================

print("\n[4] Training ML Classifiers...")

classifiers = {
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
}

results = []

for model_name in models:
    print(f"\n  Using {model_name} features:")
    
    train_feat = features_dict[model_name]['train']
    val_feat = features_dict[model_name]['val']
    test_feat = features_dict[model_name]['test']
    
    # Combine val and test for evaluation
    eval_feat = np.vstack([val_feat, test_feat])
    eval_labels = np.concatenate([y_val, y_test])
    
    for clf_name, clf in classifiers.items():
        print(f"    Training {clf_name}...")
        
        # Train
        clf.fit(train_feat, y_train)
        
        # Predict
        y_pred = clf.predict(eval_feat)
        y_pred_proba = clf.predict_proba(eval_feat) if hasattr(clf, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(eval_labels, y_pred)
        precision = precision_score(eval_labels, y_pred, average='weighted', zero_division=0)
        recall = recall_score(eval_labels, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(eval_labels, y_pred, average='weighted', zero_division=0)
        
        results.append({
            'CNN Model': model_name,
            'Classifier': clf_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        print(f"      Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f}")

# ============================================================================
# 5. VISUALIZATION AND EVALUATION
# ============================================================================

print("\n[5] Generating Visualizations...")

# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv('outputs/results/ml_classifier_results.csv', index=False)
print(f"\n✓ Results saved to: outputs/results/ml_classifier_results.csv")

# Plot 1: Performance Comparison Bar Chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('ML Classifiers Performance Comparison', fontsize=16, fontweight='bold')

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    pivot_df = results_df.pivot(index='Classifier', columns='CNN Model', values=metric)
    pivot_df.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric, fontsize=12)
    ax.set_xlabel('Classifier', fontsize=12)
    ax.legend(title='CNN Model', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)

plt.tight_layout()
plt.savefig('outputs/plots/performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Performance comparison saved to: outputs/plots/performance_comparison.png")
plt.close()

# Plot 2: Best model confusion matrices
print("\nGenerating confusion matrices for best performers...")

# Find best combination
best_combo = results_df.loc[results_df['Accuracy'].idxmax()]
best_cnn = best_combo['CNN Model']
best_clf_name = best_combo['Classifier']

print(f"\nBest combination: {best_cnn} + {best_clf_name}")
print(f"Accuracy: {best_combo['Accuracy']:.4f}")

# Train best model and get predictions
train_feat = features_dict[best_cnn]['train']
eval_feat = np.vstack([features_dict[best_cnn]['val'], features_dict[best_cnn]['test']])
eval_labels = np.concatenate([y_val, y_test])

best_clf = classifiers[best_clf_name]
best_clf.fit(train_feat, y_train)
y_pred = best_clf.predict(eval_feat)

# Confusion Matrix
cm = confusion_matrix(eval_labels, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASSES, yticklabels=CLASSES,
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix: {best_cnn} + {best_clf_name}\nAccuracy: {best_combo["Accuracy"]:.4f}', 
          fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/plots/confusion_matrix_best.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved to: outputs/plots/confusion_matrix_best.png")
plt.close()

# Classification Report
print("\nClassification Report (Best Model):")
print("=" * 80)
print(classification_report(eval_labels, y_pred, target_names=CLASSES))

# Save classification report
report = classification_report(eval_labels, y_pred, target_names=CLASSES, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('outputs/results/classification_report_best.csv')

# Plot 3: ROC Curves (for multi-class)
print("\nGenerating ROC curves...")

if hasattr(best_clf, 'predict_proba'):
    y_score = best_clf.predict_proba(eval_feat)
    y_test_bin = label_binarize(eval_labels, classes=range(NUM_CLASSES))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'ROC Curves: {best_cnn} + {best_clf_name}', fontsize=16, fontweight='bold')
    
    for i, class_name in enumerate(CLASSES):
        ax = axes[i // 3, i % 3]
        
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'Class: {class_name}', fontsize=12, fontweight='bold')
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/roc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ ROC curves saved to: outputs/plots/roc_curves.png")
    plt.close()

# Plot 4: Overall comparison heatmap
print("\nGenerating overall comparison heatmap...")

plt.figure(figsize=(14, 10))
for idx, metric in enumerate(['Accuracy', 'F1-Score']):
    plt.subplot(2, 1, idx + 1)
    pivot_df = results_df.pivot(index='CNN Model', columns='Classifier', values=metric)
    sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='YlGnBu', 
                cbar_kws={'label': metric}, vmin=0, vmax=1)
    plt.title(f'{metric} Heatmap: CNN Models vs Classifiers', fontsize=14, fontweight='bold')
    plt.xlabel('Classifier', fontsize=12)
    plt.ylabel('CNN Model', fontsize=12)

plt.tight_layout()
plt.savefig('outputs/plots/performance_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Performance heatmap saved to: outputs/plots/performance_heatmap.png")
plt.close()

# ============================================================================
# 6. SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nDataset: Intel Image Classification")
print(f"Classes: {NUM_CLASSES} ({', '.join(CLASSES)})")
print(f"Total images: {len(X_all)}")
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"\nBest Combination: {best_cnn} + {best_clf_name}")
print(f"  Accuracy:  {best_combo['Accuracy']:.4f}")
print(f"  Precision: {best_combo['Precision']:.4f}")
print(f"  Recall:    {best_combo['Recall']:.4f}")
print(f"  F1-Score:  {best_combo['F1-Score']:.4f}")

print("\n✓ All outputs saved to 'outputs/' directory")
print("  - Plots: outputs/plots/")
print("  - Features: outputs/features/")
print("  - Results: outputs/results/")
print("\n" + "=" * 80)
print("Part 1 Complete!")
print("=" * 80)