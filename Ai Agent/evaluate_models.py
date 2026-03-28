"""
Model Evaluation & Comparison Script
=====================================
Compares the fine-tuned intent classifier with the baseline zero-shot model.
Generates comprehensive evaluation metrics and visualizations.
"""

import json
import os
import time
import numpy as np
import torch
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    pipeline
)
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Configuration
# ============================================
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Zero-shot candidate labels (matching our intents)
ZERO_SHOT_LABELS = [
    "Schedule or book a meeting or event",
    "Send an email to someone",
    "Check or show my calendar or meetings",
    "Cancel or delete a meeting",
    "Add or set a reminder or task",
    "Check or show my reminders or tasks",
    "Notify, inform, or message someone about an update",
    "General question or small talk"
]

# Mapping from zero-shot labels to our intent names
ZEROSHOT_TO_INTENT = {
    "Schedule or book a meeting or event": "schedule_meeting",
    "Send an email to someone": "send_email",
    "Check or show my calendar or meetings": "check_calendar",
    "Cancel or delete a meeting": "cancel_meeting",
    "Add or set a reminder or task": "add_reminder",
    "Check or show my reminders or tasks": "check_reminders",
    "Notify, inform, or message someone about an update": "send_message",
    "General question or small talk": "general_query"
}

# ============================================
# Load Models
# ============================================
def load_fine_tuned_model():
    """Load the fine-tuned intent classifier."""
    print("📦 Loading fine-tuned model...")
    
    model_path = "models/intent_classifier"
    
    if not os.path.exists(model_path):
        print("⚠️  Fine-tuned model not found. Please run train_intent_classifier.py first.")
        return None, None, None, None
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    
    # Load label mappings
    with open(os.path.join(model_path, "label_mappings.json"), 'r') as f:
        mappings = json.load(f)
    
    label2id = mappings['label2id']
    id2label = {int(k): v for k, v in mappings['id2label'].items()}
    
    print("✅ Fine-tuned model loaded")
    return model, tokenizer, label2id, id2label

def load_zero_shot_model():
    """Load the baseline zero-shot classifier."""
    print("📦 Loading zero-shot model...")
    
    classifier = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
        device=-1  # CPU
    )
    
    print("✅ Zero-shot model loaded")
    return classifier

# ============================================
# Prediction Functions
# ============================================
def predict_fine_tuned(text, model, tokenizer, id2label):
    """Predict using fine-tuned model."""
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=64,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    
    return id2label[pred_idx], confidence

def predict_zero_shot(text, classifier):
    """Predict using zero-shot model."""
    result = classifier(text, ZERO_SHOT_LABELS)
    top_label = result['labels'][0]
    confidence = result['scores'][0]
    
    return ZEROSHOT_TO_INTENT.get(top_label, "general_query"), confidence

# ============================================
# Evaluation Functions
# ============================================
def load_test_data():
    """Load test data from the dataset."""
    print("📂 Loading test data...")
    
    with open("data/intent_dataset.json", 'r') as f:
        data = json.load(f)
    
    texts = []
    labels = []
    intent_names = []
    
    for intent_data in data['intents']:
        intent_name = intent_data['intent']
        if intent_name not in intent_names:
            intent_names.append(intent_name)
        
        for example in intent_data['examples']:
            texts.append(example)
            labels.append(intent_name)
    
    print(f"✅ Loaded {len(texts)} test examples")
    return texts, labels, intent_names

def evaluate_model(texts, true_labels, predict_fn, model_name):
    """Evaluate a model and return metrics."""
    print(f"\n🔄 Evaluating {model_name}...")
    
    predictions = []
    confidences = []
    inference_times = []
    
    for text in tqdm(texts, desc=f"Predicting ({model_name})"):
        start_time = time.time()
        pred, conf = predict_fn(text)
        inference_times.append(time.time() - start_time)
        
        predictions.append(pred)
        confidences.append(conf)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    f1_macro = f1_score(true_labels, predictions, average='macro')
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    avg_time = np.mean(inference_times) * 1000  # Convert to ms
    avg_confidence = np.mean(confidences)
    
    return {
        'predictions': predictions,
        'confidences': confidences,
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall,
        'avg_inference_time_ms': avg_time,
        'avg_confidence': avg_confidence
    }

# ============================================
# Visualization Functions
# ============================================
def plot_comparison_metrics(ft_metrics, zs_metrics, save_path):
    """Plot comparison bar chart of metrics."""
    metrics = ['Accuracy', 'F1 (Weighted)', 'F1 (Macro)', 'Precision', 'Recall']
    ft_values = [
        ft_metrics['accuracy'],
        ft_metrics['f1_weighted'],
        ft_metrics['f1_macro'],
        ft_metrics['precision'],
        ft_metrics['recall']
    ]
    zs_values = [
        zs_metrics['accuracy'],
        zs_metrics['f1_weighted'],
        zs_metrics['f1_macro'],
        zs_metrics['precision'],
        zs_metrics['recall']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, ft_values, width, label='Fine-tuned Model', color='#2ecc71')
    bars2 = ax.bar(x + width/2, zs_values, width, label='Zero-shot Baseline', color='#3498db')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison: Fine-tuned vs Zero-shot')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Comparison chart saved to {save_path}")

def plot_inference_time_comparison(ft_metrics, zs_metrics, save_path):
    """Plot inference time comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = ['Fine-tuned\nModel', 'Zero-shot\nBaseline']
    times = [ft_metrics['avg_inference_time_ms'], zs_metrics['avg_inference_time_ms']]
    colors = ['#2ecc71', '#3498db']
    
    bars = ax.bar(models, times, color=colors)
    ax.set_ylabel('Average Inference Time (ms)')
    ax.set_title('Inference Speed Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, time in zip(bars, times):
        ax.annotate(f'{time:.2f} ms',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add speedup annotation
    speedup = zs_metrics['avg_inference_time_ms'] / ft_metrics['avg_inference_time_ms']
    if speedup > 1:
        ax.text(0.5, 0.95, f'Fine-tuned is {speedup:.1f}x faster!',
                transform=ax.transAxes, ha='center', fontsize=12,
                color='#27ae60', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Inference time chart saved to {save_path}")

def plot_dual_confusion_matrix(true_labels, ft_preds, zs_preds, intent_names, save_path):
    """Plot confusion matrices for both models side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Fine-tuned confusion matrix
    cm_ft = confusion_matrix(true_labels, ft_preds, labels=intent_names)
    sns.heatmap(cm_ft, annot=True, fmt='d', cmap='Greens',
                xticklabels=intent_names, yticklabels=intent_names, ax=axes[0])
    axes[0].set_title('Fine-tuned Model', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Zero-shot confusion matrix
    cm_zs = confusion_matrix(true_labels, zs_preds, labels=intent_names)
    sns.heatmap(cm_zs, annot=True, fmt='d', cmap='Blues',
                xticklabels=intent_names, yticklabels=intent_names, ax=axes[1])
    axes[1].set_title('Zero-shot Baseline', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrices saved to {save_path}")

def plot_confidence_distribution(ft_confs, zs_confs, save_path):
    """Plot confidence score distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Fine-tuned confidence distribution
    axes[0].hist(ft_confs, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(ft_confs), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(ft_confs):.3f}')
    axes[0].set_xlabel('Confidence Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Fine-tuned Model\nConfidence Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Zero-shot confidence distribution
    axes[1].hist(zs_confs, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(zs_confs), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(zs_confs):.3f}')
    axes[1].set_xlabel('Confidence Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Zero-shot Baseline\nConfidence Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Confidence distribution saved to {save_path}")

# ============================================
# Report Generation
# ============================================
def generate_comparison_report(ft_metrics, zs_metrics, save_path):
    """Generate detailed comparison report."""
    
    improvement = {
        'accuracy': (ft_metrics['accuracy'] - zs_metrics['accuracy']) * 100,
        'f1_weighted': (ft_metrics['f1_weighted'] - zs_metrics['f1_weighted']) * 100,
        'f1_macro': (ft_metrics['f1_macro'] - zs_metrics['f1_macro']) * 100,
        'precision': (ft_metrics['precision'] - zs_metrics['precision']) * 100,
        'recall': (ft_metrics['recall'] - zs_metrics['recall']) * 100,
    }
    
    speedup = zs_metrics['avg_inference_time_ms'] / ft_metrics['avg_inference_time_ms']
    
    report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MODEL COMPARISON REPORT                                   ║
║                  Fine-tuned vs Zero-shot Baseline                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│                           PERFORMANCE METRICS                                │
├─────────────────────┬──────────────────┬──────────────────┬─────────────────┤
│ Metric              │ Fine-tuned       │ Zero-shot        │ Improvement     │
├─────────────────────┼──────────────────┼──────────────────┼─────────────────┤
│ Accuracy            │ {ft_metrics['accuracy']:.4f}           │ {zs_metrics['accuracy']:.4f}           │ {improvement['accuracy']:+.2f}%          │
│ F1 (Weighted)       │ {ft_metrics['f1_weighted']:.4f}           │ {zs_metrics['f1_weighted']:.4f}           │ {improvement['f1_weighted']:+.2f}%          │
│ F1 (Macro)          │ {ft_metrics['f1_macro']:.4f}           │ {zs_metrics['f1_macro']:.4f}           │ {improvement['f1_macro']:+.2f}%          │
│ Precision           │ {ft_metrics['precision']:.4f}           │ {zs_metrics['precision']:.4f}           │ {improvement['precision']:+.2f}%          │
│ Recall              │ {ft_metrics['recall']:.4f}           │ {zs_metrics['recall']:.4f}           │ {improvement['recall']:+.2f}%          │
└─────────────────────┴──────────────────┴──────────────────┴─────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE SPEED                                    │
├─────────────────────┬──────────────────┬──────────────────┬─────────────────┤
│ Model               │ Avg Time (ms)    │ Speedup          │                 │
├─────────────────────┼──────────────────┼──────────────────┼─────────────────┤
│ Fine-tuned          │ {ft_metrics['avg_inference_time_ms']:>8.2f}         │ {speedup:.2f}x faster     │                 │
│ Zero-shot           │ {zs_metrics['avg_inference_time_ms']:>8.2f}         │ (baseline)       │                 │
└─────────────────────┴──────────────────┴──────────────────┴─────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                           CONFIDENCE SCORES                                  │
├─────────────────────┬──────────────────┬──────────────────┬─────────────────┤
│ Model               │ Avg Confidence   │                  │                 │
├─────────────────────┼──────────────────┼──────────────────┼─────────────────┤
│ Fine-tuned          │ {ft_metrics['avg_confidence']:.4f}           │                  │                 │
│ Zero-shot           │ {zs_metrics['avg_confidence']:.4f}           │                  │                 │
└─────────────────────┴──────────────────┴──────────────────┴─────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                              SUMMARY                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  🏆 WINNER: {"Fine-tuned Model" if ft_metrics['accuracy'] > zs_metrics['accuracy'] else "Zero-shot Baseline"}                                             │
│                                                                              │
│  Key Findings:                                                               │
│  • Accuracy improved by {improvement['accuracy']:+.2f}%                                           │
│  • F1 Score improved by {improvement['f1_weighted']:+.2f}%                                           │
│  • Inference is {speedup:.2f}x faster with fine-tuned model                            │
│  • Fine-tuned model shows {"higher" if ft_metrics['avg_confidence'] > zs_metrics['avg_confidence'] else "lower"} average confidence                              │
│                                                                              │
│  Recommendation:                                                             │
│  {"✅ Use fine-tuned model for production - better accuracy & speed" if ft_metrics['accuracy'] > zs_metrics['accuracy'] else "⚠️ Consider more training data to improve fine-tuned model"}          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"📄 Comparison report saved to {save_path}")
    print(report)
    
    return improvement, speedup

# ============================================
# Main Comparison Pipeline
# ============================================
def run_comparison():
    """Run the full model comparison pipeline."""
    print("\n" + "="*70)
    print("🔬 MODEL COMPARISON: Fine-tuned vs Zero-shot Baseline")
    print("="*70 + "\n")
    
    os.makedirs("reports", exist_ok=True)
    
    # Load models
    ft_model, ft_tokenizer, ft_label2id, ft_id2label = load_fine_tuned_model()
    zs_classifier = load_zero_shot_model()
    
    if ft_model is None:
        print("❌ Cannot run comparison without fine-tuned model.")
        print("   Please run: python train_intent_classifier.py")
        return
    
    # Load test data
    texts, true_labels, intent_names = load_test_data()
    
    # Create prediction functions
    ft_predict = lambda text: predict_fine_tuned(text, ft_model, ft_tokenizer, ft_id2label)
    zs_predict = lambda text: predict_zero_shot(text, zs_classifier)
    
    # Evaluate both models
    ft_metrics = evaluate_model(texts, true_labels, ft_predict, "Fine-tuned")
    zs_metrics = evaluate_model(texts, true_labels, zs_predict, "Zero-shot")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    
    plot_comparison_metrics(
        ft_metrics, zs_metrics, 
        'reports/model_comparison_metrics.png'
    )
    
    plot_inference_time_comparison(
        ft_metrics, zs_metrics,
        'reports/inference_time_comparison.png'
    )
    
    plot_dual_confusion_matrix(
        true_labels, ft_metrics['predictions'], zs_metrics['predictions'],
        intent_names, 'reports/confusion_matrices_comparison.png'
    )
    
    plot_confidence_distribution(
        ft_metrics['confidences'], zs_metrics['confidences'],
        'reports/confidence_distribution.png'
    )
    
    # Generate report
    generate_comparison_report(
        ft_metrics, zs_metrics,
        'reports/model_comparison_report.txt'
    )
    
    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE!")
    print("="*70)
    print("\n📁 All reports saved to: reports/")
    print("\nGenerated files:")
    print("  • model_comparison_metrics.png")
    print("  • inference_time_comparison.png")
    print("  • confusion_matrices_comparison.png")
    print("  • confidence_distribution.png")
    print("  • model_comparison_report.txt")

# ============================================
# Run
# ============================================
if __name__ == "__main__":
    run_comparison()

