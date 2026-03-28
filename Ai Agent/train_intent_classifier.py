"""
Intent Classifier Training Script
=================================
Trains a fine-tuned DistilBERT model for intent classification.
Includes comprehensive evaluation metrics and comparison with baseline.
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
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
CONFIG = {
    "model_name": "distilbert-base-uncased",
    "max_length": 64,
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "random_seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
}

print(f"🖥️  Using device: {CONFIG['device']}")

# ============================================
# Dataset Class
# ============================================
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ============================================
# Data Loading & Preprocessing
# ============================================
def load_dataset(filepath):
    """Load and preprocess the intent dataset."""
    print("📂 Loading dataset...")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    texts = []
    labels = []
    intent_names = []
    
    for intent_data in data['intents']:
        intent_name = intent_data['intent']
        intent_names.append(intent_name)
        intent_idx = len(intent_names) - 1
        
        for example in intent_data['examples']:
            texts.append(example)
            labels.append(intent_idx)
    
    print(f"✅ Loaded {len(texts)} examples across {len(intent_names)} intents")
    
    # Create label mappings
    label2id = {name: idx for idx, name in enumerate(intent_names)}
    id2label = {idx: name for idx, name in enumerate(intent_names)}
    
    return texts, labels, label2id, id2label

def split_data(texts, labels, config):
    """Split data into train, validation, and test sets."""
    print("✂️  Splitting dataset...")
    
    # First split: train + val vs test
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels,
        test_size=config['test_split'],
        random_state=config['random_seed'],
        stratify=labels
    )
    
    # Second split: train vs val
    val_ratio = config['val_split'] / (config['train_split'] + config['val_split'])
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels,
        test_size=val_ratio,
        random_state=config['random_seed'],
        stratify=train_val_labels
    )
    
    print(f"   Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

# ============================================
# Model Training
# ============================================
def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy

def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return avg_loss, accuracy, f1, predictions, true_labels

# ============================================
# Visualization & Reporting
# ============================================
def plot_training_history(history, save_path):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'], label='Validation', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train', marker='o')
    axes[1].plot(history['val_acc'], label='Validation', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score plot
    axes[2].plot(history['val_f1'], label='Validation F1', marker='s', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Validation F1 Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Training history saved to {save_path}")

def plot_confusion_matrix(true_labels, predictions, id2label, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=[id2label[i] for i in range(len(id2label))],
        yticklabels=[id2label[i] for i in range(len(id2label))]
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Intent Classification - Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrix saved to {save_path}")

def generate_report(true_labels, predictions, id2label, save_path):
    """Generate detailed classification report."""
    report = classification_report(
        true_labels, 
        predictions, 
        target_names=[id2label[i] for i in range(len(id2label))],
        digits=4
    )
    
    # Calculate overall metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    f1_macro = f1_score(true_labels, predictions, average='macro')
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    
    report_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║            INTENT CLASSIFIER - EVALUATION REPORT                 ║
╚══════════════════════════════════════════════════════════════════╝

📊 OVERALL METRICS
─────────────────────────────────────────────────────────────────────
   Accuracy:           {accuracy:.4f}  ({accuracy*100:.2f}%)
   F1 Score (Weighted): {f1_weighted:.4f}
   F1 Score (Macro):    {f1_macro:.4f}
   Precision:          {precision:.4f}
   Recall:             {recall:.4f}

📋 DETAILED CLASSIFICATION REPORT
─────────────────────────────────────────────────────────────────────
{report}

💡 INTERPRETATION
─────────────────────────────────────────────────────────────────────
   • Accuracy > 90%: Excellent performance
   • F1 > 0.85: Good balance between precision and recall
   • Check confusion matrix for common misclassifications
"""
    
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    print(f"📄 Classification report saved to {save_path}")
    print(report_text)
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall
    }

# ============================================
# Main Training Pipeline
# ============================================
def train_model():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("🚀 INTENT CLASSIFIER TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # Load data
    texts, labels, label2id, id2label = load_dataset("data/intent_dataset.json")
    
    # Split data
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = \
        split_data(texts, labels, CONFIG)
    
    # Initialize tokenizer and model
    print("🤖 Loading DistilBERT model...")
    tokenizer = DistilBertTokenizer.from_pretrained(CONFIG['model_name'])
    model = DistilBertForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    model.to(CONFIG['device'])
    
    # Create datasets
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer, CONFIG['max_length'])
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer, CONFIG['max_length'])
    test_dataset = IntentDataset(test_texts, test_labels, tokenizer, CONFIG['max_length'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    total_steps = len(train_loader) * CONFIG['epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    best_val_f1 = 0
    best_epoch = 0
    
    # Training loop
    print("\n📚 Starting training...\n")
    for epoch in range(CONFIG['epochs']):
        print(f"Epoch {epoch + 1}/{CONFIG['epochs']}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, CONFIG['device'])
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, CONFIG['device'])
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'label2id': label2id,
                'id2label': id2label
            }, 'models/intent_classifier_best.pt')
            print(f"   ✅ New best model saved! (F1: {val_f1:.4f})")
        
        print()
    
    # Load best model for final evaluation
    print("="*60)
    print("📊 FINAL EVALUATION ON TEST SET")
    print("="*60 + "\n")
    
    checkpoint = torch.load('models/intent_classifier_best.pt', map_location=CONFIG['device'], weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_loss, test_acc, test_f1, test_preds, test_true = evaluate(model, test_loader, CONFIG['device'])
    
    print(f"🎯 Test Results (Best Model from Epoch {best_epoch}):")
    print(f"   Test Loss:     {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Test F1 Score: {test_f1:.4f}")
    
    # Generate visualizations and reports
    plot_training_history(history, 'reports/training_history.png')
    plot_confusion_matrix(test_true, test_preds, id2label, 'reports/confusion_matrix.png')
    metrics = generate_report(test_true, test_preds, id2label, 'reports/classification_report.txt')
    
    # Save final model with tokenizer
    print("\n💾 Saving final model...")
    model.save_pretrained('models/intent_classifier')
    tokenizer.save_pretrained('models/intent_classifier')
    
    # Save label mappings
    with open('models/intent_classifier/label_mappings.json', 'w') as f:
        json.dump({'label2id': label2id, 'id2label': id2label}, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📁 Model saved to: models/intent_classifier/")
    print(f"📊 Reports saved to: reports/")
    print(f"\n🏆 Best Performance:")
    print(f"   Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   F1 Score: {metrics['f1_weighted']:.4f}")
    
    return model, tokenizer, label2id, id2label, metrics

# ============================================
# Inference Function
# ============================================
def predict_intent(text, model, tokenizer, id2label, device=CONFIG['device']):
    """Predict intent for a single text input."""
    model.eval()
    
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=CONFIG['max_length'],
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    
    return {
        'intent': id2label[pred_idx],
        'confidence': confidence,
        'all_scores': {id2label[i]: probs[0][i].item() for i in range(len(id2label))}
    }

# ============================================
# Run Training
# ============================================
if __name__ == "__main__":
    model, tokenizer, label2id, id2label, metrics = train_model()
    
    # Test with sample inputs
    print("\n" + "="*60)
    print("🧪 TESTING WITH SAMPLE INPUTS")
    print("="*60 + "\n")
    
    test_inputs = [
        "Schedule a meeting with John at 3pm tomorrow",
        "Send an email to sarah@company.com",
        "What meetings do I have today?",
        "Cancel my 2pm appointment",
        "Remind me to call the doctor at 5pm",
        "What reminders do I have?",
        "Tell Mike that the project is delayed",
        "Hello, how are you?"
    ]
    
    for text in test_inputs:
        result = predict_intent(text, model, tokenizer, id2label)
        print(f"📝 Input: \"{text}\"")
        print(f"   🎯 Intent: {result['intent']} ({result['confidence']*100:.1f}%)")
        print()

