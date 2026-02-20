import torch
import numpy as np
import os
from sklearn.metrics import (
    roc_curve, auc,
    f1_score, confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import torch.distributed as dist

import torch
import numpy as np
import os
from sklearn.metrics import (
    roc_curve, auc,
    f1_score, confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

def plot_roc_and_f1(ensemble_model, test_loader, device, save_dir, model_names, is_main=True):
    if not is_main:
        return
       
    ensemble_model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
   
    print("="*70)
    print("Generating ROC and F1-Score Plots (Fake = Positive Class)...")
    print("="*70)
   
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Collecting predictions"):
            images, labels = images.to(device), labels.to(device).float()
           
            try:
                outputs, weights, _, _ = ensemble_model(images, return_details=True)
            except Exception as e:
                print(f"Error in forward pass: {e}")
                continue
           
            probs = torch.sigmoid(outputs).cpu().numpy().ravel()
            pred = (outputs > 0).long().cpu().numpy().ravel()
           
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)
            all_preds.append(pred)
   
    all_labels = np.concatenate(all_labels).ravel()
    all_probs = np.concatenate(all_probs).ravel()
    all_preds = np.concatenate(all_preds).ravel()
   
    # ====================== اصلاح اصلی ======================
    # حالا مثبت = فیک (لیبل ۰)
    fpr, tpr, _ = roc_curve(all_labels, all_probs, pos_label=0)   # ← اینجا تغییر کرد
    roc_auc = auc(fpr, tpr)
   
    precision, recall, _ = precision_recall_curve(all_labels, all_probs, pos_label=0)  # ← اینجا تغییر کرد
    pr_auc = average_precision_score(all_labels, all_probs, pos_label=0)
   
    f1 = f1_score(all_labels, all_preds, pos_label=0)  # F1 برای فیک
    cm = confusion_matrix(all_labels, all_preds)
   
    tn, fp, fn, tp = cm.ravel()   # tn=Real→Real, fp=Real→Fake, fn=Fake→Real, tp=Fake→Fake
   
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
   
    if is_main:
        print(f"\n{'='*70}")
        print(f"METRICS (Pos Label = Fake)")
        print(f"{'='*70}")
        print(f"ROC AUC (Fake): {roc_auc:.4f}")
        print(f"PR AUC  (Fake): {pr_auc:.4f}")
        print(f"F1 Score (Fake): {f1:.4f}")
        print(f"Precision (Fake): {precision_val:.4f}")
        print(f"Recall (Fake): {recall_val:.4f}")
        print(f"\nConfusion Matrix (Fake = Positive):")
        print(f"                  Pred Real    Pred Fake")
        print(f"Actual Real     {tn:<10}     {fp:<10}")
        print(f"Actual Fake     {fn:<10}     {tp:<10}")
        print(f"{'='*70}")
       
        # ====================== رسم نمودارها ======================
        plt.figure(figsize=(18, 6))
       
        # ROC Curve
        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity for Fake)')
        plt.title('ROC Curve (Fake as Positive Class)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
       
        # Precision-Recall Curve
        plt.subplot(1, 3, 2)
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensitivity for Fake)')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Fake as Positive)')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
       
        # Confusion Matrix
        plt.subplot(1, 3, 3)
        cm_display = np.array([[tn, fp], [fn, tp]])  # ردیف: Real / Fake
        ax = plt.gca()
        sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred Real', 'Pred Fake'],
                    yticklabels=['Actual Real', 'Actual Fake'], ax=ax)
        plt.title('Confusion Matrix (Fake = Positive)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
       
        model_name_str = " + ".join(model_names)
        plt.suptitle(f"Ensemble Performance - Fake as Positive Class\n({model_name_str})", 
                     fontsize=16, y=1.02)
       
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'roc_f1_analysis_fake_positive.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        plt.close()
       
        # Classification Report
        print("\nClassification Report (Fake=0, Real=1):")
        target_names = ['Fake', 'Real']
        report = classification_report(all_labels, all_preds, target_names=target_names, digits=4, zero_division=0)
        print(report)
       
        # ذخیره متریک‌ها
        metrics = {
            'roc_auc_fake': float(roc_auc),
            'pr_auc_fake': float(pr_auc),
            'f1_fake': float(f1),
            'precision_fake': float(precision_val),
            'recall_fake': float(recall_val),
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            }
        }
        metrics_path = os.path.join(save_dir, 'roc_f1_metrics_fake_positive.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to: {metrics_path}")
        return metrics
