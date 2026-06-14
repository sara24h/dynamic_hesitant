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

def plot_roc_and_f1(y_true, y_score, save_dir, model_names, is_main=True, ensemble_type="Ensemble"):
    """
    رسم منحنی ROC و محاسبه F1-Score
    از داده‌های از پیش محاسبه‌شده استفاده می‌کند (نه اجرای مجدد مدل)
    
    Args:
        y_true: لیبل‌های واقعی (array-like)
        y_score: احتمال‌های پیش‌بینی (array-like)
        save_dir: مسیر ذخیره تصاویر
        model_names: لیست نام مدل‌ها
        is_main: آیا پردازش اصلی است
        ensemble_type: نوع انسامبل (مثلاً "Fuzzy Hesitant", "Simple Averaging", "Majority Voting" و غیره)
    """
    if not is_main:
        return

    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    y_pred = (y_score > 0.5).astype(int)

    # محاسبه ROC
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # محاسبه PR
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=1)
    pr_auc = average_precision_score(y_true, y_score, pos_label=1)

    # متریک‌ها
    f1 = f1_score(y_true, y_pred, pos_label=1)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"METRICS (Pos Label=Real)")
    print(f"{'='*70}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision_val:.4f}")
    print(f"Recall: {recall_val:.4f}")
    print(f"Confusion Matrix:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")
    print(f"{'='*70}")
    
    # رسم نمودارها
    plt.figure(figsize=(18, 6))
    
    # نمودار 1: ROC Curve
    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # نمودار 2: Precision-Recall Curve
    plt.subplot(1, 3, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    # نمودار 3: Confusion Matrix (Heatmap)
    plt.subplot(1, 3, 3)
    cm_display = np.array([[tp, fn], [fp, tn]])
    
    ax = plt.gca()
    sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'], ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # تنظیمات کلی نمودار
    model_name_str = " + ".join(model_names)
    full_title = f"{ensemble_type} Ensemble Performance ({model_name_str})"
    plt.suptitle(full_title, fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    # ذخیره نمودار
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'roc_f1_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()
    
    # چاپ گزارش متنی کلاس‌بندی
    print("\nClassification Report:")
    target_names = ['Fake', 'Real']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)
    print(report)
    
    # ذخیره متریک‌ها در فایل JSON
    metrics = {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'f1_score': float(f1),
        'precision': float(precision_val),
        'recall': float(recall_val),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        }
    }
    
    metrics_path = os.path.join(save_dir, 'roc_f1_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")

    return metrics
