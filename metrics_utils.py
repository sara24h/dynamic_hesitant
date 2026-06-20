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
import json

def plot_roc_and_f1(y_true, y_score, save_dir, model_names, is_main=True):
    """
    نسخه DDP-Safe این تابع.
    به جای گرفتن test_loader، مستقیماً لیست‌های کامل شده را می‌گیرد.
    """
    if not is_main:
        return

    # تبدیل لیست‌ها به آرایه‌های نامپای
    all_labels = np.array(y_true)
    all_probs = np.array(y_score)
    all_preds = (all_probs > 0.5).astype(int)

    print("="*70)
    print("Generating ROC and F1-Score Plots...")
    print("="*70)

    # 2. محاسبه ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    # 3. محاسبه Precision-Recall
    precision, recall, _ = precision_recall_curve(all_labels, all_probs, pos_label=1)
    pr_auc = average_precision_score(all_labels, all_probs, pos_label=1)
    
    # 4. محاسبه متریک‌های دقیق
    f1 = f1_score(all_labels, all_preds, pos_label=1)
    cm = confusion_matrix(all_labels, all_preds)
    
    # متریک‌های دقیق
    tn, fp, fn, tp = cm.ravel()
    
    # محاسبه Precision و Recall از CM
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
    
    # 5. رسم نمودارها
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
    cm_display = np.array([[tp, fn], [fp, tn]]) # Row: Actual Real, Fake | Col: Pred Real, Fake
    
    ax = plt.gca()
    sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'], ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # تنظیمات کلی نمودار
    model_name_str = " + ".join(model_names)
    full_title = f"Stacking Ensemble Performance ({model_name_str})"
    plt.suptitle(full_title, fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    # ذخیره نمودار
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'roc_f1_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {save_path}")
    plt.close()
    
    # 6. چاپ گزارش متنی کلاس‌بندی
    print("\nClassification Report:")
    target_names = ['Fake', 'Real']
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4, zero_division=0)
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
    print(f"✅ Metrics saved to: {metrics_path}")

    return metrics
