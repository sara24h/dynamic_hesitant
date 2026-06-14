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

def plot_roc_and_f1(ensemble_model, test_loader, device, save_dir, model_names, is_main=True):
    """
    تابع رسم منحنی ROC و محاسبه F1-Score
    
    Args:
        ensemble_model: مدل انسامبل (فازی یا Majority Voting)
        test_loader: دیتالودر تست
        device: دستگاه (cuda/cpu)
        save_dir: مسیر ذخیره تصاویر
        model_names: لیست نام مدل‌ها (برای نمایش در عنوان)
    """
    if not is_main:
        return
        
    ensemble_model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    
    # 1. جمع‌آوری Probabilities و Labels
    print("="*70)
    print("Generating ROC and F1-Score Plots...")
    print("="*70)
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Collecting predictions"):
            images, labels = images.to(device), labels.to(device).float()
            
            # دریافت خروجی با جزئیات
            # برای Majority Voting: outputs, weights, dummy, votes
            # برای Fuzzy Hesitant: outputs, weights, memberships, raw_outputs
            # ما فقط logits (outputs) را می‌خواهیم
            try:
                outputs, weights, _, _ = ensemble_model(images, return_details=True)
            except Exception as e:
                print(f"Error in forward pass: {e}")
                continue
            
            # outputs: (Batch, 1) -> Logits
            # probabilities: استفاده از Sigmoid برای تبدیل به 0 تا 1
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # تبدیل logits به پیش‌بینی (0 یا 1)
            pred = (outputs > 0).long().cpu().numpy()
            
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)
            all_preds.append(pred)

    # تبدیل لیست‌ها به آرایه‌های یک‌پارچه (Flat)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)

    # 2. محاسبه ROC Curve
    # pos_label=1 برای تشخیص "Real"
    fpr, tpr, _ = roc_curve(all_labels, all_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    # 3. محاسبه Precision-Recall
    precision, recall, _ = precision_recall_curve(all_labels, all_probs, pos_label=1)
    pr_auc = average_precision_score(all_labels, all_probs, pos_label=1)
    
    # 4. محاسبه متریک‌های دقیق
    # pos_label=1 برای تشخیص کلاس "Real"
    f1 = f1_score(all_labels, all_preds, pos_label=1)
    cm = confusion_matrix(all_labels, all_preds)
    
    # متریک‌های دقیق
    tn, fp, fn, tp = cm.ravel()
    
    # محاسبه Precision و Recall از CM (اختیاری، برای چاپ)
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if is_main:
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
        # ردیف‌ها: Real (1), Fake (0) - ستون‌ها: Real (1), Fake (0)
        # برای رسم تمیز، از لیبل‌های استاندارد استفاده می‌کنیم
        cm_display = np.array([[tp, fn], [fp, tn]]) # Row: Actual Real, Fake | Col: Pred Real, Fake
        
        # استفاده از seaborn برای رسم Heatmap بهتر
        ax = plt.gca()
        sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'], 
                    yticklabels=['Real', 'Fake'], ax=ax)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # تنظیمات کلی نمودار
        model_name_str = " + ".join(model_names)
        ensemble_type = "Fuzzy" if hasattr(ensemble_model, 'hesitant_fuzzy') else "Majority Voting"
        full_title = f"{ensemble_type} Ensemble Performance ({model_name_str})"
        plt.suptitle(full_title, fontsize=16, y=1.02)
        
        plt.tight_layout()
        
        # ذخیره نمودار
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'roc_f1_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        plt.close()
        
        # 6. چاپ گزارش متنی کلاس‌بندی
        print("\nClassification Report:")
        target_names = ['Fake', 'Real']
        # دقت کلی برای هر کلاس
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
        print(f"Metrics saved to: {metrics_path}")

        return metrics
