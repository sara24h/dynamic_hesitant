import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.distributed as dist

def plot_roc_and_f1(ensemble_model, test_loader, device, save_dir, model_names, is_main=True):

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
            
            # دریافت خروجی با جزئیات (return_details=True)
            outputs, weights, memberships, raw_outputs = ensemble_model(images, return_details=True)
            
            # outputs: (Batch, 1) -> Logits
            # probabilities: استفاده از Sigmoid برای تبدیل به 0 تا 1
            probs = torch.sigmoid(outputs).cpu().numpy()
            pred = (outputs > 0).long().cpu().numpy()
            
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)
            all_preds.append(pred)

    # تبدیل لیست‌ها به آرایه‌های یک‌پارچه (Flat)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)

    # 2. محاسبه ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # 3. محاسبه Precision-Recall (اختیاری برای دیتاست نامتعادل)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    
    # 4. محاسبه متریک‌های F1, Precision, Recall
    # pos_label=1 برای تشخیص "Real"
    f1 = f1_score(all_labels, all_preds, pos_label=1)
    cm = confusion_matrix(all_labels, all_preds)
    
    # متریک‌های دقیق
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if is_main:
        print(f"\n{'='*70}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")
        print(f"F1 Score (Real): {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Confusion Matrix:")
        print(f"  TN={tn}, FP={fp}")
        print(f"  FN={fn}, TP={tp}")
        print(f"{'='*70}")
        
        # 5. رسم نمودارها
        plt.figure(figsize=(18, 6))
        
        # نمودار 1: ROC Curve
        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # نمودار 2: Precision-Recall Curve
        plt.subplot(1, 3, 2)
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        # نمودار 3: Confusion Matrix (Heatmap)
        plt.subplot(1, 3, 3)
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # تنظیمات کلی
        model_name_str = " + ".join(model_names)
        full_title = f"Fuzzy Ensemble Performance on {model_name_str}"
        plt.suptitle(full_title, fontsize=16, y=1.02)
        
        plt.tight_layout()
        
        # ذخیره
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'roc_f1_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        plt.show()
        plt.close()

        # 6. چاپ گزارش متنی کلاس‌بندی
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=['Fake', 'Real'], digits=4))
        
        # ذخیره متریک‌ها در فایل JSON
        metrics = {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            }
        }
        
        metrics_path = os.path.join(save_dir, 'roc_f1_metrics.json')
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to: {metrics_path}")
