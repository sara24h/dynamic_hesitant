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
        return None

    ensemble_model.eval()
    all_labels = []
    all_prob_fake = []
    all_preds = []

    print("=" * 70)
    print("Collecting predictions → Fake = positive class (label 0)")
    print("=" * 70)

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Predictions"):
            images = images.to(device)
            labels = labels.to(device).float()

            try:
                outputs, _, _, _ = ensemble_model(images, return_details=True)
            except Exception as e:
                print(f"Forward pass error: {e}")
                continue

            # outputs → logits (معمولاً بالاتر از 0 → Real)
            prob_real = torch.sigmoid(outputs).squeeze().cpu().numpy()
            prob_fake = 1.0 - prob_real

            pred = (outputs.squeeze() > 0).long().cpu().numpy()

            all_labels.append(labels.cpu().numpy())
            all_prob_fake.append(prob_fake)
            all_preds.append(pred)

    # flatten
    y_true = np.concatenate(all_labels).ravel()
    y_score_fake = np.concatenate(all_prob_fake).ravel()
    y_pred = np.concatenate(all_preds).ravel()

    # ────────────────────────────────────────────────
    # محاسبه معیارها با فرض Fake = positive (pos_label=0)
    # ────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_score_fake, pos_label=0)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_score_fake, pos_label=0)
    pr_auc = average_precision_score(y_true, y_score_fake, pos_label=0)

    f1 = f1_score(y_true, y_pred, pos_label=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()   # tn: Real→Real, fp: Real→Fake, fn: Fake→Real, tp: Fake→Fake

    prec_fake = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_fake  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # ────────────────────────────────────────────────
    # چاپ نتایج
    # ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("METRICS ─ Fake is positive class (label = 0)")
    print("=" * 70)
    print(f"ROC AUC (Fake)     : {roc_auc:.4f}")
    print(f"PR AUC  (Fake)     : {pr_auc:.4f}")
    print(f"F1-score (Fake)    : {f1:.4f}")
    print(f"Precision (Fake)   : {prec_fake:.4f}")
    print(f"Recall    (Fake)   : {rec_fake:.4f}")
    print()
    print("Confusion Matrix:")
    print("                Predicted")
    print("              Real     Fake")
    print(f"Actual Real   {tn:6d}   {fp:6d}")
    print(f"Actual Fake   {fn:6d}   {tp:6d}")
    print("=" * 70)

    # ────────────────────────────────────────────────
    # رسم
    # ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ROC
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate (Recall Fake)')
    axes[0].set_title('ROC Curve – Fake positive')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # PR curve
    axes[1].plot(recall, precision, color='blue', lw=2, label=f'AP = {pr_auc:.4f}')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall (Fake)')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve – Fake positive')
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                xticklabels=['Pred Real', 'Pred Fake'],
                yticklabels=['Actual Real', 'Actual Fake'])
    axes[2].set_title('Confusion Matrix')
    axes[2].set_ylabel('True label')
    axes[2].set_xlabel('Predicted label')

    fig.suptitle(f"Ensemble: {' + '.join(model_names)} – Fake as Positive Class", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(save_dir, 'roc_pr_cm_fake_positive.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved → {save_path}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=['Fake (0)', 'Real (1)'],
                                digits=4, zero_division=0))

    # ذخیره json
    metrics = {
        "roc_auc_fake": float(roc_auc),
        "pr_auc_fake": float(pr_auc),
        "f1_fake": float(f1),
        "precision_fake": float(prec_fake),
        'recall_fake': float(recall_val),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        }
    }

    json_path = os.path.join(save_dir, "metrics_fake_positive.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved → {json_path}")

    return metrics
