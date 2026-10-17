import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import json
from old_dataset_utils import create_dataloaders
from efficiency_utils import Stopwatch, benchmark_efficiency  # اگر دارید

def evaluate_model2_only(
    model2: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    mean: tuple,
    std: tuple,
    save_dir: str = "./model2_only_results"
):
    """
    فقط Model 2 را ارزیابی می‌کند (بدون هیچ ensemble).
    """
    os.makedirs(save_dir, exist_ok=True)
    model2.eval()
    
    # نرمال‌سازی مخصوص Model 2
    mean_t = torch.tensor(mean).view(1, 3, 1, 1).to(device)
    std_t  = torch.tensor(std).view(1, 3, 1, 1).to(device)
    
    all_y_true, all_y_score, all_y_pred = [], [], []
    TP = TN = FP = FN = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Model-2-only Eval"):
            images = images.to(device)
            labels = labels.to(device).long()
            
            # نرمال‌سازی
            images_norm = (images - mean_t) / std_t
            
            logits = model2(images_norm)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            
            probs = torch.sigmoid(logits.squeeze(1))
            preds = (probs > 0.5).long()
            
            all_y_true.extend(labels.cpu().numpy().tolist())
            all_y_score.extend(probs.cpu().numpy().tolist())
            all_y_pred.extend(preds.cpu().numpy().tolist())
            
            for t, p in zip(labels, preds):
                if t == 1:
                    if p == 1: TP += 1
                    else: FN += 1
                else:
                    if p == 1: FP += 1
                    else: TN += 1
    
    total = len(all_y_true)
    acc = 100.0 * (TP + TN) / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results = {
        "method": "Model-2-only",
        "accuracy": float(acc),
        "precision_real": float(precision),
        "recall_real": float(recall),
        "f1_real": float(f1),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "num_samples": total
    }
    
    print("\n" + "="*60)
    print("MODEL-2-ONLY RESULTS")
    print("="*60)
    print(f"Accuracy  : {acc:.2f}%")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("="*60)
    
    with open(os.path.join(save_dir, "model2_only_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # ذخیره برای ROC
    np.savez(os.path.join(save_dir, "roc_data_model2.npz"),
             y_true=np.array(all_y_true),
             y_score=np.array(all_y_score),
             y_pred=np.array(all_y_pred))
    
    return results


# ------------------ نحوه استفاده ------------------
if __name__ == "__main__":
    # فرض: base_models از قبل لود شده‌اند و Model 2 ایندکس 1 است
    # MEANS و STDS هم از قبل تعریف شده‌اند
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # لود دیتاست تست
    _, _, test_loader = create_dataloaders(
        data_dir="PATH_TO_DATA",
        batch_size=32,
        dataset_type="YOUR_DATASET",   # مثلاً 'hard_fake_real' یا 'wild'
        is_distributed=False,
        seed=42
    )
    
    # Model 2 (ایندکس 1)
    model2 = base_models[1]
    mean2  = MEANS[1]
    std2   = STDS[1]
    
    results_m2 = evaluate_model2_only(
        model2=model2,
        test_loader=test_loader,
        device=device,
        mean=mean2,
        std=std2,
        save_dir="./ablation_model2_only"
    )
