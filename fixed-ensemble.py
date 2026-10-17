import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import json

class FixedWeightEnsemble(nn.Module):
   
    def __init__(self, models, means, stds, fixed_weights, device="cuda"):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.device = device
        
        self.means = [torch.tensor(m, dtype=torch.float32).view(1, 3, 1, 1).to(device) for m in means]
        self.stds  = [torch.tensor(s, dtype=torch.float32).view(1, 3, 1, 1).to(device) for s in stds]
        
        # وزن‌های ثابت (نرمال‌سازی شده)
        weights = torch.tensor(fixed_weights, dtype=torch.float32)
        weights = weights / weights.sum()          # مطمئن شو جمع = 1
        self.register_buffer("fixed_weights", weights)
        
        # فریز کردن مدل‌ها
        for m in self.models:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        batch_size = x.size(0)
        outputs = torch.zeros(batch_size, self.num_models, device=x.device)
        
        for i in range(self.num_models):
            x_norm = (x - self.means[i]) / self.stds[i]
            with torch.no_grad():
                out = self.models[i](x_norm)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            outputs[:, i] = out.squeeze(1)
        
        # وزن‌دهی ثابت
        final_logit = (outputs * self.fixed_weights.unsqueeze(0)).sum(dim=1)
        return final_logit


def evaluate_fixed_ensemble(ensemble, test_loader, device, save_dir="./fixed_mean_weights"):
    os.makedirs(save_dir, exist_ok=True)
    ensemble.eval()
    
    all_y_true, all_y_score, all_y_pred = [], [], []
    TP = TN = FP = FN = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Fixed Ensemble (mean HFDE weights)"):
            images = images.to(device)
            labels = labels.to(device).long()
            
            logits = ensemble(images)
            probs = torch.sigmoid(logits)
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
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    results = {
        "method": "Fixed_Ensemble_Mean_HFDE_Weights",
        "fixed_weights": ensemble.fixed_weights.cpu().tolist(),
        "accuracy": float(acc),
        "precision_real": float(precision),
        "recall_real": float(recall),
        "f1_real": float(f1),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "num_samples": total
    }
    
    print("\n" + "="*65)
    print("FIXED ENSEMBLE (Mean HFDE Weights) RESULTS")
    print("="*65)
    print(f"Fixed weights : {[round(w, 4) for w in results['fixed_weights']]}")
    print(f"Accuracy      : {acc:.2f}%")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-Score      : {f1:.4f}")
    print("="*65)
    
    # ذخیره نتایج
    with open(os.path.join(save_dir, "fixed_mean_weights_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    np.savez(os.path.join(save_dir, "roc_data_fixed.npz"),
             y_true=np.array(all_y_true),
             y_score=np.array(all_y_score),
             y_pred=np.array(all_y_pred))
    
    print(f"\nنتایج ذخیره شد در: {save_dir}")
    return results


# ============================================================
# نحوه استفاده (در همان محیطی که مدل‌ها را لود کرده‌ای)
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # --- این قسمت‌ها را با متغیرهای خودت جایگزین کن ---
    # base_models : لیست ۳ مدل لود شده
    # MEANS و STDS : نرمال‌سازی‌های مربوط به هر مدل
    # test_loader : دیتالودر تست
    
    # میانگین وزن‌هایی که از خروجی HFDE گرفتی
    fixed_weights = [0.130522, 0.775484, 0.093994]
    
    fixed_ensemble = FixedWeightEnsemble(
        models=base_models,
        means=MEANS,
        stds=STDS,
        fixed_weights=fixed_weights,
        device=device
    ).to(device)
    
    results = evaluate_fixed_ensemble(
        ensemble=fixed_ensemble,
        test_loader=test_loader,
        device=device,
        save_dir="./ablation_fixed_mean_weights"
    )
