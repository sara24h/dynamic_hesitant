import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

from old_dataset_utils import create_dataloaders

# ---------- کلاس‌های مورد نیاز از کد اصلی ----------
class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


def load_pruned_models(model_paths: List[str], device: torch.device) -> List[nn.Module]:
    try:
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        raise ImportError("Cannot import ResNet_50_pruned_hardfakevsreal")

    models = []
    print(f"Loading {len(model_paths)} pruned models...")
    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            print(f" [WARNING] File not found: {path}")
            continue
        print(f" [{i+1}/{len(model_paths)}] Loading: {os.path.basename(path)}")
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device).eval()
            models.append(model)
        except Exception as e:
            print(f" [ERROR] Failed to load {path}: {e}")
    if len(models) == 0:
        raise ValueError("No models loaded!")
    return models


# ---------- Fixed Top-k Ensemble ----------
class FixedTopKEnsemble(nn.Module):
    """
    Ensemble ثابت که همیشه فقط k مدل مشخص‌شده را فعال می‌کند
    (برای match کردن بودجه محاسباتی با HFDE)
    """
    def __init__(
        self,
        models: List[nn.Module],
        means: List[Tuple[float]],
        stds: List[Tuple[float]],
        active_indices: List[int],          # ایندکس مدل‌هایی که همیشه فعال‌اند
        fixed_weights: List[float] = None,  # وزن‌های مربوط به active_indices (اختیاری)
        device: str = "cuda"
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.active_indices = active_indices
        self.normalizations = MultiModelNormalization(means, stds)
        
        # ساخت وزن نهایی (فقط برای مدل‌های فعال)
        full_weights = torch.zeros(self.num_models, dtype=torch.float32)
        
        if fixed_weights is None:
            # وزن یکنواخت روی مدل‌های فعال
            w = 1.0 / len(active_indices)
            for idx in active_indices:
                full_weights[idx] = w
        else:
            # وزن‌های داده‌شده را نرمال کن و فقط روی active_indices بگذار
            assert len(fixed_weights) == len(active_indices), \
                "طول fixed_weights باید با active_indices برابر باشد"
            weights = torch.tensor(fixed_weights, dtype=torch.float32)
            weights = weights / weights.sum()
            for i, idx in enumerate(active_indices):
                full_weights[idx] = weights[i]
        
        self.register_buffer("fixed_weights", full_weights)
        
        # فریز کردن مدل‌ها
        for model in self.models:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

        print(f"Active models indices : {self.active_indices}")
        print(f"Final fixed weights   : {self.fixed_weights.tolist()}")

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        outputs = torch.zeros(batch_size, self.num_models, device=x.device)
        
        # فقط مدل‌های فعال را اجرا کن (صرفه‌جویی در FLOPs)
        for i in self.active_indices:
            x_n = self.normalizations(x, i)
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            outputs[:, i] = out.squeeze(1)
        
        final_logit = (outputs * self.fixed_weights.unsqueeze(0)).sum(dim=1)
        return final_logit


# ---------- ارزیابی ----------
@torch.no_grad()
def evaluate_fixed(ensemble, test_loader, device, save_dir, model_names, method_name):
    os.makedirs(save_dir, exist_ok=True)
    ensemble.eval()
    
    all_y_true, all_y_score, all_y_pred = [], [], []
    TP = TN = FP = FN = 0
    
    print(f"\nEvaluating {method_name} on {len(test_loader.dataset)} samples...")
    
    for images, labels in tqdm(test_loader, desc=method_name):
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
        "method": method_name,
        "active_indices": ensemble.active_indices,
        "fixed_weights": ensemble.fixed_weights.cpu().tolist(),
        "model_names": model_names,
        "accuracy": float(acc),
        "precision_real": float(precision),
        "recall_real": float(recall),
        "f1_real": float(f1),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "num_samples": total
    }
    
    print("\n" + "="*70)
    print(f"{method_name} RESULTS")
    print("="*70)
    print(f"Active models  : {[model_names[i] for i in ensemble.active_indices]}")
    print(f"Fixed weights  : {[round(w, 6) for w in results['fixed_weights']]}")
    print(f"Accuracy       : {acc:.2f}%")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-Score       : {f1:.4f}")
    print("="*70)
    
    with open(os.path.join(save_dir, f"{method_name.lower().replace(' ', '_')}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    np.savez(
        os.path.join(save_dir, f"roc_data_{method_name.lower().replace(' ', '_')}.npz"),
        y_true=np.array(all_y_true),
        y_score=np.array(all_y_score),
        y_pred=np.array(all_y_pred)
    )
    
    print(f"\nنتایج ذخیره شد در: {save_dir}")
    return results


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Fixed Top-2 Ensemble (matched computational budget)")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./ablation_fixed_top2')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    
    # ایندکس مدل‌های فعال (پیش‌فرض: Model 2 + Model 1 → indices 1 و 0)
    parser.add_argument('--active_indices', type=int, nargs='+', default=[1, 0],
                        help="Indices of models to always activate (0-based). Default: [1, 0] = Model2 + Model1")
    
    # وزن‌های مربوط به active_indices (اختیاری)
    # اگر ندهی، وزن یکنواخت استفاده می‌شود
    parser.add_argument('--fixed_weights', type=float, nargs='+', default=None,
                        help="Weights corresponding to active_indices. If None, uniform weights are used.")
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # نرمال‌سازی (همان مقادیر کد اصلی)
    DEFAULT_MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    DEFAULT_STDS  = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    
    num_models = len(args.model_paths)
    MEANS = DEFAULT_MEANS[:num_models]
    STDS  = DEFAULT_STDS[:num_models]
    
    # لود مدل‌ها
    base_models = load_pruned_models(args.model_paths, device)
    MODEL_NAMES = args.model_names[:len(base_models)]
    
    # ساخت Fixed Top-2
    fixed_ensemble = FixedTopKEnsemble(
        models=base_models,
        means=MEANS,
        stds=STDS,
        active_indices=args.active_indices,
        fixed_weights=args.fixed_weights,
        device=device
    ).to(device)
    
    # دیتالودر تست
    _, _, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=False, seed=args.seed, is_main=True
    )
    
    # ارزیابی
    method_name = f"Fixed_Top{len(args.active_indices)}"
    results = evaluate_fixed(
        ensemble=fixed_ensemble,
        test_loader=test_loader,
        device=device,
        save_dir=args.save_dir,
        model_names=MODEL_NAMES,
        method_name=method_name
    )


if __name__ == "__main__":
    main()
