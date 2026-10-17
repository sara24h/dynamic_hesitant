import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
import json
import argparse
import random
import warnings
warnings.filterwarnings("ignore")

from old_dataset_utils import create_dataloaders

# ================== UTILITY ==================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ================== METRIC HELPERS ==================
def _safe_div(a, b):
    return float(a) / float(b) if b > 0 else 0.0


def _class_metrics(tp, fp, fn, tn):
   
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    s = _safe_div(tn, tn + fp)
    f = _safe_div(2 * p * r, p + r)
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "support": int(tp + fn),
        "precision": p, "recall": r, "specificity": s, "f1": f,
    }


def compute_all_metrics(y_true, y_score, threshold=0.5):
   
    y_true  = np.asarray(y_true,  dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float32)
    y_pred  = (y_score > threshold).astype(np.int64)

    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    total = TP + TN + FP + FN

    real_m = _class_metrics(TP, FP, FN, TN)          # positive = real
    fake_m = _class_metrics(TN, FN, FP, TP)          # positive = fake

    macro_p  = (real_m["precision"] + fake_m["precision"]) / 2
    macro_r  = (real_m["recall"]    + fake_m["recall"])    / 2
    macro_f1 = (real_m["f1"]        + fake_m["f1"])        / 2

    s_r, s_f = real_m["support"], fake_m["support"]
    weighted_p  = _safe_div(real_m["precision"] * s_r + fake_m["precision"] * s_f, total)
    weighted_r  = _safe_div(real_m["recall"]    * s_r + fake_m["recall"]    * s_f, total)
    weighted_f1 = _safe_div(real_m["f1"]        * s_r + fake_m["f1"]        * s_f, total)

    balanced_acc = (real_m["recall"] + fake_m["recall"]) / 2
    accuracy     = _safe_div(TP + TN, total)

    roc_auc = None
    fpr, tpr = None, None
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        roc_auc = float(roc_auc_score(y_true, y_score))
        fpr, tpr, _ = roc_curve(y_true, y_score)
    except ImportError:
        pass

    return {
        "confusion": {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
        "num_samples": total,
        "num_per_class": {"fake": s_f, "real": s_r},
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "class_0_fake": fake_m,
        "class_1_real": real_m,
        "macro_precision": float(macro_p),
        "macro_recall":    float(macro_r),
        "macro_f1":        float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall":    float(weighted_r),
        "weighted_f1":        float(weighted_f1),
        "roc_auc": roc_auc,
        "threshold": float(threshold),
        "roc_curve": {"fpr": fpr, "tpr": tpr},
    }

def _print_metric_table(m, title):
    fr, rl = m["class_0_fake"], m["class_1_real"]
    cm = m["confusion"]
    print(f"\n{'='*76}\n{title}\n{'='*76}")
    print(f"Samples: {m['num_samples']}  (fake={m['num_per_class']['fake']}, real={m['num_per_class']['real']})")
    print('-'*76)
    print(f"{'Metric':<20}{'Fake (0)':>15}{'Real (1)':>15}{'Macro Avg':>15}")
    print('-'*76)
    print(f"{'Precision':<20}{fr['precision']:>15.4f}{rl['precision']:>15.4f}{m['macro_precision']:>15.4f}")
    print(f"{'Recall (TPR)':<20}{fr['recall']:>15.4f}{rl['recall']:>15.4f}{m['macro_recall']:>15.4f}")
    print(f"{'Specificity (TNR)':<20}{fr['specificity']:>15.4f}{rl['specificity']:>15.4f}{'-':>15}")
    print(f"{'F1-Score':<20}{fr['f1']:>15.4f}{rl['f1']:>15.4f}{m['macro_f1']:>15.4f}")
    print(f"{'Support':<20}{fr['support']:>15d}{rl['support']:>15d}{m['num_samples']:>15d}")
    print('-'*76)
    print(f"{'Accuracy':<20}: {m['accuracy']*100:.2f}%")
    print(f"{'Balanced Accuracy':<20}: {m['balanced_accuracy']*100:.2f}%")
    print(f"{'Weighted Precision':<20}: {m['weighted_precision']:.4f}")
    print(f"{'Weighted Recall':<20}: {m['weighted_recall']:.4f}")
    print(f"{'Weighted F1':<20}: {m['weighted_f1']:.4f}")
    if m["roc_auc"] is not None:
        print(f"{'ROC-AUC':<20}: {m['roc_auc']:.4f}")
    print('='*76)
    print(f"Confusion: TP={cm['TP']} TN={cm['TN']} FP={cm['FP']} FN={cm['FN']}")
    print('='*76)


# ================== MODEL CLASSES ==================
class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


class NonFuzzyGatingNetwork(nn.Module):
    
    def __init__(self, num_models: int = 3, dropout: float = 0.3):
        super().__init__()
        self.num_models = num_models

        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.gating_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_models)
        )

    def forward(self, x: torch.Tensor):
        features = self.feature_net(x).flatten(1)
        logits = self.gating_head(features)
        weights = F.softmax(logits, dim=1)
        return weights


class NonFuzzyEnsemble(nn.Module):
    def __init__(
        self,
        models: List[nn.Module],
        means: List[Tuple[float]],
        stds: List[Tuple[float]],
        freeze_models: bool = True,
        cum_weight_threshold: float = 0.9,
        hesitancy_threshold: float = 0.2
    ):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.gating = NonFuzzyGatingNetwork(num_models=self.num_models)
        self.cum_weight_threshold = cum_weight_threshold
        self.hesitancy_threshold = hesitancy_threshold

        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        for model in self.models:
            model.eval()
        return self

    def _compute_mask_vectorized(self, weights: torch.Tensor, uncertainty: torch.Tensor):
        sorted_weights, sorted_indices = torch.sort(weights, dim=1, descending=True)
        cum_weights = torch.cumsum(sorted_weights, dim=1)

        mask = (cum_weights <= self.cum_weight_threshold).float()
        mask[:, 0] = 1.0

        high_unc_mask = (uncertainty > self.hesitancy_threshold).unsqueeze(1)
        mask = torch.where(high_unc_mask, torch.ones_like(mask), mask)

        final_mask = torch.zeros_like(weights)
        final_mask.scatter_(1, sorted_indices, mask)
        return final_mask

    def forward(self, x: torch.Tensor, return_details: bool = False):
        weights = self.gating(x)   # [B, N]

        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1)
        uncertainty = entropy / 1.0986   # ~ max entropy for 3 models

        mask = self._compute_mask_vectorized(weights, uncertainty)
        weights = weights * mask
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
        active_models = torch.any(weights > 1e-4, dim=0).nonzero(as_tuple=True)[0]

        for i in active_models:
            x_n = self.normalizations(x, i.item())
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            outputs[:, i] = out

        final_output = (outputs * weights.unsqueeze(-1)).sum(dim=1)

        if return_details:
            return final_output, weights, uncertainty, outputs
        return final_output, weights


# ================== MODEL LOADING ==================
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


# ================== TRAINING ==================
def train_non_fuzzy(ensemble, train_loader, val_loader, num_epochs, lr, device, save_dir, model_names):
    os.makedirs(save_dir, exist_ok=True)
    gating_net = ensemble.gating
    optimizer = torch.optim.AdamW(gating_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    total_samples = len(train_loader.dataset)

    print("="*70)
    print("Training Non-Fuzzy Gating Network")
    print("="*70)
    print(f"Trainable params: {sum(p.numel() for p in gating_net.parameters()):,}")
    print(f"Epochs: {num_epochs} | LR: {lr}\n")

    for epoch in range(num_epochs):
        ensemble.train()
        train_loss = 0.0
        train_correct = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()

            outputs, weights = ensemble(images)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            pred = (outputs.squeeze(1) > 0).long()
            train_correct += pred.eq(labels.long()).sum().item()

        train_loss /= total_samples
        train_acc = 100. * train_correct / total_samples

        ensemble.eval()
        val_correct = 0
        val_total = len(val_loader.dataset)
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs, _ = ensemble(images)
                pred = (outputs.squeeze(1) > 0).long()
                val_correct += pred.eq(labels.long()).sum().item()
        val_acc = 100. * val_correct / val_total

        scheduler.step()

        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'gating_state_dict': gating_net.state_dict()},
                       os.path.join(save_dir, 'best_non_fuzzy.pt'))
            print(f"  → Best model saved ({val_acc:.2f}%)")

    return best_val_acc


# ================== EVALUATION (UPDATED METRICS) ==================
@torch.no_grad()
def evaluate_non_fuzzy(ensemble, test_loader, device, save_dir, model_names):
    os.makedirs(save_dir, exist_ok=True)
    ensemble.eval()

    all_y_true, all_y_score = [], []
    all_weights = []

    print(f"\nEvaluating Non-Fuzzy Gating on {len(test_loader.dataset)} samples...")

    for images, labels in tqdm(test_loader, desc="Non-Fuzzy Eval"):
        images = images.to(device)
        labels = labels.to(device).long()

        logits, weights = ensemble(images)
        probs = torch.sigmoid(logits)

        all_y_true.extend(labels.cpu().numpy().tolist())
        all_y_score.extend(probs.cpu().numpy().tolist())
        all_weights.append(weights.cpu().numpy())

    # ============ همه متریک‌ها ============
    metrics = compute_all_metrics(all_y_true, all_y_score, threshold=0.5)

    weights_np = np.concatenate(all_weights, axis=0)
    avg_weights = weights_np.mean(axis=0)
    active_rates = (weights_np > 1e-4).mean(axis=0) * 100

    # ============ چاپ جدول ============
    _print_metric_table(metrics, "NON-FUZZY GATING NETWORK — FINAL RESULTS")
    print(f"\nAvg ensemble weights : {[round(w, 4) for w in avg_weights]}")
    print(f"Activation rates (%) : {[round(a, 1) for a in active_rates]}")
    print("="*76)

    # ============ ساخت نتایج JSON ============
    fr, rl = metrics["class_0_fake"], metrics["class_1_real"]
    results = {
        "method": "Non_Fuzzy_Gating",
        "model_names": model_names,

        # overall
        "accuracy": metrics["accuracy"] * 100.0,
        "balanced_accuracy": metrics["balanced_accuracy"] * 100.0,

        # macro
        "macro_precision": metrics["macro_precision"],
        "macro_recall":    metrics["macro_recall"],
        "macro_f1":        metrics["macro_f1"],

        # weighted
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall":    metrics["weighted_recall"],
        "weighted_f1":        metrics["weighted_f1"],

        # per-class — fake (0)
        "precision_fake": fr["precision"],
        "recall_fake":    fr["recall"],
        "specificity_fake": fr["specificity"],
        "f1_fake":        fr["f1"],
        "support_fake":   fr["support"],

        # per-class — real (1)
        "precision_real": rl["precision"],
        "recall_real":    rl["recall"],
        "specificity_real": rl["specificity"],
        "f1_real":        rl["f1"],
        "support_real":   rl["support"],

        # raw confusion
        "TP": metrics["confusion"]["TP"], "TN": metrics["confusion"]["TN"],
        "FP": metrics["confusion"]["FP"], "FN": metrics["confusion"]["FN"],
        "num_samples": metrics["num_samples"],

        # AUC + threshold
        "roc_auc": metrics["roc_auc"],
        "decision_threshold": 0.5,

        # gating behaviour
        "average_weights": avg_weights.tolist(),
        "activation_rates_pct": active_rates.tolist(),
    }

    with open(os.path.join(save_dir, "non_fuzzy_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ============ ذخیره داده ROC ============
    save_dict = {
        "y_true":  np.array(all_y_true,  dtype=np.int64),
        "y_score": np.array(all_y_score, dtype=np.float32),
        "y_pred":  (np.array(all_y_score) > 0.5).astype(np.int64),
    }
    if metrics["roc_curve"]["fpr"] is not None:
        save_dict["fpr"] = np.array(metrics["roc_curve"]["fpr"])
        save_dict["tpr"] = np.array(metrics["roc_curve"]["tpr"])
    np.savez(os.path.join(save_dir, "roc_data_non_fuzzy.npz"), **save_dict)

    print(f"\nنتایج ذخیره شد در: {save_dir}")
    return results

# ================== MAIN ==================
def main():
    parser = argparse.ArgumentParser(description="Non-Fuzzy Gating Network Baseline")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./ablation_non_fuzzy')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cum_weight_threshold', type=float, default=0.8)
    parser.add_argument('--hesitancy_threshold', type=float, default=0.0001)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    DEFAULT_MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    DEFAULT_STDS  = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]

    num_models = len(args.model_paths)
    MEANS = DEFAULT_MEANS[:num_models]
    STDS  = DEFAULT_STDS[:num_models]

    base_models = load_pruned_models(args.model_paths, device)
    MODEL_NAMES = args.model_names[:len(base_models)]

    ensemble = NonFuzzyEnsemble(
        models=base_models,
        means=MEANS,
        stds=STDS,
        freeze_models=True,
        cum_weight_threshold=args.cum_weight_threshold,
        hesitancy_threshold=args.hesitancy_threshold
    ).to(device)

    trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
    print(f"Trainable params (gating only): {trainable:,}")

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=False, seed=args.seed, is_main=True
    )

    best_val = train_non_fuzzy(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_dir, MODEL_NAMES
    )

    ckpt_path = os.path.join(args.save_dir, 'best_non_fuzzy.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        ensemble.gating.load_state_dict(ckpt['gating_state_dict'])
        print("Best non-fuzzy model loaded.\n")

    results = evaluate_non_fuzzy(
        ensemble, test_loader, device, args.save_dir, MODEL_NAMES
    )

if __name__ == "__main__":
    main()
