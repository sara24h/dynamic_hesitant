import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
import warnings
import argparse
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns

from old_dataset_utils import create_dataloaders, get_sample_info
from efficiency_utils import Stopwatch

warnings.filterwarnings("ignore")

# ================== UTILITY ==================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("WARNING: CUDA not available, using CPU")
    return device

# ================== MODEL CLASSES ==================
class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


class NonFuzzyGating(nn.Module):
    """
    Non-fuzzy version of HesitantFuzzyMembership.
    Same feature extractor, same parameter count roughly,
    but directly predicts model weights (no membership values, no aggregation weights α).
    """
    def __init__(self, num_models: int = 3, dropout: float = 0.3):
        super().__init__()
        self.num_models = num_models

        # دقیقاً همان feature_net موجود در HFDE
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # Gating head: مستقیم N وزن تولید می‌کند
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
        self.gating = NonFuzzyGating(num_models=self.num_models)
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
        """همان منطق dual selection موجود در HFDE"""
        sorted_weights, sorted_indices = torch.sort(weights, dim=1, descending=True)
        cum_weights = torch.cumsum(sorted_weights, dim=1)

        mask = (cum_weights <= self.cum_weight_threshold).float()
        mask[:, 0] = 1.0   # همیشه حداقل یک مدل

        high_unc_mask = (uncertainty > self.hesitancy_threshold).unsqueeze(1)
        mask = torch.where(high_unc_mask, torch.ones_like(mask), mask)

        final_mask = torch.zeros_like(weights)
        final_mask.scatter_(1, sorted_indices, mask)
        return final_mask

    def forward(self, x: torch.Tensor, return_details: bool = False):
        weights = self.gating(x)   # [B, N]

        # Uncertainty جایگزین: entropy وزن‌ها (چون membership نداریم)
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1)
        # نرمال‌سازی تقریبی به [0,1] (max entropy برای 3 کلاس ≈ ln(3) ≈ 1.0986)
        uncertainty = entropy / 1.0986

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


# ================== EVALUATION (sample-by-sample مثل کد اصلی) ==================
@torch.no_grad()
def final_evaluation_non_fuzzy(model, test_loader, device, save_dir, model_names, args):
    model.eval()
    test_dataset = test_loader.dataset
    num_samples = len(test_dataset)

    all_y_true, all_y_score, all_y_pred = [], [], []
    all_model_weights = []
    TP = TN = FP = FN = 0

    print(f"\nRunning Non-Fuzzy Final Evaluation on {num_samples} samples...")

    for i in tqdm(range(num_samples), desc="Non-Fuzzy Eval"):
        try:
            image, label = test_dataset[i]
            original_idx = test_dataset.indices[i]
            path, _ = get_sample_info(test_dataset.dataset, original_idx)
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue

        image = image.unsqueeze(0).to(device)
        label_int = int(label)

        output, weights, _, _ = model(image, return_details=True)
        prob = torch.sigmoid(output.squeeze()).item()
        pred_int = int(prob > 0.5)

        all_y_true.append(label_int)
        all_y_score.append(prob)
        all_y_pred.append(pred_int)
        all_model_weights.append(weights.squeeze(0).cpu().numpy())

        if label_int == 1:
            if pred_int == 1: TP += 1
            else: FN += 1
        else:
            if pred_int == 1: FP += 1
            else: TN += 1

    total_samples = len(all_y_true)
    acc = 100.0 * (TP + TN) / total_samples if total_samples > 0 else 0.0

    precision_real = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall_real    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_real        = 2 * precision_real * recall_real / (precision_real + recall_real) if (precision_real + recall_real) > 0 else 0.0

    precision_fake = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    recall_fake    = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1_fake        = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0.0

    macro_precision = (precision_real + precision_fake) / 2
    macro_recall    = (recall_real + recall_fake) / 2
    macro_f1        = (f1_real + f1_fake) / 2
    balanced_acc    = ((recall_real + recall_fake) / 2) * 100

    # Confusion matrix
    cm = np.array([[TN, FP], [FN, TP]])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Fake', 'Predicted Real'],
                yticklabels=['Actual Fake', 'Actual Real'])
    plt.title(f'Non-Fuzzy Gating\nAcc: {acc:.2f}% | Balanced: {balanced_acc:.2f}% | Macro-F1: {macro_f1:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_non_fuzzy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*70}")
    print("NON-FUZZY GATING RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy          : {acc:.2f}%")
    print(f"Balanced Accuracy : {balanced_acc:.2f}%")
    print(f"Macro-F1          : {macro_f1:.4f}")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 45)
    print(f"{'Fake (0)':<12} {precision_fake:10.4f} {recall_fake:10.4f} {f1_fake:10.4f}")
    print(f"{'Real (1)':<12} {precision_real:10.4f} {recall_real:10.4f} {f1_real:10.4f}")
    print("-" * 45)
    print(f"{'Macro Avg':<12} {macro_precision:10.4f} {macro_recall:10.4f} {macro_f1:10.4f}")
    print("="*70)

    # Activation statistics
    if len(all_model_weights) > 0:
        weights_np = np.array(all_model_weights)
        active_mask = (weights_np > 1e-4).astype(float)
        activation_rates = active_mask.mean(axis=0) * 100
        avg_weights = weights_np.mean(axis=0)

        print(f"\nMODEL ACTIVATION STATISTICS")
        print(f"  {'Model':<30} {'Active %':>10} {'Avg Weight':>12}")
        print(f"  {'-'*55}")
        for i, name in enumerate(model_names):
            print(f"  {i+1}. {name:<27}: {activation_rates[i]:6.2f}%  | {avg_weights[i]:.6f}")
        print(f"  Mean active models per sample: {active_mask.sum(axis=1).mean():.2f}")
        print("="*70)

    # Save results
    results = {
        "method": "Non_Fuzzy_Gating",
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "precision_real": float(precision_real),
        "recall_real": float(recall_real),
        "f1_real": float(f1_real),
        "precision_fake": float(precision_fake),
        "recall_fake": float(recall_fake),
        "f1_fake": float(f1_fake),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "num_samples": total_samples,
        "average_weights": avg_weights.tolist() if len(all_model_weights) > 0 else [],
        "activation_rates_pct": activation_rates.tolist() if len(all_model_weights) > 0 else []
    }
    with open(os.path.join(save_dir, "non_fuzzy_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    np.savez(os.path.join(save_dir, "roc_data_non_fuzzy.npz"),
             y_true=np.array(all_y_true),
             y_score=np.array(all_y_score),
             y_pred=np.array(all_y_pred))

    return acc


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

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs, _ = ensemble(images)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            pred = (outputs.squeeze(1) > 0).long()
            train_correct += pred.eq(labels.long()).sum().item()

        train_loss /= total_samples
        train_acc = 100. * train_correct / total_samples

        # Validation
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


# ================== MAIN ==================
def main():
    parser = argparse.ArgumentParser(description="Non-Fuzzy Gating Network Baseline (Reviewer 2)")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./ablation_non_fuzzy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cum_weight_threshold', type=float, default=0.80)
    parser.add_argument('--hesitancy_threshold', type=float, default=0.0001)

    args = parser.parse_args()
    set_seed(args.seed)
    device = setup_device()

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

    # Load best
    ckpt_path = os.path.join(args.save_dir, 'best_non_fuzzy.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        ensemble.gating.load_state_dict(ckpt['gating_state_dict'])
        print("Best non-fuzzy model loaded.\n")

    # Final evaluation (sample-by-sample مثل کد اصلی)
    final_acc = final_evaluation_non_fuzzy(
        ensemble, test_loader, device, args.save_dir, MODEL_NAMES, args
    )

    print(f"\nFinal Non-Fuzzy Accuracy: {final_acc:.2f}%")


if __name__ == "__main__":
    main()
