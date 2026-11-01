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
import time
import matplotlib.pyplot as plt
import seaborn as sns

from old_dataset_utils import create_dataloaders, get_sample_info
from efficiency_utils import Stopwatch, summarize_training_times, benchmark_efficiency, print_headline

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


# ================== NORMALIZATION ==================
class MultiModelNormalization(nn.Module):
    def __init__(self, means, stds):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x, idx):
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


# ================== NON-FUZZY GATING ==================
class NonFuzzyGating(nn.Module):
    """
    Non-fuzzy counterpart of HesitantFuzzyMembership.

    - Same feature extractor (identical conv stack, identical capacity).
    - Instead of producing M memberships per model + aggregation,
      directly outputs one weight per model (softmax).

    Parameter budget:
      Fuzzy head (trainable, on top of shared conv features):
          Linear(128,128):    16,512
          Linear(128,9):       1,161
          aggregation_weights:     3
          Total head:         17,676
      Non-fuzzy head:
          Linear(128,128):    16,512
          Linear(128,3):         387
          Total head:         16,899
      Difference = 777 params = 0.7% of TOTAL trainable params (~111k).
      This is negligible; if the reviewer insists on exact parity,
      pass --match_params to close it further.
    """
    def __init__(self, num_models: int, dropout: float = 0.3, match_params: bool = False):
        super().__init__()
        self.num_models = num_models

        # SAME feature extractor as HesitantFuzzyMembership
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # Single weight per model (NOT M memberships)
        self.gate_hidden = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.gate_out = nn.Linear(128, num_models)

        # Optional: add a small projection to equalize parameter count
        self.match_params = match_params
        if match_params:
            # Fuzzy head has 777 more trainable params. Add Linear(128, 6, bias=False) = 768
            # → remaining diff = 9 params (0.008%).
            self.pad = nn.Linear(128, 6, bias=False)

    def forward(self, x):
        feats = self.feature_net(x).flatten(1)
        h = self.gate_hidden(feats)
        logits = self.gate_out(h)
        if self.match_params:
            pad_out = self.pad(h)                 # [B, 6]
            # Fold 6 → 3 via simple reshape+mean, purely to consume the params
            logits = logits + pad_out.view(pad_out.size(0), 2, 3).mean(dim=1)
        return F.softmax(logits, dim=1)


# ================== NON-FUZZY ENSEMBLE ==================
class NonFuzzyEnsemble(nn.Module):
    """
    Reviewer-required baseline: same architecture, same inputs, same
    parameter budget, same selection threshold (cum_weight_threshold).
    Only difference: no memberships, no hesitancy fallback (main mode).
    """
    def __init__(self, models, means, stds, freeze_models=True,
                 cum_weight_threshold: float = 0.9,
                 hesitancy_threshold: float = 0.2,
                 fallback_on: bool = False,
                 match_params: bool = False):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.gating = NonFuzzyGating(num_models=self.num_models, match_params=match_params)
        self.cum_weight_threshold = cum_weight_threshold
        self.hesitancy_threshold = hesitancy_threshold
        self.fallback_on = fallback_on

        if freeze_models:
            for m in self.models:
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        for m in self.models:
            m.eval()
        return self

    def _compute_mask_vectorized(self, weights, uncertainty):
        sorted_w, sorted_idx = torch.sort(weights, dim=1, descending=True)
        cum_w = torch.cumsum(sorted_w, dim=1)
        mask = (cum_w <= self.cum_weight_threshold).float()
        mask[:, 0] = 1.0  # always keep at least one model

        if self.fallback_on:
            # Same-style fallback as fuzzy: when "uncertainty" is high,
            # keep all models. Uncertainty = var(weights) (comparable scale
            # to var(memberships) in the fuzzy model).
            high_unc = (uncertainty > self.hesitancy_threshold).unsqueeze(1)
            mask = torch.where(high_unc, torch.ones_like(mask), mask)

        final_mask = torch.zeros_like(weights)
        final_mask.scatter_(1, sorted_idx, mask)
        return final_mask

    def forward(self, x, return_details=False):
        weights = self.gating(x)                            # [B, N]
        uncertainty = weights.var(dim=1, unbiased=False)    # [B]
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
def load_pruned_models(model_paths, device):
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
            m = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
            m.load_state_dict(ckpt['model_state_dict'])
            m = m.to(device).eval()
            models.append(m)
        except Exception as e:
            print(f" [ERROR] Failed to load {path}: {e}")
    if len(models) == 0:
        raise ValueError("No models loaded!")
    return models


# ================== EVALUATION (identical format to fuzzy) ==================
def final_evaluation_unified(model, test_loader, device, save_dir, model_names, args,
                              method_name="Non_Fuzzy_Gating"):
    model.eval()
    test_dataset = test_loader.dataset
    num_samples = len(test_dataset)

    lines = []
    lines.append("=" * 100)
    lines.append("SAMPLE-BY-SAMPLE PREDICTIONS (For McNemar Test Comparison):")
    lines.append("=" * 100)
    lines.append(f"{'ID':<10} {'Filename':<60} {'True':<10} {'Pred':<10} {'Correct':<10}")
    lines.append("-" * 100)

    TP = TN = FP = FN = 0
    all_y_true, all_y_score, all_y_pred = [], [], []
    all_model_weights, all_uncertainties = [], []

    print(f"\nRunning Unified Final Evaluation on {num_samples} samples...")

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Final Eval"):
            try:
                image, label = test_dataset[i]
                original_idx = test_dataset.indices[i]
                path, _ = get_sample_info(test_dataset.dataset, original_idx)
            except Exception as e:
                print(f"Error loading sample {i}: {e}")
                continue

            image = image.unsqueeze(0).to(device)
            label_int = int(label)

            output, final_weights, unc, _ = model(image, return_details=True)
            prob = torch.sigmoid(output.squeeze()).item()
            pred_int = int(prob > 0.5)

            all_model_weights.append(final_weights.squeeze(0).cpu().numpy())
            all_uncertainties.append(float(unc.squeeze().item()))

            all_y_true.append(label_int)
            all_y_score.append(prob)
            all_y_pred.append(pred_int)

            if label_int == 1:
                if pred_int == 1: TP += 1
                else: FN += 1
            else:
                if pred_int == 1: FP += 1
                else: TN += 1

            filename = os.path.basename(path)
            if len(filename) > 55:
                filename = filename[:25] + "..." + filename[-27:]
            lines.append(f"{i+1:<10} {filename:<60} {label_int:<10} {pred_int:<10} "
                         f"{'Yes' if pred_int == label_int else 'No':<10}")

    total_samples = len(all_y_true)
    correct_count = TP + TN
    acc = 100.0 * correct_count / total_samples if total_samples else 0.0

    precision_real = TP / (TP + FP) if (TP + FP) else 0
    recall_real    = TP / (TP + FN) if (TP + FN) else 0
    f1_real        = 2 * precision_real * recall_real / (precision_real + recall_real) \
                     if (precision_real + recall_real) else 0

    precision_fake = TN / (TN + FN) if (TN + FN) else 0
    recall_fake    = TN / (TN + FP) if (TN + FP) else 0
    f1_fake        = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) \
                     if (precision_fake + recall_fake) else 0

    macro_precision = (precision_real + precision_fake) / 2
    macro_recall    = (recall_real + recall_fake) / 2
    macro_f1        = (f1_real + f1_fake) / 2
    balanced_acc    = ((recall_real + recall_fake) / 2) * 100

    # Confusion matrix
    cm = np.array([[TN, FP], [FN, TP]])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Fake', 'Predicted Real'],
                yticklabels=['Actual Fake', 'Actual Real'],
                cbar=True, annot_kws={"size": 16})
    plt.title(f'{method_name}\nAcc: {acc:.2f}% | Balanced: {balanced_acc:.2f}% | Macro-F1: {macro_f1:.4f}',
              fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Console summary
    print(f"\n{'='*70}\n{method_name} RESULTS\n{'='*70}")
    print(f"Accuracy          : {acc:.2f}%")
    print(f"Balanced Accuracy : {balanced_acc:.2f}%")
    print(f"Macro-F1          : {macro_f1:.4f}")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 45)
    print(f"{'Fake (0)':<12} {precision_fake:10.4f} {recall_fake:10.4f} {f1_fake:10.4f}")
    print(f"{'Real (1)':<12} {precision_real:10.4f} {recall_real:10.4f} {f1_real:10.4f}")
    print(f"{'Macro Avg':<12} {macro_precision:10.4f} {macro_recall:10.4f} {macro_f1:10.4f}")
    print("=" * 70)

    # Activation statistics (same format as fuzzy)
    weights_np = np.array(all_model_weights)
    active_mask = (weights_np > 1e-4).astype(float)
    activation_rates = active_mask.mean(axis=0) * 100
    avg_weights = weights_np.mean(axis=0)
    active_per_sample = active_mask.sum(axis=1)

    print(f"\nMODEL ACTIVATION STATISTICS")
    print(f"  {'Model':<30} {'Active %':>10} {'Avg Weight':>12}")
    print(f"  {'-'*55}")
    for i, name in enumerate(model_names):
        print(f"  {i+1}. {name:<27}: {activation_rates[i]:6.2f}%  | {avg_weights[i]:.6f}")
    print(f"  Mean active models per sample: {active_per_sample.mean():.2f}")
    print(f"  Distribution:")
    uniq, freq = np.unique(active_per_sample.astype(int), return_counts=True)
    for c, f in zip(uniq, freq):
        print(f"    {c} model(s): {f} samples ({100.0*f/len(active_per_sample):.2f}%)")
    print("=" * 70)

    # Save JSON (SAME schema as fuzzy)
    roc_json = {
        "metadata": {
            "seed": args.seed,
            "dataset": args.dataset,
            "num_samples": int(total_samples),
            "positive_count": int(np.sum(all_y_true)),
            "negative_count": int(total_samples - np.sum(all_y_true)),
            "model": method_name,
        },
        "y_true": [int(v) for v in all_y_true],
        "y_score": [float(v) for v in all_y_score],
        "y_pred": [int(v) for v in all_y_pred],
    }
    with open(os.path.join(save_dir, "roc_data_test.json"), "w") as f:
        json.dump(roc_json, f, indent=2)
    with open(os.path.join(save_dir, "roc_data_test.txt"), "w") as f:
        f.write("y_true\ty_score\ty_pred\n")
        for t, s, p in zip(all_y_true, all_y_score, all_y_pred):
            f.write(f"{t}\t{s:.6f}\t{p}\n")

    # Activation stats + uncertainties (important for the paper's analysis)
    activation_stats = {
        "activation_rates_pct": {n: float(activation_rates[i]) for i, n in enumerate(model_names)},
        "average_weights": {n: float(avg_weights[i]) for i, n in enumerate(model_names)},
        "active_per_sample_stats": {
            "min": int(active_per_sample.min()),
            "max": int(active_per_sample.max()),
            "mean": float(active_per_sample.mean()),
            "median": float(np.median(active_per_sample)),
        },
        "per_sample_weights": weights_np.tolist(),
        "per_sample_uncertainty": all_uncertainties,   # var(weights) — for reviewer analysis
    }
    with open(os.path.join(save_dir, "model_activation_stats.json"), "w") as f:
        json.dump(activation_stats, f, indent=2)

    # Prediction log
    summary = [
        "-" * 100, "SUMMARY STATISTICS:", "-" * 100,
        f"Overall Accuracy: {acc:.2f}%",
        f"Balanced Accuracy: {balanced_acc:.2f}%",
        f"Macro F1-Score: {macro_f1:.4f}",
        f"Precision (Real): {precision_real:.4f}",
        f"Recall (Real):    {recall_real:.4f}",
        f"F1 (Real):        {f1_real:.4f}",
        f"Precision (Fake): {precision_fake:.4f}",
        f"Recall (Fake):    {recall_fake:.4f}",
        f"F1 (Fake):        {f1_fake:.4f}",
        f"Correct: {correct_count} ({acc:.2f}%)",
        f"Incorrect: {total_samples - correct_count} ({100-acc:.2f}%)",
        *lines,
    ]
    with open(os.path.join(save_dir, 'prediction_log.txt'), 'w') as f:
        f.write("\n".join(summary))

    # Also save a compact results JSON for later analysis
    with open(os.path.join(save_dir, "final_results.json"), 'w') as f:
        json.dump({
            "seed": args.seed,
            "method": method_name,
            "test_accuracy": float(acc),
            "balanced_accuracy": float(balanced_acc),
            "macro_f1": float(macro_f1),
            "f1_real": float(f1_real),
            "f1_fake": float(f1_fake),
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        }, f, indent=2)

    return acc, np.array(all_y_true), np.array(all_y_score)


# ================== TRAINING ==================
@torch.no_grad()
def evaluate_accuracy_and_loss(model, loader, device, criterion):
    model.eval()
    correct, total_loss = 0, 0.0
    total_n = len(loader.dataset)
    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device).float()
        outputs, _ = model(images)
        loss = criterion(outputs.squeeze(1), labels)
        total_loss += loss.item() * images.size(0)
        pred = (outputs.squeeze(1) > 0).long()
        correct += pred.eq(labels.long()).sum().item()
    return 100.0 * correct / total_n, total_loss / total_n


def train_non_fuzzy(ensemble, train_loader, val_loader, num_epochs, lr,
                    device, save_dir, model_names):
    os.makedirs(save_dir, exist_ok=True)
    gating_net = ensemble.gating
    optimizer = torch.optim.AdamW(gating_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()
    best_val_acc = 0.0

    total_real_samples = len(train_loader.dataset)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("=" * 70)
    print("Training Non-Fuzzy Gating Network (fair baseline)")
    print("=" * 70)
    print(f"Trainable params: {sum(p.numel() for p in gating_net.parameters()):,}")
    print(f"Epochs: {num_epochs} | Initial LR: {lr}")
    print(f"Num models: {len(model_names)}")
    print(f"Fallback: {'ON' if ensemble.fallback_on else 'OFF (clean comparison)'}")
    print("=" * 70 + "\n")

    for epoch in range(num_epochs):
        ensemble.train()
        train_loss, train_correct = 0.0, 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs, _ = ensemble(images)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            pred = (outputs.squeeze(1) > 0).long()
            train_correct += pred.eq(labels.long()).sum().item()

        train_loss /= total_real_samples
        train_acc = 100. * train_correct / total_real_samples

        val_acc, val_loss = evaluate_accuracy_and_loss(ensemble, val_loader, device, criterion)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:3d} | Train L: {train_loss:.4f} | Train A: {train_acc:.2f}% | "
              f"Val L: {val_loss:.4f} | Val A: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'gating_state_dict': gating_net.state_dict()},
                       os.path.join(save_dir, 'best_non_fuzzy.pt'))
            print(f"  ✓ Best saved → {val_acc:.2f}%")

    # Training curves
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), history['train_loss'], label='train', linewidth=2)
    plt.plot(range(1, num_epochs + 1), history['val_loss'], label='val', linewidth=2)
    plt.title('Loss', fontsize=16); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend(fontsize=12)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), history['train_acc'], label='train', linewidth=2)
    plt.plot(range(1, num_epochs + 1), history['val_acc'], label='val', linewidth=2)
    plt.title('Accuracy', fontsize=16); plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return best_val_acc


# ================== MAIN ==================
def main():
    parser = argparse.ArgumentParser(
        description="Non-Fuzzy Gating Baseline (Reviewer-required comparison)")
    # SAME defaults as the fuzzy script
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['wild', 'artiface', 'hard_fake_real', 'deepfake_lab',
                                 'uadfV', 'custom_genai', 'custom_genai_v2',
                                 'real_fake_dataset', '20k'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./ablation_non_fuzzy')
    parser.add_argument('--seed', type=int, default=42)

    # SAME selection thresholds as the fuzzy script
    parser.add_argument('--cum_weight_threshold', type=float, default=0.9)
    parser.add_argument('--hesitancy_threshold', type=float, default=0.2)

    # Fallback control — OFF by default (cleanest baseline)
    parser.add_argument('--fallback_on', action='store_true',
                        help='Enable uncertainty-based fallback (uses var(weights)). '
                             'OFF by default for clean comparison with Fuzzy.')
    parser.add_argument('--match_params', action='store_true',
                        help='Add a small projection to match fuzzy head param count exactly.')

    args = parser.parse_args()

    if len(args.model_names) != len(args.model_paths):
        raise ValueError("Number of model_names must match model_paths")

    set_seed(args.seed)
    device = setup_device()
    pipeline_sw = Stopwatch(device).start()

    print("=" * 70)
    print("NON-FUZZY GATING BASELINE (Reviewer Comparison)")
    print(f"Seed: {args.seed} | Fallback: {'ON' if args.fallback_on else 'OFF'}")
    print("=" * 70)
    print(f"Dataset     : {args.dataset}")
    print(f"Data dir    : {args.data_dir}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Epochs / LR : {args.epochs} / {args.lr}")
    print(f"cum_weight_threshold: {args.cum_weight_threshold}")
    if args.fallback_on:
        print(f"hesitancy_threshold : {args.hesitancy_threshold} (var(weights))")
    else:
        print(f"hesitancy_threshold : — (fallback disabled)")
    print("=" * 70 + "\n")

    DEFAULT_MEANS = [(0.5207, 0.4258, 0.3806),
                     (0.4460, 0.3622, 0.3416),
                     (0.4668, 0.3816, 0.3414)]
    DEFAULT_STDS = [(0.2490, 0.2239, 0.2212),
                    (0.2057, 0.1849, 0.1761),
                    (0.2410, 0.2161, 0.2081)]

    num_models = len(args.model_paths)
    if num_models > len(DEFAULT_MEANS):
        MEANS = DEFAULT_MEANS + [DEFAULT_MEANS[-1]] * (num_models - len(DEFAULT_MEANS))
        STDS = DEFAULT_STDS + [DEFAULT_STDS[-1]] * (num_models - len(DEFAULT_STDS))
    else:
        MEANS = DEFAULT_MEANS[:num_models]
        STDS = DEFAULT_STDS[:num_models]

    base_models = load_pruned_models(args.model_paths, device)
    MODEL_NAMES = args.model_names[:len(base_models)]

    ensemble = NonFuzzyEnsemble(
        models=base_models,
        means=MEANS,
        stds=STDS,
        freeze_models=True,
        cum_weight_threshold=args.cum_weight_threshold,
        hesitancy_threshold=args.hesitancy_threshold,
        fallback_on=args.fallback_on,
        match_params=args.match_params,
    ).to(device)

    total = sum(p.numel() for p in ensemble.parameters())
    trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,} | Frozen: {total - trainable:,}")
    print(f"(Fuzzy head trainable = 17,676 ; Non-fuzzy = {trainable - (total - trainable):,}"
          f"  → diff = {17676 - (trainable - (total - trainable))})\n")

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=False, seed=args.seed, is_main=True,
    )

    best_val = train_non_fuzzy(ensemble, train_loader, val_loader,
                               args.epochs, args.lr, device, args.save_dir, MODEL_NAMES)

    # Load best
    ckpt_path = os.path.join(args.save_dir, 'best_non_fuzzy.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        ensemble.gating.load_state_dict(ckpt['gating_state_dict'])
        print("Best model loaded.\n")

    method_name = "Non_Fuzzy_Gating_with_Fallback" if args.fallback_on else "Non_Fuzzy_Gating"
    final_acc, y_true, y_score = final_evaluation_unified(
        ensemble, test_loader, device, args.save_dir, MODEL_NAMES, args,
        method_name=method_name,
    )

    # Efficiency report (same structure as fuzzy)
    efficiency = benchmark_efficiency(
        ensemble, test_loader, device, MODEL_NAMES,
        latency_samples=len(test_loader.dataset),
    )
    efficiency['final_evaluation_time_s'] = pipeline_sw.stop()
    with open(os.path.join(args.save_dir, 'efficiency_report.json'), 'w') as f:
        json.dump(efficiency, f, indent=2)

    print(f"\n{'='*70}\nFINAL: {method_name}\n{'='*70}")
    print(f"Test Accuracy: {final_acc:.2f}%")
    print(f"Results saved to: {args.save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
