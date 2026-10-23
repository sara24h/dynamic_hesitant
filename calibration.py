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

from old_dataset_utils import (
    UADFVDataset, CustomGenAIDataset, NewGenAIDataset,
    create_dataloaders, get_sample_info
)
from visualization_utils import generate_visualizations
from efficiency_utils import Stopwatch, summarize_training_times, benchmark_efficiency, print_headline

warnings.filterwarnings("ignore")

# ================== UTILITY FUNCTIONS ==================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ================== DEVICE SETUP (Single GPU) ==================
def setup_device():
    """تنظیم دستگاه روی یک GPU واحد"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("WARNING: CUDA not available, using CPU")
    return device


# ================== ECE METRIC ==================
def compute_ece(y_true, y_prob, n_bins: int = 15) -> float:
    """Expected Calibration Error برای طبقه‌بندی دودویی."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    pred = (y_prob > 0.5).astype(int)
    conf = np.where(pred == 1, y_prob, 1.0 - y_prob)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_true)) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


# ================== UNIFIED FINAL EVALUATION ==================
def final_evaluation_unified(model, test_loader, device, save_dir, model_names, args):

    model.eval()

    test_dataset = test_loader.dataset

    num_samples = len(test_dataset)

    lines = []
    lines.append("="*100)
    lines.append("SAMPLE-BY-SAMPLE PREDICTIONS (For McNemar Test Comparison):")
    lines.append("="*100)
    header = f"{'ID':<10} {'Filename':<60} {'True':<10} {'Pred':<10} {'Correct':<10}"
    lines.append(header)
    lines.append("-"*100)

    TP, TN, FP, FN = 0, 0, 0, 0

    all_y_true = []
    all_y_score = []
    all_y_pred = []
    all_model_weights = []

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

            output, final_weights, _, _ = model(image, return_details=True)

            prob = torch.sigmoid(output.squeeze()).item()

            all_model_weights.append(final_weights.squeeze(0).cpu().numpy())

            pred_int = int(prob > 0.5)

            all_y_true.append(label_int)
            all_y_score.append(prob)
            all_y_pred.append(pred_int)

            if label_int == 1:
                if pred_int == 1: TP += 1
                else: FN += 1
            else:
                if pred_int == 1: FP += 1
                else: TN += 1

            correct_str = "Yes" if pred_int == label_int else "No"
            filename = os.path.basename(path)
            if len(filename) > 55:
                filename = filename[:25] + "..." + filename[-27:]
            line = f"{i+1:<10} {filename:<60} {label_int:<10} {pred_int:<10} {correct_str:<10}"
            lines.append(line)

    total_samples = len(all_y_true)
    correct_count = TP + TN
    acc = 100.0 * correct_count / total_samples if total_samples > 0 else 0.0

    precision_real = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_real    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_real        = 2 * precision_real * recall_real / (precision_real + recall_real) if (precision_real + recall_real) > 0 else 0

    precision_fake = TN / (TN + FN) if (TN + FN) > 0 else 0
    recall_fake    = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_fake        = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0

    macro_precision = (precision_real + precision_fake) / 2
    macro_recall    = (recall_real + recall_fake) / 2
    macro_f1        = (f1_real + f1_fake) / 2

    balanced_acc = ((recall_real + recall_fake) / 2) * 100

    # ================== Confusion Matrix Plot ==================
    cm = np.array([[TN, FP],
               [FN, TP]])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Predicted Fake', 'Predicted Real'],
        yticklabels=['Actual Fake', 'Actual Real'],
        cbar=True,
        annot_kws={"size": 16}
    )
    plt.title(f'Confusion Matrix\nAccuracy: {acc:.2f}% | Balanced Acc: {balanced_acc:.2f}% | Macro-F1: {macro_f1:.4f}',
          fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion Matrix saved to: {cm_path}")

    # ================== چاپ نتایج نهایی ==================
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy          : {acc:.2f}%")
    print(f"Balanced Accuracy : {balanced_acc:.2f}%")
    print(f"Macro-F1          : {macro_f1:.4f}")
    print()
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 45)
    print(f"{'Fake (0)':<12} {precision_fake:10.4f} {recall_fake:10.4f} {f1_fake:10.4f}")
    print(f"{'Real (1)':<12} {precision_real:10.4f} {recall_real:10.4f} {f1_real:10.4f}")
    print("-" * 45)
    print(f"{'Macro Avg':<12} {macro_precision:10.4f} {macro_recall:10.4f} {macro_f1:10.4f}")
    print(f"\nCorrect Predictions  : {correct_count} ({acc:.2f}%)")
    print(f"Incorrect Predictions: {total_samples - correct_count} ({100 - acc:.2f}%)")
    print("="*70)

    if len(all_model_weights) > 0:
        weights_np = np.array(all_model_weights)

        print(f"\n{'='*70}")
        print("MODEL ACTIVATION STATISTICS (Final Evaluation)")
        print(f"{'='*70}")

        activation_threshold = 1e-4
        active_mask = (weights_np > activation_threshold).astype(float)

        activation_rates = active_mask.mean(axis=0) * 100
        avg_weights = weights_np.mean(axis=0)
        max_weights = weights_np.max(axis=0)

        print(f"\n  {'Model':<35} {'Active %':>10} {'Avg Weight':>12} {'Max Weight':>12}")
        print(f"  {'-'*70}")
        for i, name in enumerate(model_names):
            act_pct = activation_rates[i]
            avg_w = avg_weights[i]
            max_w = max_weights[i]
            bar_length = int(act_pct / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"  {i+1:2d}. {name:<30}: {act_pct:6.2f}%  | {avg_w:.6f}  | {max_w:.6f}")
            print(f"      [{bar}]")

        print(f"  {'-'*70}")
        print(f"  {'Overall Average':<35} {activation_rates.mean():6.2f}%")

        active_per_sample = active_mask.sum(axis=1)
        print(f"\n  Active Models Per Sample:")
        print(f"    Min:    {int(active_per_sample.min())}")
        print(f"    Max:    {int(active_per_sample.max())}")
        print(f"    Mean:   {active_per_sample.mean():.2f}")
        print(f"    Median: {np.median(active_per_sample):.1f}")

        unique_counts, count_freq = np.unique(active_per_sample.astype(int), return_counts=True)
        print(f"\n  Distribution of Active Model Count:")
        for count, freq in zip(unique_counts, count_freq):
            pct = 100.0 * freq / len(all_model_weights)
            bar = '█' * int(pct / 2)
            print(f"    {count} model(s): {freq:5d} samples ({pct:5.2f}%) {bar}")

        print(f"{'='*70}\n")

        activation_stats = {
            "activation_rates_pct": {name: float(activation_rates[i]) for i, name in enumerate(model_names)},
            "average_weights": {name: float(avg_weights[i]) for i, name in enumerate(model_names)},
            "max_weights": {name: float(max_weights[i]) for i, name in enumerate(model_names)},
            "active_per_sample_stats": {
                "min": int(active_per_sample.min()),
                "max": int(active_per_sample.max()),
                "mean": float(active_per_sample.mean()),
                "median": float(np.median(active_per_sample))
            },
            "per_sample_weights": weights_np.tolist()
        }
        activation_json_path = os.path.join(save_dir, "model_activation_stats.json")
        with open(activation_json_path, 'w', encoding='utf-8') as f:
            json.dump(activation_stats, f, indent=2, ensure_ascii=False)
        print(f"Model activation stats saved: {activation_json_path}")

    output_str = []

    output_str.append("-" * 100)
    output_str.append("SUMMARY STATISTICS:")
    output_str.append("-" * 100)
    output_str.append(f"Overall Accuracy: {acc:.2f}%")
    output_str.append(f"Balanced Accuracy: {balanced_acc:.2f}%")
    output_str.append(f"Macro F1-Score: {macro_f1:.4f}")

    output_str.append("\n--- Real (1) Class Metrics ---")
    output_str.append(f"Precision (Real): {precision_real:.4f}")
    output_str.append(f"Recall (Real):    {recall_real:.4f}")
    output_str.append(f"F1-Score (Real):  {f1_real:.4f}")

    output_str.append("\n--- Fake (0) Class Metrics ---")
    output_str.append(f"Precision (Fake): {precision_fake:.4f}")
    output_str.append(f"Recall (Fake):    {recall_fake:.4f}")
    output_str.append(f"F1-Score (Fake):  {f1_fake:.4f}")

    output_str.append(f"\nCorrect Predictions: {correct_count} ({acc:.2f}%)")
    output_str.append(f"Incorrect Predictions: {total_samples - correct_count} ({100-acc:.2f}%)")
    output_str.append(f"\nConfusion Matrix saved to: confusion_matrix.png")
    output_str.extend(lines)

    log_path = os.path.join(save_dir, 'prediction_log.txt')
    with open(log_path, 'w') as f:
        f.write("\n".join(output_str))
    print(f"✅ Prediction log saved to: {log_path}")

    print("\nCollecting ROC data (y_true & y_score) ...")

    y_true_np = np.array(all_y_true)
    y_score_np = np.array(all_y_score)
    y_pred_np = np.array(all_y_pred)

    roc_json_path = os.path.join(save_dir, "roc_data_test.json")
    roc_data_json = {
        "metadata": {
            "seed": args.seed,
            "dataset": args.dataset,
            "num_samples": int(total_samples),
            "positive_count": int(np.sum(y_true_np)),
            "negative_count": int(total_samples - np.sum(y_true_np)),
            "model": "fuzzy_hesitant_ensemble",
            "calibration_enabled": bool(getattr(args, 'calibrate', True)),
            "temperatures": list(getattr(model, 'temperatures', []))
        },
        "y_true": y_true_np.tolist(),
        "y_score": y_score_np.tolist(),
        "y_pred": y_pred_np.tolist()
    }
    with open(roc_json_path, 'w', encoding='utf-8') as f:
        json.dump(roc_data_json, f, indent=2, ensure_ascii=False)
    print(f"ROC data saved (JSON): {roc_json_path}")

    roc_txt_path = os.path.join(save_dir, "roc_data_test.txt")
    with open(roc_txt_path, 'w', encoding='utf-8') as f:
        f.write("y_true\ty_score\ty_pred\n")
        for t, s, p in zip(y_true_np, y_score_np, y_pred_np):
            f.write(f"{int(t)}\t{s:.6f}\t{int(p)}\n")
    print(f"ROC data saved (TXT):  {roc_txt_path}")

    vis_dir = os.path.join(save_dir, 'visualizations')
    generate_visualizations(
        model, test_loader, device, vis_dir, model_names,
        args.num_grad_cam_samples, args.num_lime_samples,
        args.dataset, is_main=True)

    return acc, y_true_np, y_score_np


# ================== MODEL CLASSES ==================
class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


class HesitantFuzzyMembership(nn.Module):
    def __init__(self, input_dim: int, num_models: int, num_memberships: int = 3, dropout: float = 0.3):
        super().__init__()
        self.num_models = num_models
        self.num_memberships = num_memberships
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.membership_generator = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_models * num_memberships)
        )
        self.aggregation_weights = nn.Parameter(torch.ones(num_memberships) / num_memberships)

    def forward(self, x: torch.Tensor):
        features = self.feature_net(x).flatten(1)
        memberships = self.membership_generator(features)
        memberships = memberships.view(-1, self.num_models, self.num_memberships)
        memberships = torch.sigmoid(memberships)
        agg_weights = F.softmax(self.aggregation_weights, dim=0)
        final_weights = (memberships * agg_weights.view(1, 1, -1)).sum(dim=2)
        final_weights = F.softmax(final_weights, dim=1)
        return final_weights, memberships


class FuzzyHesitantEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]],
                 stds: List[Tuple[float]], num_memberships: int = 3, freeze_models: bool = True,
                 cum_weight_threshold: float = 0.9, hesitancy_threshold: float = 0.2,
                 temperatures: List[float] = None):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.hesitant_fuzzy = HesitantFuzzyMembership(
            input_dim=128, num_models=self.num_models, num_memberships=num_memberships)
        self.cum_weight_threshold = cum_weight_threshold
        self.hesitancy_threshold = hesitancy_threshold

        # 🆕 Per-model temperature scaling (fit post-training on validation set)
        if temperatures is None:
            temperatures = [1.0] * self.num_models
        self.temperatures = [float(t) for t in temperatures]

        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # Force base models to remain in eval mode
        for model in self.models:
            model.eval()
        return self

    def _compute_mask_vectorized(self, final_weights: torch.Tensor, avg_hesitancy: torch.Tensor):
        batch_size = final_weights.size(0)
        sorted_weights, sorted_indices = torch.sort(final_weights, dim=1, descending=True)
        cum_weights = torch.cumsum(sorted_weights, dim=1)
        mask = (cum_weights <= self.cum_weight_threshold).float()
        mask[:, 0] = 1.0
        high_hesitancy_mask = (avg_hesitancy > self.hesitancy_threshold).unsqueeze(1)
        mask = torch.where(high_hesitancy_mask, torch.ones_like(mask), mask)
        final_mask = torch.zeros_like(final_weights)
        final_mask.scatter_(1, sorted_indices, mask)
        return final_mask

    def forward(self, x: torch.Tensor, return_details: bool = False):
        final_weights, all_memberships = self.hesitant_fuzzy(x)
        hesitancy = all_memberships.var(dim=2, unbiased=False)
        avg_hesitancy = hesitancy.mean(dim=1)
        mask = self._compute_mask_vectorized(final_weights, avg_hesitancy)
        final_weights = final_weights * mask
        final_weights = final_weights / (final_weights.sum(dim=1, keepdim=True) + 1e-8)

        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
        active_models = torch.any(final_weights > 0, dim=0).nonzero(as_tuple=True)[0]

        for i in active_models:
            i_idx = int(i.item())
            x_n = self.normalizations(x, i_idx)
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            # 🆕 Apply per-model temperature calibration before aggregation
            outputs[:, i] = out / self.temperatures[i_idx]

        final_output = (outputs * final_weights.unsqueeze(-1)).sum(dim=1)
        if return_details:
            return final_output, final_weights, all_memberships, outputs
        return final_output, final_weights


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


# ================== TEMPERATURE CALIBRATION ==================
def fit_temperatures(base_models, val_loader, device, means, stds, model_names=None):
    """
    Fit per-model Temperature Scaling on the validation set via NLL minimization.
    Calibrated logit = raw_logit / T_i.
    Returns list of temperatures, one per model.
    """
    temperatures = []
    criterion = nn.BCEWithLogitsLoss()
    for i, model in enumerate(base_models):
        model.eval()
        normalizer = MultiModelNormalization([means[i]], [stds[i]]).to(device)

        all_logits, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = normalizer(images.to(device), 0)
                out = model(images)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                all_logits.append(out.squeeze(1).detach().cpu())
                all_labels.append(labels.float().detach().cpu())

        logits = torch.cat(all_logits).float()
        labels = torch.cat(all_labels).float()

        T = torch.ones(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=100)

        def closure():
            optimizer.zero_grad()
            loss = criterion(logits / T.clamp(min=0.05), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        T_val = float(T.item())
        temperatures.append(T_val)
        name = model_names[i] if model_names else f"Model {i+1}"
        print(f"  {name:<30}: T = {T_val:.4f}")

    return temperatures


# ================== EVALUATION FUNCTIONS ==================
@torch.no_grad()
def evaluate_single_model_exact(model: nn.Module, loader: DataLoader, device: torch.DeviceObjType,
                                name: str, mean: Tuple[float, float, float],
                                std: Tuple[float, float, float]) -> float:
    model.eval()
    normalizer = MultiModelNormalization([mean], [std]).to(device)
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=f"Evaluating {name}"):
        images, labels = images.to(device), labels.to(device).float()
        images = normalizer(images, 0)
        out = model(images)
        if isinstance(out, (tuple, list)): out = out[0]
        pred = (out.squeeze(1) > 0).long()
        correct += pred.eq(labels.long()).sum().item()
        total += labels.size(0)

    acc = 100. * correct / total
    print(f" {name}: {acc:.2f}%")
    return acc


@torch.no_grad()
def evaluate_accuracy_and_loss(model, loader, device, criterion):

    model.eval()
    correct = 0
    total_loss = 0.0
    total_real_samples = len(loader.dataset)

    for images, labels in tqdm(loader, desc="Validation"):
        images, labels = images.to(device), labels.to(device).float()
        outputs, _ = model(images)

        loss = criterion(outputs.squeeze(1), labels)
        total_loss += loss.item() * images.size(0)

        pred = (outputs.squeeze(1) > 0).long()
        correct += pred.eq(labels.long()).sum().item()

    acc = 100. * correct / total_real_samples
    avg_loss = total_loss / total_real_samples
    return acc, avg_loss


# ================== TRAINING (Single GPU) ==================
def train_hesitant_fuzzy(ensemble_model, train_loader, val_loader, num_epochs, lr,
                        device, save_dir, model_names):
    os.makedirs(save_dir, exist_ok=True)
    hesitant_net = ensemble_model.hesitant_fuzzy
    optimizer = torch.optim.AdamW(hesitant_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()
    best_val_acc = 0.0

    total_real_samples = len(train_loader.dataset)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    epoch_times, train_times, val_times = [], [], []
    total_sw = Stopwatch(device).start()

    print("="*70)
    print("Training Fuzzy Hesitant Network (Single GPU)")
    print("="*70)
    print(f"Trainable params: {sum(p.numel() for p in hesitant_net.parameters()):,}")
    print(f"Epochs: {num_epochs} | Initial LR: {lr}")
    print(f"Hesitant memberships per model: {hesitant_net.num_memberships}")
    print(f"Number of models: {len(model_names)}")
    print(f"Total real training samples: {total_real_samples:,}\n")

    for epoch in range(num_epochs):
        epoch_sw = Stopwatch(device).start()
        ensemble_model.train()

        train_loss = 0.0
        train_correct = 0

        sum_per_model_hesitancy = torch.zeros(len(model_names), device=device)
        sum_cumsum_used = 0
        sum_active_models = torch.zeros(len(model_names), device=device)

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()

            outputs, weights, memberships, _ = ensemble_model(images, return_details=True)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            pred = (outputs.squeeze(1) > 0).long()
            train_correct += pred.eq(labels.long()).sum().item()

            per_model_hesitancy = memberships.var(dim=2, unbiased=False)
            sum_per_model_hesitancy += per_model_hesitancy.sum(dim=0)

            active_mask = (weights > 1e-4).float()
            num_active_per_sample = active_mask.sum(dim=1)
            cumsum_used_samples = (num_active_per_sample < len(model_names)).sum().item()
            sum_cumsum_used += cumsum_used_samples
            sum_active_models += active_mask.sum(dim=0)

        t_train = epoch_sw.stop()

        train_loss = train_loss / total_real_samples
        train_acc = 100. * train_correct / total_real_samples

        avg_per_model_hesitancy = sum_per_model_hesitancy / total_real_samples
        avg_cumsum_usage = (sum_cumsum_used / total_real_samples) * 100
        avg_model_activation = (sum_active_models / total_real_samples) * 100
        overall_mean_hesitancy = avg_per_model_hesitancy.mean().item()

        val_sw = Stopwatch(device).start()
        val_acc, val_loss = evaluate_accuracy_and_loss(ensemble_model, val_loader, device, criterion)
        t_val = val_sw.stop()
        t_epoch = epoch_sw.stop()
        epoch_times.append(t_epoch); train_times.append(t_train); val_times.append(t_val)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*70}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Time: epoch {t_epoch:.1f}s (train {t_train:.1f}s | val {t_val:.1f}s)")

        print(f"\n{'Hesitancy (Variance) per Model:':^70}")
        print(f"{'-'*70}")
        for i, name in enumerate(model_names):
            hesitancy_val = avg_per_model_hesitancy[i].item()
            print(f"  {i+1:2d}. {name:<30}: {hesitancy_val:.6f}")
        print(f"{'-'*70}")
        print(f"  {'Overall Mean Hesitancy:':<30}  {overall_mean_hesitancy:.6f}")

        print(f"\n{'Cumulative Weight Threshold (Cumsum) Analysis:':^70}")
        print(f"{'-'*70}")
        print(f"  {'Cumsum activated in:':<30}  {avg_cumsum_usage:.2f}% of samples")
        if avg_cumsum_usage > 50:
            print(f"  {'Status:':<30}  ✓ Efficient")
        elif avg_cumsum_usage > 20:
            print(f"  {'Status:':<30}  ≈ Moderate")
        else:
            print(f"  {'Status:':<30}  ✗ Low efficiency")
        print(f"  {'All models used in:':<30}  {100-avg_cumsum_usage:.2f}% of samples")

        print(f"\n{'Model Activation Frequency:':^70}")
        print(f"{'-'*70}")
        for i, name in enumerate(model_names):
            activation_pct = avg_model_activation[i].item()
            bar_length = int(activation_pct / 2)
            bar = '█' * bar_length
            print(f"  {i+1:2d}. {name:<30}: {activation_pct:5.1f}% {bar}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, 'best_hesitant_fuzzy.pt')
            torch.save({'hesitant_state_dict': hesitant_net.state_dict()}, save_path)
            print(f"\n✓ Best model saved → {val_acc:.2f}%")

        print()

    total_time = total_sw.stop()
    timing = summarize_training_times(epoch_times, train_times, val_times, total_time)
    print(f"Total training time: {total_time/60:.2f} min "
          f"(mean epoch: {timing['mean_epoch_time_s']:.1f}s)")

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), history['train_loss'], label='train', color='blue', linewidth=2)
    plt.plot(range(1, num_epochs + 1), history['val_loss'], label='validation', color='orange', linewidth=2)
    plt.title('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), history['train_acc'], label='train', color='blue', linewidth=2)
    plt.plot(range(1, num_epochs + 1), history['val_acc'], label='validation', color='orange', linewidth=2)
    plt.title('Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()
    curves_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Training curves (Loss & Acc) saved to: {curves_path}")

    return best_val_acc, timing


# ================== MAIN FUNCTION ==================
def main():
    parser = argparse.ArgumentParser(description="Optimized Fuzzy Hesitant Ensemble Training (Single GPU)")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, required=True, choices=['wild','artiface', 'hard_fake_real', 'deepfake_lab', 'uadfV', 'custom_genai', 'custom_genai_v2', 'real_fake_dataset', '20k'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--num_memberships', type=int, default=3)
    parser.add_argument('--cum_weight_threshold', type=float, default=0.9)
    parser.add_argument('--hesitancy_threshold', type=float, default=0.2)

    # 🆕 Calibration flag (default ON; use --no_calibrate for ablation)
    parser.add_argument('--calibrate', action='store_true', default=True,
                        help='Apply per-model Temperature Scaling (default: enabled)')
    parser.add_argument('--no_calibrate', action='store_false', dest='calibrate',
                        help='Disable per-model Temperature Scaling (ablation baseline)')

    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)

    args = parser.parse_args()

    if len(args.model_names) != len(args.model_paths):
        raise ValueError("Number of model_names must match model_paths")

    set_seed(args.seed)

    device = setup_device()
    pipeline_sw = Stopwatch(device).start()

    print("="*70)
    print(f"OPTIMIZED FUZZY HESITANT ENSEMBLE TRAINING")
    print(f"Single GPU Mode | Seed: {args.seed}")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Models: {len(args.model_paths)}")
    print(f"Calibration enabled: {args.calibrate}")
    print("="*70 + "\n")

    DEFAULT_MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    DEFAULT_STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]

    num_models = len(args.model_paths)
    if num_models > len(DEFAULT_MEANS):
        MEANS = DEFAULT_MEANS + [DEFAULT_MEANS[-1]] * (num_models - len(DEFAULT_MEANS))
        STDS = DEFAULT_STDS + [DEFAULT_STDS[-1]] * (num_models - len(DEFAULT_STDS))
    else:
        MEANS = DEFAULT_MEANS[:num_models]
        STDS = DEFAULT_STDS[:num_models]

    base_models = load_pruned_models(args.model_paths, device)
    MODEL_NAMES = args.model_names[:len(base_models)]

    ensemble = FuzzyHesitantEnsemble(
        base_models, MEANS, STDS,
        num_memberships=args.num_memberships,
        freeze_models=True,
        cum_weight_threshold=args.cum_weight_threshold,
        hesitancy_threshold=args.hesitancy_threshold
    ).to(device)

    trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
    total = sum(p.numel() for p in ensemble.parameters())
    print(f"Total params: {total:,} | Trainable: {trainable:,} | Frozen: {total-trainable:,}\n")

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=False, seed=args.seed, is_main=True)

    print("\n" + "="*70)
    print("INDIVIDUAL MODEL PERFORMANCE (Before Training)")
    print("="*70)

    individual_accs = []
    for i, model in enumerate(base_models):
        acc = evaluate_single_model_exact(model, test_loader, device,
                                          f"Model {i+1} ({MODEL_NAMES[i]})",
                                          MEANS[i], STDS[i])
        individual_accs.append(acc)

    best_single = max(individual_accs)
    best_idx = individual_accs.index(best_single)
    print(f"\nBest Single: Model {best_idx+1} ({MODEL_NAMES[best_idx]}) → {best_single:.2f}%")
    print("="*70)

    best_val_acc, train_timing = train_hesitant_fuzzy(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_dir, MODEL_NAMES
    )

    # لود بهترین مدل
    ckpt_path = os.path.join(args.save_dir, 'best_hesitant_fuzzy.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        ensemble.hesitant_fuzzy.load_state_dict(ckpt['hesitant_state_dict'])
        print("Best model loaded.\n")

    # 🆕 Fit per-model temperature scaling on validation set (post-hoc, no retrain)
    if args.calibrate:
        print("\n" + "="*70)
        print("FITTING PER-MODEL TEMPERATURE SCALING (on validation set)")
        print("="*70)
        temperatures = fit_temperatures(
            base_models, val_loader, device, MEANS, STDS, MODEL_NAMES)
        ensemble.temperatures = temperatures
        print(f"\n✅ Calibration applied. Temperatures: "
              f"{[f'{t:.4f}' for t in temperatures]}\n")
    else:
        temperatures = [1.0] * len(MODEL_NAMES)
        print("\n⚠️  Calibration disabled (T = 1.0 for all models).\n")

    print("\n" + "="*70)
    print("FINAL ENSEMBLE EVALUATION")
    print("="*70)

    efficiency = benchmark_efficiency(ensemble, test_loader, device, MODEL_NAMES,
                                  latency_samples=len(test_loader.dataset))
    efficiency['training_time'] = train_timing
    print_headline(efficiency)

    eval_sw = Stopwatch(device).start()
    final_acc, y_true, y_scores = final_evaluation_unified(
        ensemble, test_loader, device, args.save_dir, MODEL_NAMES, args
    )
    final_eval_time = eval_sw.stop()

    # 🆕 ECE report
    test_ece = compute_ece(y_true, y_scores)
    print(f"\n📊 Test ECE (Expected Calibration Error): {test_ece:.4f}")

    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"Best Single Model: {best_single:.2f}%")
    print(f"Ensemble Accuracy: {final_acc:.2f}%")
    print(f"Improvement: {final_acc - best_single:+.2f}%")
    print(f"Test ECE: {test_ece:.4f}")
    print("="*70)

    # ذخیره نتایج JSON
    final_results = {
        'seed': args.seed,
        'method': 'Fuzzy_Hesitant',
        'calibration': {
            'enabled': bool(args.calibrate),
            'temperatures': list(ensemble.temperatures)
        },
        'best_single_model': {'name': MODEL_NAMES[best_idx], 'accuracy': float(best_single)},
        'ensemble': {
            'test_accuracy': float(final_acc),
            'test_ece': float(test_ece)
        },
        'improvement': float(final_acc - best_single)
    }
    with open(os.path.join(args.save_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)

    torch.save({
        'ensemble_state_dict': ensemble.state_dict(),
        'model_names': MODEL_NAMES,
        'means': MEANS,
        'stds': STDS,
        'temperatures': ensemble.temperatures,   # 🆕
        'seed': args.seed
    }, os.path.join(args.save_dir, 'final_ensemble_model.pt'))

    efficiency['final_evaluation_time_s'] = final_eval_time
    efficiency['total_pipeline_time_s'] = pipeline_sw.stop()
    with open(os.path.join(args.save_dir, 'efficiency_report.json'), 'w') as f:
        json.dump(efficiency, f, indent=2)
    print(f"Efficiency report saved: {os.path.join(args.save_dir, 'efficiency_report.json')}")


if __name__ == "__main__":
    main()
