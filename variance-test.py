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

# ================== REVIEWER RESPONSE UTILS ==================
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
# =============================================================


# ================== UTILITY FUNCTIONS ==================
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


# ================== UNIFIED FINAL EVALUATION ==================
def final_evaluation_unified(model, test_loader, device, save_dir, model_names, args):
    model.eval()
    test_dataset = test_loader.dataset
    num_samples = len(test_dataset)

    lines = []
    lines.append("=" * 100)
    lines.append("SAMPLE-BY-SAMPLE PREDICTIONS (For McNemar Test Comparison):")
    lines.append("=" * 100)
    header = f"{'ID':<10} {'Filename':<60} {'True':<10} {'Pred':<10} {'Correct':<10}"
    lines.append(header)
    lines.append("-" * 100)

    TP, TN, FP, FN = 0, 0, 0, 0
    all_y_true, all_y_score, all_y_pred, all_model_weights = [], [], [], []

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
            lines.append(f"{i+1:<10} {filename:<60} {label_int:<10} {pred_int:<10} {correct_str:<10}")

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
    balanced_acc    = ((recall_real + recall_fake) / 2) * 100

    cm = np.array([[TN, FP], [FN, TP]])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Fake', 'Predicted Real'],
                yticklabels=['Actual Fake', 'Actual Real'],
                cbar=True, annot_kws={"size": 16})
    plt.title(f'Confusion Matrix\nAccuracy: {acc:.2f}% | Balanced Acc: {balanced_acc:.2f}% | Macro-F1: {macro_f1:.4f}',
              fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion Matrix saved to: {cm_path}")

    print(f"\n{'='*70}\nFINAL RESULTS\n{'='*70}")
    print(f"Accuracy          : {acc:.2f}%")
    print(f"Balanced Accuracy : {balanced_acc:.2f}%")
    print(f"Macro-F1          : {macro_f1:.4f}\n")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 45)
    print(f"{'Fake (0)':<12} {precision_fake:10.4f} {recall_fake:10.4f} {f1_fake:10.4f}")
    print(f"{'Real (1)':<12} {precision_real:10.4f} {recall_real:10.4f} {f1_real:10.4f}")
    print("-" * 45)
    print(f"{'Macro Avg':<12} {macro_precision:10.4f} {macro_recall:10.4f} {macro_f1:10.4f}")
    print(f"\nCorrect Predictions  : {correct_count} ({acc:.2f}%)")
    print(f"Incorrect Predictions: {total_samples - correct_count} ({100 - acc:.2f}%)")
    print("=" * 70)

    if len(all_model_weights) > 0:
        weights_np = np.array(all_model_weights)
        print(f"\n{'='*70}\nMODEL ACTIVATION STATISTICS (Final Evaluation)\n{'='*70}")

        activation_threshold = 1e-4
        active_mask = (weights_np > activation_threshold).astype(float)
        activation_rates = active_mask.mean(axis=0) * 100
        avg_weights = weights_np.mean(axis=0)
        max_weights = weights_np.max(axis=0)

        print(f"\n  {'Model':<35} {'Active %':>10} {'Avg Weight':>12} {'Max Weight':>12}")
        print(f"  {'-'*70}")
        for i, name in enumerate(model_names):
            act_pct = activation_rates[i]
            bar_length = int(act_pct / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"  {i+1:2d}. {name:<30}: {act_pct:6.2f}%  | {avg_weights[i]:.6f}  | {max_weights[i]:.6f}")
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
        with open(os.path.join(save_dir, "model_activation_stats.json"), 'w', encoding='utf-8') as f:
            json.dump(activation_stats, f, indent=2, ensure_ascii=False)

    output_str = [
        "-" * 100, "SUMMARY STATISTICS:", "-" * 100,
        f"Overall Accuracy: {acc:.2f}%",
        f"Balanced Accuracy: {balanced_acc:.2f}%",
        f"Macro F1-Score: {macro_f1:.4f}",
        "\n--- Real (1) Class Metrics ---",
        f"Precision (Real): {precision_real:.4f}",
        f"Recall (Real):    {recall_real:.4f}",
        f"F1-Score (Real):  {f1_real:.4f}",
        "\n--- Fake (0) Class Metrics ---",
        f"Precision (Fake): {precision_fake:.4f}",
        f"Recall (Fake):    {recall_fake:.4f}",
        f"F1-Score (Fake):  {f1_fake:.4f}",
        f"\nCorrect Predictions: {correct_count} ({acc:.2f}%)",
        f"Incorrect Predictions: {total_samples - correct_count} ({100-acc:.2f}%)",
        f"\nConfusion Matrix saved to: confusion_matrix.png",
    ]
    output_str.extend(lines)
    with open(os.path.join(save_dir, 'prediction_log.txt'), 'w') as f:
        f.write("\n".join(output_str))

    y_true_np = np.array(all_y_true)
    y_score_np = np.array(all_y_score)
    y_pred_np = np.array(all_y_pred)

    roc_data_json = {
        "metadata": {
            "seed": args.seed, "dataset": args.dataset,
            "num_samples": int(total_samples),
            "positive_count": int(np.sum(y_true_np)),
            "negative_count": int(total_samples - np.sum(y_true_np)),
            "model": "fuzzy_hesitant_ensemble",
            "hesitancy_measure": args.hesitancy_measure,
        },
        "y_true": y_true_np.tolist(),
        "y_score": y_score_np.tolist(),
        "y_pred": y_pred_np.tolist()
    }
    with open(os.path.join(save_dir, "roc_data_test.json"), 'w', encoding='utf-8') as f:
        json.dump(roc_data_json, f, indent=2, ensure_ascii=False)

    with open(os.path.join(save_dir, "roc_data_test.txt"), 'w', encoding='utf-8') as f:
        f.write("y_true\ty_score\ty_pred\n")
        for t, s, p in zip(y_true_np, y_score_np, y_pred_np):
            f.write(f"{int(t)}\t{s:.6f}\t{int(p)}\n")

    vis_dir = os.path.join(save_dir, 'visualizations')
    generate_visualizations(model, test_loader, device, vis_dir, model_names,
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
                 hesitancy_measure: str = 'variance'):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.hesitant_fuzzy = HesitantFuzzyMembership(
            input_dim=128, num_models=self.num_models, num_memberships=num_memberships)
        self.cum_weight_threshold = cum_weight_threshold
        self.hesitancy_threshold = hesitancy_threshold
        self.hesitancy_measure = hesitancy_measure

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

    # ================== REVIEWER RESPONSE ==================
    def _compute_hesitancy(self, all_memberships: torch.Tensor) -> torch.Tensor:
        if self.hesitancy_measure == 'variance':
            return all_memberships.var(dim=2, unbiased=False).mean(dim=1)
        elif self.hesitancy_measure == 'cv':
            mean_mem = all_memberships.mean(dim=2)
            std_mem = all_memberships.std(dim=2, unbiased=False)
            return (std_mem / (mean_mem + 1e-8)).mean(dim=1)
        elif self.hesitancy_measure == 'entropy':
            p = all_memberships / (all_memberships.sum(dim=2, keepdim=True) + 1e-8)
            K = all_memberships.size(2)
            ent = -(p * torch.log(p + 1e-8)).sum(dim=2) / np.log(K)
            return ent.mean(dim=1)
        elif self.hesitancy_measure == 'range':
            rng = (all_memberships.max(dim=2).values - all_memberships.min(dim=2).values)
            return rng.mean(dim=1)
        elif self.hesitancy_measure == 'combined':
            var = all_memberships.var(dim=2, unbiased=False).mean(dim=1)
            mean_level = all_memberships.mean(dim=2).mean(dim=1)
            return var + (1.0 - mean_level) * 0.5
        else:
            raise ValueError(f"Unknown hesitancy_measure: {self.hesitancy_measure}")

    # ============ NEW: correct per-model outputs independent of mask ============
    @torch.no_grad()
    def compute_all_model_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute outputs from ALL base models (ignoring the mask).
        Used for correctly validating 'model competence' in the reviewer response.
        """
        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
        for i in range(self.num_models):
            x_n = self.normalizations(x, i)
            out = self.models[i](x_n)
            if isinstance(out, (tuple, list)):
                out = out[0]
            outputs[:, i] = out
        return outputs
    # ============================================================================

    def _compute_mask_vectorized(self, final_weights: torch.Tensor, avg_hesitancy: torch.Tensor):
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
        avg_hesitancy = self._compute_hesitancy(all_memberships)
        mask = self._compute_mask_vectorized(final_weights, avg_hesitancy)
        final_weights = final_weights * mask
        final_weights = final_weights / (final_weights.sum(dim=1, keepdim=True) + 1e-8)

        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
        active_models = torch.any(final_weights > 0, dim=0).nonzero(as_tuple=True)[0]
        for i in active_models:
            x_n = self.normalizations(x, i.item())
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            outputs[:, i] = out

        final_output = (outputs * final_weights.unsqueeze(-1)).sum(dim=1)
        if return_details:
            return final_output, final_weights, all_memberships, outputs
        return final_output, final_weights


# ================== HESITANCY UTILITIES ==================
def compute_uncertainty_measures(all_memberships: torch.Tensor) -> dict:
    var = all_memberships.var(dim=2, unbiased=False).mean(dim=1)
    mean_mem = all_memberships.mean(dim=2)
    std_mem = all_memberships.std(dim=2, unbiased=False)
    cv = (std_mem / (mean_mem + 1e-8)).mean(dim=1)
    p = all_memberships / (all_memberships.sum(dim=2, keepdim=True) + 1e-8)
    K = all_memberships.size(2)
    ent = -(p * torch.log(p + 1e-8)).sum(dim=2) / np.log(K)
    ent = ent.mean(dim=1)
    rng = (all_memberships.max(dim=2).values - all_memberships.min(dim=2).values).mean(dim=1)
    mean_level = mean_mem.mean(dim=1)
    combined = var + (1.0 - mean_level) * 0.5
    return {
        'variance': var, 'cv': cv, 'entropy': ent, 'range': rng,
        'mean_level': mean_level, 'combined': combined,
    }


@torch.no_grad()
def calibrate_hesitancy_threshold(ensemble, loader, device, measure, percentile=75.0):
    """Calibrate the hesitancy threshold from a chosen measure on a validation set.
    Uses percentile-based threshold (default: 75th percentile = ~25% of samples get override).
    """
    ensemble.eval()
    original = ensemble.hesitancy_measure
    ensemble.hesitancy_measure = measure

    ds = loader.dataset
    all_vals = []
    for i in tqdm(range(len(ds)), desc=f"Calibrating [{measure}]"):
        try:
            image, _ = ds[i]
        except Exception:
            continue
        image = image.unsqueeze(0).to(device)
        _, _, mem, _ = ensemble(image, return_details=True)
        m = compute_uncertainty_measures(mem)
        all_vals.append(float(m[measure].item()))

    ensemble.hesitancy_measure = original

    if len(all_vals) == 0:
        raise ValueError("No samples collected for calibration")

    vals = np.array(all_vals)
    threshold = float(np.percentile(vals, percentile))
    return threshold, vals


@torch.no_grad()
def validate_hesitancy_measure(ensemble, loader, device, model_names, save_dir, tag="val"):
    """Validate hesitancy measures against prediction errors and model competence
    on a held-out set (should be VALIDATION set, not test, to avoid leakage).
    """
    ensemble.eval()
    ds = loader.dataset
    keys = ['variance', 'cv', 'entropy', 'range', 'mean_level', 'combined']
    collected = {k: [] for k in keys}
    errors, per_model_errors = [], []

    for i in tqdm(range(len(ds)), desc=f"Validate hesitancy [{tag}]"):
        try:
            image, label = ds[i]
        except Exception:
            continue
        image = image.unsqueeze(0).to(device)
        label_int = int(label)

        out, _, mem, _ = ensemble(image, return_details=True)
        m = compute_uncertainty_measures(mem)
        for k in keys:
            collected[k].append(float(m[k].item()))

        p = torch.sigmoid(out.squeeze()).item()
        pred = int(p > 0.5)
        errors.append(0 if pred == label_int else 1)

        # CORRECT per-model error computation (independent of mask)
        all_logits = ensemble.compute_all_model_logits(image)
        m_errs = [
            (0 if int(all_logits[0, j, 0].item() > 0) == label_int else 1)
            for j in range(ensemble.num_models)
        ]
        per_model_errors.append(m_errs)

    errors_np = np.array(errors)
    model_errs_np = np.array(per_model_errors)
    results = {}

    print("\n" + "=" * 110)
    print(f"VALIDATION OF HESITANCY MEASURES ON {tag.upper()} SET")
    print("=" * 110)
    print(f"#samples: {len(errors_np)} | Error rate: {errors_np.mean()*100:.2f}%")
    print("-" * 110)
    print(f"{'Measure':<12} {'Pearson':>10} {'p-val':>9} {'Spearman':>10} {'p-val':>9} "
          f"{'AUC(err)':>10} {'Spear(model-err)':>18}")
    print("-" * 110)

    for name, vals in collected.items():
        v = np.array(vals)
        if name == 'mean_level':
            try:
                r_correct, p_correct = pearsonr(v, 1 - errors_np)
            except Exception:
                r_correct, p_correct = 0.0, 1.0
            results[name] = {
                'note': 'higher is more confident',
                'pearson_with_correctness': float(r_correct),
                'p_value': float(p_correct),
            }
            print(f"{name:<12} {'(confidence)':>10} {p_correct:>9.4f} {'-':>10} {'-':>9} {'-':>10} {'-':>18}")
            continue

        try:
            pr, pr_p = pearsonr(v, errors_np)
        except Exception:
            pr, pr_p = 0.0, 1.0
        try:
            sr, sr_p = spearmanr(v, errors_np)
        except Exception:
            sr, sr_p = 0.0, 1.0

        if len(np.unique(errors_np)) > 1:
            auc = roc_auc_score(errors_np, v)
        else:
            auc = 0.5

        try:
            sm, sm_p = spearmanr(v, model_errs_np.mean(axis=1))
        except Exception:
            sm, sm_p = 0.0, 1.0

        mean_correct   = float(v[errors_np == 0].mean()) if (errors_np == 0).any() else 0.0
        mean_incorrect = float(v[errors_np == 1].mean()) if (errors_np == 1).any() else 0.0

        results[name] = {
            'pearson_with_error': float(pr),
            'pearson_p_value': float(pr_p),
            'spearman_with_error': float(sr),
            'spearman_p_value': float(sr_p),
            'auc_error_detection': float(auc),
            'spearman_with_model_errors': float(sm),
            'spearman_model_p_value': float(sm_p),
            'mean_hesitancy_correct': mean_correct,
            'mean_hesitancy_incorrect': mean_incorrect,
        }

        print(f"{name:<12} {pr:>+10.3f} {pr_p:>9.4f} {sr:>+10.3f} {sr_p:>9.4f} "
              f"{auc:>10.3f} {sm:>+18.3f}")

    # Mean hesitancy (correct vs incorrect) -- direct answer to reviewer's concern
    print("-" * 110)
    print("Mean hesitancy (correct vs incorrect):")
    for name, vals in collected.items():
        if name == 'mean_level':
            continue
        v = np.array(vals)
        mc = float(v[errors_np == 0].mean()) if (errors_np == 0).any() else 0.0
        mi = float(v[errors_np == 1].mean()) if (errors_np == 1).any() else 0.0
        print(f"  {name:<12} correct={mc:>8.4f}  incorrect={mi:>8.4f}  delta={mi-mc:>+8.4f}")

    candidates = [n for n in keys if n != 'mean_level']
    best = max(candidates, key=lambda n: results[n]['auc_error_detection'])
    print("-" * 110)
    print(f"Best measure by AUC(err): {best} (AUC = {results[best]['auc_error_detection']:.4f})")
    print(f"Active measure: {ensemble.hesitancy_measure} | Threshold: {ensemble.hesitancy_threshold:.4f}")
    print("=" * 110 + "\n")

    with open(os.path.join(save_dir, f'hesitancy_validation_{tag}.json'), 'w') as f:
        json.dump({
            'split': tag,
            'num_samples': int(len(errors_np)),
            'error_rate': float(errors_np.mean()),
            'active_measure': ensemble.hesitancy_measure,
            'active_threshold': float(ensemble.hesitancy_threshold),
            'per_model_error_rates': {
                name: float(model_errs_np[:, j].mean())
                for j, name in enumerate(model_names)
            },
            'measures': results,
            'best_measure_by_auc': best,
        }, f, indent=2)

    _plot_hesitancy_hist(collected, errors_np, save_dir, tag)
    return results


def _plot_hesitancy_hist(collected, errors_np, save_dir, tag="val"):
    to_plot = [n for n in ['variance', 'entropy', 'cv', 'combined'] if n in collected]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, name in zip(axes, to_plot):
        v = np.array(collected[name])
        ax.hist(v[errors_np == 0], bins=30, alpha=0.6, density=True, label='Correct', color='green')
        ax.hist(v[errors_np == 1], bins=30, alpha=0.6, density=True, label='Incorrect', color='red')
        ax.set_xlabel(name.capitalize())
        ax.set_ylabel('Density')
        ax.set_title(f'{name.capitalize()}')
        ax.legend()
        ax.grid(alpha=0.3)
    for ax in axes[len(to_plot):]:
        ax.axis('off')
    plt.tight_layout()
    path = os.path.join(save_dir, f'hesitancy_validation_distributions_{tag}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Distributions plot saved: {path}")


@torch.no_grad()
def validate_model_competence(ensemble, loader, device, model_names, save_dir, tag="val"):
    """Validate that the learned final_weights correlate with actual per-model correctness.

    Direct test for "model competence estimation": does the ensemble give higher
    weight to models that are actually correct on each sample?
    """
    ensemble.eval()
    ds = loader.dataset
    N = ensemble.num_models

    all_weights = []           # [S, N]
    all_per_model_correct = [] # [S, N]
    all_ensemble_correct = []  # [S]

    for i in tqdm(range(len(ds)), desc=f"Validate competence [{tag}]"):
        try:
            image, label = ds[i]
        except Exception:
            continue
        image = image.unsqueeze(0).to(device)
        label_int = int(label)

        out, final_weights, _, _ = ensemble(image, return_details=True)
        w = final_weights.squeeze(0).cpu().numpy()

        p = torch.sigmoid(out.squeeze()).item()
        pred = int(p > 0.5)
        all_ensemble_correct.append(1 if pred == label_int else 0)

        all_logits = ensemble.compute_all_model_logits(image)
        per_correct = [
            (1 if int(all_logits[0, j, 0].item() > 0) == label_int else 0)
            for j in range(N)
        ]
        all_weights.append(w)
        all_per_model_correct.append(per_correct)

    weights_np = np.array(all_weights)
    correct_np = np.array(all_per_model_correct)
    ens_correct_np = np.array(all_ensemble_correct)
    S = weights_np.shape[0]

    print("\n" + "=" * 110)
    print(f"VALIDATION OF MODEL-COMPETENCE ESTIMATION ON {tag.upper()} SET")
    print("=" * 110)
    print(f"#samples: {S} | Ensemble accuracy: {ens_correct_np.mean()*100:.2f}%")
    print(f"#models : {N}")
    print("-" * 110)

    results = {
        'split': tag,
        'num_samples': int(S),
        'num_models': int(N),
        'ensemble_accuracy': float(ens_correct_np.mean()),
        'per_model_accuracy': {model_names[j]: float(correct_np[:, j].mean())
                                for j in range(N)},
        'per_model_mean_weight': {model_names[j]: float(weights_np[:, j].mean())
                                   for j in range(N)},
    }

    print("\n[1] Per-model correlation: final weight vs correctness (across samples)")
    print(f"    {'Model':<30} {'Acc':>8} {'MeanW':>8} {'Pearson':>10} {'p-val':>10} "
          f"{'Spearman':>10} {'p-val':>10}")
    print("    " + "-" * 96)

    per_model_corr = {}
    for j, name in enumerate(model_names):
        w = weights_np[:, j]
        c = correct_np[:, j]
        try:
            pr, pr_p = pearsonr(w, c)
        except Exception:
            pr, pr_p = 0.0, 1.0
        try:
            sr, sr_p = spearmanr(w, c)
        except Exception:
            sr, sr_p = 0.0, 1.0
        per_model_corr[name] = {
            'accuracy': float(c.mean()),
            'mean_weight': float(w.mean()),
            'pearson_weight_vs_correct': float(pr),
            'pearson_p_value': float(pr_p),
            'spearman_weight_vs_correct': float(sr),
            'spearman_p_value': float(sr_p),
        }
        print(f"    {name:<30} {c.mean()*100:>7.2f}% {w.mean():>8.4f} "
              f"{pr:>+10.3f} {pr_p:>10.4f} {sr:>+10.3f} {sr_p:>10.4f}")

    print("\n[2] Pooled correlation over all (sample, model) pairs")
    w_flat = weights_np.flatten()
    c_flat = correct_np.flatten()
    try:
        pr, pr_p = pearsonr(w_flat, c_flat)
    except Exception:
        pr, pr_p = 0.0, 1.0
    try:
        sr, sr_p = spearmanr(w_flat, c_flat)
    except Exception:
        sr, sr_p = 0.0, 1.0
    print(f"    Pearson:  r = {pr:+.4f}  (p = {pr_p:.4f})")
    print(f"    Spearman: r = {sr:+.4f}  (p = {sr_p:.4f})")
    results['pooled_correlation'] = {
        'pearson': float(pr), 'pearson_p': float(pr_p),
        'spearman': float(sr), 'spearman_p': float(sr_p),
    }

    print("\n[3] Top-1 weight accuracy: is the highest-weighted model correct?")
    top1_model_idx = weights_np.argmax(axis=1)
    top1_model_correct = correct_np[np.arange(S), top1_model_idx]
    top1_acc = float(top1_model_correct.mean())
    top1_selections = np.bincount(top1_model_idx, minlength=N)

    print(f"    Top-1 model correct rate: {top1_acc*100:.2f}%")
    print(f"    {'Model':<30} {'#Top1':>8} {'%Top1':>8} {'AccWhenTop1':>14}")
    print("    " + "-" * 62)
    for j, name in enumerate(model_names):
        n_top1 = int(top1_selections[j])
        acc_when_top1 = float(correct_np[top1_model_idx == j, j].mean()) if n_top1 > 0 else 0.0
        print(f"    {name:<30} {n_top1:>8d} {n_top1/S*100:>7.2f}% {acc_when_top1*100:>13.2f}%")
        per_model_corr[name]['times_top1'] = n_top1
        per_model_corr[name]['pct_top1'] = float(n_top1 / S)
        per_model_corr[name]['accuracy_when_top1'] = acc_when_top1

    results['top1_weight_accuracy'] = {
        'top1_correct_rate': top1_acc,
        'per_model_top1_count': {model_names[j]: int(top1_selections[j]) for j in range(N)},
    }

    weighted_competence = (weights_np * correct_np).sum(axis=1)
    print("\n[4] Weighted competence score (sum_j w_j * correct_j):")
    print(f"    Mean:   {weighted_competence.mean():.4f}")
    print(f"    Median: {np.median(weighted_competence):.4f}")
    print(f"    Std:    {weighted_competence.std():.4f}")
    results['weighted_competence'] = {
        'mean': float(weighted_competence.mean()),
        'median': float(np.median(weighted_competence)),
        'std': float(weighted_competence.std()),
    }

    any_correct = correct_np.max(axis=1)
    oracle_acc = float(any_correct.mean())
    print("\n[5] Oracle vs achieved:")
    print(f"    Oracle (pick any correct model): {oracle_acc*100:.2f}%")
    print(f"    Top-1 weight accuracy:            {top1_acc*100:.2f}%")
    print(f"    Ensemble accuracy:                {ens_correct_np.mean()*100:.2f}%")
    print(f"    Gap (oracle - top1):              {(oracle_acc-top1_acc)*100:+.2f}%")
    results['oracle'] = {
        'oracle_accuracy': oracle_acc,
        'top1_weight_accuracy': top1_acc,
        'ensemble_accuracy': float(ens_correct_np.mean()),
        'gap_oracle_top1': float(oracle_acc - top1_acc),
    }

    print("\n[6] Mean final weight on samples where model was correct vs incorrect:")
    print(f"    {'Model':<30} {'W|correct':>12} {'W|incorrect':>12} {'delta':>10}")
    print("    " + "-" * 66)
    for j, name in enumerate(model_names):
        w = weights_np[:, j]
        c = correct_np[:, j]
        wc = float(w[c == 1].mean()) if (c == 1).any() else 0.0
        wi = float(w[c == 0].mean()) if (c == 0).any() else 0.0
        print(f"    {name:<30} {wc:>12.4f} {wi:>12.4f} {wc-wi:>+10.4f}")
        per_model_corr[name]['mean_weight_correct'] = wc
        per_model_corr[name]['mean_weight_incorrect'] = wi
        per_model_corr[name]['delta_weight'] = wc - wi
    results['per_model_correlation'] = per_model_corr

    active_mask = (weights_np > 1e-4)
    active_per_sample = active_mask.sum(axis=1)
    print("\n[7] Active models per sample (weight > 1e-4):")
    print(f"    Mean: {active_per_sample.mean():.2f} | Median: {np.median(active_per_sample):.1f}")
    unique_counts, count_freq = np.unique(active_per_sample, return_counts=True)
    for count, freq in zip(unique_counts, count_freq):
        print(f"    {int(count)} model(s): {int(freq):5d} samples ({freq/S*100:5.2f}%)")

    per_sample_data = []
    for i in range(S):
        per_sample_data.append({
            'weights': [float(x) for x in weights_np[i]],
            'model_correct': [int(x) for x in correct_np[i]],
            'ensemble_correct': int(ens_correct_np[i]),
            'top1_model_idx': int(top1_model_idx[i]),
            'top1_model_correct': int(top1_model_correct[i]),
            'weighted_competence': float(weighted_competence[i]),
        })
    results['per_sample'] = per_sample_data

    print("\n" + "=" * 110)
    print("VERDICT — Does the learned weighting reflect model competence?")
    print("=" * 110)
    n_positive_delta = sum(1 for m in per_model_corr.values() if m.get('delta_weight', 0) > 0)
    print(f"  Models with higher weight when correct: {n_positive_delta}/{N}")

    pooled_sr = results['pooled_correlation']['spearman']
    if pooled_sr > 0.05:
        print(f"  Pooled Spearman = {pooled_sr:+.4f} (positive → weight tracks competence)")
    else:
        print(f"  Pooled Spearman = {pooled_sr:+.4f} (weak/near-zero)")

    chance_top1 = float(correct_np.mean())
    print(f"  Top-1 weight accuracy: {top1_acc*100:.2f}%  "
          f"(chance reference: {chance_top1*100:.2f}%)")
    random_baseline = float(correct_np.mean())
    print(f"  Weighted competence: {weighted_competence.mean():.4f}  "
          f"(uniform-weight reference: {random_baseline:.4f})")

    if top1_acc > chance_top1 + 0.05 or weighted_competence.mean() > random_baseline + 0.05:
        print("\n  ✅ POSITIVE EVIDENCE: The learned weights show competence awareness.")
    elif top1_acc < chance_top1 - 0.05:
        print("\n  ❌ NEGATIVE EVIDENCE: Weights anti-correlate with competence.")
    else:
        print("\n  ⚠️  WEAK EVIDENCE: Weights are near chance-level for competence tracking.")
    print("=" * 110 + "\n")

    json_path = os.path.join(save_dir, f'competence_validation_{tag}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Competence validation saved: {json_path}")

    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        x = np.arange(N)
        width = 0.35
        w_correct = [per_model_corr[name].get('mean_weight_correct', 0) for name in model_names]
        w_incorrect = [per_model_corr[name].get('mean_weight_incorrect', 0) for name in model_names]
        ax.bar(x - width/2, w_correct, width, label='Correct', color='green', alpha=0.7)
        ax.bar(x + width/2, w_incorrect, width, label='Incorrect', color='red', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f'M{i+1}' for i in range(N)])
        ax.set_ylabel('Mean final weight')
        ax.set_title('Weight on correct vs incorrect\n(per model)')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        ax = axes[1]
        ax.hist(weighted_competence, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(weighted_competence.mean(), color='red', linestyle='--',
                   label=f'Mean = {weighted_competence.mean():.3f}')
        ax.axvline(random_baseline, color='orange', linestyle=':',
                   label=f'Uniform baseline = {random_baseline:.3f}')
        ax.set_xlabel('Weighted competence')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of weighted competence')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        ax = axes[2]
        labels = ['Ensemble', 'Top-1 weight', 'Oracle']
        values = [ens_correct_np.mean()*100, top1_acc*100, oracle_acc*100]
        colors = ['steelblue', 'green', 'gold']
        bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, v + 1, f'{v:.1f}%',
                    ha='center', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Ensemble vs Top-1 weight vs Oracle')
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = os.path.join(save_dir, f'competence_validation_{tag}.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Competence plot saved: {plot_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")

    return results


@torch.no_grad()
def validate_variance_vs_competence(ensemble, loader, device, model_names, save_dir, tag="val"):
    """Directly correlate the variance-based hesitancy score with per-model competence.

    This is the DIRECT test requested by the reviewer:
        "Does the variance measure track model competence?"

    For each sample we compute:
        - H_var: variance-based hesitancy (the paper's measure)
        - correct_i: whether base model i was actually correct (mask-independent)
        - mean_correct: fraction of base models that were correct
        - ensemble_correct: whether the ensemble prediction was correct
        - top1_correct: whether the highest-weighted model was correct

    Then we compute correlations and AUC to answer whether high variance
    co-occurs with competent models (paper's claim) or with incompetent
    models (uncertainty inversion).
    """
    ensemble.eval()
    ds = loader.dataset
    N = ensemble.num_models

    original_measure = ensemble.hesitancy_measure
    ensemble.hesitancy_measure = 'variance'

    all_var = []
    all_per_model_correct = []
    all_ensemble_correct = []
    all_mean_correct = []
    all_any_correct = []
    all_all_correct = []
    all_weights = []

    for i in tqdm(range(len(ds)), desc=f"Variance vs Competence [{tag}]"):
        try:
            image, label = ds[i]
        except Exception:
            continue
        image = image.unsqueeze(0).to(device)
        label_int = int(label)

        out, final_weights, mem, _ = ensemble(image, return_details=True)

        # 1) Variance-based hesitancy
        m = compute_uncertainty_measures(mem)
        h_var = float(m['variance'].item())
        all_var.append(h_var)

        # 2) Ensemble correctness
        p = torch.sigmoid(out.squeeze()).item()
        pred = int(p > 0.5)
        all_ensemble_correct.append(1 if pred == label_int else 0)

        # 3) Per-model correctness (mask-independent)
        all_logits = ensemble.compute_all_model_logits(image)
        per_correct = [
            (1 if int(all_logits[0, j, 0].item() > 0) == label_int else 0)
            for j in range(N)
        ]
        all_per_model_correct.append(per_correct)
        all_mean_correct.append(float(np.mean(per_correct)))
        all_any_correct.append(int(any(per_correct)))
        all_all_correct.append(int(all(per_correct)))
        all_weights.append(final_weights.squeeze(0).cpu().numpy())

    ensemble.hesitancy_measure = original_measure

    h_var_np = np.array(all_var)
    correct_np = np.array(all_per_model_correct)
    ens_correct_np = np.array(all_ensemble_correct)
    mean_correct_np = np.array(all_mean_correct)
    any_correct_np = np.array(all_any_correct)
    all_correct_np = np.array(all_all_correct)
    weights_np = np.array(all_weights)

    top1_idx = weights_np.argmax(axis=1)
    top1_correct_np = correct_np[np.arange(len(h_var_np)), top1_idx]

    S = len(h_var_np)

    print("\n" + "=" * 110)
    print(f"VARIANCE-BASED HESITANCY vs MODEL COMPETENCE — {tag.upper()} SET")
    print("=" * 110)
    print(f"#samples: {S} | Ensemble accuracy: {ens_correct_np.mean()*100:.2f}%")
    print(f"#models : {N}")
    print(f"Variance  mean={h_var_np.mean():.6f}  median={np.median(h_var_np):.6f}  "
          f"min={h_var_np.min():.6f}  max={h_var_np.max():.6f}")
    print("-" * 110)

    results = {
        'split': tag,
        'num_samples': int(S),
        'num_models': int(N),
        'variance_stats': {
            'mean': float(h_var_np.mean()),
            'median': float(np.median(h_var_np)),
            'std': float(h_var_np.std()),
            'min': float(h_var_np.min()),
            'max': float(h_var_np.max()),
        },
        'ensemble_accuracy': float(ens_correct_np.mean()),
    }

    # ============ [1] Correlation: variance vs competence targets ============
    print("\n[1] Spearman/Pearson: variance vs competence targets")
    print("    Positive r = high variance co-occurs with competent models")
    print("    Negative r = high variance co-occurs with incompetent models (inversion)")
    print(f"    {'Target':<35} {'Pearson':>10} {'p-val':>10} {'Spearman':>10} {'p-val':>10}")
    print("    " + "-" * 82)

    corr_results = {}
    targets = {
        'mean_correct': mean_correct_np,
        'ensemble_correct': ens_correct_np,
        'any_correct': any_correct_np,
        'all_correct': all_correct_np,
        'top1_correct': top1_correct_np,
    }
    for name, target in targets.items():
        try:
            pr, pr_p = pearsonr(h_var_np, target)
        except Exception:
            pr, pr_p = 0.0, 1.0
        try:
            sr, sr_p = spearmanr(h_var_np, target)
        except Exception:
            sr, sr_p = 0.0, 1.0
        corr_results[name] = {
            'pearson': float(pr), 'pearson_p': float(pr_p),
            'spearman': float(sr), 'spearman_p': float(sr_p),
        }
        print(f"    {name:<35} {pr:>+10.3f} {pr_p:>10.4f} {sr:>+10.3f} {sr_p:>10.4f}")

    results['correlations_with_competence'] = corr_results

    # ============ [2] AUC: can variance discriminate correct vs incorrect? ============
    print("\n[2] AUC: can variance discriminate correct from incorrect?")
    print("    AUC > 0.5 → high variance = correct (desired)")
    print("    AUC < 0.5 → high variance = incorrect (inversion)")

    auc_results = {}
    for name, target in [('ensemble_correct', ens_correct_np),
                          ('any_correct', any_correct_np),
                          ('all_correct', all_correct_np),
                          ('top1_correct', top1_correct_np)]:
        if len(np.unique(target)) > 1:
            try:
                auc = roc_auc_score(target, h_var_np)
            except Exception:
                auc = 0.5
        else:
            auc = 0.5
        auc_results[name] = float(auc)
        print(f"    AUC(variance → {name:<20}) = {auc:.4f}")

    results['auc'] = auc_results

    # ============ [3] Per-model correlation ============
    print("\n[3] Per-model: variance vs correct_i for each base model")
    print(f"    {'Model':<30} {'Acc':>8} {'Pearson':>10} {'p-val':>10} {'Spearman':>10} {'p-val':>10}")
    print("    " + "-" * 82)

    per_model_results = {}
    for j, name in enumerate(model_names):
        c = correct_np[:, j]
        try:
            pr, pr_p = pearsonr(h_var_np, c)
        except Exception:
            pr, pr_p = 0.0, 1.0
        try:
            sr, sr_p = spearmanr(h_var_np, c)
        except Exception:
            sr, sr_p = 0.0, 1.0
        per_model_results[name] = {
            'accuracy': float(c.mean()),
            'pearson_variance_vs_correct': float(pr),
            'pearson_p': float(pr_p),
            'spearman_variance_vs_correct': float(sr),
            'spearman_p': float(sr_p),
        }
        print(f"    {name:<30} {c.mean()*100:>7.2f}% {pr:>+10.3f} {pr_p:>10.4f} "
              f"{sr:>+10.3f} {sr_p:>10.4f}")

    results['per_model'] = per_model_results

    # ============ [4] Group comparison ============
    print("\n[4] Mean variance on correct vs incorrect samples")
    print(f"    {'Target':<35} {'var|correct':>15} {'var|incorrect':>15} {'delta':>10}")
    print("    " + "-" * 82)

    group_results = {}
    for name, target in targets.items():
        if target.sum() == 0 or (1 - target).sum() == 0:
            print(f"    {name:<35} {'N/A (single class)':>42}")
            continue
        var_correct = float(h_var_np[target == 1].mean())
        var_incorrect = float(h_var_np[target == 0].mean())
        delta = var_correct - var_incorrect
        group_results[name] = {
            'variance_on_correct': var_correct,
            'variance_on_incorrect': var_incorrect,
            'delta': delta,
        }
        print(f"    {name:<35} {var_correct:>15.6f} {var_incorrect:>15.6f} {delta:>+10.6f}")

    results['group_comparison'] = group_results

    # ============ [5] Verdict ============
    print("\n" + "=" * 110)
    print("VERDICT — Does variance track model competence?")
    print("=" * 110)

    key_sr = corr_results['mean_correct']['spearman']
    key_auc = auc_results['ensemble_correct']
    key_delta = group_results.get('ensemble_correct', {}).get('delta', 0.0)

    print(f"  Spearman(variance, mean_correctness)  = {key_sr:+.4f}")
    print(f"  AUC(variance → ensemble_correct)      = {key_auc:.4f}")
    print(f"  Delta variance (correct − incorrect)  = {key_delta:+.6f}")

    if key_sr > 0.10 and key_auc > 0.55:
        print("\n  ✅ STRONG POSITIVE: variance is positively associated with competence.")
        verdict = "strong_positive"
    elif key_sr > 0.03 or key_auc > 0.52:
        print("\n  🟡 WEAK POSITIVE: variance is weakly associated with competence.")
        verdict = "weak_positive"
    elif key_sr < -0.10 or key_auc < 0.45:
        print("\n  ❌ NEGATIVE: variance is ANTI-correlated with competence.")
        print("     Interpretation: low variance co-occurs with incorrect models —")
        print("     this is the 'uncertainty inversion' phenomenon discussed by the reviewer.")
        verdict = "negative"
    else:
        print("\n  ⚠️  NEUTRAL: variance is essentially uncorrelated with competence.")
        verdict = "neutral"

    results['verdict'] = verdict
    print("=" * 110 + "\n")

    json_path = os.path.join(save_dir, f'variance_competence_validation_{tag}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Variance-vs-competence validation saved: {json_path}")

    # ============ Plots ============
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        if 'ensemble_correct' in group_results:
            data = [h_var_np[ens_correct_np == 1], h_var_np[ens_correct_np == 0]]
            ax.boxplot(data, labels=['Correct', 'Incorrect'])
            ax.set_ylabel('Variance')
            ax.set_title('Variance on correct vs incorrect\n(ensemble level)')
        ax.grid(alpha=0.3, axis='y')

        ax = axes[1]
        ax.scatter(mean_correct_np + np.random.normal(0, 0.02, S), h_var_np,
                   alpha=0.4, s=20)
        z = np.polyfit(mean_correct_np, h_var_np, 1)
        p = np.poly1d(z)
        x_line = np.linspace(mean_correct_np.min(), mean_correct_np.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', label=f'trend (slope={z[0]:+.6f})')
        ax.set_xlabel('Mean per-model correctness')
        ax.set_ylabel('Variance')
        ax.set_title(f'Variance vs mean correctness\nSpearman = {corr_results["mean_correct"]["spearman"]:+.3f}')
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[2]
        names = list(per_model_results.keys())
        srs = [per_model_results[n]['spearman_variance_vs_correct'] for n in names]
        colors = ['green' if s > 0 else 'red' for s in srs]
        bars = ax.bar(range(len(names)), srs, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f'M{i+1}' for i in range(len(names))])
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel('Spearman(variance, correct_i)')
        ax.set_title('Per-model variance–competence correlation')
        for bar, s in zip(bars, srs):
            ax.text(bar.get_x() + bar.get_width()/2, s + (0.01 if s >= 0 else -0.03),
                    f'{s:+.3f}', ha='center', fontsize=10)
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = os.path.join(save_dir, f'variance_competence_validation_{tag}.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {plot_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")

    return results


@torch.no_grad()
def run_measure_ablation(ensemble, val_loader, device, save_dir, percentile=75.0):
    """Ablation on all measures, using the VALIDATION set only (no leakage).
    Each measure gets its own calibrated threshold. Best measure picked by val accuracy.
    """
    measures = ['variance', 'cv', 'entropy', 'range', 'combined']
    results = {}

    original_measure = ensemble.hesitancy_measure
    original_threshold = ensemble.hesitancy_threshold

    for measure in measures:
        print(f"\n{'='*60}\nABLATION: {measure}\n{'='*60}")
        try:
            threshold, _ = calibrate_hesitancy_threshold(
                ensemble, val_loader, device, measure, percentile
            )
        except Exception as e:
            print(f"  [SKIP] Calibration failed: {e}")
            continue

        ensemble.hesitancy_measure = measure
        ensemble.hesitancy_threshold = threshold

        criterion = nn.BCEWithLogitsLoss()
        acc, loss = evaluate_accuracy_and_loss(ensemble, val_loader, device, criterion)

        results[measure] = {
            'threshold': threshold,
            'val_accuracy': acc,
            'val_loss': loss,
        }
        print(f"  Threshold: {threshold:.4f} | Val Acc: {acc:.2f}% | Val Loss: {loss:.4f}")

    ensemble.hesitancy_measure = original_measure
    ensemble.hesitancy_threshold = original_threshold

    print(f"\n{'='*60}\nABLATION SUMMARY (Val, percentile={percentile})\n{'='*60}")
    print(f"{'Measure':<12} {'Threshold':>12} {'Val Acc':>10} {'Val Loss':>10}")
    print("-" * 48)
    for m, r in results.items():
        print(f"{m:<12} {r['threshold']:>12.4f} {r['val_accuracy']:>10.2f} {r['val_loss']:>10.4f}")

    best_measure = max(results, key=lambda m: results[m]['val_accuracy'])
    print(f"\nBest measure (val accuracy): {best_measure} ({results[best_measure]['val_accuracy']:.2f}%)")

    with open(os.path.join(save_dir, 'ablation_measures.json'), 'w') as f:
        json.dump({
            'split': 'validation',
            'percentile': percentile,
            'results': results,
            'best_measure': best_measure,
        }, f, indent=2)

    return best_measure, results


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
            models.append(model.to(device).eval())
        except Exception as e:
            print(f" [ERROR] Failed to load {path}: {e}")
    if len(models) == 0:
        raise ValueError("No models loaded!")
    return models


# ================== EVALUATION FUNCTIONS ==================
@torch.no_grad()
def evaluate_single_model_exact(model, loader, device, name, mean, std):
    model.eval()
    normalizer = MultiModelNormalization([mean], [std]).to(device)
    correct, total = 0, 0
    for images, labels in tqdm(loader, desc=f"Evaluating {name}"):
        images, labels = images.to(device), labels.to(device).float()
        images = normalizer(images, 0)
        out = model(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = (out.squeeze(1) > 0).long()
        correct += pred.eq(labels.long()).sum().item()
        total += labels.size(0)
    acc = 100. * correct / total
    print(f" {name}: {acc:.2f}%")
    return acc


@torch.no_grad()
def evaluate_accuracy_and_loss(model, loader, device, criterion):
    model.eval()
    correct, total_loss = 0, 0.0
    total_samples = len(loader.dataset)
    for images, labels in tqdm(loader, desc="Validation"):
        images, labels = images.to(device), labels.to(device).float()
        outputs, _ = model(images)
        loss = criterion(outputs.squeeze(1), labels)
        total_loss += loss.item() * images.size(0)
        pred = (outputs.squeeze(1) > 0).long()
        correct += pred.eq(labels.long()).sum().item()
    return 100. * correct / total_samples, total_loss / total_samples


# ================== TRAINING ==================
def train_hesitant_fuzzy(ensemble_model, train_loader, val_loader, num_epochs, lr,
                        device, save_dir, model_names):
    os.makedirs(save_dir, exist_ok=True)
    hesitant_net = ensemble_model.hesitant_fuzzy
    optimizer = torch.optim.AdamW(hesitant_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()
    best_val_acc = 0.0

    total_real_samples = len(train_loader.dataset)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    epoch_times, train_times, val_times = [], [], []
    total_sw = Stopwatch(device).start()

    print("=" * 70)
    print("Training Fuzzy Hesitant Network (Single GPU)")
    print("=" * 70)
    print(f"Trainable params: {sum(p.numel() for p in hesitant_net.parameters()):,}")
    print(f"Epochs: {num_epochs} | Initial LR: {lr}")
    print(f"Hesitant memberships per model: {hesitant_net.num_memberships}")
    print(f"Hesitancy measure: {ensemble_model.hesitancy_measure}")
    print(f"Number of models: {len(model_names)}")
    print(f"Total real training samples: {total_real_samples:,}\n")

    for epoch in range(num_epochs):
        epoch_sw = Stopwatch(device).start()
        ensemble_model.train()
        train_loss, train_correct = 0.0, 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs, _, _, _ = ensemble_model(images, return_details=True)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            pred = (outputs.squeeze(1) > 0).long()
            train_correct += pred.eq(labels.long()).sum().item()

        t_train = epoch_sw.stop()
        train_loss /= total_real_samples
        train_acc = 100. * train_correct / total_real_samples

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

        print(f"\n{'='*70}\nEpoch {epoch+1}/{num_epochs}\n{'='*70}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Time: epoch {t_epoch:.1f}s (train {t_train:.1f}s | val {t_val:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'hesitant_state_dict': hesitant_net.state_dict()},
                       os.path.join(save_dir, 'best_hesitant_fuzzy.pt'))
            print(f"\n✓ Best model saved → {val_acc:.2f}%")

    total_time = total_sw.stop()
    timing = summarize_training_times(epoch_times, train_times, val_times, total_time)
    print(f"Total training time: {total_time/60:.2f} min (mean epoch: {timing['mean_epoch_time_s']:.1f}s)")

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), history['train_loss'], label='train', color='blue', linewidth=2)
    plt.plot(range(1, num_epochs + 1), history['val_loss'], label='validation', color='orange', linewidth=2)
    plt.title('Loss', fontsize=16); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), history['train_acc'], label='train', color='blue', linewidth=2)
    plt.plot(range(1, num_epochs + 1), history['val_acc'], label='validation', color='orange', linewidth=2)
    plt.title('Accuracy', fontsize=16); plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    return best_val_acc, timing


# ================== MAIN ==================
def main():
    parser = argparse.ArgumentParser(description="Fuzzy Hesitant Ensemble with Hesitancy Validation (Single GPU)")
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
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--num_memberships', type=int, default=3)
    parser.add_argument('--cum_weight_threshold', type=float, default=0.9)
    parser.add_argument('--hesitancy_threshold', type=float, default=0.2)

    # ===== REVIEWER RESPONSE ARGS =====
    parser.add_argument('--hesitancy_measure', type=str, default='variance',
                        choices=['variance', 'cv', 'entropy', 'range', 'combined'],
                        help="Uncertainty measure for hesitancy override.")
    parser.add_argument('--calibrate_threshold', action='store_true',
                        help="Calibrate hesitancy threshold on validation set (per measure).")
    parser.add_argument('--threshold_percentile', type=float, default=75.0,
                        help="Percentile for threshold calibration (default: 75).")
    parser.add_argument('--run_ablation', action='store_true',
                        help="Run ablation over all measures on validation set and pick the best.")
    parser.add_argument('--validate_hesitancy', action='store_true',
                        help="Detailed validation of hesitancy measures on VALIDATION set.")
    parser.add_argument('--validate_competence', action='store_true',
                        help="Validate that learned final weights reflect model competence.")
    parser.add_argument('--validate_variance_competence', action='store_true',
                        help="Directly correlate variance with per-model competence.")
    # ==================================

    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)

    args = parser.parse_args()

    if len(args.model_names) != len(args.model_paths):
        raise ValueError("Number of model_names must match model_paths")

    set_seed(args.seed)
    device = setup_device()
    pipeline_sw = Stopwatch(device).start()

    print("=" * 70)
    print("FUZZY HESITANT ENSEMBLE TRAINING + HESITANCY VALIDATION")
    print(f"Single GPU | Seed: {args.seed}")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Models: {len(args.model_paths)}")
    print(f"Initial hesitancy measure: {args.hesitancy_measure}")
    print("=" * 70 + "\n")

    DEFAULT_MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    DEFAULT_STDS  = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]

    num_models = len(args.model_paths)
    if num_models > len(DEFAULT_MEANS):
        MEANS = DEFAULT_MEANS + [DEFAULT_MEANS[-1]] * (num_models - len(DEFAULT_MEANS))
        STDS  = DEFAULT_STDS  + [DEFAULT_STDS[-1]]  * (num_models - len(DEFAULT_STDS))
    else:
        MEANS = DEFAULT_MEANS[:num_models]
        STDS  = DEFAULT_STDS[:num_models]

    base_models = load_pruned_models(args.model_paths, device)
    MODEL_NAMES = args.model_names[:len(base_models)]

    ensemble = FuzzyHesitantEnsemble(
        base_models, MEANS, STDS,
        num_memberships=args.num_memberships,
        freeze_models=True,
        cum_weight_threshold=args.cum_weight_threshold,
        hesitancy_threshold=args.hesitancy_threshold,
        hesitancy_measure=args.hesitancy_measure,
    ).to(device)

    trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
    total = sum(p.numel() for p in ensemble.parameters())
    print(f"Total params: {total:,} | Trainable: {trainable:,} | Frozen: {total-trainable:,}\n")

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=False, seed=args.seed, is_main=True)

    print("\n" + "=" * 70 + "\nINDIVIDUAL MODEL PERFORMANCE (Before Training)\n" + "=" * 70)
    individual_accs = []
    for i, model in enumerate(base_models):
        acc = evaluate_single_model_exact(model, test_loader, device,
                                          f"Model {i+1} ({MODEL_NAMES[i]})",
                                          MEANS[i], STDS[i])
        individual_accs.append(acc)

    best_single = max(individual_accs)
    best_idx = individual_accs.index(best_single)
    print(f"\nBest Single: Model {best_idx+1} ({MODEL_NAMES[best_idx]}) → {best_single:.2f}%\n" + "=" * 70)

    best_val_acc, train_timing = train_hesitant_fuzzy(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_dir, MODEL_NAMES
    )

    ckpt_path = os.path.join(args.save_dir, 'best_hesitant_fuzzy.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        ensemble.hesitant_fuzzy.load_state_dict(ckpt['hesitant_state_dict'])
        print("Best model loaded.\n")

    # ===== REVIEWER RESPONSE: on VALIDATION set only (no leakage) =====
    if args.run_ablation:
        print("\n" + "=" * 70 + "\nABLATION OVER HESITANCY MEASURES (Validation)\n" + "=" * 70)
        best_measure, abl = run_measure_ablation(
            ensemble, val_loader, device, args.save_dir, args.threshold_percentile
        )
        ensemble.hesitancy_measure = best_measure
        ensemble.hesitancy_threshold = abl[best_measure]['threshold']
        print(f"[MAIN] Using best measure: {best_measure}, threshold={abl[best_measure]['threshold']:.4f}")

    elif args.calibrate_threshold:
        print("\n" + "=" * 70 + f"\nCALIBRATING THRESHOLD FOR {args.hesitancy_measure}\n" + "=" * 70)
        threshold, _ = calibrate_hesitancy_threshold(
            ensemble, val_loader, device, args.hesitancy_measure, args.threshold_percentile
        )
        ensemble.hesitancy_threshold = threshold
        print(f"[MAIN] Calibrated threshold: {threshold:.4f} (percentile={args.threshold_percentile})")

    if args.validate_hesitancy:
        print("\n" + "=" * 70 + "\nVALIDATING HESITANCY MEASURES (Validation Set)\n" + "=" * 70)
        validate_hesitancy_measure(ensemble, val_loader, device, MODEL_NAMES, args.save_dir, tag="val")

    if args.validate_competence:
        print("\n" + "=" * 70 + "\nVALIDATING MODEL-COMPETENCE ESTIMATION (Validation Set)\n" + "=" * 70)
        validate_model_competence(ensemble, val_loader, device, MODEL_NAMES,
                                  args.save_dir, tag="val")

    if args.validate_variance_competence:
        print("\n" + "=" * 70 + "\nVARIANCE vs MODEL COMPETENCE (Validation Set)\n" + "=" * 70)
        validate_variance_vs_competence(ensemble, val_loader, device, MODEL_NAMES,
                                        args.save_dir, tag="val")
    # ===================================================================

    print("\n" + "=" * 70 + "\nFINAL ENSEMBLE EVALUATION (TEST SET)\n" + "=" * 70)

    efficiency = benchmark_efficiency(ensemble, test_loader, device, MODEL_NAMES,
                                      latency_samples=len(test_loader.dataset))
    efficiency['training_time'] = train_timing
    print_headline(efficiency)

    eval_sw = Stopwatch(device).start()
    final_acc, y_true, y_scores = final_evaluation_unified(
        ensemble, test_loader, device, args.save_dir, MODEL_NAMES, args
    )
    final_eval_time = eval_sw.stop()

    print("\n" + "=" * 70 + "\nFINAL COMPARISON\n" + "=" * 70)
    print(f"Best Single Model: {best_single:.2f}%")
    print(f"Ensemble Accuracy: {final_acc:.2f}%")
    print(f"Improvement: {final_acc - best_single:+.2f}%")
    print("=" * 70)

    final_results = {
        'seed': args.seed,
        'method': 'Fuzzy_Hesitant',
        'hesitancy_measure_final': ensemble.hesitancy_measure,
        'hesitancy_threshold_final': float(ensemble.hesitancy_threshold),
        'best_single_model': {'name': MODEL_NAMES[best_idx], 'accuracy': float(best_single)},
        'ensemble': {'test_accuracy': float(final_acc)},
        'improvement': float(final_acc - best_single),
    }
    with open(os.path.join(args.save_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)

    torch.save({
        'ensemble_state_dict': ensemble.state_dict(),
        'model_names': MODEL_NAMES,
        'means': MEANS, 'stds': STDS, 'seed': args.seed,
        'hesitancy_measure': ensemble.hesitancy_measure,
        'hesitancy_threshold': float(ensemble.hesitancy_threshold),
    }, os.path.join(args.save_dir, 'final_ensemble_model.pt'))

    efficiency['final_evaluation_time_s'] = final_eval_time
    efficiency['total_pipeline_time_s'] = pipeline_sw.stop()
    with open(os.path.join(args.save_dir, 'efficiency_report.json'), 'w') as f:
        json.dump(efficiency, f, indent=2)
    print(f"Efficiency report saved: {os.path.join(args.save_dir, 'efficiency_report.json')}")


if __name__ == "__main__":
    main()
