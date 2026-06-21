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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from metrics_utils import plot_roc_and_f1
from dataset_utils import (
    UADFVDataset, CustomGenAIDataset, NewGenAIDataset,
    create_dataloaders, create_adabn_dataloader, get_sample_info
)
from visualization_utils import generate_visualizations

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


def is_dist():
    return dist.is_initialized()


# ================== MODEL CLASSES ==================
class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


# =====================================================================
# ================== DISTRIBUTED BATCHNORM ADAPTATION ==================
# =====================================================================
@torch.no_grad()
def adapt_batchnorm_for_new_dataset(models, means, stds, adabn_loader, device, is_main=True, is_distributed=False):
    """
    اعمال AdaBN به صورت موازی روی چندین GPU.
    اگر is_distributed=True باشد، از فرمول محاسبه واریانس موازی (Parallel Variance) استفاده می‌کند.
    """
    if is_main:
        print("\n" + "="*70)
        print(f"STARTING BATCHNORM ADAPTATION (AdaBN) - Mode: {'Distributed' if is_distributed else 'Single GPU'}")
        print("="*70)
        
    normalizer = MultiModelNormalization(means, stds).to(device)
    
    if is_distributed:
        # =================== منطق چند GPU ===================
        accumulators = [{} for _ in models]
        hooks = []
        
        for i, model in enumerate(models):
            model.eval() # مدل در حالت eval میماند تا dropout خاموش شود
            for name, m in model.named_modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    # ساخت بافرهای accumulator برای هر لایه BN
                    accumulators[i][name] = {
                        'n': torch.tensor(0.0, device=device),
                        'sum_mean': torch.zeros(m.num_features, device=device),
                        'sum_x2': torch.zeros(m.num_features, device=device) # مجموع مربعات برای محاسبه واریانس
                    }
                    
                    def get_hook(model_idx, layer_name):
                        def hook(module, inp, out):
                            x = inp[0]
                            # N در اینجا برابر با (Batch_size * H * W) است
                            n = x.numel() / x.size(1) 
                            
                            # محاسبه mean و var دقیقاً مشابه نحوه محاسبه داخلی PyTorch (unbiased=False)
                            mean = x.mean(dim=[0, 2, 3])
                            var = x.var(dim=[0, 2, 3], unbiased=False) 
                            
                            # آپدیت مقادیر تجمعی
                            accumulators[model_idx][layer_name]['n'] += n
                            accumulators[model_idx][layer_name]['sum_mean'] += mean * n
                            # فرمول ریاضی: sum(x^2) = Var(x) * N + Mean(x)^2 * N
                            accumulators[model_idx][layer_name]['sum_x2'] += (var * n) + (mean ** 2 * n)
                        return hook
                    # ثبت Hook روی لایه
                    hooks.append(m.register_forward_hook(get_hook(i, name)))

        # تنظیم epoch برای sampler در صورت وجود
        if hasattr(adabn_loader.sampler, 'set_epoch'):
            adabn_loader.sampler.set_epoch(0)

        # پاس دادن داده‌ها (هر GPU فقط بخشی از داده را می‌بیند)
        for images, _ in tqdm(adabn_loader, desc="Adapting BN (Dist)", disable=not is_main):
            images = images.to(device)
            for i, model in enumerate(models):
                x_n = normalizer(images, i)
                model(x_n) # Hook ها به صورت خودکار اجرا و مقادیر را جمع می‌کنند

        # حذف Hook ها
        for h in hooks: h.remove()

        if is_main: print("Synchronizing BN statistics across GPUs...")
        
        # جمع کردن آمارهای محاسبه شده از تمام GPUها
        for i, model in enumerate(models):
            for name, m in model.named_modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    acc = accumulators[i][name]
                    
                    # جمع کردن مقادیر روی تمام پردازنده‌ها
                    dist.all_reduce(acc['n'], op=dist.ReduceOp.SUM)
                    dist.all_reduce(acc['sum_mean'], op=dist.ReduceOp.SUM)
                    dist.all_reduce(acc['sum_x2'], op=dist.ReduceOp.SUM)
                    
                    # محاسبه آمار نهایی و جامع (Global Mean & Variance)
                    global_mean = acc['sum_mean'] / acc['n']
                    global_var = (acc['sum_x2'] / acc['n']) - (global_mean ** 2)
                    global_var = torch.clamp(global_var, min=1e-5) # جلوگیری از صفر شدن واریانس
                    
                    # ست کردن آمارهای محاسبه شده
                    m.running_mean.copy_(global_mean)
                    m.running_var.copy_(global_var)
                    m.num_batches_tracked.copy_(torch.tensor(1, dtype=torch.long, device=device))
        
        if is_main: print("✅ Distributed BatchNorm Adaptation Completed!\n")

    else:
        # =================== منطق تک GPU (کد قبلی) ===================
        for model in models:
            model.eval()  
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.reset_running_stats()
                    m.momentum = None  
                    m.train()          
                    
        for images, _ in tqdm(adabn_loader, desc="Adapting BN", disable=not is_main):
            images = images.to(device)
            for i, model in enumerate(models):
                x_n = normalizer(images, i)
                model(x_n) 
                
        for model in models:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.momentum = 0.1
            model.eval()
            
        if is_main: print("✅ BatchNorm Adaptation Completed!\n")
# =====================================================================


class HesitantFuzzyMembership(nn.Module):
    def __init__(self, input_dim: int, num_models: int,
                 num_memberships: int = 3, dropout: float = 0.3):
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
        self.aggregation_weights = nn.Parameter(
            torch.ones(num_memberships) / num_memberships
        )

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
                 stds: List[Tuple[float]], num_memberships: int = 3,
                 freeze_models: bool = True, cum_weight_threshold: float = 0.9,
                 hesitancy_threshold: float = 0.2):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.hesitant_fuzzy = HesitantFuzzyMembership(
            input_dim=128, num_models=self.num_models,
            num_memberships=num_memberships
        )
        self.cum_weight_threshold = cum_weight_threshold
        self.hesitancy_threshold = hesitancy_threshold

        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

    def _compute_mask_vectorized(self, final_weights: torch.Tensor,
                                  avg_hesitancy: torch.Tensor):
        sorted_weights, sorted_indices = torch.sort(
            final_weights, dim=1, descending=True
        )
        cum_weights = torch.cumsum(sorted_weights, dim=1)
        mask = (cum_weights <= self.cum_weight_threshold).float()
        mask[:, 0] = 1.0
        high_hesitancy_mask = (
            avg_hesitancy > self.hesitancy_threshold
        ).unsqueeze(1)
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
        final_weights = final_weights / (
            final_weights.sum(dim=1, keepdim=True) + 1e-8
        )

        outputs = torch.zeros(
            x.size(0), self.num_models, 1, device=x.device
        )
        active_models = torch.any(
            final_weights > 0, dim=0
        ).nonzero(as_tuple=True)[0]

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


# ================== MODEL LOADING ==================
def load_pruned_models(model_paths: List[str], device: torch.device,
                       is_main: bool) -> List[nn.Module]:
    try:
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        raise ImportError("Cannot import ResNet_50_pruned_hardfakevsreal")

    models = []
    if is_main:
        print(f"Loading {len(model_paths)} pruned models...")
    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            if is_main:
                print(f" [WARNING] File not found: {path}")
            continue
        if is_main:
            print(f" [{i+1}/{len(model_paths)}] Loading: {os.path.basename(path)}")
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device).eval()
            models.append(model)
        except Exception as e:
            if is_main:
                print(f" [ERROR] Failed to load {path}: {e}")
    if len(models) == 0:
        raise ValueError("No models loaded!")
    return models


# ================== EVALUATION FUNCTIONS ==================

@torch.no_grad()
def evaluate_single_model(model: nn.Module, loader: DataLoader,
                          device: torch.device, name: str,
                          mean: Tuple[float, float, float],
                          std: Tuple[float, float, float],
                          is_main: bool) -> float:
  
    model.eval()
    normalizer = MultiModelNormalization([mean], [std]).to(device)
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc=f"Evaluating {name}",
                                disable=not is_main):
        images, labels = images.to(device), labels.to(device).float()
        images = normalizer(images, 0)
        out = model(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = (out.squeeze(1) > 0).long()
        correct += pred.eq(labels.long()).sum().item()
        total += labels.size(0)

    acc = 100.0 * correct / total if total > 0 else 0.0
    if is_main:
        print(f"  {name}: {acc:.2f}%")
    return acc


@torch.no_grad()
def evaluate_accuracy_ddp(model, loader, device, is_main=False):
    # اگر GPU اصلی نیستیم، کاری نکن تا از محاسبات تکراری و Hang جلوگیری شود
    if not is_main:
        return 0.0 
        
    # استخراج مدل اصلی از داخل پوشش DDP (جلوگیری از Deadlock در Broadcast)
    base_model = model.module if hasattr(model, 'module') else model
    base_model.eval()
    
    local_correct = 0
    local_total = 0
    
    # ساخت یک لودر موقت روی کل دیتاست برای GPU اصلی (بدون توزیع و بدون تکرار)
    from torch.utils.data import SequentialSampler
    temp_loader = DataLoader(loader.dataset, batch_size=loader.batch_size, 
                             sampler=SequentialSampler(range(len(loader.dataset))),
                             num_workers=loader.num_workers, pin_memory=True)

    for images, labels in temp_loader:
        images, labels = images.to(device), labels.to(device).float()
        # حتماً از base_model استفاده شود نه model
        outputs, _ = base_model(images)
        pred = (outputs.squeeze(1) > 0).long()
        local_correct += pred.eq(labels.long()).sum().item()
        local_total += labels.size(0)

    return 100.0 * local_correct / local_total if local_total > 0 else 0.0


# ================== UNIFIED FINAL EVALUATION ==================
def final_evaluation_unified(model, test_loader_full, device, save_dir,
                             model_names, args, is_main):
  
    if not is_main:
        return 0.0, None, None

    model.eval()

    base_dataset = test_loader_full.dataset
    if hasattr(base_dataset, 'dataset'):
        base_dataset = base_dataset.dataset

    if (hasattr(test_loader_full, 'sampler')
            and hasattr(test_loader_full.sampler, 'indices')):
        test_indices = test_loader_full.sampler.indices
    elif hasattr(test_loader_full.dataset, 'indices'):
        test_indices = test_loader_full.dataset.indices
    else:
        test_indices = list(range(len(base_dataset)))

    lines = []
    lines.append("=" * 100)
    lines.append(
        "SAMPLE-BY-SAMPLE PREDICTIONS (For McNemar Test Comparison):"
    )
    lines.append("=" * 100)
    header = (f"{'ID':<10} {'Filename':<60} {'True':<10} "
              f"{'Pred':<10} {'Correct':<10}")
    lines.append(header)
    lines.append("-" * 100)

    TP = TN = FP = FN = 0
    all_y_true, all_y_score, all_y_pred = [], [], []
    all_model_weights = []

    print(f"\nRunning Unified Final Evaluation on "
          f"{len(test_indices)} samples...")

    with torch.no_grad():
        for i, global_idx in enumerate(
                tqdm(test_indices, desc="Final Eval")):
            try:
                image, label = base_dataset[global_idx]
                path, _ = get_sample_info(base_dataset, global_idx)
            except Exception:
                continue

            image = image.unsqueeze(0).to(device)
            label_int = int(label)

            output, final_weights, _, _ = model(
                image, return_details=True
            )
            prob = torch.sigmoid(output.squeeze()).item()
            all_model_weights.append(
                final_weights.squeeze(0).cpu().numpy()
            )

            pred_int = int(prob > 0.5)

            all_y_true.append(label_int)
            all_y_score.append(prob)
            all_y_pred.append(pred_int)

            if label_int == 1:
                if pred_int == 1:
                    TP += 1
                else:
                    FN += 1
            else:
                if pred_int == 1:
                    FP += 1
                else:
                    TN += 1

            correct_str = "Yes" if pred_int == label_int else "No"
            filename = os.path.basename(path)
            if len(filename) > 55:
                filename = filename[:25] + "..." + filename[-27:]
            line = (f"{i+1:<10} {filename:<60} {label_int:<10} "
                    f"{pred_int:<10} {correct_str:<10}")
            lines.append(line)

    total_samples = len(all_y_true)
    correct_count = TP + TN
    acc = (100.0 * correct_count / total_samples
           if total_samples > 0 else 0.0)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    # ──────────── یک بلوک چاپ (بدون تکرار) ────────────
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted Real  Predicted Fake")
    print(f"    Actual Real      {TP:<15} {FN:<15}")
    print(f"    Actual Fake      {FP:<15} {TN:<15}")
    print(f"\nCorrect Predictions: {correct_count} ({acc:.2f}%)")
    print(f"Incorrect Predictions: "
          f"{total_samples - correct_count} ({100.0 - acc:.2f}%)")
    print("=" * 70)

    # ═══════════ آمار فعال‌سازی هر مدل ═══════════
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

        print(f"\n  {'Model':<35} {'Active %':>10} "
              f"{'Avg Weight':>12} {'Max Weight':>12}")
        print(f"  {'-'*70}")
        for i, name in enumerate(model_names):
            act_pct = activation_rates[i]
            avg_w = avg_weights[i]
            max_w = max_weights[i]
            bar_length = int(act_pct / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"  {i+1:2d}. {name:<30}: {act_pct:6.2f}%  "
                  f"| {avg_w:.6f}  | {max_w:.6f}")
            print(f"      [{bar}]")

        print(f"  {'-'*70}")
        print(f"  {'Overall Average':<35} "
              f"{activation_rates.mean():6.2f}%")

        active_per_sample = active_mask.sum(axis=1)
        print(f"\n  Active Models Per Sample:")
        print(f"    Min:    {int(active_per_sample.min())}")
        print(f"    Max:    {int(active_per_sample.max())}")
        print(f"    Mean:   {active_per_sample.mean():.2f}")
        print(f"    Median: {np.median(active_per_sample):.1f}")

        unique_counts, count_freq = np.unique(
            active_per_sample.astype(int), return_counts=True
        )
        print(f"\n  Distribution of Active Model Count:")
        for count, freq in zip(unique_counts, count_freq):
            pct = 100.0 * freq / len(all_model_weights)
            bar = '█' * int(pct / 2)
            print(f"    {count} model(s): {freq:5d} samples "
                  f"({pct:5.2f}%) {bar}")

        print(f"{'='*70}\n")

        activation_stats = {
            "activation_rates_pct": {
                name: float(activation_rates[i])
                for i, name in enumerate(model_names)
            },
            "average_weights": {
                name: float(avg_weights[i])
                for i, name in enumerate(model_names)
            },
            "max_weights": {
                name: float(max_weights[i])
                for i, name in enumerate(model_names)
            },
            "active_per_sample_stats": {
                "min": int(active_per_sample.min()),
                "max": int(active_per_sample.max()),
                "mean": float(active_per_sample.mean()),
                "median": float(np.median(active_per_sample))
            },
            "per_sample_weights": weights_np.tolist()
        }
        activation_json_path = os.path.join(
            save_dir, "model_activation_stats.json"
        )
        with open(activation_json_path, 'w', encoding='utf-8') as f:
            json.dump(activation_stats, f, indent=2, ensure_ascii=False)
        print(f"Model activation stats saved: {activation_json_path}")

    # ──────── ذخیره لاگ متنی (McNemar Format) ────────
    output_str = []
    output_str.append("-" * 100)
    output_str.append("SUMMARY STATISTICS:")
    output_str.append("-" * 100)
    output_str.append(f"Accuracy: {acc:.2f}%")
    output_str.append(f"Precision: {precision:.4f}")
    output_str.append(f"Recall: {recall:.4f}")
    output_str.append(f"Specificity: {specificity:.4f}")
    output_str.append("\nConfusion Matrix:")
    output_str.append(
        f"                 {'Predicted Real':<15} {'Predicted Fake':<15}"
    )
    output_str.append(
        f"    Actual Real   {TP:<15} {FN:<15}"
    )
    output_str.append(
        f"    Actual Fake   {FP:<15} {TN:<15}"
    )
    output_str.append(
        f"\nCorrect Predictions: {correct_count} ({acc:.2f}%)"
    )
    output_str.append(
        f"Incorrect Predictions: "
        f"{total_samples - correct_count} ({100.0 - acc:.2f}%)"
    )
    output_str.extend(lines)

    log_path = os.path.join(save_dir, 'prediction_log.txt')
    with open(log_path, 'w') as f:
        f.write("\n".join(output_str))
    print(f"✅ Prediction log saved to: {log_path}")

    # ──────── ذخیره داده‌های ROC ────────
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
            "model": "fuzzy_hesitant_ensemble_AdaBN"
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

    # ویژوالایزیشن
    vis_dir = os.path.join(save_dir, 'visualizations')
    generate_visualizations(
        model, test_loader_full, device, vis_dir, model_names,
        args.num_grad_cam_samples, args.num_lime_samples,
        args.dataset, is_main
    )

    return acc, y_true_np, y_score_np


# ================== TRAINING ==================
def train_hesitant_fuzzy(ensemble_model, train_loader, val_loader,
                         num_epochs, lr, device, save_dir, is_main,
                         model_names):
    os.makedirs(save_dir, exist_ok=True)
    hesitant_net = (
        ensemble_model.module.hesitant_fuzzy
        if hasattr(ensemble_model, 'module')
        else ensemble_model.hesitant_fuzzy
    )
    optimizer = torch.optim.AdamW(
        hesitant_net.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    criterion = nn.BCEWithLogitsLoss()
    best_val_acc = 0.0
    num_model = len(model_names)

    if is_main:
        print("=" * 70)
        print("Training Fuzzy Hesitant Network")
        print("=" * 70)
        print(f"Trainable params: "
              f"{sum(p.numel() for p in hesitant_net.parameters()):,}")
        print(f"Epochs: {num_epochs} | Initial LR: {lr}")
        print(f"Hesitant memberships per model: "
              f"{hesitant_net.num_memberships}")
        print(f"Number of models: {num_model}\n")

    for epoch in range(num_epochs):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        ensemble_model.train()

        # ──── متغیرهای محلی هر GPU ────
        local_loss = 0.0
        local_correct = 0
        local_total = 0
        local_hesitancy = torch.zeros(num_model, device=device)
        local_cumsum_used = 0
        local_active = torch.zeros(num_model, device=device)

        pbar = tqdm(train_loader,
                     desc=f'Epoch {epoch+1}/{num_epochs}',
                     disable=not is_main)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()

            outputs, weights, memberships, _ = ensemble_model(
                images, return_details=True
            )
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            local_loss += loss.item() * bs
            pred = (outputs.squeeze(1) > 0).long()
            local_correct += pred.eq(labels.long()).sum().item()
            local_total += bs

            per_model_hes = memberships.var(dim=2, unbiased=False)
            local_hesitancy += per_model_hes.sum(dim=0)

            act_mask = (weights > 1e-4).float()
            n_active = act_mask.sum(dim=1)
            local_cumsum_used += (n_active < num_model).sum().item()
            local_active += act_mask.sum(dim=0)

        # ──── All-reduce برای متریک‌های آموزش ────
        if is_dist():
            correct_t = torch.tensor(
                local_correct, dtype=torch.float, device=device
            )
            total_t = torch.tensor(
                local_total, dtype=torch.float, device=device
            )
            loss_t = torch.tensor(
                local_loss, dtype=torch.float, device=device
            )
            cumsum_t = torch.tensor(
                local_cumsum_used, dtype=torch.float, device=device
            )

            for tensor in [correct_t, total_t, loss_t, cumsum_t,
                           local_hesitancy, local_active]:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

            g_total = total_t.item()
            train_acc = 100.0 * correct_t.item() / g_total
            train_loss = loss_t.item() / g_total
            avg_hesitancy = local_hesitancy / g_total
            avg_cumsum = (cumsum_t.item() / g_total) * 100
            avg_activation = (local_active / g_total) * 100
        else:
            g_total = local_total
            train_acc = 100.0 * local_correct / local_total
            train_loss = local_loss / local_total
            avg_hesitancy = local_hesitancy / local_total
            avg_cumsum = (local_cumsum_used / local_total) * 100
            avg_activation = (local_active / local_total) * 100

        overall_mean_hes = avg_hesitancy.mean().item()
        val_acc = evaluate_accuracy_ddp(ensemble_model, val_loader, device, is_main)
    
        # ---------- این دو خط را اضافه کنید ----------
        if is_dist():
            dist.barrier()
        # ---------------------------------------------

        scheduler.step()

        if is_main:
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*70}")
            print(f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}%")
            print(f"Val Acc: {val_acc:.2f}% | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            print(f"\n{'Hesitancy (Variance) per Model:':^70}")
            print(f"{'-'*70}")
            for i, name in enumerate(model_names):
                hv = avg_hesitancy[i].item()
                print(f"  {i+1:2d}. {name:<30}: {hv:.6f}")
            print(f"{'-'*70}")
            print(f"  {'Overall Mean Hesitancy:':<30}  "
                  f"{overall_mean_hes:.6f}")

            print(f"\n{'Cumulative Weight Threshold Analysis:':^70}")
            print(f"{'-'*70}")
            print(f"  {'Cumsum activated in:':<30}  "
                  f"{avg_cumsum:.2f}% of samples")
            if avg_cumsum > 50:
                print(f"  {'Status:':<30}  ✓ Efficient")
            elif avg_cumsum > 20:
                print(f"  {'Status:':<30}  ≈ Moderate")
            else:
                print(f"  {'Status:':<30}  ✗ Low efficiency")
            print(f"  {'All models used in:':<30}  "
                  f"{100 - avg_cumsum:.2f}% of samples")

            print(f"\n{'Model Activation Frequency:':^70}")
            print(f"{'-'*70}")
            for i, name in enumerate(model_names):
                pct = avg_activation[i].item()
                bar = '█' * int(pct / 2)
                print(f"  {i+1:2d}. {name:<30}: {pct:5.1f}% {bar}")

        if is_main and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, 'best_hesitant_fuzzy.pt')
            torch.save(
                {'hesitant_state_dict': hesitant_net.state_dict()},
                save_path
            )
            print(f"\n✓ Best model saved → {val_acc:.2f}%")

        if is_main:
            print()

    return best_val_acc


# ================== DISTRIBUTED SETUP ==================
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        if rank == 0:
            print(f"Distributed: rank {rank}/{world_size}, "
                  f"local_rank {local_rank}")
        return device, local_rank, rank, world_size
    else:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        return device, 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ================== MAIN FUNCTION ==================
def main():
    parser = argparse.ArgumentParser(
        description="Optimized Fuzzy Hesitant Ensemble Training with Distributed AdaBN"
    )
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['wild', 'hard_fake_real', 'deepfake_lab',
                                 'uadfV', 'custom_genai', 'custom_genai_v2',
                                 'real_fake_dataset'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_memberships', type=int, default=3)
    parser.add_argument('--cum_weight_threshold', type=float, default=0.9)
    parser.add_argument('--hesitancy_threshold', type=float, default=0.2)
    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)
    args = parser.parse_args()

    if len(args.model_names) != len(args.model_paths):
        raise ValueError("Number of model_names must match model_paths")

    set_seed(args.seed)
    device, local_rank, rank, world_size = setup_distributed()
    is_main = (rank == 0)
    is_distributed = (world_size > 1)

    if is_main:
        print("=" * 70)
        print("OPTIMIZED FUZZY HESITANT ENSEMBLE TRAINING")
        print(f"Distributed on {world_size} GPU(s) | Seed: {args.seed}")
        print("=" * 70)
        print(f"Dataset: {args.dataset}")
        print(f"Data directory: {args.data_dir}")
        print(f"Batch size: {args.batch_size}")
        print(f"Models: {len(args.model_paths)}")
        print("=" * 70 + "\n")

    # ──── نرمالیزیشن ────
    DEFAULT_MEANS = [
        (0.5207, 0.4258, 0.3806),
        (0.4460, 0.3622, 0.3416),
        (0.4668, 0.3816, 0.3414)
    ]
    DEFAULT_STDS = [
        (0.2490, 0.2239, 0.2212),
        (0.2057, 0.1849, 0.1761),
        (0.2410, 0.2161, 0.2081)
    ]
    num_models = len(args.model_paths)
    if num_models > len(DEFAULT_MEANS):
        MEANS = (DEFAULT_MEANS
                 + [DEFAULT_MEANS[-1]] * (num_models - len(DEFAULT_MEANS)))
        STDS = (DEFAULT_STDS
                + [DEFAULT_STDS[-1]] * (num_models - len(DEFAULT_STDS)))
    else:
        MEANS = DEFAULT_MEANS[:num_models]
        STDS = DEFAULT_STDS[:num_models]

    base_models = load_pruned_models(args.model_paths, device, is_main)
    MODEL_NAMES = args.model_names[:len(base_models)]

    # =================== فراخوانی AdaBN به صورت موازی ===================
    adabn_loader = create_adabn_dataloader(
        args.data_dir, args.batch_size, num_workers=0,
        dataset_type=args.dataset, seed=args.seed, is_main=is_main,
        is_distributed=is_distributed
    )
    adapt_batchnorm_for_new_dataset(
        base_models, MEANS, STDS, adabn_loader, device, is_main,
        is_distributed=is_distributed
    )
    del adabn_loader
    if is_main: torch.cuda.empty_cache()
    # ====================================================================

    ensemble = FuzzyHesitantEnsemble(
        base_models, MEANS, STDS,
        num_memberships=args.num_memberships,
        freeze_models=True,
        cum_weight_threshold=args.cum_weight_threshold,
        hesitancy_threshold=args.hesitancy_threshold
    ).to(device)

    if is_distributed:
        ensemble = DDP(
            ensemble, device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

    if is_main:
        trainable = sum(
            p.numel() for p in ensemble.parameters() if p.requires_grad
        )
        total_p = sum(p.numel() for p in ensemble.parameters())
        print(f"Total params: {total_p:,} | Trainable: {trainable:,} | "
              f"Frozen: {total_p - trainable:,}\n")

    # ──── ساخت دیتالودرهای توزیع‌شده (برای آموزش) ────
    train_loader, val_loader, _ = create_dataloaders(
        args.data_dir, args.batch_size,
        dataset_type=args.dataset,
        is_distributed=is_distributed,
        seed=args.seed, is_main=is_main
    )

    # ──── ساخت دیتالودر تست غیرتوزیع‌شده روی GPU اصلی ────
    test_loader_full = None
    if is_main:
        _, _, test_loader_full = create_dataloaders(
            args.data_dir, args.batch_size,
            dataset_type=args.dataset,
            is_distributed=False,
            seed=args.seed, is_main=True
        )

    # ──── ارزیابی مدل‌های منفرد با loader غیرتوزیع‌شده ────
    individual_accs = []
    best_single = 0.0
    best_idx = 0

    if is_main:
        print("\n" + "=" * 70)
        print("INDIVIDUAL MODEL PERFORMANCE (After AdaBN)")
        print("=" * 70)
        for i, model in enumerate(base_models):
            acc = evaluate_single_model(
                model, test_loader_full, device,
                f"Model {i+1} ({MODEL_NAMES[i]})",
                MEANS[i], STDS[i], is_main
            )
            individual_accs.append(acc)
        best_single = max(individual_accs)
        best_idx = individual_accs.index(best_single)
        print(f"\nBest Single: Model {best_idx+1} "
              f"({MODEL_NAMES[best_idx]}) → {best_single:.2f}%")
        print("=" * 70)

    # همگام‌سازی قبل از شروع آموزش
    if is_dist():
        dist.barrier()

    # ──── آموزش ────
    best_val_acc = train_hesitant_fuzzy(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device,
        args.save_dir, is_main, MODEL_NAMES
    )

    # همگام‌سازی بعد از آموزش و قبل از بارگذاری
    if is_dist():
        dist.barrier()

    # بارگذاری بهترین مدل (فقط GPU اصلی)
    ensemble_module = (
        ensemble.module if hasattr(ensemble, 'module') else ensemble
    )
    ckpt_path = os.path.join(args.save_dir, 'best_hesitant_fuzzy.pt')

    if is_main and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        ensemble_module.hesitant_fuzzy.load_state_dict(
            ckpt['hesitant_state_dict']
        )
        print("Best model loaded.\n")

    # ──── ارزیابی نهایی (GPU اصلی، غیرتوزیع‌شده) ────
    if is_main:
        print("\n" + "=" * 70)
        print("FINAL ENSEMBLE EVALUATION")
        print("=" * 70)

        final_acc, y_true, y_scores = final_evaluation_unified(
            ensemble_module, test_loader_full, device,
            args.save_dir, MODEL_NAMES, args, is_main
        )

        print("\n" + "=" * 70)
        print("FINAL COMPARISON")
        print("=" * 70)
        print(f"Best Single Model: {best_single:.2f}%")
        print(f"Ensemble Accuracy: {final_acc:.2f}%")
        print(f"Improvement: {final_acc - best_single:+.2f}%")
        print("=" * 70)

        # ذخیره نتایج JSON
        final_results = {
            'seed': args.seed,
            'method': 'Fuzzy_Hesitant_Distributed_AdaBN',
            'best_single_model': {
                'name': MODEL_NAMES[best_idx],
                'accuracy': float(best_single)
            },
            'ensemble': {
                'test_accuracy': float(final_acc)
            },
            'improvement': float(final_acc - best_single)
        }
        with open(os.path.join(args.save_dir, 'final_results.json'),
                  'w') as f:
            json.dump(final_results, f, indent=4)

        torch.save({
            'ensemble_state_dict': ensemble_module.state_dict(),
            'model_names': MODEL_NAMES,
            'means': MEANS, 'stds': STDS,
            'seed': args.seed
        }, os.path.join(args.save_dir, 'final_ensemble_model.pt'))

    # ──── Cleanup قبل از ROC ────
    cleanup_distributed()

    # ──── رسم ROC (فقط GPU اصلی) ────
    if is_main:
        plot_roc_and_f1(
            ensemble_module, test_loader_full,
            device, args.save_dir, MODEL_NAMES, is_main
        )


if __name__ == "__main__":
    main()
