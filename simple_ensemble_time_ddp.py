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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from old_dataset_utils import (
    UADFVDataset,
    create_dataloaders,
    get_sample_info,
    worker_init_fn
)
from visualization_utils import GradCAM, generate_lime_explanation, generate_visualizations
from efficiency_utils import Stopwatch, benchmark_efficiency

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
            print(f"Distributed: rank {rank}/{world_size}, local_rank {local_rank}")
        return device, local_rank, rank, world_size
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return device, 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ================== NORMALIZATION ==================
class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


# ================== SIMPLE AVERAGING ENSEMBLE ==================
class SimpleAveragingEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)

        # فریز کامل مدل‌های پایه
        for model in self.models:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor, return_details: bool = False):
        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
        raw_logits = torch.zeros(x.size(0), self.num_models, 1, device=x.device)

        for i in range(self.num_models):
            x_n = self.normalizations(x, i)
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            raw_logits[:, i] = out
            outputs[:, i] = torch.sigmoid(out)

        final_output = outputs.mean(dim=1)  # میانگین احتمال‌ها

        if return_details:
            weights = torch.ones(x.size(0), self.num_models, device=x.device) / self.num_models
            return final_output, weights, None, raw_logits
        return final_output, None


# ================== BENCHMARK ADAPTER ==================
class _SimpleAvgBenchmarkAdapter(nn.Module):
    def __init__(self, ensemble: 'SimpleAveragingEnsemble'):
        super().__init__()
        self.ensemble = ensemble
        self.models = ensemble.models
        self.normalizations = ensemble.normalizations
        self.num_models = ensemble.num_models

    def eval(self):
        self.ensemble.eval()
        return self

    def hesitant_fuzzy(self, x: torch.Tensor):
        return torch.full((x.size(0), self.num_models),
                          1.0 / self.num_models, device=x.device)

    def forward(self, x: torch.Tensor):
        out, _ = self.ensemble(x)
        w = torch.full((x.size(0), self.num_models),
                       1.0 / self.num_models, device=x.device)
        return out, w


def print_headline_simple_avg(results):
    h = results['methods'].get('Simple Averaging', {})
    if not h:
        return
    thr = h['throughput']
    bs = thr['batch_size']
    fp = results.get('full_test_pass', {})
    print("=" * 70)
    print("SIMPLE AVERAGING EFFICIENCY SUMMARY")
    print("=" * 70)
    print(f"Latency (bs=1)          : {h['latency']['mean_ms']:.2f} ± {h['latency']['std_ms']:.2f} ms / image")
    print(f"Throughput (bs={bs})".ljust(24) +
          f": {thr['throughput_img_per_s']:.1f} ± {thr['throughput_std_img_per_s']:.1f} images/s "
          f"(mean ± std of {thr['repeats']} runs)")
    if fp:
        print(f"Computational time      : {fp['model_compute_time_s']:.2f} ± {fp['model_compute_time_std_s']:.2f} s "
              f"(model inference on the whole test set, {fp['num_images']} images, "
              f"mean of {fp['repeats']} passes; data loading excluded)")
    print("=" * 70 + "\n")


# ================== MODEL LOADING ==================
def load_pruned_models(model_paths: List[str], device: torch.device, is_main: bool) -> List[nn.Module]:
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
            for param in model.parameters():
                param.requires_grad = False
            if is_main:
                param_count = sum(p.numel() for p in model.parameters())
                print(f" → Parameters: {param_count:,} (FROZEN)")
            models.append(model)
        except Exception as e:
            if is_main:
                print(f" [ERROR] Failed to load {path}: {e}")
            continue

    if len(models) == 0:
        raise ValueError("No models loaded!")
    if is_main:
        print(f"All {len(models)} models loaded and frozen successfully!\n")
    return models


# ================== SINGLE MODEL EVAL (DDP-aware) ==================
@torch.no_grad()
def evaluate_single_model_ddp(model: nn.Module, loader: DataLoader, device,
                              name: str, mean: Tuple[float, float, float],
                              std: Tuple[float, float, float], is_main: bool) -> float:
    model.eval()
    normalizer = MultiModelNormalization([mean], [std]).to(device)

    correct = 0
    total_real_samples = len(loader.dataset)

    for images, labels in tqdm(loader, desc=f"Evaluating {name}", disable=not is_main):
        images, labels = images.to(device), labels.to(device).float()
        images = normalizer(images, 0)
        out = model(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = (out.squeeze(1) > 0).long()
        correct += pred.eq(labels.long()).sum().item()

    if dist.is_initialized():
        correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        correct = correct_tensor.item()

    acc = 100. * correct / total_real_samples if total_real_samples > 0 else 0.0
    if is_main:
        print(f" {name}: {acc:.2f}% (Real samples: {total_real_samples})")
    return acc


# ================== UNIFIED FINAL EVALUATION (DDP-aware) ==================
@torch.no_grad()
def evaluate_ensemble_unified(model, loader, device, name, model_names,
                              save_dir, args, is_main=True):
    model.eval()
    total_correct = 0
    total_samples_seen = 0
    ws = dist.get_world_size() if dist.is_initialized() else 1

    all_y_true = []
    all_y_score = []
    all_y_pred = []
    last_weights = None

    total_real_samples = len(loader.dataset)

    if is_main:
        print(f"\nEvaluating {name} set (Simple Averaging)...")

    for images, labels in tqdm(loader, desc=f"Evaluating {name}", disable=not is_main):
        images, labels = images.to(device), labels.to(device)

        outputs, weights, _, raw_logits = model(images, return_details=True)

        if last_weights is None:
            last_weights = weights.detach().cpu()

        # outputs = میانگین احتمال‌ها → برای threshold از 0.5 استفاده می‌کنیم
        probs = outputs.squeeze(1)
        pred_int = (probs > 0.5).long()

        all_y_true.extend(labels.cpu().numpy().tolist())
        all_y_score.extend(probs.cpu().numpy().tolist())
        all_y_pred.extend(pred_int.cpu().numpy().tolist())

        total_correct += pred_int.eq(labels.long()).sum().item()
        total_samples_seen += labels.size(0)

    stats = torch.tensor([total_correct, total_samples_seen],
                         dtype=torch.long, device=device)
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    if ws > 1:
        gathered_true = [None for _ in range(ws)]
        gathered_score = [None for _ in range(ws)]
        gathered_pred = [None for _ in range(ws)]
        if dist.is_initialized():
            dist.all_gather_object(gathered_true, all_y_true)
            dist.all_gather_object(gathered_score, all_y_score)
            dist.all_gather_object(gathered_pred, all_y_pred)

        if is_main:
            all_y_true = [item for sublist in gathered_true for item in sublist]
            all_y_score = [item for sublist in gathered_score for item in sublist]
            all_y_pred = [item for sublist in gathered_pred for item in sublist]

    if not is_main:
        return 0.0, []

    # فقط روی rank 0
    all_y_true = all_y_true[:total_real_samples]
    all_y_score = all_y_score[:total_real_samples]
    all_y_pred = all_y_pred[:total_real_samples]

    total = total_real_samples
    TP = sum(1 for yt, yp in zip(all_y_true, all_y_pred) if yt == 1 and yp == 1)
    TN = sum(1 for yt, yp in zip(all_y_true, all_y_pred) if yt == 0 and yp == 0)
    FP = sum(1 for yt, yp in zip(all_y_true, all_y_pred) if yt == 0 and yp == 1)
    FN = sum(1 for yt, yp in zip(all_y_true, all_y_pred) if yt == 1 and yp == 0)

    correct_count = TP + TN
    acc = 100.0 * correct_count / total if total > 0 else 0.0

    # Real (1)
    precision_real = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall_real = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_real = (2 * precision_real * recall_real / (precision_real + recall_real)
               if (precision_real + recall_real) > 0 else 0.0)

    # Fake (0)
    precision_fake = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    recall_fake = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1_fake = (2 * precision_fake * recall_fake / (precision_fake + recall_fake)
               if (precision_fake + recall_fake) > 0 else 0.0)

    macro_precision = (precision_real + precision_fake) / 2
    macro_recall = (recall_real + recall_fake) / 2
    macro_f1 = (f1_real + f1_fake) / 2
    balanced_acc = ((recall_real + recall_fake) / 2) * 100

    if last_weights is not None:
        final_weights = last_weights.mean(dim=0).numpy()
    else:
        final_weights = np.ones(len(model_names)) / len(model_names)

    # Confusion Matrix Plot
    os.makedirs(save_dir, exist_ok=True)
    cm = np.array([[TN, FP],
                   [FN, TP]])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Predicted Fake', 'Predicted Real'],
        yticklabels=['Actual Fake', 'Actual Real'],
        cbar=True, annot_kws={"size": 16}
    )
    plt.title(
        f'Confusion Matrix\nAccuracy: {acc:.2f}% | '
        f'Balanced Acc: {balanced_acc:.2f}% | Macro-F1: {macro_f1:.4f}',
        fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion Matrix saved to: {cm_path}")

    # گزارش کنسول
    print(f"\n{'='*70}")
    print(f"{name.upper()} SET RESULTS (Simple Averaging)")
    print(f"{'='*70}")
    print(f"Accuracy          : {acc:.2f}%")
    print(f"Balanced Accuracy : {balanced_acc:.2f}%")
    print(f"Macro-F1          : {macro_f1:.4f}")
    print()
    print("--- Real (1) Class Metrics ---")
    print(f"Precision (Real): {precision_real:.4f}")
    print(f"Recall (Real):    {recall_real:.4f}")
    print(f"F1-Score (Real):  {f1_real:.4f}")
    print()
    print("--- Fake (0) Class Metrics ---")
    print(f"Precision (Fake): {precision_fake:.4f}")
    print(f"Recall (Fake):    {recall_fake:.4f}")
    print(f"F1-Score (Fake):  {f1_fake:.4f}")
    print()
    print(f"Macro Avg Precision : {macro_precision:.4f}")
    print(f"Macro Avg Recall    : {macro_recall:.4f}")
    print(f"Macro Avg F1-Score  : {macro_f1:.4f}")
    print()
    print(f"Confusion Matrix:")
    print(f"                 Predicted Fake  Predicted Real")
    print(f"    Actual Fake      {int(TN):<15} {int(FP):<15}")
    print(f"    Actual Real      {int(FN):<15} {int(TP):<15}")
    print()
    print(f"Correct Predictions  : {correct_count} ({acc:.2f}%)")
    print(f"Incorrect Predictions: {total - correct_count} ({100 - acc:.2f}%)")
    print(f" → Total Real Samples: {total:,}")
    print()
    print(f"Model Weights (Uniform):")
    for i, (w, mname) in enumerate(zip(final_weights, model_names)):
        print(f" {i+1:2d}. {mname:<25}: {w:6.4f} ({w*100:5.2f}%)")
    print(f"{'='*70}")

    # prediction_log.txt
    lines = []
    lines.append("-" * 100)
    lines.append("SUMMARY STATISTICS:")
    lines.append("-" * 100)
    lines.append(f"Overall Accuracy   : {acc:.2f}%")
    lines.append(f"Balanced Accuracy  : {balanced_acc:.2f}%")
    lines.append(f"Macro F1-Score     : {macro_f1:.4f}")
    lines.append("\n--- Real (1) Class Metrics ---")
    lines.append(f"Precision (Real): {precision_real:.4f}")
    lines.append(f"Recall (Real):    {recall_real:.4f}")
    lines.append(f"F1-Score (Real):  {f1_real:.4f}")
    lines.append("\n--- Fake (0) Class Metrics ---")
    lines.append(f"Precision (Fake): {precision_fake:.4f}")
    lines.append(f"Recall (Fake):    {recall_fake:.4f}")
    lines.append(f"F1-Score (Fake):  {f1_fake:.4f}")
    lines.append(f"\nCorrect Predictions: {correct_count} ({acc:.2f}%)")
    lines.append(f"Incorrect Predictions: {total - correct_count} ({100 - acc:.2f}%)")
    lines.append(f"\nConfusion Matrix saved to: confusion_matrix.png")
    lines.append("")
    lines.append("=" * 100)
    lines.append("SAMPLE-BY-SAMPLE PREDICTIONS:")
    lines.append("=" * 100)
    lines.append(f"{'ID':<10} {'True':<10} {'Pred':<10} {'Score':<12} {'Status':<10}")
    lines.append("-" * 100)
    for i, (yt, yp, ys) in enumerate(zip(all_y_true, all_y_pred, all_y_score)):
        status = "Correct" if yt == yp else "Wrong"
        lines.append(f"{i+1:<10} {int(yt):<10} {int(yp):<10} {ys:<12.6f} {status:<10}")

    log_path = os.path.join(save_dir, 'prediction_log.txt')
    with open(log_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"✅ Prediction log saved to: {log_path}")

    # ROC data
    y_true_np = np.array(all_y_true)
    y_score_np = np.array(all_y_score)
    y_pred_np = np.array(all_y_pred)

    roc_json_path = os.path.join(save_dir, "roc_data_test.json")
    roc_data_json = {
        "metadata": {
            "seed": args.seed,
            "dataset": args.dataset,
            "num_samples": int(total),
            "positive_count": int(np.sum(y_true_np)),
            "negative_count": int(total - np.sum(y_true_np)),
            "model": "simple_averaging_ensemble"
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

    # Visualizations
    try:
        from visualization_utils import generate_visualizations
        vis_dir = os.path.join(save_dir, 'visualizations')
        generate_visualizations(
            model, loader, device, vis_dir, model_names,
            args.num_grad_cam_samples, args.num_lime_samples,
            args.dataset, is_main)
    except Exception as e:
        print(f"[WARN] Visualizations failed: {e}")

    return acc, final_weights.tolist()


# ================== MAIN ==================
def main():
    parser = argparse.ArgumentParser(description="Simple Averaging Ensemble (Multi-GPU ready)")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['wild','20k','artiface', 'deepfake_lab', 'hard_fake_real',
                                 'real_fake_dataset', 'uadfV', 'custom_genai'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if len(args.model_names) != len(args.model_paths):
        raise ValueError("Number of model_names must match model_paths")

    set_seed(args.seed)
    device, local_rank, rank, world_size = setup_distributed()
    is_main = rank == 0

    pipeline_sw = Stopwatch(device).start()

    if is_main:
        print("=" * 70)
        print("SIMPLE AVERAGING ENSEMBLE EVALUATION")
        print(f"Distributed on {world_size} GPU(s) | Seed: {args.seed}")
        print("=" * 70)

    DEFAULT_MEANS = [
        (0.5207, 0.4258, 0.3806),
        (0.4460, 0.3622, 0.3416),
        (0.4668, 0.3816, 0.3414),
    ]
    DEFAULT_STDS = [
        (0.2490, 0.2239, 0.2212),
        (0.2057, 0.1849, 0.1761),
        (0.2410, 0.2161, 0.2081),
    ]

    base_models = load_pruned_models(args.model_paths, device, is_main)
    n_loaded = len(base_models)

    if n_loaded > len(DEFAULT_MEANS):
        MEANS = DEFAULT_MEANS + [DEFAULT_MEANS[-1]] * (n_loaded - len(DEFAULT_MEANS))
        STDS = DEFAULT_STDS + [DEFAULT_STDS[-1]] * (n_loaded - len(DEFAULT_STDS))
    else:
        MEANS = DEFAULT_MEANS[:n_loaded]
        STDS = DEFAULT_STDS[:n_loaded]

    MODEL_NAMES = args.model_names[:n_loaded]

    ensemble = SimpleAveragingEnsemble(base_models, MEANS, STDS).to(device)

    # نیازی به DDP نیست چون training نداریم (فقط evaluation)
    # اما اگر بخواهید می‌توانید uncomment کنید:
    # if world_size > 1:
    #     ensemble = DDP(ensemble, device_ids=[local_rank], output_device=local_rank)

    set_seed(args.seed)

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            args.data_dir, args.batch_size, dataset_type=args.dataset,
            is_distributed=(world_size > 1), seed=args.seed, is_main=is_main)
    except Exception as e:
        if is_main:
            print(f"[ERROR] Dataset loading failed: {e}")
        cleanup_distributed()
        return

    # ============ INDIVIDUAL MODELS ============
    if is_main:
        print("\n" + "=" * 70)
        print("INDIVIDUAL MODEL PERFORMANCE")
        print("=" * 70)

    individual_accs = []
    individual_times = []
    set_seed(args.seed)

    for i, model in enumerate(base_models):
        sw = Stopwatch(device).start()
        acc = evaluate_single_model_ddp(
            model, test_loader, device,
            f"Model {i+1} ({MODEL_NAMES[i]})",
            MEANS[i], STDS[i], is_main)
        t = sw.stop()
        individual_accs.append(acc)
        individual_times.append(t)
        if is_main:
            print(f" ⏱ Evaluation time: {t:.1f}s\n")

    best_single = max(individual_accs) if individual_accs else 0.0
    best_idx = individual_accs.index(best_single) if individual_accs else 0

    if is_main:
        print(f"\nBest Single: Model {best_idx+1} "
              f"({MODEL_NAMES[best_idx]}) → {best_single:.2f}%")

    ensemble_module = ensemble.module if hasattr(ensemble, 'module') else ensemble

    # ============ EFFICIENCY BENCHMARK (فقط rank 0) ============
    os.makedirs(args.save_dir, exist_ok=True)
    efficiency = {}
    bench_time = 0.0

    if is_main:
        try:
            adapter = _SimpleAvgBenchmarkAdapter(ensemble_module).to(device)
            bench_sw = Stopwatch(device).start()
            efficiency = benchmark_efficiency(
                adapter, test_loader, device, MODEL_NAMES,
                latency_samples=min(300, len(test_loader.dataset)),
                verbose=True,
            )
            bench_time = bench_sw.stop()

            if 'HFDE (dynamic)' in efficiency.get('methods', {}):
                efficiency['methods']['Simple Averaging'] = efficiency['methods'].pop('HFDE (dynamic)')
            efficiency.get('methods', {}).pop('Router only', None)
            if 'full_test_pass_hfde' in efficiency:
                efficiency['full_test_pass'] = efficiency.pop('full_test_pass_hfde')
            efficiency['method'] = 'Simple Averaging'
            efficiency['benchmark_time_s'] = float(bench_time)
            print_headline_simple_avg(efficiency)
        except Exception as e:
            print(f"[WARNING] Efficiency benchmark failed: {e}")
            efficiency = {}
            bench_time = 0.0

    # ============ FINAL ENSEMBLE EVALUATION ============
    if is_main:
        print("\n" + "=" * 70)
        print("FINAL ENSEMBLE EVALUATION (Simple Averaging)")
        print("=" * 70)

    ensemble_sw = Stopwatch(device).start()
    final_acc, ensemble_weights = evaluate_ensemble_unified(
        ensemble_module, test_loader, device, "Test",
        MODEL_NAMES, args.save_dir, args, is_main
    )
    ensemble_eval_time = ensemble_sw.stop()

    if is_main:
        print(f"\n⏱ Final ensemble evaluation time: {ensemble_eval_time:.1f}s")

        print("\n" + "=" * 70)
        print("FINAL COMPARISON")
        print("=" * 70)
        print(f"Best Single Model: {best_single:.2f}%")
        print(f"Ensemble Accuracy (Simple Averaging): {final_acc:.2f}%")
        print(f"Improvement: {final_acc - best_single:+.2f}%")
        print("=" * 70)

        # ذخیره مدل
        final_model_path = os.path.join(args.save_dir, 'final_ensemble_model_simple_avg.pt')
        torch.save({
            'ensemble_state_dict': ensemble_module.state_dict(),
            'test_accuracy': final_acc,
            'model_names': MODEL_NAMES,
            'means': MEANS,
            'stds': STDS,
            'seed': args.seed,
        }, final_model_path)
        print(f"Model saved: {final_model_path}")

        # TIME SUMMARY
        total_pipeline_time = pipeline_sw.stop()
        print("\n" + "=" * 70)
        print("TIME SUMMARY")
        print("=" * 70)
        for i, name in enumerate(MODEL_NAMES):
            print(f"  Model {i+1} ({name}) evaluation : {individual_times[i]:.1f}s")
        print(f"  Efficiency benchmark          : {bench_time:.1f}s")
        print(f"  Final ensemble evaluation     : {ensemble_eval_time:.1f}s")
        print(f"  Total pipeline time           : {total_pipeline_time/60:.2f} min ({total_pipeline_time:.1f}s)")
        print("=" * 70)

        final_results = {
            'seed': args.seed,
            'method': 'Simple Averaging',
            'dataset': args.dataset,
            'best_single_model': {
                'name': MODEL_NAMES[best_idx],
                'accuracy': float(best_single)
            },
            'ensemble': {
                'test_accuracy': float(final_acc),
                'model_weights': {name: float(w) for name, w in zip(MODEL_NAMES, ensemble_weights)}
            },
            'improvement': float(final_acc - best_single),
            'timing': {
                'individual_model_eval_times_s': {
                    name: float(t) for name, t in zip(MODEL_NAMES, individual_times)
                },
                'benchmark_time_s': float(bench_time),
                'ensemble_eval_time_s': float(ensemble_eval_time),
                'total_pipeline_time_s': float(total_pipeline_time),
            }
        }
        with open(os.path.join(args.save_dir, 'final_results.json'), 'w') as f:
            json.dump(final_results, f, indent=4)

        if efficiency:
            efficiency['individual_model_eval_times_s'] = {
                name: float(t) for name, t in zip(MODEL_NAMES, individual_times)
            }
            efficiency['ensemble_eval_time_s'] = float(ensemble_eval_time)
            efficiency['total_pipeline_time_s'] = float(total_pipeline_time)
            with open(os.path.join(args.save_dir, 'efficiency_report.json'), 'w') as f:
                json.dump(efficiency, f, indent=2)
            print(f"Efficiency report saved: {os.path.join(args.save_dir, 'efficiency_report.json')}")

    cleanup_distributed()

    if is_main:
        try:
            from metrics_utils import plot_roc_and_f1
            plot_roc_and_f1(
                ensemble_module,
                test_loader,
                device,
                args.save_dir,
                MODEL_NAMES,
                is_main
            )
        except Exception as e:
            print(f"[WARN] plot_roc_and_f1 failed: {e}")


if __name__ == "__main__":
    main()
