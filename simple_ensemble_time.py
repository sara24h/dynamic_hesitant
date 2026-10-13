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

from old_dataset_utils import (
    UADFVDataset, 
    create_dataloaders, 
    get_sample_info, 
    worker_init_fn
)
from visualization_utils import GradCAM, generate_lime_explanation, generate_visualizations
from evaluate_utils import Stopwatch, benchmark_efficiency

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


# ================== UNIFIED FINAL EVALUATION & REPORT ==================
@torch.no_grad()
def final_evaluation_and_report(model, loader, device, save_dir, model_name, args):
    if loader is None: return 0.0, None, None

    model.eval()
    
    # استخراج دیتاست پایه و اندیس‌ها
    base_dataset = loader.dataset
    if hasattr(base_dataset, 'dataset'):
        base_dataset = base_dataset.dataset
        
    if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'indices'):
        test_indices = loader.sampler.indices
    elif hasattr(loader.dataset, 'indices'):
        test_indices = loader.dataset.indices
    else:
        test_indices = list(range(len(base_dataset)))

    all_y_true = []
    all_y_score = []
    all_y_pred = []
    lines = []

    lines.append("="*100)
    lines.append("SAMPLE-BY-SAMPLE PREDICTIONS (For McNemar Test Comparison):")
    lines.append("="*100)
    header = f"{'Sample_ID':<10} {'Sample_Path':<60} {'True_Label':<12} {'Predicted_Label':<15} {'Correct':<10}"
    lines.append(header)
    lines.append("-"*100)

    TP, TN, FP, FN = 0, 0, 0, 0
    correct_count = 0
    total_samples = 0

    print(f"\nRunning Final Evaluation on {len(test_indices)} samples...")
    
    for i, global_idx in enumerate(tqdm(test_indices, desc="Final Eval")):
        try:
            image, label = base_dataset[global_idx]
            path, _ = get_sample_info(base_dataset, global_idx)
        except Exception as e:
            continue

        image = image.unsqueeze(0).to(device)
        label_int = int(label)
        
        # پیش‌بینی مدل
        output, _, _, stacked_logits = model(image, return_details=True)

        # تبدیل لاجیت‌ها به احتمالات
        probs = torch.sigmoid(stacked_logits).mean(dim=1).item()
        pred_int = int(probs > 0.5)
        
        all_y_true.append(label_int)
        all_y_score.append(probs)
        all_y_pred.append(pred_int)
        
        is_correct = (pred_int == label_int)
        
        # محاسبه ماتریس آشفتگی (Real=1, Fake=0)
        if label_int == 1:          # Real
            if pred_int == 1: TP += 1
            else: FN += 1
        else:                       # Fake
            if pred_int == 1: FP += 1
            else: TN += 1
            
        total_samples += 1
        
        filename = os.path.basename(path)
        if len(filename) > 55: filename = filename[:25] + "..." + filename[-27:]
        line = f"{i+1:<10} {filename:<60} {label_int:<12} {pred_int:<15} {'Yes' if is_correct else 'No':<10}"
        lines.append(line)

    total_samples = len(all_y_true)
    correct_count = TP + TN
    acc = 100.0 * correct_count / total_samples if total_samples > 0 else 0.0
    
    # ================== محاسبه معیارهای عملکرد ==================
    # Real class (Positive class = 1)
    precision_real = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_real    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_real        = 2 * precision_real * recall_real / (precision_real + recall_real) if (precision_real + recall_real) > 0 else 0

    # Fake class (Negative class = 0)
    precision_fake = TN / (TN + FN) if (TN + FN) > 0 else 0
    recall_fake    = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_fake        = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0

    # Macro averages
    macro_precision = (precision_real + precision_fake) / 2
    macro_recall    = (recall_real + recall_fake) / 2
    macro_f1        = (f1_real + f1_fake) / 2

    # Balanced Accuracy
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
    with open(log_path, 'w', encoding='utf-8') as f:
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
    with open(roc_txt_path, 'w') as f:
        f.write("y_true\ty_score\ty_pred\n")
        for t, s, p in zip(y_true_np, y_score_np, y_pred_np):
            f.write(f"{int(t)}\t{s:.6f}\t{int(p)}\n")
    print(f"ROC data saved (TXT):  {roc_txt_path}")

    return acc, y_true_np, y_score_np


# ================== BENCHMARK ADAPTER (سازگاری با evaluate_utils) ==================
class _SimpleAvgBenchmarkAdapter(nn.Module):
    """
    benchmark_efficiency در evaluate_utils برای انسمبل HFDE نوشته شده و انتظار دارد:
      - ensemble.models / ensemble.normalizations
      - ensemble.hesitant_fuzzy(x)          (متد 'Router only')
      - ensemble(x) -> (output, weights)    (آمار مسیریابی از o[1])
    SimpleAveragingEnsemble روتر ندارد؛ این آداپتور بدون تغییر کلاس اصلی و بدون
    تغییر evaluate_utils، رابط بالا را فراهم می‌کند: وزن‌های یکنواخت (1/M) که
    همیشه مثبت‌اند؛ بنابراین آمار routing به‌درستی نشان می‌دهد که در این انسمبل
    همه مدل‌ها همیشه اجرا می‌شوند.
    """
    def __init__(self, ensemble: 'SimpleAveragingEnsemble'):
        super().__init__()
        self.ensemble = ensemble
        self.models = ensemble.models
        self.normalizations = ensemble.normalizations
        self.num_models = ensemble.num_models

    def hesitant_fuzzy(self, x: torch.Tensor):
        # این انسمبل روتر ندارد؛ وزن یکنواخت با هزینه‌ی تقریباً صفر
        return torch.full((x.size(0), self.num_models),
                          1.0 / self.num_models, device=x.device)

    def forward(self, x: torch.Tensor):
        out, _ = self.ensemble(x)
        w = torch.full((x.size(0), self.num_models),
                      1.0 / self.num_models, device=x.device)
        return out, w


def print_headline_simple_avg(results):
    """چاپ سه عدد اصلی کارایی برای Simple Averaging (دقیقاً همان فرمت HFDE)."""
    h = results['methods']['Simple Averaging']
    thr = h['throughput']
    bs = thr['batch_size']
    fp = results['full_test_pass']
    print("=" * 70)
    print("SIMPLE AVERAGING EFFICIENCY SUMMARY")
    print("=" * 70)
    print(f"Latency (bs=1)          : {h['latency']['mean_ms']:.2f} ± {h['latency']['std_ms']:.2f} ms / image")
    print(f"Throughput (bs={bs})".ljust(24) +
          f": {thr['throughput_img_per_s']:.1f} ± {thr['throughput_std_img_per_s']:.1f} images/s "
          f"(mean ± std of {thr['repeats']} runs)")
    print(f"Computational time      : {fp['model_compute_time_s']:.2f} ± {fp['model_compute_time_std_s']:.2f} s "
          f"(model inference on the whole test set, {fp['num_images']} images, "
          f"mean of {fp['repeats']} passes; data loading excluded)")
    print("=" * 70 + "\n")


class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


class SimpleAveragingEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)

    def forward(self, x: torch.Tensor, return_details: bool = False):
        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
        raw_logits = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
    
        for i in range(self.num_models):
            x_n = self.normalizations(x, i)
            with torch.no_grad(): # برای امنیت بیشتر حافظه
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            raw_logits[:, i] = out
            outputs[:, i] = torch.sigmoid(out)  
    
        final_output = outputs.mean(dim=1)  
    
        if return_details:
            weights = torch.ones(x.size(0), self.num_models, device=x.device) / self.num_models
            return final_output, weights, None, raw_logits
        return final_output, None


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
            
            # ===== فریز کردن صد در صدی مدل های پایه =====
            for param in model.parameters():
                param.requires_grad = False
            # =============================================
            
            param_count = sum(p.numel() for p in model.parameters())
            print(f" → Parameters: {param_count:,} (FROZEN)")
            models.append(model)
        except Exception as e:
            print(f" [ERROR] Failed to load {path}: {e}")
            continue

    if len(models) == 0:
        raise ValueError("No models loaded!")
    print(f"All {len(models)} models loaded and frozen successfully!\n")
    return models


# ================== EVALUATION FUNCTIONS ==================
@torch.no_grad()
def evaluate_single_model(model: nn.Module, loader: DataLoader, device: torch.device,
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
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = (out.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()

    acc = 100. * correct / total if total > 0 else 0.0
    print(f" {name}: {acc:.2f}%")
    return acc


# ================== MAIN FUNCTION ==================
def main():
    parser = argparse.ArgumentParser(description="Simple Averaging Ensemble - Single GPU")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['wild', 'deepfake_lab', 'hard_fake_real', 'uadfV','real_fake_dataset'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if len(args.model_names) != len(args.model_paths):
        raise ValueError("Number of model_names must match model_paths")

    set_seed(args.seed)
    
    # تنظیم دستگاه به صورت تک GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # زمان‌سنجی کل پایپ‌لاین
    pipeline_sw = Stopwatch(device).start()

    print("="*70)
    print(f"SIMPLE AVERAGING ENSEMBLE (Single GPU Mode)")
    print(f"Seed: {args.seed}")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Models: {len(args.model_paths)}")
    print("="*70 + "\n")

    MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    MEANS = MEANS[:len(args.model_paths)]
    STDS = STDS[:len(args.model_paths)]

    base_models = load_pruned_models(args.model_paths, device)
    MODEL_NAMES = args.model_names[:len(base_models)]

    ensemble = SimpleAveragingEnsemble(base_models, MEANS, STDS).to(device)

    trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
    total = sum(p.numel() for p in ensemble.parameters())
    print(f"Total params: {total:,} | Trainable: {trainable:,} | Frozen: {total-trainable:,}\n")

    # ساخت دیتالودر (غیرتوزیع شده)
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=False, seed=args.seed, is_main=True)

    print("\n" + "="*70)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("="*70)

    individual_accs = []
    individual_times = []
    for i, model in enumerate(base_models):
        sw = Stopwatch(device).start()
        acc = evaluate_single_model(
            model, test_loader, device,
            f"Model {i+1} ({MODEL_NAMES[i]})",
            MEANS[i], STDS[i])
        t = sw.stop()
        individual_times.append(t)
        print(f" ⏱ Evaluation time: {t:.1f}s\n")
        individual_accs.append(acc)

    best_single = max(individual_accs)
    best_idx = individual_accs.index(best_single)

    print(f"\nBest Single: Model {best_idx+1} ({MODEL_NAMES[best_idx]}) → {best_single:.2f}%")
    print("="*70)
    print("\nSkipping Training (Simple Averaging does not learn parameters)...\n")

    print("\n" + "="*70)
    print("FINAL ENSEMBLE EVALUATION")
    print("="*70)

    os.makedirs(args.save_dir, exist_ok=True)

    # ================== بنچمارک کارایی (Latency / Throughput / Computational time) ==================
    # همان پروتکل اسکریپت HFDE با استفاده از evaluate_utils.benchmark_efficiency
    efficiency = {}
    bench_time = 0.0
    try:
        adapter = _SimpleAvgBenchmarkAdapter(ensemble)
        bench_sw = Stopwatch(device).start()
        efficiency = benchmark_efficiency(
            adapter, test_loader, device, MODEL_NAMES,
            latency_samples=len(test_loader.dataset))
        bench_time = bench_sw.stop()

        # برچسب‌گذاری درست برای گزارش JSON (بنچمارک برای HFDE نام‌گذاری کرده)
        efficiency['methods']['Simple Averaging'] = efficiency['methods'].pop('HFDE (dynamic)')
        efficiency['full_test_pass'] = efficiency.pop('full_test_pass_hfde')
        # این انسمبل روتر ندارد؛ مدخل بی‌معنای 'Router only' حذف شود
        efficiency['methods'].pop('Router only', None)
        efficiency['method'] = 'Simple Averaging'
        efficiency['benchmark_time_s'] = float(bench_time)

        print_headline_simple_avg(efficiency)
    except Exception as e:
        print(f"[WARNING] Efficiency benchmark failed: {e}")
        efficiency = {}
        bench_time = 0.0

    # زمان‌سنجی ارزیابی نهایی
    eval_sw = Stopwatch(device).start()
    ensemble_test_acc, y_true, y_score = final_evaluation_and_report(
        ensemble, test_loader, device, args.save_dir, 
        "Simple Averaging Ensemble", args
    )
    final_eval_time = eval_sw.stop()
    print(f"\n⏱ Final evaluation time: {final_eval_time:.1f}s")

    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"Best Single Model: {best_single:.2f}%")
    print(f"Ensemble Accuracy: {ensemble_test_acc:.2f}%")
    print(f"Improvement: {ensemble_test_acc - best_single:+.2f}%")
    print("="*70)

    final_results = {
        'method': 'Simple Averaging',
        'best_single_model': {
            'name': MODEL_NAMES[best_idx],
            'accuracy': float(best_single)
        },
        'ensemble': {
            'test_accuracy': float(ensemble_test_acc),
            'strategy': 'Uniform Weights'
        },
        'improvement': float(ensemble_test_acc - best_single)
    }

    results_path = os.path.join(args.save_dir, 'final_results_simple_avg.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\nResults saved: {results_path}")

    final_model_path = os.path.join(args.save_dir, 'final_ensemble_model_simple_avg.pt')
    torch.save({
        'ensemble_state_dict': ensemble.state_dict(),
        'test_accuracy': ensemble_test_acc,
        'model_names': MODEL_NAMES,
        'means': MEANS,
        'stds': STDS
    }, final_model_path)
    print(f"Model saved: {final_model_path}")

    vis_dir = os.path.join(args.save_dir, 'visualizations')
    generate_visualizations(
        ensemble, test_loader, device, vis_dir, MODEL_NAMES,
        args.num_grad_cam_samples, args.num_lime_samples,
        args.dataset, is_main=True
    )
   
    # ================== خلاصه زمان‌ها ==================
    total_pipeline_time = pipeline_sw.stop()

    print("\n" + "="*70)
    print("TIME SUMMARY")
    print("="*70)
    for i, name in enumerate(MODEL_NAMES):
        print(f"  Model {i+1} ({name}) evaluation : {individual_times[i]:.1f}s")
    print(f"  Efficiency benchmark         : {bench_time:.1f}s")
    print(f"  Final ensemble evaluation    : {final_eval_time:.1f}s")
    print(f"  Total pipeline time          : {total_pipeline_time/60:.2f} min ({total_pipeline_time:.1f}s)")
    print("="*70)

    # ذخیره گزارش کارایی (Latency / Throughput / زمان‌ها)
    efficiency['individual_model_eval_times_s'] = {
        name: float(t) for name, t in zip(MODEL_NAMES, individual_times)}
    efficiency['final_evaluation_time_s'] = final_eval_time   # شامل ویژوالایزیشن نیست
    efficiency['total_pipeline_time_s'] = total_pipeline_time
    with open(os.path.join(args.save_dir, 'efficiency_report.json'), 'w') as f:
        json.dump(efficiency, f, indent=2)
    print(f"Efficiency report saved: {os.path.join(args.save_dir, 'efficiency_report.json')}")

if __name__ == "__main__":
    main()
