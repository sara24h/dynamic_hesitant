import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, SequentialSampler
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
import cv2
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP
from metrics_utils import plot_roc_and_f1
from efficiency_utils import (
    Stopwatch, summarize_training_times,
    benchmark_efficiency, print_headline
)

warnings.filterwarnings("ignore")

# =================== بخش ایمپورت دیتاست ===================
try:
    from old_dataset_utils import (
        UADFVDataset,
        CustomGenAIDataset,
        create_dataloaders,
        get_sample_info,
        worker_init_fn
    )
except ImportError:
    print("Warning: 'dataset_utils.py' not found. Using dummy functions.")
    def create_dataloaders(*args, **kwargs): return None, None, None
    def get_sample_info(*args, **kwargs): return "dummy_path", 0
    def worker_init_fn(*args, **kwargs): pass
    class UADFVDataset(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs): pass
    class CustomGenAIDataset(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs): pass

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
except ImportError:
    lime_image = None


# ================== UTILITY FUNCTIONS ==================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_ensemble_checkpoint(save_path: str, ensemble_model: nn.Module,
                             model_paths: List[str], model_names: List[str],
                             accuracy: float, means: List, stds: List):
    model_to_save = ensemble_model.module if hasattr(ensemble_model, 'module') else ensemble_model
    checkpoint = {
        'format_version': '1.0',
        'model_paths': model_paths,
        'model_names': model_names,
        'normalization_means': means,
        'normalization_stds': stds,
        'accuracy': accuracy,
        'state_dict': model_to_save.state_dict()
    }
    torch.save(checkpoint, save_path)
    print(f" Best Ensemble model saved to: {save_path}")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, score):
        self.model.zero_grad()
        score.backward()
        if self.gradients is None:
            raise RuntimeError("Gradients not captured")
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=[1, 2], keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                            size=activations.shape[1:],
                            mode='bilinear', align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.squeeze().cpu().numpy()


class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


# ================== MAJORITY VOTING ENSEMBLE (FIXED) ==================
class MajorityVotingEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]],
                 stds: List[Tuple[float]], freeze_models: bool = True):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor, return_details: bool = False):
        votes_logits = []
        for i in range(self.num_models):
            current_std = self.normalizations.__getattr__(f'std_{i}')
            x_n = x
            if not torch.all(current_std == 0):
                x_n = self.normalizations(x, i)
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            votes_logits.append(out)

        stacked_logits = torch.cat(votes_logits, dim=1)  # [B, N]

        # ================= Hard Voting =================
        hard_votes = (stacked_logits > 0).long()
        sum_votes = hard_votes.sum(dim=1, keepdim=True)
        threshold = self.num_models / 2.0
        final_decision = (sum_votes >= threshold).long().float()

        # ================= Soft Score for ROC =================
        avg_probs = torch.sigmoid(stacked_logits).mean(dim=1, keepdim=True)

        if return_details:
            batch_size = x.size(0)
            weights = torch.ones(batch_size, self.num_models, device=x.device) / self.num_models
            return final_decision, weights, avg_probs, stacked_logits

        return final_decision, hard_votes


# ================== BENCHMARK ADAPTER ==================
class _MajorityVotingBenchmarkAdapter(nn.Module):
    """
    آداپتور برای benchmark_efficiency که API آن با hesitant_fuzzy سازگار است.
    """

    def __init__(self, ensemble: 'MajorityVotingEnsemble'):
        super().__init__()
        self.ensemble = ensemble
        self.models = ensemble.models
        self.normalizations = ensemble.normalizations
        self.num_models = ensemble.num_models

    def eval(self):
        self.ensemble.eval()
        return self

    def hesitant_fuzzy(self, x: torch.Tensor):
        # وزن یکنواخت برای همه مدل‌ها (چون بنچمارک این فرمت را انتظار دارد)
        return torch.full((x.size(0), self.num_models),
                          1.0 / self.num_models, device=x.device)

    def forward(self, x: torch.Tensor):
        out, _ = self.ensemble(x)
        w = torch.full((x.size(0), self.num_models),
                       1.0 / self.num_models, device=x.device)
        return out, w


def print_headline_majority_voting(results):
    """چاپ خلاصه‌ی سه عدد اصلی کارایی برای Majority Voting."""
    h = results['methods'].get('Majority Voting', {})
    if not h:
        return
    thr = h['throughput']
    bs = thr['batch_size']
    fp = results.get('full_test_pass', {})
    print("=" * 70)
    print("MAJORITY VOTING EFFICIENCY SUMMARY")
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
        raise ImportError("Cannot import ResNet_50_pruned_hardfakevsreal.")

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
    if is_main:
        print(f"All {len(models)} models loaded!\n")
    return models


# ================== EVALUATION FUNCTIONS ==================
@torch.no_grad()
def evaluate_single_model_ddp(model: nn.Module, loader: DataLoader, device,
                              name: str, mean: Tuple[float, float, float],
                              std: Tuple[float, float, float], is_main: bool) -> float:
    model.eval()
    normalizer = MultiModelNormalization([mean], [std]).to(device)
    correct = 0
    total_real_samples = len(loader.dataset)

    if loader is None:
        return 0.0

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


# ================== UNIFIED EVALUATION (Single Source of Truth) ==================
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
        print(f"\nEvaluating {name} set (Majority Voting - Hard)...")

    for images, labels in tqdm(loader, desc=f"Evaluating {name}", disable=not is_main):
        images, labels = images.to(device), labels.to(device)
        outputs, weights, avg_probs, stacked_logits = model(images, return_details=True)

        if last_weights is None:
            last_weights = weights.detach().cpu()

        pred_int = outputs.squeeze(1).long()
        score_for_roc = avg_probs.squeeze(1)

        all_y_true.extend(labels.cpu().numpy().tolist())
        all_y_score.extend(score_for_roc.cpu().numpy().tolist())
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

    # ============ بعد از این خط فقط روی rank 0 ============
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

    # --- Real (1) Class Metrics ---
    precision_real = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_real = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_real = (2 * precision_real * recall_real / (precision_real + recall_real)
               if (precision_real + recall_real) > 0 else 0)

    # --- Fake (0) Class Metrics ---
    precision_fake = TN / (TN + FN) if (TN + FN) > 0 else 0
    recall_fake = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_fake = (2 * precision_fake * recall_fake / (precision_fake + recall_fake)
               if (precision_fake + recall_fake) > 0 else 0)

    # Macro
    macro_precision = (precision_real + precision_fake) / 2
    macro_recall = (recall_real + recall_fake) / 2
    macro_f1 = (f1_real + f1_fake) / 2

    # Balanced Accuracy
    balanced_acc = ((recall_real + recall_fake) / 2) * 100

    if last_weights is not None:
        final_weights = last_weights.mean(dim=0).numpy()
    else:
        final_weights = np.zeros(len(model_names))

    # ================== Confusion Matrix Plot ==================
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

    # ================== گزارش واحد ==================
    print(f"\n{'='*70}")
    print(f"{name.upper()} SET RESULTS (Majority Voting - Hard)")
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
    print(f"                 Predicted Real  Predicted Fake")
    print(f"    Actual Real      {int(TP):<15} {int(FN):<15}")
    print(f"    Actual Fake      {int(FP):<15} {int(TN):<15}")
    print()
    print(f"Correct Predictions  : {correct_count} ({acc:.2f}%)")
    print(f"Incorrect Predictions: {total - correct_count} ({100 - acc:.2f}%)")
    print(f" → Total Real Samples: {total:,}")
    print()
    print(f"Learned Model Weights (Global):")
    for i, (w, mname) in enumerate(zip(final_weights, model_names)):
        print(f" {i+1:2d}. {mname:<25}: {w:6.4f} ({w*100:5.2f}%)")
    print(f"{'='*70}")

    # ================== ذخیره prediction_log.txt ==================
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
    lines.append("SAMPLE-BY-SAMPLE PREDICTIONS (Hard Voting Logic):")
    lines.append("=" * 100)
    lines.append(f"{'ID':<10} {'True':<10} {'Pred':<10} {'Score':<12} {'Status':<10}")
    lines.append("-" * 100)
    for i, (yt, yp, ys) in enumerate(zip(all_y_true, all_y_pred, all_y_score)):
        status = "Correct" if yt == yp else "Wrong"
        lines.append(f"{i+1:<10} {int(yt):<10} {int(yp):<10} {ys:<12.6f} {status:<10}")

    log_path = os.path.join(save_dir, 'prediction_log.txt')
    with open(log_path, 'w') as f:
        f.write("\n".join(lines))
    print(f" Prediction log saved to: {log_path}")

    # ================== ذخیره ROC data (JSON + TXT) ==================
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
            "model": "majority_voting_ensemble_HARD"
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

    # ================== Visualizations ==================
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ================== MAIN FUNCTION ==================
def main():
    parser = argparse.ArgumentParser(
        description="Majority Voting Ensemble Evaluation (HARD VOTING)")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['wild', 'deepfake_lab', 'hard_fake_real',
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
        print("MAJORITY VOTING ENSEMBLE EVALUATION (HARD VOTING)")
        print(f"Distributed on {world_size} GPU(s) | Seed: {args.seed}")
        print("=" * 70)

    MEANS = [(0.5207, 0.4258, 0.3806),
             (0.4460, 0.3622, 0.3416),
             (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212),
            (0.2057, 0.1849, 0.1761),
            (0.2410, 0.2161, 0.2081)]

    base_models = load_pruned_models(args.model_paths, device, is_main)
    MODEL_NAMES = args.model_names[:len(base_models)]
    MEANS = MEANS[:len(base_models)]
    STDS = STDS[:len(base_models)]

    ensemble = MajorityVotingEnsemble(base_models, MEANS, STDS,
                                      freeze_models=True).to(device)
    set_seed(args.seed)

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            args.data_dir, args.batch_size, dataset_type=args.dataset,
            is_distributed=(world_size > 1), seed=args.seed, is_main=is_main)
    except FileNotFoundError as e:
        if is_main:
            print(f"\n[ERROR] Dataset loading failed: {e}")
            print(f"[HINT] Please check if your --data_dir ('{args.data_dir}') "
                  f"contains the correct folder structure.")
        cleanup_distributed()
        return
    except Exception as e:
        if is_main:
            print(f"Unexpected error: {e}")
        cleanup_distributed()
        return

    # ================== ارزیابی مدل‌های تکی ==================
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

    # ================== EFFICIENCY BENCHMARK ==================
    os.makedirs(args.save_dir, exist_ok=True)
    efficiency = {}
    bench_time = 0.0
    if is_main:
        try:
            adapter = _MajorityVotingBenchmarkAdapter(ensemble_module).to(device)
            bench_sw = Stopwatch(device).start()
            efficiency = benchmark_efficiency(
                adapter, test_loader, device, MODEL_NAMES,
                latency_samples=min(300, len(test_loader.dataset)),
                verbose=True,
            )
            bench_time = bench_sw.stop()

            # تغییر برچسب‌ها از HFDE به Majority Voting
            if 'HFDE (dynamic)' in efficiency.get('methods', {}):
                efficiency['methods']['Majority Voting'] = efficiency['methods'].pop('HFDE (dynamic)')
            efficiency.get('methods', {}).pop('Router only', None)
            if 'full_test_pass_hfde' in efficiency:
                efficiency['full_test_pass'] = efficiency.pop('full_test_pass_hfde')

            efficiency['method'] = 'Majority Voting'
            efficiency['benchmark_time_s'] = float(bench_time)

            print_headline_majority_voting(efficiency)
        except Exception as e:
            print(f"[WARNING] Efficiency benchmark failed: {e}")
            efficiency = {}
            bench_time = 0.0

    # ================== ارزیابی یکپارچه ensemble ==================
    if is_main:
        print("\n" + "=" * 70)
        print("FINAL ENSEMBLE EVALUATION (Unified - Hard Voting)")
        print("=" * 70)

    ensemble_sw = Stopwatch(device).start()
    final_acc, ensemble_weights = evaluate_ensemble_unified(
        ensemble_module, test_loader, device, "Test",
        MODEL_NAMES, args.save_dir, args, is_main
    )
    ensemble_eval_time = ensemble_sw.stop()
    if is_main:
        print(f"\n⏱ Final ensemble evaluation time: {ensemble_eval_time:.1f}s")

    # ================== FINAL COMPARISON ==================
    if is_main:
        print("\n" + "=" * 70)
        print("FINAL COMPARISON")
        print("=" * 70)
        print(f"Best Single Model: {best_single:.2f}%")
        print(f"Ensemble Accuracy (Hard Voting): {final_acc:.2f}%")
        print(f"Improvement: {final_acc - best_single:+.2f}%")
        print("=" * 70)

        save_ensemble_checkpoint(
            os.path.join(args.save_dir, 'best_ensemble_model.pt'),
            ensemble, args.model_paths, MODEL_NAMES, final_acc, MEANS, STDS
        )

        # ================== TIME SUMMARY ==================
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
            'method': 'Majority Voting (Hard)',
            'dataset': args.dataset,
            'best_single_model': {
                'name': MODEL_NAMES[best_idx],
                'accuracy': float(best_single)
            },
            'ensemble': {
                'test_accuracy': float(final_acc),
                'model_weights': {name: float(w)
                                  for name, w in zip(MODEL_NAMES, ensemble_weights)}
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

        # ================== Efficiency Report ==================
        if efficiency:
            efficiency['individual_model_eval_times_s'] = {
                name: float(t) for name, t in zip(MODEL_NAMES, individual_times)
            }
            efficiency['individual_models_eval_time_s'] = float(sum(individual_times))
            efficiency['ensemble_eval_time_s'] = float(ensemble_eval_time)
            efficiency['total_pipeline_time_s'] = float(total_pipeline_time)

            with open(os.path.join(args.save_dir, 'efficiency_report.json'), 'w') as f:
                json.dump(efficiency, f, indent=2)
            print(f"Efficiency report saved: "
                  f"{os.path.join(args.save_dir, 'efficiency_report.json')}")

    cleanup_distributed()

    if is_main:
        try:
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
