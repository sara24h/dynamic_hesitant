import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
import warnings
import argparse
import json
import random
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
# from metrics_utils import plot_roc_and_f1

warnings.filterwarnings("ignore")

from old_dataset_utils import (
    UADFVDataset,
    create_dataloaders,
    get_sample_info,
    worker_init_fn
)

from visualization_utils import GradCAM, generate_lime_explanation, generate_visualizations

# ================== EFFICIENCY BENCHMARK ==================
from efficiency_utils import (
    Stopwatch,
    benchmark_efficiency,
    print_headline,
    summarize_training_times,
)

# ================== UTILITY FUNCTIONS ==================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')

# ================== WEIGHTED AVERAGING ENSEMBLE ==================
class WeightedAverageEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]],
                 stds: List[Tuple[float]], freeze_models: bool = True):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)

        initial_weights = torch.ones(self.num_models) / self.num_models
        self.weights = nn.Parameter(initial_weights)

        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False


    def hesitant_fuzzy(self, x: torch.Tensor):
 
        return torch.ones(x.size(0), self.num_models, device=x.device) / self.num_models

    def forward(self, x: torch.Tensor, return_details: bool = False):
        norm_weights = F.softmax(self.weights, dim=0)
        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)

        for i in range(self.num_models):
            x_n = self.normalizations(x, i)
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            outputs[:, i] = out

        final_output = (outputs * norm_weights.view(1, -1, 1)).sum(dim=1)

        if return_details:
            return final_output, norm_weights, None, outputs
        return final_output, norm_weights

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
            if is_main: print(f" [WARNING] File not found: {path}")
            continue
        if is_main: print(f" [{i+1}/{len(model_paths)}] Loading: {os.path.basename(path)}")
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device).eval()
            models.append(model)
        except Exception as e:
            if is_main: print(f" [ERROR] Failed to load {path}: {e}")

    if len(models) == 0: raise ValueError("No models loaded!")
    if is_main: print(f"All {len(models)} models loaded!\n")
    return models


# ================== EVALUATION FUNCTIONS ==================
@torch.no_grad()
def evaluate_single_model_ddp(model: nn.Module, loader: DataLoader, device: torch.device,
                              name: str, mean: Tuple[float, float, float],
                              std: Tuple[float, float, float], is_main: bool) -> float:
    model.eval()
    normalizer = MultiModelNormalization([mean], [std]).to(device)
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc=f"Evaluating {name}", disable=not is_main):
        images, labels = images.to(device), labels.to(device).float()
        images = normalizer(images, 0)
        out = model(images)
        if isinstance(out, (tuple, list)): out = out[0]
        pred = (out.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()

    correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
    total_tensor = torch.tensor(total, dtype=torch.long, device=device)
    if dist.is_initialized():
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    acc = 100. * correct_tensor.item() / total_tensor.item() if total_tensor.item() > 0 else 0.0
    if is_main: print(f" {name}: {acc:.2f}%")
    return acc


@torch.no_grad()
def evaluate_accuracy_ddp(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs, _ = model(images)
        pred = (outputs.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()

    correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
    total_tensor = torch.tensor(total, dtype=torch.long, device=device)
    if dist.is_initialized():
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    acc = 100. * correct_tensor.item() / total_tensor.item() if total_tensor.item() > 0 else 0.0
    return acc


@torch.no_grad()
def evaluate_ensemble_final_ddp(model, loader, device, name, model_names, is_main=True):
    model.eval()
    total_correct = 0
    total_samples = 0

    ws = dist.get_world_size() if dist.is_initialized() else 1

    all_y_true = []
    all_y_score = []
    all_y_pred = []

    last_weights = None

    if is_main:
        print(f"\nEvaluating {name} set (Weighted Average)...")

    for images, labels in tqdm(loader, desc=f"Evaluating {name}", disable=not is_main):
        images, labels = images.to(device), labels.to(device)
        outputs, weights, _, _ = model(images, return_details=True)

        if last_weights is None:
            last_weights = weights.detach().cpu()

        probs = torch.sigmoid(outputs.squeeze(1))
        pred_int = (probs > 0.5).long()

        all_y_true.extend(labels.cpu().numpy().tolist())
        all_y_score.extend(probs.cpu().numpy().tolist())
        all_y_pred.extend(pred_int.cpu().numpy().tolist())

        total_correct += pred_int.eq(labels.long()).sum().item()
        total_samples += labels.size(0)

    stats = torch.tensor([total_correct, total_samples], dtype=torch.long, device=device)
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

    if is_main:
        total_correct = stats[0].item()
        total_samples = stats[1].item()
        acc = 100. * total_correct / total_samples

        final_weights = last_weights.numpy() if last_weights is not None else np.zeros(len(model_names))

        print(f"\n{'='*70}")
        print(f"{name.upper()} SET RESULTS (Weighted Average)")
        print(f"{'='*70}")
        print(f" → Accuracy: {acc:.3f}%")
        print(f" → Total Samples: {total_samples:,}")
        print(f"\nLearned Model Weights (Global):")
        for i, (w, mname) in enumerate(zip(final_weights, model_names)):
            print(f" {i+1:2d}. {mname:<25}: {w:6.4f} ({w*100:5.2f}%)")
        print(f"{'='*70}")
        return acc, final_weights.tolist(), (all_y_true, all_y_score, all_y_pred)

    return 0.0, [0.0]*len(model_names), ([], [], [])


def train_weighted_ensemble(ensemble_model, train_loader, val_loader, num_epochs, lr,
                        device, save_dir, is_main, model_names):
    os.makedirs(save_dir, exist_ok=True)

    weights_param = ensemble_model.module.weights if hasattr(ensemble_model, 'module') else ensemble_model.weights

    optimizer = torch.optim.AdamW([weights_param], lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'model_weights': []}

    # ---------- Training time tracking ----------
    epoch_times, train_times, val_times = [], [], []
    total_sw = Stopwatch(device).start()
    # -------------------------------------------

    if is_main:
        print("="*70)
        print("Training Weighted Average Ensemble")
        print("="*70)
        print(f"Trainable params: {sum(p.numel() for p in [weights_param]):,}")
        print(f"Epochs: {num_epochs} | Initial LR: {lr}")
        print(f"Number of models: {len(model_names)}")
        print(f"Total training samples: {len(train_loader.dataset):,}\n")

    for epoch in range(num_epochs):
        epoch_sw = Stopwatch(device).start()

        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        ensemble_model.train()

        actual_module = ensemble_model.module if hasattr(ensemble_model, 'module') else ensemble_model
        for model in actual_module.models:
            model.eval()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', disable=not is_main):
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()

            outputs, _, _, _ = ensemble_model(images, return_details=True)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            pred = (outputs.squeeze(1) > 0).long()
            train_correct += pred.eq(labels.long()).sum().item()
            train_total += batch_size

        t_train = epoch_sw.stop()

        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / train_total

        current_weights = F.softmax(weights_param, dim=0).detach().cpu()

        val_sw = Stopwatch(device).start()
        val_acc = evaluate_accuracy_ddp(ensemble_model, val_loader, device)
        t_val = val_sw.stop()

        t_epoch = epoch_sw.stop()
        epoch_times.append(t_epoch)
        train_times.append(t_train)
        val_times.append(t_val)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['model_weights'].append(current_weights.numpy().tolist())

        if is_main:
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*70}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"Time: epoch {t_epoch:.1f}s (train {t_train:.1f}s | val {t_val:.1f}s)")
            print(f"Current weights: {[f'{w:.4f}' for w in current_weights.numpy()]}")

        if is_main and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, 'best_weighted_ensemble.pt')
            torch.save({
                'epoch': epoch + 1,
                'weights': weights_param.data.cpu(),
                'val_acc': val_acc,
                'history': history
            }, save_path)
            print(f"\n✓ Best model saved → {val_acc:.2f}%")

    # ---------- Summarize training time ----------
    total_time = total_sw.stop()
    timing = summarize_training_times(epoch_times, train_times, val_times, total_time)

    if is_main:
        print(f"\n{'='*70}")
        print("TRAINING TIME SUMMARY")
        print(f"{'='*70}")
        print(f"Total training time       : {timing['total_training_time_s']:.1f}s "
              f"({timing['total_training_time_min']:.2f} min)")
        print(f"Mean epoch time           : {timing['mean_epoch_time_s']:.1f}s")
        print(f"Mean epoch (excl. first)  : {timing['mean_epoch_time_excl_first_s']:.1f}s  "
              f"[CUDA/cuDNN init excluded]")
        print(f"Mean train phase          : {timing['mean_train_phase_s']:.1f}s")
        print(f"Mean val phase            : {timing['mean_val_phase_s']:.1f}s")
        print(f"{'='*70}\n")

        # ---------- Training curves (Loss & Accuracy) ----------
        try:
            plt.figure(figsize=(14, 5))
            epochs_range = range(1, num_epochs + 1)

            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, history['train_loss'], label='train', color='blue', linewidth=2)
            plt.title('Loss', fontsize=16)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)

            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, history['train_acc'], label='train', color='blue', linewidth=2)
            plt.plot(epochs_range, history['val_acc'], label='validation', color='orange', linewidth=2)
            plt.title('Accuracy', fontsize=16)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)

            plt.tight_layout()
            curves_path = os.path.join(save_dir, 'training_curves.png')
            plt.savefig(curves_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Training curves saved to: {curves_path}")
        except Exception as e:
            print(f"[WARNING] Could not save training curves: {e}")

    return best_val_acc, history, timing


# ================== DISTRIBUTED SETUP ==================
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        return device, local_rank, rank, world_size
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return device, 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ================== REAL/FAKE METRICS ==================
def compute_real_fake_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    real_precision, real_recall, real_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average=None, zero_division=0
    )
    real_precision, real_recall, real_f1 = real_precision[0], real_recall[0], real_f1[0]

    fake_precision, fake_recall, fake_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0], average=None, zero_division=0
    )
    fake_precision, fake_recall, fake_f1 = fake_precision[0], fake_recall[0], fake_f1[0]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # Fake=0, Real=1
    TN, FP, FN, TP = cm.ravel()

    real_specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    fake_specificity = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    metrics = {
        'real': {
            'precision': float(real_precision),
            'recall': float(real_recall),
            'f1': float(real_f1),
            'specificity': float(real_specificity),
            'support': int(np.sum(y_true == 1))
        },
        'fake': {
            'precision': float(fake_precision),
            'recall': float(fake_recall),
            'f1': float(fake_f1),
            'specificity': float(fake_specificity),
            'support': int(np.sum(y_true == 0))
        },
        'macro_f1': float((real_f1 + fake_f1) / 2),
        'confusion_matrix': {
            'TN': int(TN), 'FP': int(FP), 'FN': int(FN), 'TP': int(TP)
        }
    }
    return metrics


def print_real_fake_metrics(metrics, title="REAL/FAKE METRICS"):
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Specificity':<12} {'Support':<10}")
    print(f"{'-'*70}")
    for cls in ['real', 'fake']:
        m = metrics[cls]
        print(f"{cls.upper():<10} {m['precision']:<12.4f} {m['recall']:<12.4f} "
              f"{m['f1']:<12.4f} {m['specificity']:<12.4f} {m['support']:<10}")
    print(f"{'-'*70}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Confusion Matrix: TN={metrics['confusion_matrix']['TN']}, "
          f"FP={metrics['confusion_matrix']['FP']}, "
          f"FN={metrics['confusion_matrix']['FN']}, "
          f"TP={metrics['confusion_matrix']['TP']}")
    print(f"{'='*70}")


# ================== SAVE LOG / JSON HELPERS ==================
def save_accuracy_log_from_results(y_true, y_pred, save_path, model_name):
    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    acc = 100.0 * correct / total if total > 0 else 0.0

    TP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    TN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    FP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    FN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    lines = []
    lines.append("="*70)
    lines.append(f"SUMMARY STATISTICS (Exact Console Match)")
    lines.append(f"Model: {model_name}")
    lines.append("="*70)
    lines.append(f"Accuracy: {acc:.3f}%")
    lines.append(f"Total Samples: {total}")
    lines.append(f"Correct Predictions: {correct}")
    lines.append(f"Incorrect Predictions: {total - correct}")

    lines.append("\nConfusion Matrix:")
    lines.append(f"                 Predicted Fake  Predicted Real")
    lines.append(f"    Actual Fake   {TN:<15} {FP:<15}")
    lines.append(f"    Actual Real   {FN:<15} {TP:<15}")

    lines.append("\nClassification Metrics:")
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    lines.append(f"Precision: {precision:.4f}")
    lines.append(f"Recall: {recall:.4f}")
    lines.append(f"Specificity: {specificity:.4f}")

    rf = compute_real_fake_metrics(y_true, y_pred)
    lines.append("\n" + "="*70)
    lines.append("REAL/FAKE METRICS")
    lines.append("="*70)
    lines.append(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Specificity':<12}")
    lines.append("-"*70)
    for cls in ['real', 'fake']:
        m = rf[cls]
        lines.append(f"{cls.upper():<10} {m['precision']:<12.4f} {m['recall']:<12.4f} "
                     f"{m['f1']:<12.4f} {m['specificity']:<12.4f}")
    lines.append("-"*70)
    lines.append(f"Macro F1: {rf['macro_f1']:.4f}")

    lines.append("-" * 70)
    lines.append("SAMPLE-BY-SAMPLE PREDICTIONS (For McNemar Test):")
    lines.append(f"{'ID':<10} {'True':<10} {'Pred':<10} {'Status':<10}")
    lines.append("-" * 70)

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        status = "Correct" if yt == yp else "Wrong"
        lines.append(f"{i+1:<10} {yt:<10} {yp:<10} {status:<10}")

    with open(save_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"✅ Accuracy log (Console-Accurate) saved to: {save_path}")


def save_predictions_json(y_true, y_score, y_pred, save_path, seed, dataset_name):
    pos_count = sum(y_true)
    neg_count = len(y_true) - pos_count

    output_data = {
        "metadata": {
            "seed": seed,
            "dataset": dataset_name,
            "num_samples": len(y_true),
            "positive_count": pos_count,
            "negative_count": neg_count,
        },
        "y_true": y_true,
        "y_score": y_score,
        "y_pred": y_pred
    }

    with open(save_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"✅ Predictions JSON saved to: {save_path}")


# ================== MAIN FUNCTION ==================
def main():
    parser = argparse.ArgumentParser(description="Weighted Average Ensemble Training")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['wild', 'deepfake_lab', 'hard_fake_real', 'real_fake_dataset', 'uadfV'])
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

    # ============ PIPELINE STOPWATCH ============
    pipeline_sw = Stopwatch(device).start()

    if is_main:
        print("="*70)
        print(f"WEIGHTED AVERAGE ENSEMBLE TRAINING")
        print(f"Distributed on {world_size} GPU(s) | Seed: {args.seed}")
        print("="*70)

    MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]

    if len(args.model_paths) > len(MEANS):
        MEANS = MEANS * (len(args.model_paths) // len(MEANS) + 1)
        STDS = STDS * (len(args.model_paths) // len(STDS) + 1)

    MEANS = MEANS[:len(args.model_paths)]
    STDS = STDS[:len(args.model_paths)]

    base_models = load_pruned_models(args.model_paths, device, is_main)
    MODEL_NAMES = args.model_names[:len(base_models)]

    ensemble = WeightedAverageEnsemble(
        base_models, MEANS, STDS,
        freeze_models=True
    ).to(device)

    if world_size > 1:
        ensemble = DDP(ensemble, device_ids=[local_rank], output_device=local_rank)

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=(world_size > 1), seed=args.seed, is_main=is_main)

    if is_main:
        print("\n" + "="*70); print("INDIVIDUAL MODEL PERFORMANCE"); print("="*70)

    individual_accs = []
    individual_times = []
    for i, model in enumerate(base_models):
        sw = Stopwatch(device).start()
        acc = evaluate_single_model_ddp(
            model, test_loader, device,
            f"Model {i+1} ({MODEL_NAMES[i]})",
            MEANS[i], STDS[i], is_main)
        t = sw.stop()
        individual_times.append(t)
        if is_main:
            print(f" ⏱ Evaluation time: {t:.1f}s\n")
        individual_accs.append(acc)

    best_single = max(individual_accs)
    best_idx = individual_accs.index(best_single)

    # ---------- TRAINING (with timing) ----------
    best_val_acc, history, train_timing = train_weighted_ensemble(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_dir, is_main, MODEL_NAMES)

    ckpt_path = os.path.join(args.save_dir, 'best_weighted_ensemble.pt')
    if is_main and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if hasattr(ensemble, 'module'): ensemble.module.weights.data.copy_(ckpt['weights'])
        else: ensemble.weights.data.copy_(ckpt['weights'])
        print("Best model weights loaded.\n")

    ensemble_module = ensemble.module if hasattr(ensemble, 'module') else ensemble

    # ============ EFFICIENCY BENCHMARK ============
    os.makedirs(args.save_dir, exist_ok=True)

    efficiency = {}
    bench_time = 0.0
    if is_main:
        try:
            bench_sw = Stopwatch(device).start()
            efficiency = benchmark_efficiency(
                ensemble_module, test_loader, device, MODEL_NAMES,
                latency_samples=len(test_loader.dataset),
            )
            bench_time = bench_sw.stop()
            print_headline(efficiency)
        except Exception as e:
            print(f"[WARNING] Efficiency benchmark failed: {e}")
            efficiency = {}
            bench_time = 0.0

    # ============ FINAL EVALUATION ============
    eval_sw = Stopwatch(device).start()
    ensemble_test_acc, ensemble_weights, predictions_data = evaluate_ensemble_final_ddp(
        ensemble_module, test_loader, device, "Test", MODEL_NAMES, is_main)
    final_eval_time = eval_sw.stop()

    if is_main:
        print(f"\n⏱ Final evaluation time: {final_eval_time:.1f}s")
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        print(f"Best Single Model: {best_single:.2f}%")
        print(f"Ensemble Accuracy: {ensemble_test_acc:.2f}%")
        print(f"Improvement: {ensemble_test_acc - best_single:+.2f}%")
        print("="*70)

        y_true, y_score, y_pred = predictions_data

        # REAL/FAKE METRICS
        rf_metrics = compute_real_fake_metrics(y_true, y_pred)
        print_real_fake_metrics(rf_metrics, title="WEIGHTED ENSEMBLE — REAL/FAKE METRICS")

        # Save JSON + log
        json_path = os.path.join(args.save_dir, 'test_predictions.json')
        save_predictions_json(y_true, y_score, y_pred, json_path, args.seed, args.dataset)

        log_path = os.path.join(args.save_dir, 'prediction_log.txt')
        save_accuracy_log_from_results(y_true, y_pred, log_path, "Weighted Ensemble")

        # ============ TIME SUMMARY ============
        total_pipeline_time = pipeline_sw.stop()

        print("\n" + "="*70)
        print("TIME SUMMARY")
        print("="*70)
        print(f"  Training time                : {train_timing['total_training_time_s']:.1f}s "
              f"({train_timing['total_training_time_min']:.2f} min, "
              f"mean epoch {train_timing['mean_epoch_time_s']:.1f}s)")
        for i, name in enumerate(MODEL_NAMES):
            print(f"  Model {i+1} ({name}) evaluation : {individual_times[i]:.1f}s")
        print(f"  Efficiency benchmark          : {bench_time:.1f}s")
        print(f"  Final ensemble evaluation     : {final_eval_time:.1f}s")
        print(f"  Total pipeline time           : {total_pipeline_time/60:.2f} min ({total_pipeline_time:.1f}s)")
        print("="*70)

        # ============ SAVE FINAL RESULTS ============
        final_results = {
            'seed': args.seed,
            'method': 'Weighted_Average',
            'best_single_model': {'name': MODEL_NAMES[best_idx], 'accuracy': float(best_single)},
            'ensemble': {
                'test_accuracy': float(ensemble_test_acc),
                'model_weights': {name: float(w) for name, w in zip(MODEL_NAMES, ensemble_weights)}
            },
            'improvement': float(ensemble_test_acc - best_single),
            'training_history': history,
            'real_fake_metrics': rf_metrics,
            'performance_metrics': efficiency,
            'timing': {
                'training_time': train_timing,
                'individual_model_eval_times_s': {
                    name: float(t) for name, t in zip(MODEL_NAMES, individual_times)
                },
                'benchmark_time_s': float(bench_time),
                'final_evaluation_time_s': float(final_eval_time),
                'total_pipeline_time_s': float(total_pipeline_time),
            }
        }

        results_path = os.path.join(args.save_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4)

        final_model_path = os.path.join(args.save_dir, 'final_ensemble_model.pt')
        torch.save({
            'ensemble_state_dict': ensemble_module.state_dict(),
            'test_accuracy': ensemble_test_acc,
            'model_names': MODEL_NAMES,
            'means': MEANS,
            'stds': STDS,
            'seed': args.seed
        }, final_model_path)

        # Save efficiency report separately
        if efficiency:
            efficiency['training_time'] = train_timing
            efficiency['individual_model_eval_times_s'] = {
                name: float(t) for name, t in zip(MODEL_NAMES, individual_times)
            }
            efficiency['final_evaluation_time_s'] = float(final_eval_time)
            efficiency['total_pipeline_time_s'] = float(total_pipeline_time)
            eff_path = os.path.join(args.save_dir, 'efficiency_report.json')
            with open(eff_path, 'w') as f:
                json.dump(efficiency, f, indent=2)
            print(f"Efficiency report saved: {eff_path}")

        # ============ VISUALIZATIONS ============
        vis_dir = os.path.join(args.save_dir, 'visualizations')
        generate_visualizations(
            ensemble_module, test_loader, device, vis_dir, MODEL_NAMES,
            args.num_grad_cam_samples, args.num_lime_samples,
            args.dataset, is_main)

    cleanup_distributed()

if __name__ == "__main__":
    main()
