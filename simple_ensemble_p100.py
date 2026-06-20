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
from metrics_utils import plot_roc_and_f1

warnings.filterwarnings("ignore")

from dataset_utils_p100 import (
    UADFVDataset, 
    create_dataloaders, 
    get_sample_info, 
    worker_init_fn
)

from visualization_utils import GradCAM, generate_lime_explanation, generate_visualizations

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

from torchvision import transforms  # ← بالای فایل

@torch.no_grad()
def final_evaluation_and_report(model, loader, device, save_dir, model_name, args, is_main):
    if not is_main or loader is None:
        return 0.0, None, None

    model.eval()
    all_y_true, all_y_score, all_y_pred = [], [], []
    TP, TN, FP, FN = 0, 0, 0, 0

    for images, labels in tqdm(loader, desc="Final Eval"):
        images = images.to(device)
        labels_int = labels.long()

        outputs, _, _, _ = model(images, return_details=True)
        probs = torch.sigmoid(outputs.squeeze(1)).cpu()
        preds = (probs > 0.5).long()

        all_y_true.extend(labels_int.tolist())
        all_y_score.extend(probs.tolist())
        all_y_pred.extend(preds.tolist())

        for yt, yp in zip(labels_int.tolist(), preds.tolist()):
            if yt == 1:
                if yp == 1: TP += 1
                else:       FN += 1
            else:
                if yp == 1: FP += 1
                else:       TN += 1

    total = TP + TN + FP + FN
    correct_count = TP + TN
    acc   = correct_count / total if total > 0 else 0
    prec  = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec   = TP / (TP + FN) if (TP + FN) > 0 else 0
    spec  = TN / (TN + FP) if (TN + FP) > 0 else 0

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy:    {acc*100:.2f}%")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted Real  Predicted Fake")
    print(f"    Actual Real      {TP:<15} {FN:<15}")
    print(f"    Actual Fake      {FP:<15} {TN:<15}")
    print(f"\nCorrect:   {correct_count} ({acc*100:.2f}%)")
    print(f"Incorrect: {total - correct_count} ({(1-acc)*100:.2f}%)")
    print("="*70)

    os.makedirs(save_dir, exist_ok=True)

    # ذخیره log
    output_str = [
        "-"*100, "SUMMARY STATISTICS:", "-"*100,
        f"Accuracy: {acc*100:.2f}%",
        f"Precision: {prec:.4f}",
        f"Recall: {rec:.4f}",
        f"Specificity: {spec:.4f}",
        "\nConfusion Matrix:",
        f"                 {'Predicted Real':<15} {'Predicted Fake':<15}",
        f"    Actual Real   {TP:<15} {FN:<15}",
        f"    Actual Fake   {FP:<15} {TN:<15}",
        f"\nCorrect Predictions: {correct_count} ({acc*100:.2f}%)",
        f"Incorrect Predictions: {total - correct_count} ({(1-acc)*100:.2f}%)",
        "", "-"*100, "SAMPLE-BY-SAMPLE PREDICTIONS:",
        f"{'ID':<10} {'True':<10} {'Pred':<10} {'Status':<10}", "-"*100,
    ]
    for i, (yt, yp) in enumerate(zip(all_y_true, all_y_pred)):
        output_str.append(f"{i+1:<10} {yt:<10} {yp:<10} {'Correct' if yt==yp else 'Wrong':<10}")

    log_path = os.path.join(save_dir, 'prediction_log.txt')
    with open(log_path, 'w') as f:
        f.write("\n".join(output_str))
    print(f"✅ Prediction log saved: {log_path}")

    # ذخیره JSON و TXT
    y_true_np  = np.array(all_y_true)
    y_score_np = np.array(all_y_score)
    y_pred_np  = np.array(all_y_pred)

    roc_json_path = os.path.join(save_dir, "roc_data_test.json")
    with open(roc_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "seed": args.seed, "dataset": args.dataset,
                "num_samples": total,
                "positive_count": int(np.sum(y_true_np)),
                "negative_count": int(total - np.sum(y_true_np)),
                "model": model_name
            },
            "y_true": y_true_np.tolist(),
            "y_score": y_score_np.tolist(),
            "y_pred": y_pred_np.tolist()
        }, f, indent=2, ensure_ascii=False)
    print(f"✅ ROC JSON saved: {roc_json_path}")

    roc_txt_path = os.path.join(save_dir, "roc_data_test.txt")
    with open(roc_txt_path, 'w', encoding='utf-8') as f:
        f.write("y_true\ty_score\ty_pred\n")
        for t, s, p in zip(y_true_np, y_score_np, y_pred_np):
            f.write(f"{int(t)}\t{s:.6f}\t{int(p)}\n")
    print(f"✅ ROC TXT saved: {roc_txt_path}")

    return acc * 100, y_true_np, y_score_np


class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


class SimpleAveragingEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]],
                 stds: List[Tuple[float]]):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)

    def forward(self, x: torch.Tensor, return_details: bool = False):
        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
        
        for i in range(self.num_models):
            x_n = self.normalizations(x, i)
            out = self.models[i](x_n)
            if isinstance(out, (tuple, list)):
                out = out[0]
            outputs[:, i] = out.view(x.size(0), 1)

        final_output = outputs.mean(dim=1)
        
        if return_details:
            weights = torch.ones(x.size(0), self.num_models, device=x.device) / self.num_models
            return final_output, weights, None, outputs
        return final_output, None


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
            
            param_count = sum(p.numel() for p in model.parameters())
            if is_main:
                print(f" → Parameters: {param_count:,}")
            models.append(model)
        except Exception as e:
            if is_main:
                print(f" [ERROR] Failed to load {path}: {e}")
            continue

    if len(models) == 0:
        raise ValueError("No models loaded!")
    if is_main:
        print(f"All {len(models)} models loaded!\n")
    return models


# ================== EVALUATION FUNCTIONS ==================
@torch.no_grad()
def evaluate_single_model(model: nn.Module, loader: DataLoader, device: torch.device,
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
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = (out.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()

    acc = 100. * correct / total
    if is_main:
        print(f" {name}: {acc:.2f}%")
    return acc


# ================== MAIN FUNCTION ==================
def main():
    parser = argparse.ArgumentParser(description="Simple Averaging Ensemble (Single GPU)")
    parser.add_argument('--epochs', type=int, default=1, help="Unused in Simple Averaging")
    parser.add_argument('--lr', type=float, default=0.0001, help="Unused in Simple Averaging")
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_main = True  # تک پردازشی - همیشه main

    print("="*70)
    print(f"SIMPLE AVERAGING ENSEMBLE (Single GPU)")
    print(f"Device: {device} | Seed: {args.seed}")
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

    base_models = load_pruned_models(args.model_paths, device, is_main)
    MODEL_NAMES = args.model_names[:len(base_models)]

    ensemble = SimpleAveragingEnsemble(
        base_models, MEANS, STDS
    ).to(device)

    trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
    total = sum(p.numel() for p in ensemble.parameters())
    print(f"Total params: {total:,} | Trainable: {trainable:,}\n")

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=False, seed=args.seed, is_main=is_main)

    print("\n" + "="*70)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("="*70)

    individual_accs = []
    for i, model in enumerate(base_models):
        acc = evaluate_single_model(
            model, test_loader, device,
            f"Model {i+1} ({MODEL_NAMES[i]})",
            MEANS[i], STDS[i], is_main)
        individual_accs.append(acc)

    best_single = max(individual_accs)
    best_idx = individual_accs.index(best_single)

    print(f"\nBest Single: Model {best_idx+1} ({MODEL_NAMES[best_idx]}) → {best_single:.2f}%")
    print("="*70)
    print("\nSkipping Training (Simple Averaging does not learn parameters)...\n")

    print("\n" + "="*70)
    print("FINAL ENSEMBLE EVALUATION")
    print("="*70)

    ensemble_module = ensemble  # بدون DDP، wrapper وجود ندارد

    os.makedirs(args.save_dir, exist_ok=True)

    # تک GPU است، همان test_loader قابل استفاده است (غیرتوزیع‌شده)
    test_loader_full = test_loader

    # اجرای ارزیابی یکپارچه
    ensemble_test_acc, y_true, y_score = final_evaluation_and_report(
        ensemble_module, test_loader_full, device, args.save_dir, 
        "Simple Averaging Ensemble", args, is_main
    )

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
        'ensemble_state_dict': ensemble_module.state_dict(),
        'test_accuracy': ensemble_test_acc,
        'model_names': MODEL_NAMES,
        'means': MEANS,
        'stds': STDS
    }, final_model_path)
    print(f"Model saved: {final_model_path}")

    vis_dir = os.path.join(args.save_dir, 'visualizations')
    generate_visualizations(
        ensemble_module, test_loader_full, device, vis_dir, MODEL_NAMES,
        args.num_grad_cam_samples, args.num_lime_samples,
        args.dataset, is_main)

    plot_roc_and_f1(
        ensemble_module,
        test_loader_full, 
        device, 
        args.save_dir, 
        MODEL_NAMES,
        is_main
    )

if __name__ == "__main__":
    main()
