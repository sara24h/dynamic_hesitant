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
    
    if len(models) == 0: raise ValueError("No models loaded!")
    print(f"All {len(models)} models loaded!\n")
    return models


# ================== EVALUATION FUNCTIONS (Single GPU) ==================
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
        if isinstance(out, (tuple, list)): out = out[0]
        pred = (out.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()

    acc = 100. * correct / total if total > 0 else 0.0
    print(f" {name}: {acc:.2f}%")
    return acc


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs, _ = model(images)
        pred = (outputs.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()
        
    acc = 100. * correct / total if total > 0 else 0.0
    return acc


@torch.no_grad()
def evaluate_ensemble_final(model, loader, device, name, model_names):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    all_y_true = []
    all_y_score = []
    all_y_pred = []
    
    # اصلاح نکته 2: وزن‌ها یک پارامتر Global هستند و به ورودی وابسته نیستند.
    # بنابراین فقط یک‌بار قبل از حلقه محاسبه می‌شوند (جلوگیری از گمراهی و سربار محاسباتی)
    learned_weights = F.softmax(model.weights, dim=0).detach().cpu()

    print(f"\nEvaluating {name} set (Weighted Average)...")
        
    for images, labels in tqdm(loader, desc=f"Evaluating {name}"):
        images, labels = images.to(device), labels.to(device)
        
        # چون وزن‌ها را از قبل داریم، return_details=False کافی است
        outputs, _ = model(images, return_details=False)
            
        probs = torch.sigmoid(outputs.squeeze(1))
        pred_int = (probs > 0.5).long()
        
        all_y_true.extend(labels.cpu().numpy().tolist())
        all_y_score.extend(probs.cpu().numpy().tolist())
        all_y_pred.extend(pred_int.cpu().numpy().tolist())
        
        total_correct += pred_int.eq(labels.long()).sum().item()
        total_samples += labels.size(0)

    acc = 100. * total_correct / total_samples if total_samples > 0 else 0.0
    final_weights = learned_weights.numpy()

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


def train_weighted_ensemble(ensemble_model, train_loader, val_loader, num_epochs, lr,
                        device, save_dir, model_names):
    os.makedirs(save_dir, exist_ok=True)
    
    weights_param = ensemble_model.weights
        
    optimizer = torch.optim.AdamW([weights_param], lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'model_weights': []}

    print("="*70)
    print("Training Weighted Average Ensemble")
    print("="*70)

    for epoch in range(num_epochs):
        # اصلاح باگ BatchNorm: خط ensemble_model.train() کاملاً حذف شد.
        # چون فقط یک nn.Parameter خام trainable داریم، نیازی به تنظیم مد نداریم
        # و مدل‌های فریز شده در حالت eval() می‌مانند تا BatchNorm آن‌ها درست کار کند.
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
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

        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / train_total
        
        current_weights = F.softmax(weights_param, dim=0).detach().cpu()
        val_acc = evaluate_accuracy(ensemble_model, val_loader, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['model_weights'].append(current_weights.numpy().tolist())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, 'best_weighted_ensemble.pt')
            torch.save({
                'epoch': epoch + 1,
                'weights': weights_param.data.cpu(),
                'val_acc': val_acc,
                'history': history
            }, save_path)
            print(f"\n✓ Best model saved → {val_acc:.2f}%")

    return best_val_acc, history


# ================== HELPER: SAVE ACCURACY LOG FROM RESULTS ==================
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
    lines.append(f"                 Predicted Real  Predicted Fake")
    lines.append(f"    Actual Real   {TP:<15} {FN:<15}")
    lines.append(f"    Actual Fake   {FP:<15} {TN:<15}")
    
    lines.append("\nClassification Metrics:")
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    lines.append(f"Precision: {precision:.4f}")
    lines.append(f"Recall: {recall:.4f}")
    lines.append(f"Specificity: {specificity:.4f}")
    
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
    parser = argparse.ArgumentParser(description="Weighted Average Ensemble Training - Single GPU")
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
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("="*70)
    print(f"WEIGHTED AVERAGE ENSEMBLE TRAINING (Single GPU)")
    print(f"Seed: {args.seed}")
    print("="*70)

    MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    
    if len(args.model_paths) > len(MEANS):
        MEANS = MEANS * (len(args.model_paths) // len(MEANS) + 1)
        STDS = STDS * (len(args.model_paths) // len(STDS) + 1)
        
    MEANS = MEANS[:len(args.model_paths)]
    STDS = STDS[:len(args.model_paths)]

    base_models = load_pruned_models(args.model_paths, device)
    MODEL_NAMES = args.model_names[:len(base_models)]

    ensemble = WeightedAverageEnsemble(
        base_models, MEANS, STDS,
        freeze_models=True
    ).to(device)

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=False, seed=args.seed, is_main=True)

    print("\n" + "="*70); print("INDIVIDUAL MODEL PERFORMANCE"); print("="*70)

    individual_accs = []
    for i, model in enumerate(base_models):
        acc = evaluate_single_model(
            model, test_loader, device,
            f"Model {i+1} ({MODEL_NAMES[i]})",
            MEANS[i], STDS[i])
        individual_accs.append(acc)

    best_single = max(individual_accs)
    best_idx = individual_accs.index(best_single)

    best_val_acc, history = train_weighted_ensemble(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_dir, MODEL_NAMES)

    ckpt_path = os.path.join(args.save_dir, 'best_weighted_ensemble.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ensemble.weights.data.copy_(ckpt['weights'])
        print("Best model weights loaded.\n")

    ensemble_test_acc, ensemble_weights, predictions_data = evaluate_ensemble_final(
        ensemble, test_loader, device, "Test", MODEL_NAMES)

    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"Best Single Model: {best_single:.2f}%")
    print(f"Ensemble Accuracy: {ensemble_test_acc:.2f}%")
    print(f"Improvement: {ensemble_test_acc - best_single:+.2f}%")
    print("="*70)

    y_true, y_score, y_pred = predictions_data
    
    json_path = os.path.join(args.save_dir, 'test_predictions.json')
    save_predictions_json(y_true, y_score, y_pred, json_path, args.seed, args.dataset)

    log_path = os.path.join(args.save_dir, 'prediction_log.txt')
    save_accuracy_log_from_results(y_true, y_pred, log_path, "Weighted Ensemble")

    final_results = {
        'seed': args.seed,
        'method': 'Weighted_Average',
        'best_single_model': {'name': MODEL_NAMES[best_idx], 'accuracy': float(best_single)},
        'ensemble': {'test_accuracy': float(ensemble_test_acc), 'model_weights': {name: float(w) for name, w in zip(MODEL_NAMES, ensemble_weights)}},
        'improvement': float(ensemble_test_acc - best_single),
        'training_history': history
    }

    results_path = os.path.join(args.save_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    
    final_model_path = os.path.join(args.save_dir, 'final_ensemble_model.pt')
    torch.save({
        'ensemble_state_dict': ensemble.state_dict(),
        'test_accuracy': ensemble_test_acc,
        'model_names': MODEL_NAMES,
        'means': MEANS,
        'stds': STDS,
        'seed': args.seed
    }, final_model_path)

    vis_dir = os.path.join(args.save_dir, 'visualizations')
    generate_visualizations(
        ensemble, test_loader, device, vis_dir, MODEL_NAMES,
        args.num_grad_cam_samples, args.num_lime_samples,
        args.dataset, is_main=True)

    plot_roc_and_f1(
        ensemble,
        test_loader, 
        device, 
        args.save_dir, 
        MODEL_NAMES,
        is_main=True
    )

if __name__ == "__main__":
    main()
