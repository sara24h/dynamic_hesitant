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

from metrics_utils import plot_roc_and_f1

# =================== اصلاح ایمپورت برای پشتیبانی از AdaBN ===================
try:
    from dataset_utils_p100 import create_dataloaders, create_adabn_dataloader
except ImportError:
    from dataset_utils_p100 import create_dataloaders
    create_adabn_dataloader = None
    print("[Warning] 'create_adabn_dataloader' not found in dataset_utils. AdaBN will be skipped.")
# ============================================================================

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


class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


# =====================================================================
# ================== BATCHNORM ADAPTATION (AdaBN) ====================
# =====================================================================
# تابع دقیقاً مشابه کد Fuzzy
@torch.no_grad()
def adapt_batchnorm_for_new_dataset(models, means, stds, adabn_loader, device, is_main=True):
    """
    اعمال AdaBN با دیتالودر بدون augmentation.
    فقط آمار BN آپدیت می‌شود، نه وزن‌ها.
    """
    if is_main:
        print("\n" + "="*70)
        print("STARTING BATCHNORM ADAPTATION (AdaBN) - CLEAN DATA (NO AUGMENTATION)")
        print("="*70)
        
    normalizer = MultiModelNormalization(means, stds).to(device)
    
    for model in models:
        model.eval()  # قطع شدن Dropout و لایه‌های غیرقطعی
        # ریست کردن آمار قبلی و تنظیم برای محاسبه آمار تجمعی (Cumulative)
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.reset_running_stats()
                m.momentum = None  # محاسبه میانگین/واریانس گلبال
                m.train()          # فقط BN در حالت train باشد
                
    for images, _ in tqdm(adabn_loader, desc="Adapting BN", disable=not is_main):
        images = images.to(device)
        for i, model in enumerate(models):
            x_n = normalizer(images, i)
            model(x_n)  # فقط یک forward pass برای آپدیت آمار BN
            
    # برگرداندن momentum به حالت پیش‌فرض و مدل به حالت eval
    for model in models:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.momentum = 0.1
        model.eval()
        
    if is_main:
        print("✅ BatchNorm Adaptation Completed!\n")
# =====================================================================


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

    def forward(self, x, return_details=False):
        norm_weights = F.softmax(self.weights, dim=0)
        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
    
        for i in range(self.num_models):
            x_n = self.normalizations(x, i)
            out = self.models[i](x_n)
            if isinstance(out, (tuple, list)):
                out = out[0]
            outputs[:, i] = out.view(x.size(0), 1)

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
    return 100. * correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_ensemble_final(model, loader, device, name, model_names):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    all_y_true = []
    all_y_score = []
    all_y_pred = []
    last_weights = None

    print(f"\nEvaluating {name} set (Weighted Average)...")
        
    for images, labels in tqdm(loader, desc=f"Evaluating {name}"):
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


# ================== TRAINING FUNCTION ==================
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
    print("Training Weighted Average Ensemble (Single GPU)")
    print("="*70)

    for epoch in range(num_epochs):
        ensemble_model.train()
        for model in ensemble_model.models:
            model.eval() 

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
            
            probs = torch.sigmoid(outputs.squeeze(1))
            pred = (probs > 0.5).long()
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


# ================== SAVE LOGS ==================
def save_accuracy_log_from_results(y_true, y_pred, save_path, model_name):
    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    acc = 100.0 * correct / total if total > 0 else 0.0
    
    TP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    TN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    FP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    FN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    lines = [
        "="*70, f"SUMMARY STATISTICS (Exact Console Match)", f"Model: {model_name}", "="*70,
        f"Accuracy: {acc:.3f}%", f"Total Samples: {total}", f"Correct Predictions: {correct}", f"Incorrect Predictions: {total - correct}",
        "\nConfusion Matrix:", "                 Predicted Real  Predicted Fake",
        f"    Actual Real   {TP:<15} {FN:<15}", f"    Actual Fake   {FP:<15} {TN:<15}",
        "\nClassification Metrics:",
        f"Precision: {TP / (TP + FP) if (TP + FP) > 0 else 0:.4f}",
        f"Recall: {TP / (TP + FN) if (TP + FN) > 0 else 0:.4f}",
        f"Specificity: {TN / (TN + FP) if (TN + FP) > 0 else 0:.4f}",
        "-" * 70, "SAMPLE-BY-SAMPLE PREDICTIONS (For McNemar Test):", f"{'ID':<10} {'True':<10} {'Pred':<10} {'Status':<10}", "-" * 70
    ]
    
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        lines.append(f"{i+1:<10} {yt:<10} {yp:<10} {'Correct' if yt == yp else 'Wrong':<10}")

    with open(save_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"✅ Accuracy log saved to: {save_path}")

def save_predictions_json(y_true, y_score, y_pred, save_path, seed, dataset_name):
    output_data = {
        "metadata": {"seed": seed, "dataset": dataset_name, "num_samples": len(y_true), "positive_count": sum(y_true), "negative_count": len(y_true) - sum(y_true)},
        "y_true": y_true, "y_score": y_score, "y_pred": y_pred
    }
    with open(save_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"✅ Predictions JSON saved to: {save_path}")


# ================== MAIN FUNCTION ==================
def main():
    parser = argparse.ArgumentParser(description="Weighted Average Ensemble - Single GPU (Non-Wild Datasets)")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)
    
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['deepfake_lab', 'hard_fake_real', 'real_fake_dataset', 'uadfV', 
                                'custom_genai', 'custom_genai_v2', 'real_fake', 'deepflux'])
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
    print(f"Using device: {device}")

    print("="*70)
    print(f"WEIGHTED AVERAGE ENSEMBLE TRAINING (Single GPU)")
    print(f"Dataset: {args.dataset} | Seed: {args.seed}")
    print("="*70)

    MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    
    if len(args.model_paths) != len(MEANS) or len(args.model_paths) != len(STDS):
        raise ValueError(
            f"Number of models ({len(args.model_paths)}) must exactly match "
            f"number of MEANS ({len(MEANS)}) and STDS ({len(STDS)}). "
            f"Please provide exact normalization stats for all models."
        )

    base_models = load_pruned_models(args.model_paths, device)
    MODEL_NAMES = args.model_names[:len(base_models)]

    # =================== فراخوانی دقیق تابع AdaBN ===================
    if create_adabn_dataloader is not None:
        adabn_loader = create_adabn_dataloader(
            args.data_dir, args.batch_size, num_workers=args.num_workers,
            dataset_type=args.dataset, seed=args.seed, is_main=True
        )
        adapt_batchnorm_for_new_dataset(base_models, MEANS, STDS, adabn_loader, device, is_main=True)
        
        del adabn_loader
        torch.cuda.empty_cache()
    else:
        print("\n[INFO] Skipping AdaBN because 'create_adabn_dataloader' is missing in dataset_utils.py")
    # ===============================================================

    # ساخت انسامبل (بدون DDP)
    ensemble = WeightedAverageEnsemble(
        base_models, MEANS, STDS, freeze_models=True
    ).to(device)

    # ساخت دیتالودر (حالت غیر توزیع شده)
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, num_workers=args.num_workers,
        dataset_type=args.dataset, is_distributed=False, seed=args.seed, is_main=True
    )

    print("\n" + "="*70); print("INDIVIDUAL MODEL PERFORMANCE (After AdaBN)"); print("="*70)
    individual_accs = []
    for i, model in enumerate(base_models):
        acc = evaluate_single_model(
            model, test_loader, device,
            f"Model {i+1} ({MODEL_NAMES[i]})", MEANS[i], STDS[i])
        individual_accs.append(acc)

    best_single = max(individual_accs)
    best_idx = individual_accs.index(best_single)

    # آموزش وزن‌ها
    best_val_acc, history = train_weighted_ensemble(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_dir, MODEL_NAMES)

    # لود کردن بهترین وزن‌ها
    ckpt_path = os.path.join(args.save_dir, 'best_weighted_ensemble.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ensemble.weights.data.copy_(ckpt['weights'])
        print("Best model weights loaded.\n")

    # ارزیابی نهایی
    ensemble_test_acc, ensemble_weights, predictions_data = evaluate_ensemble_final(
        ensemble, test_loader, device, "Test", MODEL_NAMES)

    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"Best Single Model: {best_single:.2f}%")
    print(f"Ensemble Accuracy: {ensemble_test_acc:.2f}%")
    print(f"Improvement: {ensemble_test_acc - best_single:+.2f}%")
    print("="*70)

    # استخراج لیست‌ها
    y_true, y_score, y_pred = predictions_data
    
    # ذخیره خروجی‌ها
    json_path = os.path.join(args.save_dir, 'test_predictions.json')
    save_predictions_json(y_true, y_score, y_pred, json_path, args.seed, args.dataset)

    log_path = os.path.join(args.save_dir, 'prediction_log.txt')
    save_accuracy_log_from_results(y_true, y_pred, log_path, "Weighted Ensemble")

    final_results = {
        'seed': args.seed, 'method': 'Weighted_Average_AdaBN',
        'best_single_model': {'name': MODEL_NAMES[best_idx], 'accuracy': float(best_single)},
        'ensemble': {'test_accuracy': float(ensemble_test_acc), 'model_weights': {name: float(w) for name, w in zip(MODEL_NAMES, ensemble_weights)}},
        'improvement': float(ensemble_test_acc - best_single), 'training_history': history
    }
    results_path = os.path.join(args.save_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    
    final_model_path = os.path.join(args.save_dir, 'final_ensemble_model.pt')
    torch.save({
        'ensemble_state_dict': ensemble.state_dict(),
        'test_accuracy': ensemble_test_acc,
        'model_names': MODEL_NAMES, 'means': MEANS, 'stds': STDS, 'seed': args.seed
    }, final_model_path)

    # تولید ویژوال‌های GradCAM و LIME
    vis_dir = os.path.join(args.save_dir, 'visualizations')
    generate_visualizations(
        ensemble, test_loader, device, vis_dir, MODEL_NAMES,
        args.num_grad_cam_samples, args.num_lime_samples, args.dataset, is_main=True
    )

    # رسم نمودارهای ROC و F1
    plot_roc_and_f1(ensemble, test_loader, device, args.save_dir, MODEL_NAMES, is_main=True)

if __name__ == "__main__":
    main()
