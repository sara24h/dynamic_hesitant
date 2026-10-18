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
from old_dataset_utils import create_dataloaders, get_sample_info
from visualization_utils import generate_visualizations
warnings.filterwarnings("ignore")

# ================== UTILITY ==================
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
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPU(s)")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        print("WARNING: GPU not found. Running on CPU.")
    return device

# ================== FULL METRICS ==================
def compute_full_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())

    prec_real = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec_real  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_real   = 2 * prec_real * rec_real / (prec_real + rec_real) if (prec_real + rec_real) > 0 else 0.0

    prec_fake = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    rec_fake  = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1_fake   = 2 * prec_fake * rec_fake / (prec_fake + rec_fake) if (prec_fake + rec_fake) > 0 else 0.0

    macro_p  = (prec_real + prec_fake) / 2
    macro_r  = (rec_real  + rec_fake)  / 2
    macro_f1 = (f1_real   + f1_fake)   / 2
    bal_acc  = (rec_real  + rec_fake)  / 2
    acc      = (TP + TN) / max(TP + TN + FP + FN, 1)

    return {
        "accuracy": acc * 100,
        "balanced_accuracy": bal_acc * 100,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "precision_real": prec_real,
        "recall_real": rec_real,
        "f1_real": f1_real,
        "precision_fake": prec_fake,
        "recall_fake": rec_fake,
        "f1_fake": f1_fake,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN
    }

# ================== HELPER: unwrap hesitant_fuzzy ==================
def get_hesitant_module(ensemble):
    """اگر hesitant_fuzzy داخل DataParallel است، module را برمی‌گرداند."""
    hf = ensemble.hesitant_fuzzy
    return hf.module if isinstance(hf, nn.DataParallel) else hf

def get_base_ensemble(ensemble):
    """اگر کل ensemble داخل DataParallel است، module را برمی‌گرداند."""
    return ensemble.module if isinstance(ensemble, nn.DataParallel) else ensemble

# ================== UNIFIED FINAL EVALUATION ==================
def final_evaluation_unified(model, test_loader, device, save_dir, model_names, args):
    eval_model = get_base_ensemble(model)
    eval_model.eval()

    base_dataset = test_loader.dataset
    if hasattr(base_dataset, 'dataset'):
        base_dataset = base_dataset.dataset

    if hasattr(test_loader, 'sampler') and hasattr(test_loader.sampler, 'indices'):
        test_indices = test_loader.sampler.indices
    elif hasattr(test_loader.dataset, 'indices'):
        test_indices = test_loader.dataset.indices
    else:
        test_indices = list(range(len(base_dataset)))

    all_y_true, all_y_score, all_y_pred = [], [], []
    lines = []
    lines.append("="*100)
    lines.append("SAMPLE-BY-SAMPLE PREDICTIONS (For McNemar Test Comparison):")
    lines.append("="*100)
    lines.append(f"{'ID':<10} {'Filename':<60} {'True':<10} {'Pred':<10} {'Correct':<10}")
    lines.append("-"*100)

    print(f"\nRunning Unified Final Evaluation on {len(test_indices)} samples...")
    with torch.no_grad():
        for i, global_idx in enumerate(tqdm(test_indices, desc="Final Eval")):
            try:
                image, label = base_dataset[global_idx]
                path, _ = get_sample_info(base_dataset, global_idx)
            except Exception:
                continue

            image = image.unsqueeze(0).to(device)
            label_int = int(label)

            output = eval_model(image)
            if isinstance(output, (tuple, list)):
                output = output[0]

            prob = torch.sigmoid(output.squeeze()).item()
            pred_int = int(prob > 0.5)

            all_y_true.append(label_int)
            all_y_score.append(prob)
            all_y_pred.append(pred_int)

            correct_str = "Yes" if pred_int == label_int else "No"
            filename = os.path.basename(path)
            if len(filename) > 55:
                filename = filename[:25] + "..." + filename[-27:]
            lines.append(f"{i+1:<10} {filename:<60} {label_int:<10} {pred_int:<10} {correct_str:<10}")

    metrics = compute_full_metrics(all_y_true, all_y_pred)

    print(f"\n{'='*70}")
    print("ABLATION: JUST-HESITANCY (2 GPU) — FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy          : {metrics['accuracy']:.2f}%")
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']:.2f}%")
    print(f"Macro-F1          : {metrics['macro_f1']:.4f}")
    print()
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-"*45)
    print(f"{'Fake (0)':<12} {metrics['precision_fake']:10.4f} {metrics['recall_fake']:10.4f} {metrics['f1_fake']:10.4f}")
    print(f"{'Real (1)':<12} {metrics['precision_real']:10.4f} {metrics['recall_real']:10.4f} {metrics['f1_real']:10.4f}")
    print("-"*45)
    print(f"{'Macro Avg':<12} {metrics['macro_precision']:10.4f} {metrics['macro_recall']:10.4f} {metrics['macro_f1']:10.4f}")
    print()
    print(f"Confusion Matrix: TP={metrics['TP']}  TN={metrics['TN']}  FP={metrics['FP']}  FN={metrics['FN']}")
    print("="*70)

    cm = np.array([[metrics['TN'], metrics['FP']],
                   [metrics['FN'], metrics['TP']]])
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Fake', 'Pred Real'],
                yticklabels=['True Fake', 'True Real'])
    plt.title(f"Just-Hesitancy (2GPU)\nAcc={metrics['accuracy']:.2f}% | BalAcc={metrics['balanced_accuracy']:.2f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix_just_hesitancy_2gpu.png"), dpi=300)
    plt.close()
    print("Confusion matrix saved")

    log_path = os.path.join(save_dir, 'prediction_log.txt')
    with open(log_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Prediction log saved to: {log_path}")

    y_true_np = np.array(all_y_true)
    y_score_np = np.array(all_y_score)
    y_pred_np = np.array(all_y_pred)

    roc_json_path = os.path.join(save_dir, "roc_data_test.json")
    with open(roc_json_path, 'w') as f:
        json.dump({
            "metadata": {
                "seed": args.seed,
                "dataset": args.dataset,
                "num_samples": len(all_y_true),
                "model": "fuzzy_hesitant_ensemble_ablation_just_hesitancy_2gpu"
            },
            "y_true": y_true_np.tolist(),
            "y_score": y_score_np.tolist(),
            "y_pred": y_pred_np.tolist()
        }, f, indent=2)
    print(f"ROC data saved (JSON): {roc_json_path}")

    vis_dir = os.path.join(save_dir, 'visualizations')
    try:
        generate_visualizations(
            eval_model, test_loader, device, vis_dir, model_names,
            args.num_grad_cam_samples, args.num_lime_samples,
            args.dataset, is_main=True)
    except Exception as e:
        print(f"[WARNING] Visualization generation failed: {e}")

    return metrics['accuracy'], y_true_np, y_score_np

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
    def __init__(self, num_models: int, num_memberships: int = 3, dropout: float = 0.3):
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
    """
    Ablation: Just-Hesitancy
    - Cumulative Weight Thresholding: REMOVED
    - Hesitancy Override: ACTIVE
    """
    def __init__(self, models, means, stds, num_memberships=3, freeze_models=True,
                 cum_weight_threshold=0.9, hesitancy_threshold=0.2):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.hesitant_fuzzy = HesitantFuzzyMembership(
            num_models=self.num_models, num_memberships=num_memberships)

        self.cum_weight_threshold = cum_weight_threshold   # استفاده نمی‌شود
        self.hesitancy_threshold = hesitancy_threshold

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

    def _compute_mask_vectorized(self, final_weights, avg_hesitancy):
        """
        Just-Hesitancy Logic:
        - No Cumulative Thresholding
        - Hesitancy Override active
        """
        batch_size = final_weights.size(0)
        device = final_weights.device

        # همه مدل‌ها فعال (بدون cumulative)
        mask = torch.ones_like(final_weights)

        # Hesitancy Override
        high_hesitancy = avg_hesitancy > self.hesitancy_threshold

        if high_hesitancy.any():
            top_idx = final_weights.argmax(dim=1)
            override_mask = torch.zeros_like(final_weights)
            override_mask[torch.arange(batch_size, device=device), top_idx] = 1.0
            mask = torch.where(high_hesitancy.unsqueeze(1), override_mask, mask)

        return mask

    def forward(self, x, return_details=False):
        final_weights, all_memberships = self.hesitant_fuzzy(x)

        hesitancy = all_memberships.var(dim=2)
        avg_hesitancy = hesitancy.mean(dim=1)

        mask = self._compute_mask_vectorized(final_weights, avg_hesitancy)
        final_weights = final_weights * mask
        final_weights = final_weights / (final_weights.sum(dim=1, keepdim=True) + 1e-8)

        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
        for i in range(self.num_models):
            x_n = self.normalizations(x, i)
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            outputs[:, i] = out

        final_output = (outputs * final_weights.unsqueeze(-1)).sum(dim=1)

        if return_details:
            return final_output, final_weights, all_memberships, outputs
        return final_output, final_weights

# ================== LOAD & EVAL HELPERS ==================
def load_pruned_models(model_paths, device):
    from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    models = []
    print(f"Loading {len(model_paths)} pruned models...")
    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            print(f" [WARNING] File not found: {path}")
            continue
        print(f" [{i+1}/{len(model_paths)}] Loading: {os.path.basename(path)}")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device).eval()
        models.append(model)
    if len(models) == 0:
        raise ValueError("No models loaded!")
    return models

@torch.no_grad()
def evaluate_single_model(model, loader, device, name, mean, std):
    model.eval()
    normalizer = MultiModelNormalization([mean], [std]).to(device)
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc=f"Evaluating {name}", leave=False):
        images, labels = images.to(device), labels.to(device).float()
        images = normalizer(images, 0)
        out = model(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = (out.squeeze(1) > 0).long()
        correct += pred.eq(labels.long()).sum().item()
        total += images.size(0)
    acc = 100. * correct / max(total, 1)
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
        correct += pred.eq(labels.long()).sum().item()
        total += images.size(0)
    return 100. * correct / max(total, 1)

# ================== TRAINING ==================
def train_hesitant_fuzzy(ensemble_model, train_loader, val_loader, num_epochs, lr,
                         device, save_dir, model_names):
    os.makedirs(save_dir, exist_ok=True)

    # hesitant_fuzzy داخل DataParallel → module را برای optimizer بگیر
    hesitant_net = get_hesitant_module(ensemble_model)

    optimizer = torch.optim.AdamW(hesitant_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    print("="*70)
    print("Training Fuzzy Hesitant Network (ABLATION: Just-Hesitancy | 2 GPU)")
    print("="*70)
    print(f"Trainable params: {sum(p.numel() for p in hesitant_net.parameters()):,}")
    print(f"Epochs: {num_epochs} | LR: {lr}")
    print(f"Using DataParallel on {torch.cuda.device_count()} GPUs\n")

    for epoch in range(num_epochs):
        ensemble_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()

            outputs, _ = ensemble_model(images)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            pred = (outputs.squeeze(1) > 0).long()
            train_correct += pred.eq(labels.long()).sum().item()
            train_total += images.size(0)

        train_acc = 100. * train_correct / max(train_total, 1)
        train_loss = train_loss / max(train_total, 1)
        val_acc = evaluate_accuracy(ensemble_model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'hesitant_state_dict': hesitant_net.state_dict()},
                       os.path.join(save_dir, 'best_hesitant_fuzzy.pt'))
            print(f"  → Best model saved ({val_acc:.2f}%)")

    return best_val_acc

# ================== MAIN ==================
def main():
    parser = argparse.ArgumentParser(description="Ablation: Just-Hesitancy (2 GPU DataParallel)")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./ablation_just_hesitancy_2gpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_memberships', type=int, default=3)
    parser.add_argument('--cum_weight_threshold', type=float, default=0.9)
    parser.add_argument('--hesitancy_threshold', type=float, default=0.2)
    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)
    device = setup_device()

    print("="*70)
    print("ABLATION: JUST-HESITANCY (No Cumulative Thresholding) | 2 GPU")
    print(f"Seed: {args.seed}")
    print(f"Hesitancy Threshold: {args.hesitancy_threshold}")
    print(f"Batch Size: {args.batch_size}")
    print("="*70)

    DEFAULT_MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    DEFAULT_STDS  = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]

    num_models = len(args.model_paths)
    MEANS = DEFAULT_MEANS[:num_models]
    STDS  = DEFAULT_STDS[:num_models]

    base_models = load_pruned_models(args.model_paths, device)
    MODEL_NAMES = args.model_names[:len(base_models)]

    # ---- ساخت ensemble ----
    ensemble = FuzzyHesitantEnsemble(
        base_models, MEANS, STDS,
        num_memberships=args.num_memberships,
        freeze_models=True,
        cum_weight_threshold=args.cum_weight_threshold,
        hesitancy_threshold=args.hesitancy_threshold
    ).to(device)

    # ---- DataParallel فقط روی hesitant_fuzzy (نه کل ensemble) ----
    # دلیل: backboneهای ResNet-50 فریز شده‌اند و replicate کردنشان OOM/کندی می‌آورد
    if torch.cuda.device_count() > 1:
        print(f"\n→ Wrapping hesitant_fuzzy with DataParallel on {torch.cuda.device_count()} GPUs")
        ensemble.hesitant_fuzzy = nn.DataParallel(ensemble.hesitant_fuzzy)
    else:
        print("\n→ Only 1 GPU detected. Running on single GPU.")

    total_trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
    print(f"Trainable params: {total_trainable:,}")

    # ---- DataLoaders ----
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=False, seed=args.seed, is_main=True)

    # ---- ارزیابی تک‌مدل‌ها روی VAL (برای انتخاب best single بدون data leakage) ----
    print("\nINDIVIDUAL MODEL PERFORMANCE (on VAL)")
    individual_val_accs = []
    for i, model in enumerate(base_models):
        acc = evaluate_single_model(model, val_loader, device,
                                    f"Model {i+1} ({MODEL_NAMES[i]})", MEANS[i], STDS[i])
        individual_val_accs.append(acc)

    best_single_val = max(individual_val_accs)
    best_idx = individual_val_accs.index(best_single_val)
    print(f"Best Single (val): Model {best_idx+1} ({MODEL_NAMES[best_idx]}) → {best_single_val:.2f}%\n")

    # ---- آموزش ----
    best_val_acc = train_hesitant_fuzzy(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_dir, MODEL_NAMES)

    # ---- بارگذاری بهترین مدل ----
    ckpt_path = os.path.join(args.save_dir, 'best_hesitant_fuzzy.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        get_hesitant_module(ensemble).load_state_dict(ckpt['hesitant_state_dict'])
        print("Best model loaded.\n")

    # ---- ارزیابی نهایی روی TEST ----
    print("\nFINAL ENSEMBLE EVALUATION (JUST-HESITANCY | 2 GPU)")
    final_acc, y_true, y_scores = final_evaluation_unified(
        ensemble, test_loader, device, args.save_dir, MODEL_NAMES, args)

    # ---- ارزیابی test بهترین تک‌مدل (برای گزارش) ----
    best_single_model = base_models[best_idx]
    best_single_test = evaluate_single_model(
        best_single_model, test_loader, device,
        f"Best single on test ({MODEL_NAMES[best_idx]})",
        MEANS[best_idx], STDS[best_idx])

    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"Best Single (val selection) on test : {best_single_test:.2f}%")
    print(f"Just-Hesitancy Ensemble on test     : {final_acc:.2f}%")
    print(f"Improvement over best single        : {final_acc - best_single_test:+.2f}%")
    print("="*70)

    final_results = {
        'seed': args.seed,
        'method': 'Fuzzy_Hesitant_Just_Hesitancy_2GPU',
        'ablation': 'Cumulative Thresholding Removed (Hesitancy Override Active)',
        'hesitancy_threshold': args.hesitancy_threshold,
        'num_gpus': torch.cuda.device_count(),
        'best_single_model': {
            'name': MODEL_NAMES[best_idx],
            'val_accuracy': float(best_single_val),
            'test_accuracy': float(best_single_test)
        },
        'ensemble': {'test_accuracy': float(final_acc)},
        'improvement_over_best_single': float(final_acc - best_single_test)
    }
    with open(os.path.join(args.save_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)

    # ---- ذخیره مدل نهایی (base ensemble، بدون DataParallel) ----
    base = get_base_ensemble(ensemble)
    state_to_save = {
        'ensemble_state_dict': base.state_dict(),
        'model_names': MODEL_NAMES,
        'means': MEANS,
        'stds': STDS,
        'seed': args.seed,
        'hesitancy_threshold': args.hesitancy_threshold
    }
    torch.save(state_to_save, os.path.join(args.save_dir, 'final_ensemble_model.pt'))
    print(f"\nModel saved to: {os.path.join(args.save_dir, 'final_ensemble_model.pt')}")


if __name__ == "__main__":
    main()
