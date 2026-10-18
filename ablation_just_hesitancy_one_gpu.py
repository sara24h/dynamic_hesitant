
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
        device = torch.device('cuda:0')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("WARNING: GPU not found. Running on CPU.")
    return device


# ================== METRICS ==================
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
 
    def __init__(self, models, means, stds, num_memberships=3, freeze_models=True,
                 hesitancy_threshold=0.2):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.hesitant_fuzzy = HesitantFuzzyMembership(
            num_models=self.num_models, num_memberships=num_memberships)
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
        Just-Hesitancy: always activate all models.
        (Cumulative removed → no selection based on weight sum)
        """
        return torch.ones_like(final_weights)

    def forward(self, x, return_details=False):
        final_weights, all_memberships = self.hesitant_fuzzy(x)

        hesitancy = all_memberships.var(dim=2, unbiased=False)
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


# ================== LOAD & EVAL ==================
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


@torch.no_grad()
def final_evaluation(model, test_loader, device, save_dir, model_names, args):
    model.eval()
    all_y_true, all_y_score, all_y_pred = [], [], []

    for images, labels in tqdm(test_loader, desc="Final Eval"):
        images = images.to(device)
        labels = labels.to(device).long()
        outputs, _ = model(images)
        probs = torch.sigmoid(outputs.squeeze(1))
        preds = (probs > 0.5).long()

        all_y_true.extend(labels.cpu().numpy().tolist())
        all_y_score.extend(probs.cpu().numpy().tolist())
        all_y_pred.extend(preds.cpu().numpy().tolist())

    metrics = compute_full_metrics(all_y_true, all_y_pred)

    print("\n" + "="*70)
    print("ABLATION: JUST-HESITANCY — FINAL RESULTS")
    print("="*70)
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

    # Confusion matrix
    cm = np.array([[metrics['TN'], metrics['FP']],
                   [metrics['FN'], metrics['TP']]])
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Fake', 'Pred Real'],
                yticklabels=['True Fake', 'True Real'])
    plt.title(f"Just-Hesitancy\nAcc={metrics['accuracy']:.2f}% | BalAcc={metrics['balanced_accuracy']:.2f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix_just_hesitancy.png"), dpi=300)
    plt.close()

    # Save results
    results = {
        "method": "Fuzzy_Hesitant_Just_Hesitancy",
        "seed": args.seed,
        "hesitancy_threshold": args.hesitancy_threshold,
        **{k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics.items()}
    }
    with open(os.path.join(save_dir, "final_results_just_hesitancy.json"), "w") as f:
        json.dump(results, f, indent=2)

    np.savez(os.path.join(save_dir, "roc_data_just_hesitancy.npz"),
             y_true=np.array(all_y_true),
             y_score=np.array(all_y_score),
             y_pred=np.array(all_y_pred))

    # Visualizations
    vis_dir = os.path.join(save_dir, 'visualizations')
    try:
        generate_visualizations(
            model, test_loader, device, vis_dir, model_names,
            args.num_grad_cam_samples, args.num_lime_samples,
            args.dataset, is_main=True)
    except Exception as e:
        print(f"[WARNING] Visualization failed: {e}")

    return metrics['accuracy']


# ================== TRAINING ==================
def train_hesitant_fuzzy(ensemble_model, train_loader, val_loader, num_epochs, lr,
                         device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    hesitant_net = ensemble_model.hesitant_fuzzy
    optimizer = torch.optim.AdamW(hesitant_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    print("="*70)
    print("Training (ABLATION: Just-Hesitancy | Single GPU)")
    print("="*70)
    print(f"Trainable params: {sum(p.numel() for p in hesitant_net.parameters()):,}")
    print(f"Epochs: {num_epochs} | LR: {lr}\n")

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
                       os.path.join(save_dir, 'best_just_hesitancy.pt'))
            print(f"  → Best model saved ({val_acc:.2f}%)")

    return best_val_acc


# ================== MAIN ==================
def main():
    parser = argparse.ArgumentParser(description="Ablation: Just-Hesitancy (Single GPU)")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./ablation_just_hesitancy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_memberships', type=int, default=3)
    parser.add_argument('--hesitancy_threshold', type=float, default=0.2)
    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)
    device = setup_device()

    print("="*70)
    print("ABLATION: JUST-HESITANCY (No Cumulative Thresholding) | Single GPU")
    print(f"Seed: {args.seed} | Hesitancy Threshold: {args.hesitancy_threshold}")
    print("="*70)

    DEFAULT_MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    DEFAULT_STDS  = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]

    num_models = len(args.model_paths)
    MEANS = DEFAULT_MEANS[:num_models]
    STDS  = DEFAULT_STDS[:num_models]

    base_models = load_pruned_models(args.model_paths, device)
    MODEL_NAMES = args.model_names[:len(base_models)]

    ensemble = FuzzyHesitantEnsemble(
        base_models, MEANS, STDS,
        num_memberships=args.num_memberships,
        freeze_models=True,
        hesitancy_threshold=args.hesitancy_threshold
    ).to(device)

    print(f"Trainable params: {sum(p.numel() for p in ensemble.parameters() if p.requires_grad):,}")

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=False, seed=args.seed, is_main=True)

    # Individual models on VAL
    print("\nINDIVIDUAL MODEL PERFORMANCE (on VAL)")
    individual_val_accs = []
    for i, model in enumerate(base_models):
        acc = evaluate_single_model(model, val_loader, device,
                                    f"Model {i+1} ({MODEL_NAMES[i]})", MEANS[i], STDS[i])
        individual_val_accs.append(acc)
    best_single_val = max(individual_val_accs)
    best_idx = individual_val_accs.index(best_single_val)
    print(f"Best Single (val): Model {best_idx+1} ({MODEL_NAMES[best_idx]}) → {best_single_val:.2f}%\n")

    # Train
    train_hesitant_fuzzy(ensemble, train_loader, val_loader,
                         args.epochs, args.lr, device, args.save_dir)

    # Load best
    ckpt_path = os.path.join(args.save_dir, 'best_just_hesitancy.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        ensemble.hesitant_fuzzy.load_state_dict(ckpt['hesitant_state_dict'])
        print("Best model loaded.\n")

    # Final eval on TEST
    print("\nFINAL ENSEMBLE EVALUATION (JUST-HESITANCY)")
    final_acc = final_evaluation(ensemble, test_loader, device, args.save_dir, MODEL_NAMES, args)

    # Best single on TEST
    best_single_test = evaluate_single_model(
        base_models[best_idx], test_loader, device,
        f"Best single on test ({MODEL_NAMES[best_idx]})",
        MEANS[best_idx], STDS[best_idx])

    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"Best Single (val selection) on test : {best_single_test:.2f}%")
    print(f"Just-Hesitancy Ensemble on test     : {final_acc:.2f}%")
    print(f"Improvement over best single        : {final_acc - best_single_test:+.2f}%")
    print("="*70)

    # Save summary
    final_results = {
        'seed': args.seed,
        'method': 'Fuzzy_Hesitant_Just_Hesitancy',
        'ablation': 'Cumulative Thresholding Removed (always all models)',
        'hesitancy_threshold': args.hesitancy_threshold,
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

    torch.save({
        'ensemble_state_dict': ensemble.state_dict(),
        'model_names': MODEL_NAMES,
        'means': MEANS,
        'stds': STDS,
        'seed': args.seed,
        'hesitancy_threshold': args.hesitancy_threshold
    }, os.path.join(args.save_dir, 'final_ensemble_model.pt'))

    print(f"\nModel saved to: {os.path.join(args.save_dir, 'final_ensemble_model.pt')}")


if __name__ == "__main__":
    main()
