import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
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


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# ================== METRICS ==================
def compute_full_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())

    # Real (positive = 1)
    prec_real = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec_real  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_real   = 2 * prec_real * rec_real / (prec_real + rec_real) if (prec_real + rec_real) > 0 else 0.0

    # Fake (positive = 0)
    prec_fake = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    rec_fake  = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1_fake   = 2 * prec_fake * rec_fake / (prec_fake + rec_fake) if (prec_fake + rec_fake) > 0 else 0.0

    macro_p  = (prec_real + prec_fake) / 2
    macro_r  = (rec_real  + rec_fake)  / 2
    macro_f1 = (f1_real   + f1_fake)   / 2
    bal_acc  = (rec_real  + rec_fake)  / 2
    acc      = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

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
    """Ablation: No Hesitancy Override (only cumulative thresholding)"""
    def __init__(self, models, means, stds, num_memberships=3, freeze_models=True,
                 cum_weight_threshold=0.9):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.hesitant_fuzzy = HesitantFuzzyMembership(
            num_models=self.num_models, num_memberships=num_memberships)
        self.cum_weight_threshold = cum_weight_threshold

        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # base models always stay in eval mode
        for model in self.models:
            model.eval()
        return self

    def _compute_mask_vectorized(self, final_weights: torch.Tensor):
        """Only Cumulative Weight Thresholding (no hesitancy override)"""
        sorted_weights, sorted_indices = torch.sort(final_weights, dim=1, descending=True)
        cum_weights = torch.cumsum(sorted_weights, dim=1)

        # keep models while cumulative sum <= T; always keep the top-1
        mask = (cum_weights <= self.cum_weight_threshold).float()
        mask[:, 0] = 1.0

        final_mask = torch.zeros_like(final_weights)
        final_mask.scatter_(1, sorted_indices, mask)
        return final_mask

    def forward(self, x: torch.Tensor, return_details: bool = False):
        final_weights, all_memberships = self.hesitant_fuzzy(x)

        # ---- Ablation: only cumulative thresholding ----
        mask = self._compute_mask_vectorized(final_weights)

        final_weights = final_weights * mask
        final_weights = final_weights / (final_weights.sum(dim=1, keepdim=True) + 1e-8)

        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)

        # only forward the active models (saves FLOPs)
        active = torch.any(final_weights > 1e-4, dim=0).nonzero(as_tuple=True)[0]
        for i in active:
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


# ================== LOAD MODELS ==================
def load_pruned_models(model_paths, device):
    from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    models = []
    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            if is_main_process():
                print(f"[WARNING] File not found: {path}")
            continue
        if is_main_process():
            print(f"[{i+1}/{len(model_paths)}] Loading: {os.path.basename(path)}")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device).eval()
        models.append(model)
    if len(models) == 0:
        raise ValueError("No models loaded!")
    return models


# ================== EVALUATION ==================
@torch.no_grad()
def evaluate_full(model, loader, device, save_dir, model_names, args):
    model.eval()
    all_y_true, all_y_score, all_y_pred = [], [], []

    for images, labels in tqdm(loader, desc="Final Eval", disable=not is_main_process()):
        images = images.to(device)
        labels = labels.to(device).long()

        outputs, _ = model(images)
        probs = torch.sigmoid(outputs.squeeze(1))
        preds = (probs > 0.5).long()

        all_y_true.extend(labels.cpu().numpy().tolist())
        all_y_score.extend(probs.cpu().numpy().tolist())
        all_y_pred.extend(preds.cpu().numpy().tolist())

    if is_main_process():
        metrics = compute_full_metrics(all_y_true, all_y_pred)

        print("\n" + "="*70)
        print("ABLATION: NO HESITANCY OVERRIDE — FINAL RESULTS")
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

        # Confusion matrix plot
        cm = np.array([[metrics['TN'], metrics['FP']],
                       [metrics['FN'], metrics['TP']]])
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred Fake', 'Pred Real'],
                    yticklabels=['True Fake', 'True Real'])
        plt.title(f"No-Override Ablation\nAcc={metrics['accuracy']:.2f}% | BalAcc={metrics['balanced_accuracy']:.2f}%")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "confusion_matrix_no_override.png"), dpi=300)
        plt.close()

        # Save results
        results = {
            "method": "Fuzzy_Hesitant_No_Override",
            "seed": args.seed,
            **{k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics.items()}
        }
        with open(os.path.join(save_dir, "final_results_no_override.json"), "w") as f:
            json.dump(results, f, indent=2)

        np.savez(os.path.join(save_dir, "roc_data_no_override.npz"),
                 y_true=np.array(all_y_true),
                 y_score=np.array(all_y_score),
                 y_pred=np.array(all_y_pred))

        return metrics['accuracy']
    return 0.0


# ================== TRAINING ==================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=f"Epoch {epoch}", disable=not is_main_process()):
        images, labels = images.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs.squeeze(1), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        pred = (outputs.squeeze(1) > 0).long()
        correct += pred.eq(labels.long()).sum().item()
        total += images.size(0)

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs, _ = model(images)
        pred = (outputs.squeeze(1) > 0).long()
        correct += pred.eq(labels.long()).sum().item()
        total += images.size(0)
    return 100. * correct / total


# ================== MAIN ==================
def main():
    parser = argparse.ArgumentParser(description="Ablation: No Hesitancy Override (DDP)")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./ablation_no_override')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_memberships', type=int, default=3)
    parser.add_argument('--cum_weight_threshold', type=float, default=0.80)
    args = parser.parse_args()

    # ---------- DDP setup ----------
    local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    set_seed(args.seed)   # same seed for reproducibility

    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)
        print("="*70)
        print("ABLATION: NO HESITANCY OVERRIDE  |  Multi-GPU DDP")
        print(f"Seed: {args.seed} | cum_weight_threshold: {args.cum_weight_threshold}")
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
        cum_weight_threshold=args.cum_weight_threshold
    ).to(device)

    # فقط قسمت trainable را DDP می‌کنیم
    ensemble.hesitant_fuzzy = DDP(
        ensemble.hesitant_fuzzy,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )

    # DataLoaders (distributed)
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=True, seed=args.seed, is_main=is_main_process()
    )

    # Optimizer فقط روی پارامترهای trainable
    optimizer = torch.optim.AdamW(
        ensemble.hesitant_fuzzy.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            ensemble, train_loader, optimizer, criterion, device, epoch)
        val_acc = validate(ensemble, val_loader, device)
        scheduler.step()

        if is_main_process():
            print(f"Epoch {epoch:3d} | Train Loss {train_loss:.4f} | "
                  f"Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}%")

            if val_acc > best_val:
                best_val = val_acc
                # ذخیره state_dict بدون DDP wrapper
                torch.save({
                    'hesitant_state_dict': ensemble.hesitant_fuzzy.module.state_dict()
                }, os.path.join(args.save_dir, 'best_no_override.pt'))
                print(f"  → Best model saved ({val_acc:.2f}%)")

    # ================================================================
    # ---------- Final evaluation (Robust for DDP) ----------
    # ================================================================

    # 1. همه processها اینجا همگام می‌شوند
    if dist.is_initialized():
        dist.barrier()

    # 2. DDP wrapper را باز می‌کنیم
    if hasattr(ensemble.hesitant_fuzzy, 'module'):
        ensemble.hesitant_fuzzy = ensemble.hesitant_fuzzy.module

    # 3. فقط rank 0 ارزیابی نهایی را انجام می‌دهد
    if is_main_process():
        ckpt_path = os.path.join(args.save_dir, 'best_no_override.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            ensemble.hesitant_fuzzy.load_state_dict(ckpt['hesitant_state_dict'])
            print("Best model loaded for final evaluation.")
        else:
            print("[WARNING] No checkpoint found. Using last model weights.")

        # ساخت loader کاملاً مستقل و غیر-distributed از کل دیتاست تست
        base_dataset = test_loader.dataset
        # اگر Subset یا wrapper بود، به dataset اصلی برویم
        while hasattr(base_dataset, 'dataset'):
            base_dataset = base_dataset.dataset

        eval_loader = DataLoader(
            base_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        print(f"Evaluating on full test set: {len(base_dataset)} samples")
        final_acc = evaluate_full(
            ensemble, eval_loader, device,
            args.save_dir, MODEL_NAMES, args
        )
        print(f"\nFinal No-Override Accuracy: {final_acc:.2f}%")

    # 4. همه processها دوباره همگام می‌شوند قبل از cleanup
    if dist.is_initialized():
        dist.barrier()

    cleanup_ddp()


if __name__ == "__main__":
    main()
