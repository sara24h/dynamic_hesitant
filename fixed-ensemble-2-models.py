"""
matched_budget.py

Matched-Budget Fixed Ensemble برای پاسخ به داور:
«آیا سود HFDE از routing هوشمند است یا از تعداد مدل‌های فعال؟»

دو حالت:
  1) --probs  : هر مدل با احتمال معین در هر نمونه فعال می‌شود (Bernoulli)
                E[budget] = sum(probs)  ← باید برابر mean_active روش دینامیک باشد
  2) --active_indices + --fixed_weights : ensemble ثابت کلاسیک (برای مقایسه)

نکته حیاتی: probs را از VALIDATION بگیر، نه test.
  از best_hesitant_fuzzy.pt → ckpt['val_routing_stats']['per_model_active_pct'] / 100
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

from old_dataset_utils import create_dataloaders


# ============================================================
# Normalization
# ============================================================
class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


# ============================================================
# Model loading
# ============================================================
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
    if len(models) == 0:
        raise ValueError("No models loaded!")
    return models


# ============================================================
# Fixed Top-K Ensemble (deterministic subset)
# ============================================================
class FixedTopKEnsemble(nn.Module):
    """
    Ensemble ثابت که همیشه فقط k مدل مشخص‌شده را فعال می‌کند.
    (برای مقایسه: Fixed-{2,3}، Fixed-{1,2,3}، ...)
    """
    def __init__(self, models, means, stds, active_indices,
                 fixed_weights=None, device="cuda"):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.active_indices = active_indices
        self.normalizations = MultiModelNormalization(means, stds)

        full_weights = torch.zeros(self.num_models, dtype=torch.float32)
        if fixed_weights is None:
            w = 1.0 / len(active_indices)
            for idx in active_indices:
                full_weights[idx] = w
        else:
            assert len(fixed_weights) == len(active_indices)
            weights = torch.tensor(fixed_weights, dtype=torch.float32)
            weights = weights / weights.sum()
            for i, idx in enumerate(active_indices):
                full_weights[idx] = weights[i]

        self.register_buffer("fixed_weights", full_weights)

        for model in self.models:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

        print(f"Active models indices : {self.active_indices}")
        print(f"Final fixed weights   : {self.fixed_weights.tolist()}")

    def forward(self, x):
        B = x.size(0)
        outputs = torch.zeros(B, self.num_models, device=x.device)
        for i in self.active_indices:
            x_n = self.normalizations(x, i)
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            outputs[:, i] = out.squeeze(1)
        return (outputs * self.fixed_weights.unsqueeze(0)).sum(dim=1)


# ============================================================
# Matched-Budget Fixed Ensemble (probabilistic, per-sample)
# ============================================================
class MatchedBudgetFixedEnsemble(nn.Module):
    """
    هر مدل با احتمال معین در هر نمونه به‌طور مستقل فعال می‌شود.
        E[budget] = sum(probs)   ← با mean_active روش دینامیک برابر شود

    probs باید از VALIDATION استخراج شوند (نه test) — بدون leakage.
    stochastic است → با چند seed اجرا شود و mean±std گزارش شود.
    """
    def __init__(self, models, means, stds, probs,
                 fixed_weights=None, seed=42):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        assert len(probs) == self.num_models

        self.probs = torch.tensor(probs, dtype=torch.float32)
        self.rng = np.random.default_rng(seed)
        self.normalizations = MultiModelNormalization(means, stds)

        if fixed_weights is not None:
            w = torch.tensor(fixed_weights, dtype=torch.float32)
            self.fixed_weights = w / w.sum()
        else:
            self.fixed_weights = None

        for m in self.models:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

        print(f"Matched-Budget mode | probs = {self.probs.tolist()}")
        print(f"  E[budget] = {float(self.probs.sum().item()):.4f}")
        if self.fixed_weights is not None:
            print(f"  Weights (normalized) = {self.fixed_weights.tolist()}")
        else:
            print(f"  Weights: uniform over active subset")
        print(f"  Seed    = {seed}")

    @torch.no_grad()
    def forward(self, x):
        B = x.size(0)
        u = torch.from_numpy(self.rng.random((B, self.num_models))).float().to(x.device)
        active = (u < self.probs.unsqueeze(0)).float()

        # fallback: حداقل یک مدل فعال
        none_active = active.sum(1) == 0
        if none_active.any():
            fallback = int(self.probs.argmax().item())
            active[none_active, fallback] = 1.0

        outputs = torch.zeros(B, self.num_models, 1, device=x.device)
        for i in range(self.num_models):
            idx = active[:, i] > 0
            if idx.any():
                x_n = self.normalizations(x[idx], i)
                o = self.models[i](x_n)
                if isinstance(o, (tuple, list)):
                    o = o[0]
                outputs[idx, i] = o

        if self.fixed_weights is None:
            w = active / active.sum(1, keepdim=True)
        else:
            w = active * self.fixed_weights.unsqueeze(0)
            w = w / w.sum(1, keepdim=True)

        return (outputs * w.unsqueeze(-1)).sum(1)

    def expected_budget(self) -> float:
        return float(self.probs.sum().item())


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_fixed(ensemble, test_loader, device, save_dir, model_names, method_name,
                   extra_meta=None):
    os.makedirs(save_dir, exist_ok=True)
    ensemble.eval()

    all_y_true, all_y_score, all_y_pred = [], [], []
    TP = TN = FP = FN = 0

    print(f"\nEvaluating {method_name} on {len(test_loader.dataset)} samples...")
    for images, labels in tqdm(test_loader, desc=method_name):
        images = images.to(device)
        labels = labels.to(device).long()
        logits = ensemble(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        all_y_true.extend(labels.cpu().numpy().tolist())
        all_y_score.extend(probs.cpu().numpy().tolist())
        all_y_pred.extend(preds.cpu().numpy().tolist())

        for t, p in zip(labels, preds):
            if t == 1:
                if p == 1: TP += 1
                else: FN += 1
            else:
                if p == 1: FP += 1
                else: TN += 1

    total = len(all_y_true)
    acc = 100.0 * (TP + TN) / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results = {
        "method": method_name,
        "model_names": model_names,
        "accuracy": float(acc),
        "precision_real": float(precision),
        "recall_real": float(recall),
        "f1_real": float(f1),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "num_samples": total,
    }
    if hasattr(ensemble, "active_indices"):
        results["active_indices"] = ensemble.active_indices
        results["fixed_weights"] = ensemble.fixed_weights.cpu().tolist()
    if hasattr(ensemble, "probs"):
        results["probs"] = ensemble.probs.tolist()
        results["expected_budget"] = float(ensemble.probs.sum().item())
    if extra_meta:
        results.update(extra_meta)

    print("\n" + "="*70)
    print(f"{method_name} RESULTS")
    print("="*70)
    print(f"Accuracy       : {acc:.2f}%")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-Score       : {f1:.4f}")
    print("="*70)

    with open(os.path.join(save_dir, f"{method_name.lower().replace(' ', '_')}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    np.savez(
        os.path.join(save_dir, f"roc_data_{method_name.lower().replace(' ', '_')}.npz"),
        y_true=np.array(all_y_true),
        y_score=np.array(all_y_score),
        y_pred=np.array(all_y_pred),
    )

    print(f"Results saved to: {save_dir}")
    return results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Matched-Budget / Fixed ensemble evaluation (rebuttal)")

    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./ablation')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)

    # Mode 1: Fixed top-k
    parser.add_argument('--active_indices', type=int, nargs='+', default=None,
                        help="Indices of always-active models (e.g., 1 2 → M2+M3).")
    parser.add_argument('--fixed_weights', type=float, nargs='+', default=None,
                        help="Weights for active_indices. If None → uniform.")

    # Mode 2: Matched-Budget (probabilistic)
    parser.add_argument('--probs', type=float, nargs='+', default=None,
                        help="Per-model activation probabilities (length = num_models). "
                             "Extract from VALIDATION: ckpt['val_routing_stats']['per_model_active_pct']/100")
    parser.add_argument('--num_seeds', type=int, default=3,
                        help="Number of seeds to average over in matched mode.")

    args = parser.parse_args()

    if args.probs is None and args.active_indices is None:
        raise ValueError("Specify either --probs (matched) or --active_indices (fixed).")
    if args.probs is not None and args.active_indices is not None:
        raise ValueError("Use --probs OR --active_indices, not both.")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    DEFAULT_MEANS = [(0.5207, 0.4258, 0.3806),
                     (0.4460, 0.3622, 0.3416),
                     (0.4668, 0.3816, 0.3414)]
    DEFAULT_STDS  = [(0.2490, 0.2239, 0.2212),
                     (0.2057, 0.1849, 0.1761),
                     (0.2410, 0.2161, 0.2081)]

    num_models = len(args.model_paths)
    MEANS = DEFAULT_MEANS[:num_models]
    STDS  = DEFAULT_STDS[:num_models]

    base_models = load_pruned_models(args.model_paths, device)
    MODEL_NAMES = args.model_names[:len(base_models)]

    _, _, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=False, seed=args.seed, is_main=True)

    # ---------- Mode 1: Fixed top-k ----------
    if args.active_indices is not None:
        ensemble = FixedTopKEnsemble(
            models=base_models, means=MEANS, stds=STDS,
            active_indices=args.active_indices,
            fixed_weights=args.fixed_weights,
            device=device).to(device)
        method_name = f"Fixed_Top{len(args.active_indices)}"
        evaluate_fixed(ensemble=ensemble, test_loader=test_loader, device=device,
                       save_dir=args.save_dir, model_names=MODEL_NAMES,
                       method_name=method_name)
        return

    # ---------- Mode 2: Matched-Budget ----------
    probs = args.probs
    assert len(probs) == num_models, \
        f"--probs length ({len(probs)}) must equal num models ({num_models})"
    expected_budget = float(np.sum(probs))

    print("\n" + "="*70)
    print("MATCHED-BUDGET MODE")
    print("="*70)
    print(f"  probs           = {probs}")
    print(f"  E[budget]       = {expected_budget:.4f}")
    print(f"  num_seeds       = {args.num_seeds}")
    if args.fixed_weights is not None:
        print(f"  fixed_weights   = {args.fixed_weights}")
    else:
        print(f"  fixed_weights   = uniform over active subset")
    print("="*70)

    seed_list = [args.seed + 100 * k for k in range(args.num_seeds)]
    accs, precs, recs, f1s = [], [], [], []

    for seed in seed_list:
        print(f"\n--- Seed {seed} ---")
        ens = MatchedBudgetFixedEnsemble(
            models=base_models, means=MEANS, stds=STDS,
            probs=probs, fixed_weights=args.fixed_weights,
            seed=seed).to(device)
        r = evaluate_fixed(
            ensemble=ens, test_loader=test_loader, device=device,
            save_dir=args.save_dir, model_names=MODEL_NAMES,
            method_name=f"MatchedMarginal_s{seed}",
            extra_meta={"seed": seed, "expected_budget": expected_budget})
        accs.append(r["accuracy"]); precs.append(r["precision_real"])
        recs.append(r["recall_real"]); f1s.append(r["f1_real"])

    accs = np.array(accs); precs = np.array(precs)
    recs = np.array(recs); f1s = np.array(f1s)

    summary = {
        "method": "MatchedMarginalFixed",
        "probs": probs,
        "expected_budget": expected_budget,
        "num_seeds": args.num_seeds,
        "seeds": seed_list,
        "accuracy_mean": float(accs.mean()),
        "accuracy_std":  float(accs.std()),
        "precision_mean": float(precs.mean()),
        "precision_std":  float(precs.std()),
        "recall_mean": float(recs.mean()),
        "recall_std":  float(recs.std()),
        "f1_mean": float(f1s.mean()),
        "f1_std":  float(f1s.std()),
        "per_seed_accuracy": accs.tolist(),
    }
    with open(os.path.join(args.save_dir, "matched_marginal_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print(f"MATCHED-BUDGET SUMMARY ({args.num_seeds} seeds)")
    print("="*70)
    print(f"  probs          = {probs}")
    print(f"  E[budget]      = {expected_budget:.4f}")
    print(f"  Accuracy       = {accs.mean():.2f} ± {accs.std():.2f}%")
    print(f"  Precision      = {precs.mean():.4f} ± {precs.std():.4f}")
    print(f"  Recall         = {recs.mean():.4f} ± {recs.std():.4f}")
    print(f"  F1             = {f1s.mean():.4f} ± {f1s.std():.4f}")
    print(f"  Per-seed Acc   = {[f'{a:.2f}' for a in accs]}")
    print("="*70)
    print(f"\nSummary saved to: {os.path.join(args.save_dir, 'matched_marginal_summary.json')}")


if __name__ == "__main__":
    main()
