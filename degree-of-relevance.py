#!/usr/bin/env python3
"""
analyze_memberships.py
======================
Standalone diagnostic for HFDE per-membership relevance.

Tests whether each h_{ik} behaves as a relevance signal for model i:

  [1] Per-(i,k) Spearman correlation between h_{ik} and correctness of model i
  [2] Mean h_{ik} on correct vs. incorrect samples of model i
  [3] Pairwise diversity among h_{i1}, h_{i2}, h_{i3}
  [4] Learned aggregation coefficients alpha_k
  [5] Alpha ablation: zero each alpha_k, measure accuracy drop

Verdict is empirical: if most (i,k) pairs show significant positive
correlation and every alpha_k ablation hurts accuracy, then each h_{ik}
is a relevance signal for its model.

Usage
-----
python analyze_memberships.py \
    --data_dir  /kaggle/input/datasets/hamzaboulahia/hardfakevsrealfaces \
    --dataset   hard_fake_real \
    --model_paths  /path/to/140k_pearson_pruned.pt \
                   /path/to/200k_final.pt \
                   /path/to/190k_pearson_pruned.pt \
    --model_names 140k_pearson 200k_pearson 190k_pearson \
    --checkpoint /path/to/best_hesitant_fuzzy.pt \
    --save_dir   ./membership_analysis \
    --batch_size 16 \
    --seed 42 \
    --num_memberships 3 \
    --cum_weight_threshold 0.9 \
    --hesitancy_threshold 0.0001
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


# ============================================================================
#                          MODEL DEFINITIONS
# ============================================================================
# These are copied verbatim from the training script so that the checkpoint
# loads without requiring the original training code.

class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f"mean_{i}", torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f"std_{i}", torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f"mean_{idx}")) / getattr(self, f"std_{idx}")


class HesitantFuzzyMembership(nn.Module):
    def __init__(self, input_dim: int, num_models: int,
                 num_memberships: int = 3, dropout: float = 0.3):
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
            nn.AdaptiveAvgPool2d(1),
        )
        self.membership_generator = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_models * num_memberships),
        )
        self.aggregation_weights = nn.Parameter(
            torch.ones(num_memberships) / num_memberships)

    def forward(self, x: torch.Tensor):
        features = self.feature_net(x).flatten(1)
        memberships = self.membership_generator(features)
        memberships = memberships.view(-1, self.num_models,
                                       self.num_memberships)
        memberships = torch.sigmoid(memberships)
        agg_weights = F.softmax(self.aggregation_weights, dim=0)
        raw = (memberships * agg_weights.view(1, 1, -1)).sum(dim=2)
        weights = F.softmax(raw, dim=1)
        return weights, memberships


class FuzzyHesitantEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module],
                 means: List[Tuple[float]], stds: List[Tuple[float]],
                 num_memberships: int = 3, freeze_models: bool = True,
                 cum_weight_threshold: float = 0.9,
                 hesitancy_threshold: float = 0.2):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.hesitant_fuzzy = HesitantFuzzyMembership(
            input_dim=128, num_models=self.num_models,
            num_memberships=num_memberships)
        self.cum_weight_threshold = cum_weight_threshold
        self.hesitancy_threshold = hesitancy_threshold
        if freeze_models:
            for m in self.models:
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        for m in self.models:
            m.eval()
        return self

    def _compute_mask_vectorized(self, final_weights: torch.Tensor,
                                 avg_hesitancy: torch.Tensor):
        sorted_weights, sorted_indices = torch.sort(
            final_weights, dim=1, descending=True)
        cum = torch.cumsum(sorted_weights, dim=1)
        mask = (cum <= self.cum_weight_threshold).float()
        mask[:, 0] = 1.0
        high = (avg_hesitancy > self.hesitancy_threshold).unsqueeze(1)
        mask = torch.where(high, torch.ones_like(mask), mask)
        out = torch.zeros_like(final_weights)
        out.scatter_(1, sorted_indices, mask)
        return out

    def forward(self, x: torch.Tensor, return_details: bool = False):
        weights, memberships = self.hesitant_fuzzy(x)
        hes = memberships.var(dim=2, unbiased=False)
        avg_hes = hes.mean(dim=1)
        mask = self._compute_mask_vectorized(weights, avg_hes)
        weights = weights * mask
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
        active = torch.any(weights > 0, dim=0).nonzero(as_tuple=True)[0]
        for i in active:
            x_n = self.normalizations(x, i.item())
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            outputs[:, i] = out
        final = (outputs * weights.unsqueeze(-1)).sum(dim=1)
        if return_details:
            return final, weights, memberships, outputs
        return final, weights


# ============================================================================
#                          MODEL LOADING
# ============================================================================

def load_pruned_models(model_paths: List[str], device) -> List[nn.Module]:
    """Load pruned ResNet-50 checkpoints. Requires model.ResNet_pruned."""
    try:
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        sys.exit("ERROR: Cannot import model.ResNet_pruned. "
                 "Run from the repository root.")
    models = []
    for i, p in enumerate(model_paths):
        if not os.path.exists(p):
            sys.exit(f"ERROR: not found: {p}")
        print(f"[{i + 1}/{len(model_paths)}] Loading {os.path.basename(p)}")
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        m = ResNet_50_pruned_hardfakevsreal(masks=ckpt["masks"])
        m.load_state_dict(ckpt["model_state_dict"])
        models.append(m.to(device).eval())
    return models


# ============================================================================
#                          DATA LOADING
# ============================================================================

def build_test_loader(data_dir: str, dataset: str,
                      batch_size: int, seed: int) -> DataLoader:
    """Build only the test loader (deterministic split via seed)."""
    try:
        from old_dataset_utils import create_dataloaders
    except ImportError:
        sys.exit("ERROR: Cannot import old_dataset_utils. "
                 "Run from the repository root.")
    _, _, test = create_dataloaders(
        data_dir, batch_size, dataset_type=dataset,
        is_distributed=False, seed=seed, is_main=True)
    return test


# ============================================================================
#                          ACCURACY EVALUATION
# ============================================================================

@torch.no_grad()
def eval_accuracy(ensemble: FuzzyHesitantEnsemble, loader: DataLoader,
                  device) -> float:
    ensemble.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()
        out, _ = ensemble(images)
        pred = (out.squeeze(-1) > 0).long()
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total if total else 0.0


# ============================================================================
#                          MEMBERSHIP COLLECTION
# ============================================================================

@torch.no_grad()
def collect_memberships_and_correctness(
        ensemble: FuzzyHesitantEnsemble,
        loader: DataLoader,
        device,
        n_models: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      memberships:  (S, N, K) float array of h_ik for all test samples
      correctness:  (S, N)    binary array of per-model correctness
    """
    ensemble.eval()
    all_m, all_c = [], []
    for images, labels in tqdm(loader, desc="Collecting memberships"):
        images = images.to(device)
        labels = labels.to(device).float()
        _, memberships = ensemble.hesitant_fuzzy(images)  # (B,N,K)

        B = images.size(0)
        model_correct = torch.zeros(B, n_models, device=device)
        for i in range(n_models):
            x_n = ensemble.normalizations(images, i)
            out = ensemble.models[i](x_n)
            if isinstance(out, (tuple, list)):
                out = out[0]
            pred = (out.squeeze(-1) > 0).float()
            model_correct[:, i] = (pred == labels).float()

        all_m.append(memberships.cpu())
        all_c.append(model_correct.cpu())

    M = torch.cat(all_m, dim=0).numpy()
    C = torch.cat(all_c, dim=0).numpy()
    return M, C


# ============================================================================
#                          MAIN ANALYSIS
# ============================================================================

def analyze(ensemble: FuzzyHesitantEnsemble,
            test_loader: DataLoader,
            device,
            model_names: List[str],
            save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    ensemble.eval()

    print("\n" + "=" * 88)
    print("STEP 0: Collecting memberships and per-model correctness")
    print("=" * 88)
    M, C = collect_memberships_and_correctness(
        ensemble, test_loader, device, len(model_names))
    S, N, K = M.shape
    print(f"  Samples: {S} | Models: {N} | K: {K}")
    print(f"  Per-model accuracy on test set:")
    for i, name in enumerate(model_names):
        print(f"    {i + 1}. {name:<20} {100 * C[:, i].mean():.2f}%")

    # ------------------------------------------------------------------------
    # [1] Per-(i,k) correlation between h_ik and model-i correctness
    # ------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("STEP 1: Does each h_{ik} track the correctness of model i?")
    print("=" * 88)
    print(f"  {'Model':<20} {'k':>2} {'mu|correct':>11} {'mu|incorr':>10} "
          f"{'Spearman':>10} {'p':>8} {'sig+':>5}")
    print("  " + "-" * 76)

    per_membership = defaultdict(dict)
    n_sig_pos = 0
    for i, name in enumerate(model_names):
        for k in range(K):
            h = M[:, i, k]
            c = C[:, i]
            mc = c > 0.5
            mu_c = float(h[mc].mean()) if mc.any() else float("nan")
            mu_nc = float(h[~mc].mean()) if (~mc).any() else float("nan")
            if c.std() < 1e-8:
                rho, p = 0.0, 1.0
            else:
                rho, p = spearmanr(h, c)
                rho, p = float(rho), float(p)
            sig = (p < 0.05) and (rho > 0)
            if sig:
                n_sig_pos += 1
            per_membership[name][f"h_{k + 1}"] = {
                "mu_correct": mu_c, "mu_incorrect": mu_nc,
                "spearman": rho, "p_value": p,
                "significant_positive": bool(sig),
            }
            print(f"  {name:<20} {k + 1:>2} {mu_c:>11.4f} {mu_nc:>10.4f} "
                  f"{rho:>+10.4f} {p:>8.4f} {'YES' if sig else 'no':>5}")
        print()

    # ------------------------------------------------------------------------
    # [2] Pairwise diversity among memberships of the same model
    # ------------------------------------------------------------------------
    print("=" * 88)
    print("STEP 2: Are memberships diverse (non-redundant)? "
          "Pairwise Spearman within each model")
    print("=" * 88)
    pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]
    hdr = "  " + f"{'Model':<20}" + "".join(
        f"{f'corr(h{a + 1},h{b + 1})':>18}" for a, b in pairs)
    print(hdr)
    print("  " + "-" * (20 + 18 * len(pairs)))
    diversity = {}
    for i, name in enumerate(model_names):
        vals = [float(spearmanr(M[:, i, a], M[:, i, b])[0])
                for a, b in pairs]
        diversity[name] = {f"h_{a + 1}_vs_h_{b + 1}": v
                           for (a, b), v in zip(pairs, vals)}
        line = f"  {name:<20}" + "".join(f"{v:>+18.4f}" for v in vals)
        print(line)
    print()

    # ------------------------------------------------------------------------
    # [3] Learned alpha coefficients
    # ------------------------------------------------------------------------
    print("=" * 88)
    print("STEP 3: Learned aggregation coefficients alpha_k (softmax)")
    print("=" * 88)
    alphas = F.softmax(
        ensemble.hesitant_fuzzy.aggregation_weights, dim=0).cpu().numpy()
    for k, a in enumerate(alphas):
        print(f"  alpha_{k + 1} = {a:.4f}")
    print(f"  (uniform reference = {1.0 / K:.4f})")
    print()

    # ------------------------------------------------------------------------
    # [4] Alpha ablation
    # ------------------------------------------------------------------------
    print("=" * 88)
    print("STEP 4: Alpha ablation (zero each alpha_k, re-evaluate accuracy)")
    print("=" * 88)
    original = ensemble.hesitant_fuzzy.aggregation_weights.detach().clone()
    base_acc = eval_accuracy(ensemble, test_loader, device)
    print(f"  Full model:          {base_acc:.2f}%")
    ablation = {"full": float(base_acc)}
    for k_abl in range(K):
        with torch.no_grad():
            new_w = original.clone()
            new_w[k_abl] = -100.0  # softmax -> ~0
            ensemble.hesitant_fuzzy.aggregation_weights.copy_(new_w)
        acc = eval_accuracy(ensemble, test_loader, device)
        drop = base_acc - acc
        mark = "HURTS" if drop > 0.5 else ("helps" if drop < -0.5 else "~neutral")
        print(f"  Without alpha_{k_abl + 1}:    {acc:.2f}%   "
              f"(drop {drop:+.2f} pp)  [{mark}]")
        ablation[f"without_alpha_{k_abl + 1}"] = {
            "accuracy": float(acc), "drop_pp": float(drop)}
        with torch.no_grad():
            ensemble.hesitant_fuzzy.aggregation_weights.copy_(original)
    all_hurt = all(ablation[f"without_alpha_{k + 1}"]["drop_pp"] > 0.5
                   for k in range(K))
    print(f"  Every alpha_k ablation hurts by > 0.5 pp: {all_hurt}")
    print()

    # ------------------------------------------------------------------------
    # [5] Verdict
    # ------------------------------------------------------------------------
    frac_sig = n_sig_pos / (N * K)
    print("=" * 88)
    print("VERDICT")
    print("=" * 88)
    print(f"  Significant positive memberships: {n_sig_pos}/{N * K} "
          f"({100 * frac_sig:.1f}%)")
    print(f"  Every alpha_k ablation hurts:     {all_hurt}")
    if frac_sig >= 0.70 and all_hurt:
        verdict = ("STRONG: each h_{ik} behaves as a distinct relevance "
                   "signal for its model.")
    elif frac_sig >= 0.50:
        verdict = ("MODERATE: most h_{ik} behave as relevance signals; "
                   "some are less informative.")
    else:
        verdict = ("WEAK: only some h_{ik} behave as relevance signals; "
                   "rephrase the claim accordingly.")
    print(f"  => {verdict}")

    out = {
        "dataset_shape": {"samples": int(S), "models": int(N), "K": int(K)},
        "per_model_accuracy_pct": {
            name: float(100 * C[:, i].mean())
            for i, name in enumerate(model_names)},
        "per_membership_correlation": {k: dict(v)
                                       for k, v in per_membership.items()},
        "membership_diversity": diversity,
        "alphas": [float(a) for a in alphas],
        "alpha_ablation": ablation,
        "summary": {
            "n_significant_positive": int(n_sig_pos),
            "n_total": int(N * K),
            "fraction": float(frac_sig),
            "all_alpha_ablations_hurt": bool(all_hurt),
            "verdict": verdict,
        },
    }
    out_path = os.path.join(save_dir, "per_membership_relevance.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # Optional: scatter plot for the paper (h_ik vs. correctness)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(N, K, figsize=(4 * K, 3 * N),
                                 squeeze=False)
        for i, name in enumerate(model_names):
            for k in range(K):
                ax = axes[i, k]
                h = M[:, i, k]
                c = C[:, i]
                ax.scatter(h[c < 0.5], np.zeros((c < 0.5).sum()),
                           alpha=0.3, s=8, label="incorrect")
                ax.scatter(h[c > 0.5], np.ones((c > 0.5).sum()),
                           alpha=0.3, s=8, label="correct")
                ax.set_title(f"{name} / h_{k + 1}")
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["incorrect", "correct"])
                ax.set_xlim(0, 1)
                if i == 0 and k == 0:
                    ax.legend(fontsize=7)
        fig.suptitle("Per-membership distribution vs. model correctness")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "per_membership_scatter.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: per_membership_scatter.png")
    except Exception as e:
        print(f"(plot skipped: {e})")

    return out


# ============================================================================
#                          ENTRY POINT
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--dataset", required=True,
                    choices=["wild", "artiface", "hard_fake_real",
                             "deepfake_lab", "uadfV", "custom_genai",
                             "custom_genai_v2", "real_fake_dataset", "20k"])
    ap.add_argument("--model_paths", nargs="+", required=True)
    ap.add_argument("--model_names", nargs="+", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="Path to best_hesitant_fuzzy.pt")
    ap.add_argument("--save_dir", default="./membership_analysis")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_memberships", type=int, default=3)
    ap.add_argument("--cum_weight_threshold", type=float, default=0.9)
    ap.add_argument("--hesitancy_threshold", type=float, default=1e-4)
    args = ap.parse_args()

    if len(args.model_paths) != len(args.model_names):
        sys.exit("ERROR: --model_paths and --model_names must match in length.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- data ---
    print("\nBuilding test loader...")
    test_loader = build_test_loader(
        args.data_dir, args.dataset, args.batch_size, args.seed)

    # --- base models ---
    print(f"\nLoading {len(args.model_paths)} base models...")
    base_models = load_pruned_models(args.model_paths, device)

    # --- ensemble ---
    DEFAULT_MEANS = [(0.5207, 0.4258, 0.3806),
                     (0.4460, 0.3622, 0.3416),
                     (0.4668, 0.3816, 0.3414)]
    DEFAULT_STDS = [(0.2490, 0.2239, 0.2212),
                    (0.2057, 0.1849, 0.1761),
                    (0.2410, 0.2161, 0.2081)]
    n = len(base_models)
    means = DEFAULT_MEANS[:n]
    stds = DEFAULT_STDS[:n]

    ensemble = FuzzyHesitantEnsemble(
        base_models, means, stds,
        num_memberships=args.num_memberships,
        freeze_models=True,
        cum_weight_threshold=args.cum_weight_threshold,
        hesitancy_threshold=args.hesitancy_threshold,
    ).to(device)

    # --- load trained router ---
    if not os.path.exists(args.checkpoint):
        sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("hesitant_state_dict", ckpt)
    ensemble.hesitant_fuzzy.load_state_dict(state)
    print(f"Loaded router from: {args.checkpoint}")

    # --- run analysis ---
    analyze(ensemble, test_loader, device, args.model_names, args.save_dir)


if __name__ == "__main__":
    main()
