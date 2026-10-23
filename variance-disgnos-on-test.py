import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json
import argparse
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Datasets — needs old_dataset_utils.py in the same folder
# ------------------------------------------------------------------
try:
    from old_dataset_utils import create_dataloaders, get_sample_info
except ImportError as e:
    print(f"\n[ERROR] Could not import old_dataset_utils: {e}")
    print("Place diagnose_override.py in the SAME folder as old_dataset_utils.py.")
    sys.exit(1)


# ==================================================================
# MODEL CLASSES (copied verbatim from the main script)
# ==================================================================
class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


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
                 cum_weight_threshold=0.9, hesitancy_threshold=0.2,
                 hesitancy_measure='variance'):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.hesitant_fuzzy = HesitantFuzzyMembership(
            input_dim=128, num_models=self.num_models,
            num_memberships=num_memberships)
        self.cum_weight_threshold = cum_weight_threshold
        self.hesitancy_threshold = hesitancy_threshold
        self.hesitancy_measure = hesitancy_measure

        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

    def _compute_hesitancy(self, all_memberships):
        if self.hesitancy_measure == 'variance':
            return all_memberships.var(dim=2, unbiased=False).mean(dim=1)
        elif self.hesitancy_measure == 'cv':
            mean_mem = all_memberships.mean(dim=2)
            std_mem = all_memberships.std(dim=2, unbiased=False)
            return (std_mem / (mean_mem + 1e-8)).mean(dim=1)
        elif self.hesitancy_measure == 'entropy':
            p = all_memberships / (all_memberships.sum(dim=2, keepdim=True) + 1e-8)
            K = all_memberships.size(2)
            ent = -(p * torch.log(p + 1e-8)).sum(dim=2) / np.log(K)
            return ent.mean(dim=1)
        elif self.hesitancy_measure == 'range':
            rng = (all_memberships.max(dim=2).values - all_memberships.min(dim=2).values)
            return rng.mean(dim=1)
        elif self.hesitancy_measure == 'combined':
            var = all_memberships.var(dim=2, unbiased=False).mean(dim=1)
            mean_level = all_memberships.mean(dim=2).mean(dim=1)
            return var + (1.0 - mean_level) * 0.5
        else:
            raise ValueError(f"Unknown hesitancy_measure: {self.hesitancy_measure}")

    def _compute_mask_vectorized(self, final_weights, avg_hesitancy):
        sorted_weights, sorted_indices = torch.sort(final_weights, dim=1, descending=True)
        cum_weights = torch.cumsum(sorted_weights, dim=1)
        mask = (cum_weights <= self.cum_weight_threshold).float()
        mask[:, 0] = 1.0
        high_hesitancy_mask = (avg_hesitancy > self.hesitancy_threshold).unsqueeze(1)
        mask = torch.where(high_hesitancy_mask, torch.ones_like(mask), mask)
        final_mask = torch.zeros_like(final_weights)
        final_mask.scatter_(1, sorted_indices, mask)
        return final_mask

    @torch.no_grad()
    def compute_all_model_logits(self, x):
        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
        for i in range(self.num_models):
            x_n = self.normalizations(x, i)
            out = self.models[i](x_n)
            if isinstance(out, (tuple, list)):
                out = out[0]
            outputs[:, i] = out
        return outputs

    def forward(self, x, return_details=False):
        final_weights, all_memberships = self.hesitant_fuzzy(x)
        avg_hesitancy = self._compute_hesitancy(all_memberships)
        mask = self._compute_mask_vectorized(final_weights, avg_hesitancy)
        final_weights = final_weights * mask
        final_weights = final_weights / (final_weights.sum(dim=1, keepdim=True) + 1e-8)

        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
        active_models = torch.any(final_weights > 0, dim=0).nonzero(as_tuple=True)[0]
        for i in active_models:
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


# ==================================================================
# MODEL LOADING
# ==================================================================
def load_pruned_models(model_paths, device):
    from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    models = []
    print(f"Loading {len(model_paths)} pruned models...")
    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            print(f"  [WARNING] Not found: {path}")
            continue
        print(f"  [{i+1}/{len(model_paths)}] {os.path.basename(path)}")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        m = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
        m.load_state_dict(ckpt['model_state_dict'])
        models.append(m.to(device).eval())
    if not models:
        raise ValueError("No models loaded!")
    return models


# ==================================================================
# DIAGNOSTIC
# ==================================================================
def compute_measures(all_memberships):
    var = all_memberships.var(dim=2, unbiased=False).mean(dim=1)
    mean_mem = all_memberships.mean(dim=2)
    std_mem = all_memberships.std(dim=2, unbiased=False)
    cv = (std_mem / (mean_mem + 1e-8)).mean(dim=1)
    p = all_memberships / (all_memberships.sum(dim=2, keepdim=True) + 1e-8)
    K = all_memberships.size(2)
    ent = -(p * torch.log(p + 1e-8)).sum(dim=2) / np.log(K)
    ent = ent.mean(dim=1)
    rng = (all_memberships.max(dim=2).values - all_memberships.min(dim=2).values).mean(dim=1)
    mean_level = mean_mem.mean(dim=1)
    return {'variance': var, 'cv': cv, 'entropy': ent,
            'range': rng, 'mean_level': mean_level}


@torch.no_grad()
def diagnose(ensemble, loader, device, model_names, save_dir,
             hesitancy_threshold, cum_weight_threshold, hesitancy_measure):
    ensemble.eval()
    ds = loader.dataset
    N = ensemble.num_models
    print(f"\nDiagnosing on {len(ds)} samples (measure = {hesitancy_measure})...")

    rec = {k: [] for k in ['filename', 'label', 'pred', 'prob', 'correct',
                            'h_var', 'h_ent', 'h_mean', 'override', 'n_active',
                            'model_errors', 'final_weights']}

    for i in tqdm(range(len(ds)), desc="Diagnosing"):
        try:
            image, label = ds[i]
        except Exception:
            continue
        image = image.unsqueeze(0).to(device)
        label_int = int(label)

        out, fw, mem, _ = ensemble(image, return_details=True)
        prob = torch.sigmoid(out.squeeze()).item()
        pred = int(prob > 0.5)

        m = compute_measures(mem)
        h_var  = float(m['variance'].item())
        h_ent  = float(m['entropy'].item())
        h_mean = float(m['mean_level'].item())

        if hesitancy_measure == 'variance':
            override = int(h_var > hesitancy_threshold)
        elif hesitancy_measure == 'entropy':
            override = int(h_ent > hesitancy_threshold)
        elif hesitancy_measure == 'mean_level':
            override = int(h_mean > hesitancy_threshold)
        else:
            override = int(h_var > hesitancy_threshold)

        fw_np = fw.squeeze(0).cpu().numpy()
        n_active = int((fw_np > 1e-8).sum())

        all_logits = ensemble.compute_all_model_logits(image)
        m_errs = [
            0 if int(all_logits[0, j, 0].item() > 0) == label_int else 1
            for j in range(N)
        ]

        try:
            orig = ds.indices[i] if hasattr(ds, 'indices') else i
            base = ds.dataset if hasattr(ds, 'dataset') else ds
            path, _ = get_sample_info(base, orig)
            fname = os.path.basename(path)
        except Exception:
            fname = f"sample_{i:05d}"

        rec['filename'].append(fname)
        rec['label'].append(label_int)
        rec['pred'].append(pred)
        rec['prob'].append(prob)
        rec['correct'].append(1 if pred == label_int else 0)
        rec['h_var'].append(h_var)
        rec['h_ent'].append(h_ent)
        rec['h_mean'].append(h_mean)
        rec['override'].append(override)
        rec['n_active'].append(n_active)
        rec['model_errors'].append(m_errs)
        rec['final_weights'].append(fw_np.tolist())

    labels   = np.array(rec['label'])
    correct  = np.array(rec['correct'])
    override = np.array(rec['override'])
    h_var    = np.array(rec['h_var'])
    h_ent    = np.array(rec['h_ent'])
    h_mean   = np.array(rec['h_mean'])
    n_active = np.array(rec['n_active'])
    model_errs = np.array(rec['model_errors'])
    n = len(correct)

    if n == 0:
        print("[ERROR] No samples processed.")
        return

    # =============================================================
    print("\n" + "="*82)
    print(" OVERRIDE DIAGNOSTICS — TEST SET")
    print("="*82)
    print(f"Samples                 : {n}")
    print(f"Overall accuracy        : {100*correct.mean():.2f}%")
    print(f"Hesitancy measure       : {hesitancy_measure}")
    print(f"Override threshold      : {hesitancy_threshold}")
    print(f"Cumulative weight T     : {cum_weight_threshold}")
    print(f"Override active rate    : {100*override.mean():.2f}%  "
          f"({int(override.sum())}/{n})")

    # ---- [1] Which branch has what accuracy? ----
    print("\n" + "-"*82)
    print(" [1] Is override targeting AMBIGUOUS (hard) samples?")
    print("-"*82)
    n_on  = int(override.sum())
    n_off = n - n_on
    acc_on = acc_off = delta = None

    if n_on > 0 and n_off > 0:
        acc_on  = 100 * correct[override == 1].mean()
        acc_off = 100 * correct[override == 0].mean()
        delta   = acc_on - acc_off
        print(f"  Override ACTIVE   (n={n_on:4d}): accuracy = {acc_on:5.2f}%")
        print(f"  Override INACTIVE (n={n_off:4d}): accuracy = {acc_off:5.2f}%")
        print(f"  Δ (active − inactive)          = {delta:+.2f}%")

        if delta < -2:
            print(f"\n  ► Verdict: ❌ Override hits WRONG (hard) samples")
            print(f"     This CONFIRMS the paper's claim.")
        elif delta > 2:
            print(f"\n  ► Verdict: ❌ Override hits EASY samples")
            print(f"     This CONTRADICTS the paper's claim.")
            print(f"     Use the 'diversity preservation' framing.")
        else:
            print(f"\n  ► Verdict: ✓ Override is roughly neutral on average")
    else:
        print(f"  Only one branch was active. Override rate = {100*override.mean():.1f}%")

    # ---- [2] Contingency table ----
    print("\n" + "-"*82)
    print(" [2] Contingency table: override × correct")
    print("-"*82)
    c11 = int(((override==1)&(correct==1)).sum())
    c10 = int(((override==1)&(correct==0)).sum())
    c01 = int(((override==0)&(correct==1)).sum())
    c00 = int(((override==0)&(correct==0)).sum())
    print(f"  {'':<16}{'correct=1':>12}{'correct=0':>12}{'total':>10}")
    print(f"  {'override=1':<16}{c11:>12}{c10:>12}{c11+c10:>10}")
    print(f"  {'override=0':<16}{c01:>12}{c00:>12}{c01+c00:>10}")
    print(f"  {'total':<16}{c11+c01:>12}{c10+c00:>12}{n:>10}")

    # ---- [3] Active model distribution ----
    print("\n" + "-"*82)
    print(" [3] Distribution of ACTIVE MODEL COUNT")
    print("-"*82)
    for k in range(1, N + 1):
        m = (n_active == k)
        cnt = int(m.sum())
        pct = 100 * cnt / n
        acc_k = 100 * correct[m].mean() if cnt > 0 else 0.0
        bar = '█' * int(pct / 2)
        print(f"  {k} model(s): {cnt:5d} samples ({pct:5.2f}%)   acc={acc_k:5.2f}%   {bar}")

    # ---- [4] Per-model accuracy by branch ----
    print("\n" + "-"*82)
    print(" [4] Base-model accuracy: override-active vs override-inactive")
    print("-"*82)
    print(f"  {'Model':<20}{'on override=1':>18}{'on override=0':>18}{'overall':>12}")
    for j, name in enumerate(model_names):
        a_on  = 100*(1-model_errs[override==1, j]).mean() if (override==1).any() else 0.0
        a_off = 100*(1-model_errs[override==0, j]).mean() if (override==0).any() else 0.0
        a_all = 100*(1-model_errs[:, j]).mean()
        print(f"  {name:<20}{a_on:>17.2f}%{a_off:>17.2f}%{a_all:>11.2f}%")

    mean_base_err = model_errs.mean(axis=1)
    e_on  = float(mean_base_err[override==1].mean()) if (override==1).any() else float('nan')
    e_off = float(mean_base_err[override==0].mean()) if (override==0).any() else float('nan')
    print(f"\n  Mean base-model error rate:")
    print(f"    Override active   : {100*e_on:5.2f}%")
    print(f"    Override inactive : {100*e_off:5.2f}%")
    if not (np.isnan(e_on) or np.isnan(e_off)):
        d = 100*(e_on - e_off)
        print(f"    Δ                 : {d:+.2f}%")
        if   e_on < e_off - 0.01:
            print("    → Base models MORE accurate on override-active samples.")
            print("      Consistent with 'diversity preservation' interpretation.")
        elif e_on > e_off + 0.01:
            print("    → Base models LESS accurate on override-active samples.")
            print("      Consistent with the paper's 'help hard samples' claim.")
        else:
            print("    → No meaningful difference.")

    # ---- [5] Correlations ----
    print("\n" + "-"*82)
    print(" [5] Correlations of hesitancy measures with error")
    print("-"*82)
    err = 1 - correct
    for name, vals, is_conf in [
        ('variance',   h_var,  False),
        ('entropy',    h_ent,  False),
        ('mean_level', h_mean, True),
    ]:
        if is_conf:
            r, p = pearsonr(vals, correct)
            print(f"  {name:<14} Pearson r with CORRECTNESS = {r:+.3f} (p={p:.4f})")
        else:
            r, p = pearsonr(vals, err)
            auc = roc_auc_score(err, vals) if len(np.unique(err)) > 1 else 0.5
            r2, p2 = spearmanr(vals, err)
            print(f"  {name:<14} Pearson={r:+.3f} (p={p:.4f})   "
                  f"Spearman={r2:+.3f} (p={p2:.4f})   AUC(err)={auc:.3f}")

    # ---- [6] By ground-truth class ----
    print("\n" + "-"*82)
    print(" [6] Override rate by ground-truth class")
    print("-"*82)
    for lab in [0, 1]:
        m = (labels == lab)
        if m.sum() > 0:
            cls = 'FAKE (0)' if lab == 0 else 'REAL (1)'
            print(f"  {cls:<10}  n={int(m.sum()):4d}   "
                  f"override rate={100*override[m].mean():5.2f}%   "
                  f"accuracy={100*correct[m].mean():5.2f}%")

    # ---- Save CSV ----
    csv_path = os.path.join(save_dir, 'override_diagnostics.csv')
    with open(csv_path, 'w') as f:
        cols = ['filename', 'label', 'pred', 'prob', 'correct',
                'h_var', 'h_ent', 'h_mean', 'override', 'n_active']
        cols += [f'fw_m{k}' for k in range(N)]
        cols += [f'err_m{k}' for k in range(N)]
        f.write(','.join(cols) + '\n')
        for i in range(n):
            row = [rec['filename'][i], labels[i], rec['pred'][i],
                   f"{rec['prob'][i]:.6f}", int(rec['correct'][i]),
                   f"{rec['h_var'][i]:.6f}", f"{rec['h_ent'][i]:.6f}",
                   f"{rec['h_mean'][i]:.6f}", int(rec['override'][i]),
                   int(rec['n_active'][i])]
            row += [f"{w:.6f}" for w in rec['final_weights'][i]]
            row += [int(e) for e in rec['model_errors'][i]]
            f.write(','.join(str(x) for x in row) + '\n')
    print(f"\n✓ CSV saved: {csv_path}")

    # ---- Plots ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    if (override == 1).any():
        ax.hist(h_var[override == 1], bins=40, alpha=0.6, density=True,
                color='C0', label=f'override=1 (n={int((override==1).sum())})')
    if (override == 0).any():
        ax.hist(h_var[override == 0], bins=40, alpha=0.6, density=True,
                color='C1', label=f'override=0 (n={int((override==0).sum())})')
    ax.axvline(hesitancy_threshold, color='k', linestyle='--',
               label=f'thr={hesitancy_threshold}')
    ax.set_xlabel('Variance')
    ax.set_ylabel('Density')
    ax.set_title('Variance distribution by branch')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    if acc_on is not None and acc_off is not None:
        bars = ax.bar(['override=1', 'override=0'], [acc_on, acc_off],
                      color=['C0', 'C1'], alpha=0.75)
        ax.set_ylim(0, 100)
        for b, v in zip(bars, [acc_on, acc_off]):
            ax.text(b.get_x()+b.get_width()/2, v+2, f'{v:.1f}%',
                    ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by branch')
    ax.grid(alpha=0.3, axis='y')

    ax = axes[2]
    xs = [str(k) for k in range(1, N+1)]
    ys = [int((n_active == k).sum()) for k in range(1, N+1)]
    bars = ax.bar(xs, ys, color='C2', alpha=0.75)
    ax.set_xlabel('# active models')
    ax.set_ylabel('# samples')
    ax.set_title('Active model count')
    for b, v in zip(bars, ys):
        ax.text(b.get_x()+b.get_width()/2, v+max(ys)*0.01,
                f'{v}', ha='center', fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'override_diagnostics.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved: {plot_path}")

    # ---- JSON ----
    summary = {
        'n_samples': n,
        'overall_accuracy_pct': float(100*correct.mean()),
        'hesitancy_measure': hesitancy_measure,
        'hesitancy_threshold': float(hesitancy_threshold),
        'cum_weight_threshold': float(cum_weight_threshold),
        'override_rate_pct': float(100*override.mean()),
        'n_override_active': n_on,
        'n_override_inactive': n_off,
        'accuracy_override_active_pct':   float(acc_on)  if acc_on  is not None else None,
        'accuracy_override_inactive_pct': float(acc_off) if acc_off is not None else None,
        'delta_accuracy_pct': float(delta) if delta is not None else None,
        'active_model_count_distribution': {
            int(k): int((n_active == k).sum()) for k in range(1, N+1)
        },
        'mean_base_error_override_active':   float(e_on)  if not np.isnan(e_on)  else None,
        'mean_base_error_override_inactive': float(e_off) if not np.isnan(e_off) else None,
    }
    with open(os.path.join(save_dir, 'override_diagnostics.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ JSON saved: {os.path.join(save_dir, 'override_diagnostics.json')}")

    print("\n" + "="*82)
    print(" HOW TO READ")
    print("="*82)
    print("  Δ_accuracy < −2%  → override hits HARD samples (paper's claim OK)")
    print("  Δ_accuracy > +2%  → override hits EASY samples (change framing)")
    print("  Δ_base_error < 0  → base models MORE accurate on override")
    print("                       (matches ρ=+0.437 competence)")
    print("  Δ_base_error > 0  → base models LESS accurate on override")
    print("="*82)


# ==================================================================
# MAIN
# ==================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, required=True)
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--model_paths', type=str, nargs='+', required=True)
    p.add_argument('--model_names', type=str, nargs='+', required=True)
    p.add_argument('--ensemble_ckpt', type=str, required=True,
                   help='Path to final_ensemble_model.pt')
    p.add_argument('--save_dir', type=str, default='./output')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--hesitancy_measure', type=str, default='variance',
                   choices=['variance', 'cv', 'entropy', 'range', 'combined'])
    p.add_argument('--hesitancy_threshold', type=float, required=True,
                   help='Threshold used in the paper (e.g., 0.0001)')
    p.add_argument('--cum_weight_threshold', type=float, default=0.9)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CPU (will be slow)")

    print(f"\nLoading checkpoint: {args.ensemble_ckpt}")
    ckpt = torch.load(args.ensemble_ckpt, map_location=device)
    means = ckpt['means']
    stds  = ckpt['stds']

    base_models = load_pruned_models(args.model_paths, device)

    print("Building FuzzyHesitantEnsemble...")
    ensemble = FuzzyHesitantEnsemble(
        base_models, means, stds,
        num_memberships=3,
        freeze_models=True,
        cum_weight_threshold=args.cum_weight_threshold,
        hesitancy_threshold=args.hesitancy_threshold,
        hesitancy_measure=args.hesitancy_measure,
    ).to(device)

    if 'ensemble_state_dict' in ckpt:
        ensemble.load_state_dict(ckpt['ensemble_state_dict'])
        print("  Loaded full ensemble state dict.")
    elif 'hesitant_state_dict' in ckpt:
        ensemble.hesitant_fuzzy.load_state_dict(ckpt['hesitant_state_dict'])
        print("  Loaded hesitant_fuzzy state dict only.")
    else:
        raise KeyError("Checkpoint has neither 'ensemble_state_dict' nor 'hesitant_state_dict'")

    ensemble.eval()
    print("Ensemble ready.\n")

    print("Loading test set...")
    _, _, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=False, seed=args.seed, is_main=False
    )

    diagnose(ensemble, test_loader, device,
             args.model_names[:len(base_models)],
             args.save_dir,
             args.hesitancy_threshold,
             args.cum_weight_threshold,
             args.hesitancy_measure)


if __name__ == '__main__':
    main()
