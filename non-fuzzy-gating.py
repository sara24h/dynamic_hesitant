import argparse
import csv
import json
import os
import random
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from old_dataset_utils import create_dataloaders, get_sample_info

warnings.filterwarnings("ignore")

try:
    from scipy.stats import binomtest
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed — McNemar test will be skipped.")


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_device():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available, using CPU")
    return device


# ============================================================
# 2. Input normalization (identical to HFDE)
# ============================================================

class MultiModelNormalization(nn.Module):
    def __init__(self, means, stds):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f"mean_{i}", torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f"std_{i}", torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x, idx):
        return (x - getattr(self, f"mean_{idx}")) / getattr(self, f"std_{idx}")


# ============================================================
# 3. Non-fuzzy multi-head gating (identical architecture to HFDE)
# ============================================================

class NonFuzzyMultiHeadGating(nn.Module):
    """
    ساختاراً منطبق بر HesitantFuzzyMembership در HFDE.
    تفاوت: softmax categorical به‌جای sigmoid fuzzy membership.
    """
    def __init__(self, num_models, num_heads=3, dropout=0.3):
        super().__init__()
        self.num_models = num_models
        self.num_heads = num_heads

        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.gating_generator = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_models * num_heads),
        )
        self.head_aggregation_logits = nn.Parameter(
            torch.ones(num_heads) / num_heads
        )

    def forward(self, x):
        features = self.feature_net(x).flatten(1)
        logits = self.gating_generator(features)
        head_logits = logits.view(-1, self.num_models, self.num_heads).transpose(1, 2)
        head_weights = F.softmax(head_logits, dim=2)
        alpha = F.softmax(self.head_aggregation_logits, dim=0)
        final_weights = (head_weights * alpha.view(1, self.num_heads, 1)).sum(dim=1)

        # این "disagreement" است، نه hesitancy فازی
        disagreement = head_weights.var(dim=1, unbiased=False).mean(dim=1)
        return final_weights, head_weights, disagreement


# ============================================================
# 4. Ensemble — mode-aware masking
# ============================================================

class NonFuzzyGatingEnsemble(nn.Module):
    def __init__(self, models, means, stds, num_heads=3, dropout=0.3,
                 freeze_models=True, cum_weight_threshold=0.9,
                 hesitancy_threshold=1e-4, mode="matched"):
        super().__init__()
        assert mode in ("matched", "pure"), f"mode={mode}"
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.gating = NonFuzzyMultiHeadGating(
            num_models=self.num_models, num_heads=num_heads, dropout=dropout)
        self.cum_weight_threshold = float(cum_weight_threshold)
        self.hesitancy_threshold = float(hesitancy_threshold)
        self.mode = mode

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

    def _compute_mask(self, weights, disagreement):
        sorted_w, sorted_idx = torch.sort(weights, dim=1, descending=True)
        cum = torch.cumsum(sorted_w, dim=1)
        mask_sorted = (cum <= self.cum_weight_threshold).float()
        mask_sorted[:, 0] = 1.0  # always keep top-1

        if self.mode == "matched":
            high_dis = (disagreement > self.hesitancy_threshold).unsqueeze(1)
            mask_sorted = torch.where(high_dis, torch.ones_like(mask_sorted), mask_sorted)
        # pure mode: hesitancy_threshold به‌کل نادیده گرفته می‌شود

        mask = torch.zeros_like(weights)
        mask.scatter_(1, sorted_idx, mask_sorted)
        return mask

    def forward(self, x, return_details=False):
        weights, head_weights, disagreement = self.gating(x)
        mask = self._compute_mask(weights, disagreement)
        selected = weights * mask
        selected = selected / (selected.sum(dim=1, keepdim=True) + 1e-8)

        B = x.size(0)
        outputs = x.new_zeros((B, self.num_models, 1))
        active = torch.any(selected > 0, dim=0).nonzero(as_tuple=True)[0]
        for idx_t in active:
            idx = int(idx_t.item())
            x_n = self.normalizations(x, idx)
            with torch.no_grad():
                out = self.models[idx](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                out = out.reshape(B, -1)[:, :1]
            outputs[:, idx, :] = out

        final = (outputs * selected.unsqueeze(-1)).sum(dim=1)
        if return_details:
            return final, selected, disagreement, outputs, head_weights
        return final, selected


# ============================================================
# 5. Model loading (identical to HFDE)
# ============================================================

def load_pruned_models(model_paths, device):
    try:
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError as e:
        raise ImportError("Cannot import ResNet_50_pruned_hardfakevsreal. "
                          "Run from same project root as the fuzzy code.") from e

    models = []
    for i, path in enumerate(model_paths):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Not found: {path}")
        print(f"Loading base model {i+1}/{len(model_paths)}: {path}")
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        model = ResNet_50_pruned_hardfakevsreal(masks=ckpt["masks"])
        model.load_state_dict(ckpt["model_state_dict"])
        models.append(model.to(device).eval())
    return models


# ============================================================
# 6. Train / Evaluate
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, lbls in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True).float().view(-1)
        out, _ = model(imgs)
        logits = out.view(-1)
        total_loss += criterion(logits, lbls).item() * lbls.numel()
        correct += (logits > 0).long().eq(lbls.long()).sum().item()
        total += lbls.numel()
    if total == 0:
        raise RuntimeError("Empty val loader")
    return 100.0 * correct / total, total_loss / total


def train_ensemble(ensemble, train_loader, val_loader, num_epochs, lr,
                    device, save_dir, args, mode_name):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(ensemble.gating.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val, best_epoch = -1.0, -1
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\n" + "=" * 76)
    print(f"TRAINING — MODE: {mode_name.upper()}")
    print("=" * 76)
    print(f"Trainable params: {sum(p.numel() for p in ensemble.gating.parameters() if p.requires_grad):,}")
    print(f"Epochs={num_epochs} | LR={lr} | Batch={args.batch_size}")
    print(f"cum_weight_threshold = {ensemble.cum_weight_threshold}")
    if mode_name == "matched":
        print(f"hesitancy_threshold = {ensemble.hesitancy_threshold} (fallback ENABLED)")
        print("⚠ 'hesitancy' here = variance of softmax probs (ANALOGUE, not fuzzy)")
    else:
        print("hesitancy_threshold = N/A (fallback DISABLED — pure MoE)")

    for epoch in range(num_epochs):
        ensemble.train()
        run_loss, run_correct, seen = 0.0, 0, 0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True).float().view(-1)
            optimizer.zero_grad(set_to_none=True)
            out, _ = ensemble(imgs)
            logits = out.view(-1)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            n = lbls.numel()
            run_loss += loss.item() * n
            run_correct += (logits.detach() > 0).long().eq(lbls.long()).sum().item()
            seen += n

        train_loss = run_loss / seen
        train_acc = 100.0 * run_correct / seen
        val_acc, val_loss = evaluate(ensemble, val_loader, device, criterion)
        scheduler.step()

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["train_acc"].append(train_acc)
        hist["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1:03d}/{num_epochs} | "
              f"Train {train_loss:.4f}/{train_acc:.2f}% | "
              f"Val {val_loss:.4f}/{val_acc:.2f}%")

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch + 1
            torch.save({
                "gating_state_dict": ensemble.gating.state_dict(),
                "best_epoch": best_epoch, "best_val_accuracy": best_val,
                "mode": mode_name, "args": vars(args),
            }, os.path.join(save_dir, "best_gating.pt"))

    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    ep = np.arange(1, num_epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(ep, hist["train_loss"], label="Train")
    ax[0].plot(ep, hist["val_loss"], label="Val")
    ax[0].set_title(f"Loss ({mode_name})"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(ep, hist["train_acc"], label="Train")
    ax[1].plot(ep, hist["val_acc"], label="Val")
    ax[1].set_title(f"Accuracy ({mode_name})")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy (%)")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=300)
    plt.close(fig)

    return best_val, best_epoch


@torch.no_grad()
def evaluate_test(ensemble, test_loader, device, save_dir, model_names, mode_name):
    ensemble.eval()
    ds = test_loader.dataset
    y_true, y_score, y_pred, filenames, per_w = [], [], [], [], []

    for i in tqdm(range(len(ds)), desc=f"Test [{mode_name}]"):
        try:
            img, lbl = ds[i]
            orig_idx = ds.indices[i]
            path, _ = get_sample_info(ds.dataset, orig_idx)
            fname = os.path.basename(str(path))
        except Exception:
            img, lbl = ds[i]
            fname = f"sample_{i:06d}"

        img = img.unsqueeze(0).to(device)
        label = int(lbl.item()) if torch.is_tensor(lbl) else int(lbl)

        out, w, _, _, _ = ensemble(img, return_details=True)
        prob = torch.sigmoid(out.reshape(-1)[0]).item()
        pred = int(prob > 0.5)

        y_true.append(label)
        y_score.append(float(prob))
        y_pred.append(pred)
        filenames.append(fname)
        per_w.append(w.squeeze(0).cpu().tolist())

    yt = np.array(y_true, dtype=np.int64)
    ys = np.array(y_score, dtype=np.float64)
    yp = np.array(y_pred, dtype=np.int64)
    W = np.array(per_w, dtype=np.float64)

    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    n = len(yt)

    acc = 100.0 * (tp + tn) / n
    pr = tp / (tp + fp) if tp + fp else 0.0
    rr = tp / (tp + fn) if tp + fn else 0.0
    f1r = 2 * pr * rr / (pr + rr) if pr + rr else 0.0
    pf = tn / (tn + fn) if tn + fn else 0.0
    rf = tn / (tn + fp) if tn + fp else 0.0
    f1f = 2 * pf * rf / (pf + rf) if pf + rf else 0.0
    macro_f1 = (f1r + f1f) / 2
    bal_acc = 100.0 * (rr + rf) / 2

    active = (W > 1e-4).astype(np.int64)
    act_rate = active.mean(axis=0) * 100
    mean_w = W.mean(axis=0)
    act_per = active.sum(axis=1)

    print("\n" + "=" * 72)
    print(f"NON-FUZZY TEST — MODE: {mode_name.upper()}")
    print("=" * 72)
    print(f"Accuracy: {acc:.2f}% | Balanced: {bal_acc:.2f}% | Macro-F1: {macro_f1:.4f}")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"Active models/sample: mean={act_per.mean():.2f} median={np.median(act_per):.0f}")
    for i, name in enumerate(model_names):
        print(f"  {name:<30} act={act_rate[i]:6.2f}% avg_w={mean_w[i]:.4f}")

    # Save predictions (order matches HFDE's roc_data_test.json)
    pred_csv = os.path.join(save_dir, f"predictions_{mode_name}.csv")
    with open(pred_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "filename", "y_true", "y_score", "y_pred"])
        for i, (fn_, t, s, p) in enumerate(zip(filenames, y_true, y_score, y_pred)):
            w.writerow([i, fn_, t, f"{s:.10f}", p])

    np.savez_compressed(
        os.path.join(save_dir, f"roc_{mode_name}.npz"),
        y_true=yt, y_score=ys, y_pred=yp, filenames=np.array(filenames),
    )

    stats = {
        "mode": mode_name,
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "precision_real": float(pr), "recall_real": float(rr), "f1_real": float(f1r),
        "precision_fake": float(pf), "recall_fake": float(rf), "f1_fake": float(f1f),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "activation_rates": {nm: float(act_rate[i]) for i, nm in enumerate(model_names)},
        "mean_weights": {nm: float(mean_w[i]) for i, nm in enumerate(model_names)},
        "active_per_sample": {
            "min": int(act_per.min()), "max": int(act_per.max()),
            "mean": float(act_per.mean()), "median": float(np.median(act_per)),
        },
    }
    with open(os.path.join(save_dir, f"stats_{mode_name}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


# ============================================================
# 7. McNemar comparison
# ============================================================

def compare_mcnemar(fuzzy_json_path, nf_csv_path, save_path):
    if not HAS_SCIPY:
        print("Skipping McNemar (scipy not installed)")
        return None

    with open(fuzzy_json_path) as f:
        fj = json.load(f)
    y_true_f = np.array(fj["y_true"])
    y_pred_f = np.array(fj["y_pred"])

    y_pred_nf, y_true_nf = [], []
    with open(nf_csv_path) as f:
        for row in csv.DictReader(f):
            y_pred_nf.append(int(row["y_pred"]))
            y_true_nf.append(int(row["y_true"]))
    y_pred_nf = np.array(y_pred_nf)
    y_true_nf = np.array(y_true_nf)

    assert len(y_true_f) == len(y_true_nf), "Sample count mismatch"
    assert np.all(y_true_f == y_true_nf), "Label order mismatch — investigate!"

    c_f = (y_pred_f == y_true_f)
    c_nf = (y_pred_nf == y_true_nf)
    b = int((c_f & ~c_nf).sum())     # fuzzy right, nf wrong
    c = int((~c_f & c_nf).sum())     # fuzzy wrong, nf right
    both_r = int((c_f & c_nf).sum())
    both_w = int((~c_f & ~c_nf).sum())
    n_d = b + c
    p = binomtest(b, n_d, 0.5).pvalue if n_d > 0 else 1.0

    result = {
        "fuzzy_accuracy": float(100.0 * c_f.mean()),
        "nonfuzzy_accuracy": float(100.0 * c_nf.mean()),
        "difference_pp": float(100.0 * (c_f.mean() - c_nf.mean())),
        "contingency": {
            "both_correct": both_r, "both_wrong": both_w,
            "fuzzy_only_correct": b, "nonfuzzy_only_correct": c,
        },
        "mcnemar_exact_p": float(p),
        "significant_at_0.05": bool(p < 0.05),
        "n_samples": int(len(y_true_f)),
    }
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 72)
    print("McNEMAR: HFDE vs Non-Fuzzy")
    print("=" * 72)
    print(f"Fuzzy acc:     {result['fuzzy_accuracy']:.2f}%")
    print(f"Non-fuzzy acc: {result['nonfuzzy_accuracy']:.2f}%")
    print(f"Difference:    {result['difference_pp']:+.2f} pp")
    print(f"Contingency: both_correct={both_r}, both_wrong={both_w}, "
          f"fuzzy_only={b}, nf_only={c}")
    print(f"McNemar exact p = {p:.4f} "
          f"({'SIGNIFICANT' if p < 0.05 else 'NOT significant'})")
    print("=" * 72)
    return result


# ============================================================
# 8. Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--dataset", type=str, required=True,
                    choices=["wild", "artiface", "hard_fake_real", "deepfake_lab",
                             "uadfV", "custom_genai", "custom_genai_v2",
                             "real_fake_dataset", "20k"])
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--model_paths", type=str, nargs="+", required=True)
    p.add_argument("--model_names", type=str, nargs="+", required=True)
    p.add_argument("--save_dir", type=str, default="./ablation_non_fuzzy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_heads", type=int, default=3,
                    help="Must equal HFDE's --num_memberships")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--cum_weight_threshold", type=float, default=0.9)
    p.add_argument("--hesitancy_threshold", type=float, default=1e-4,
                    help="Only used in matched mode. Ignored in pure mode.")
    p.add_argument("--mode", type=str, default="both",
                    choices=["matched", "pure", "both"])
    p.add_argument("--fuzzy_pred_path", type=str, default=None,
                    help="HFDE roc_data_test.json for McNemar test")
    return p.parse_args()


def run_one_mode(args, mode_name, device, loaders, means, stds, model_names):
    train_loader, val_loader, test_loader = loaders
    mode_dir = os.path.join(args.save_dir, mode_name) if args.mode == "both" else args.save_dir
    os.makedirs(mode_dir, exist_ok=True)

    # Fresh models per run to avoid ModuleList shared-state issues
    base_models = load_pruned_models(args.model_paths, device)

    ensemble = NonFuzzyGatingEnsemble(
        models=base_models, means=means, stds=stds,
        num_heads=args.num_heads, dropout=args.dropout,
        freeze_models=True,
        cum_weight_threshold=args.cum_weight_threshold,
        hesitancy_threshold=args.hesitancy_threshold,
        mode=mode_name,
    ).to(device)

    best_val, best_epoch = train_ensemble(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, mode_dir, args, mode_name
    )
    ckpt = torch.load(os.path.join(mode_dir, "best_gating.pt"),
                       map_location=device, weights_only=False)
    ensemble.gating.load_state_dict(ckpt["gating_state_dict"])

    stats = evaluate_test(ensemble, test_loader, device, mode_dir, model_names, mode_name)
    stats["best_val_accuracy"] = float(best_val)
    stats["best_epoch"] = int(best_epoch)
    with open(os.path.join(mode_dir, "final_results.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    args = parse_args()
    assert len(args.model_names) == len(args.model_paths)

    set_seed(args.seed)
    device = setup_device()

    # Same normalization means/stds as HFDE
    dmeans = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    dstds  = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    M = len(args.model_paths)
    if M > len(dmeans):
        means = dmeans + [dmeans[-1]] * (M - len(dmeans))
        stds  = dstds  + [dstds[-1]]  * (M - len(dstds))
    else:
        means, stds = dmeans[:M], dstds[:M]

    loaders = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=False, seed=args.seed, is_main=True,
    )
    model_names = args.model_names[:M]
    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == "both":
        print("\n" + "=" * 76)
        print("RUNNING BOTH MODES (matched + pure)")
        print("=" * 76)
        s_matched = run_one_mode(args, "matched", device, loaders, means, stds, model_names)
        s_pure    = run_one_mode(args, "pure",    device, loaders, means, stds, model_names)

        print("\n" + "=" * 76)
        print("FINAL COMPARISON: matched vs pure")
        print("=" * 76)
        print(f"{'Metric':<30} {'matched':>12} {'pure':>12}")
        print(f"{'Accuracy':<30} {s_matched['accuracy']:>11.2f}% {s_pure['accuracy']:>11.2f}%")
        print(f"{'Balanced accuracy':<30} {s_matched['balanced_accuracy']:>11.2f}% {s_pure['balanced_accuracy']:>11.2f}%")
        print(f"{'Macro-F1':<30} {s_matched['macro_f1']:>12.4f} {s_pure['macro_f1']:>12.4f}")
        print(f"{'Active models/sample':<30} {s_matched['active_per_sample']['mean']:>12.2f} {s_pure['active_per_sample']['mean']:>12.2f}")

        with open(os.path.join(args.save_dir, "comparison_matched_vs_pure.json"), "w") as f:
            json.dump({"matched": s_matched, "pure": s_pure,
                       "diff_acc_pp": s_matched["accuracy"] - s_pure["accuracy"],
                       "diff_active": (s_matched["active_per_sample"]["mean"]
                                       - s_pure["active_per_sample"]["mean"])}, f, indent=2)

        if args.fuzzy_pred_path and os.path.exists(args.fuzzy_pred_path):
            for mode_name in ["matched", "pure"]:
                csv_path = os.path.join(args.save_dir, mode_name, f"predictions_{mode_name}.csv")
                if os.path.exists(csv_path):
                    compare_mcnemar(args.fuzzy_pred_path, csv_path,
                                    os.path.join(args.save_dir, f"mcnemar_fuzzy_vs_{mode_name}.json"))
    else:
        run_one_mode(args, args.mode, device, loaders, means, stds, model_names)
        if args.fuzzy_pred_path and os.path.exists(args.fuzzy_pred_path):
            csv_path = os.path.join(args.save_dir, f"predictions_{args.mode}.csv")
            compare_mcnemar(args.fuzzy_pred_path, csv_path,
                            os.path.join(args.save_dir, f"mcnemar_fuzzy_vs_{args.mode}.json"))

    print(f"\nAll done. Results in: {args.save_dir}")


if __name__ == "__main__":
    main()
