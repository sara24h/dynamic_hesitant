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
from efficiency_utils import Stopwatch


# ---------- کلاس‌های مورد نیاز از کد اصلی ----------
class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


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


# ---------- Fixed Ensemble ----------
class FixedWeightEnsemble(nn.Module):
    def __init__(
        self,
        models: List[nn.Module],
        means: List[Tuple[float]],
        stds: List[Tuple[float]],
        fixed_weights: List[float],
        device: str = "cuda"
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.normalizations = MultiModelNormalization(means, stds)

        weights = torch.tensor(fixed_weights, dtype=torch.float32)
        weights = weights / weights.sum()
        self.register_buffer("fixed_weights", weights)

        for model in self.models:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        outputs = torch.zeros(batch_size, self.num_models, device=x.device)

        for i in range(self.num_models):
            x_n = self.normalizations(x, i)
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            outputs[:, i] = out.squeeze(1)

        final_logit = (outputs * self.fixed_weights.unsqueeze(0)).sum(dim=1)
        return final_logit


# ---------- متریک‌های کمکی ----------
def _safe_div(a, b):
    return float(a) / float(b) if b > 0 else 0.0


def compute_class_metrics(y_true, y_pred, cls):
    """متریک‌ها را برای یک کلاس مشخص (cls به عنوان positive) حساب می‌کند."""
    tp = int(((y_true == cls) & (y_pred == cls)).sum())
    tn = int(((y_true != cls) & (y_pred != cls)).sum())
    fp = int(((y_true != cls) & (y_pred == cls)).sum())
    fn = int(((y_true == cls) & (y_pred != cls)).sum())

    precision = _safe_div(tp, tp + fp)
    recall    = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)          # True Negative Rate (برای این کلاس)
    f1        = _safe_div(2 * precision * recall, precision + recall)

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,                 # = sensitivity (TPR) برای این کلاس
        "specificity": specificity,       # TNR برای این کلاس
        "f1": f1,
        "support": tp + fn,
    }


# ---------- ارزیابی ----------
@torch.no_grad()
def evaluate_fixed(ensemble, test_loader, device, save_dir, model_names):
    os.makedirs(save_dir, exist_ok=True)
    ensemble.eval()

    all_y_true, all_y_score, all_y_pred = [], [], []

    print(f"\nEvaluating Fixed Ensemble on {len(test_loader.dataset)} samples...")

    for images, labels in tqdm(test_loader, desc="Fixed Ensemble"):
        images = images.to(device)
        labels = labels.to(device).long()

        logits = ensemble(images)
        probs  = torch.sigmoid(logits)
        preds  = (probs > 0.5).long()

        all_y_true.extend(labels.cpu().numpy().tolist())
        all_y_score.extend(probs.cpu().numpy().tolist())
        all_y_pred.extend(preds.cpu().numpy().tolist())

    y_true  = np.array(all_y_true,  dtype=np.int64)
    y_score = np.array(all_y_score, dtype=np.float32)
    y_pred  = np.array(all_y_pred,  dtype=np.int64)
    total   = len(y_true)

    # ----- متریک‌های هر کلاس -----
    # قرارداد کد اصلی: 0 = fake (negative)   |   1 = real (positive)
    m_fake = compute_class_metrics(y_true, y_pred, cls=0)
    m_real = compute_class_metrics(y_true, y_pred, cls=1)

    # ----- Overall -----
    acc = 100.0 * float((y_true == y_pred).sum()) / total

    # ----- Macro -----
    macro_precision = (m_fake["precision"] + m_real["precision"]) / 2.0
    macro_recall    = (m_fake["recall"]    + m_real["recall"])    / 2.0
    macro_f1        = (m_fake["f1"]        + m_real["f1"])        / 2.0

    # ----- Balanced Accuracy = (recall_fake + recall_real) / 2 -----
    balanced_acc = (m_fake["recall"] + m_real["recall"]) / 2.0
    balanced_acc_pct = 100.0 * balanced_acc

    # ----- Weighted (اختیاری، بر اساس support) -----
    weighted_precision = _safe_div(
        m_fake["precision"] * m_fake["support"] + m_real["precision"] * m_real["support"], total)
    weighted_recall = _safe_div(
        m_fake["recall"] * m_fake["support"] + m_real["recall"] * m_real["support"], total)
    weighted_f1 = _safe_div(
        m_fake["f1"] * m_fake["support"] + m_real["f1"] * m_real["support"], total)

    # ----- ROC-AUC -----
    roc_auc = None
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        roc_auc = float(roc_auc_score(y_true, y_score))
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_curve_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    except ImportError:
        roc_curve_data = None

    results = {
        "method": "Fixed_Ensemble_Mean_HFDE_Weights",
        "fixed_weights": ensemble.fixed_weights.cpu().tolist(),
        "model_names": model_names,

        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc_pct),

        # --- Per-class (0 = fake, 1 = real طبق کد اصلی) ---
        "class_0_fake": m_fake,
        "class_1_real": m_real,

        # --- Macro ---
        "macro_precision": float(macro_precision),
        "macro_recall":    float(macro_recall),
        "macro_f1":        float(macro_f1),

        # --- Weighted ---
        "weighted_precision": float(weighted_precision),
        "weighted_recall":    float(weighted_recall),
        "weighted_f1":        float(weighted_f1),

        # --- AUC ---
        "roc_auc": roc_auc,

        "num_samples": int(total),
        "num_samples_per_class": {
            "fake": int(m_fake["support"]),
            "real": int(m_real["support"]),
        },
        # ماتریس درهم‌ریختگی برای شفافیت
        "confusion_matrix": {
            "TP_real": m_real["tp"], "TN_real": m_real["tn"],
            "FP_real": m_real["fp"], "FN_real": m_real["fn"],
        },
    }

    # ---------- چاپ ----------
    print("\n" + "=" * 72)
    print("FIXED ENSEMBLE (Mean HFDE Weights) RESULTS")
    print("=" * 72)
    print(f"Fixed weights : {[round(w, 6) for w in results['fixed_weights']]}")
    print(f"Samples       : {total}  (fake={m_fake['support']}, real={m_real['support']})")
    print("-" * 72)
    print(f"{'Metric':<20}{'Fake (0)':>15}{'Real (1)':>15}{'Macro Avg':>15}")
    print("-" * 72)
    print(f"{'Precision':<20}{m_fake['precision']:>15.4f}{m_real['precision']:>15.4f}{macro_precision:>15.4f}")
    print(f"{'Recall (TPR)':<20}{m_fake['recall']:>15.4f}{m_real['recall']:>15.4f}{macro_recall:>15.4f}")
    print(f"{'Specificity':<20}{m_fake['specificity']:>15.4f}{m_real['specificity']:>15.4f}{'-':>15}")
    print(f"{'F1-Score':<20}{m_fake['f1']:>15.4f}{m_real['f1']:>15.4f}{macro_f1:>15.4f}")
    print(f"{'Support':<20}{m_fake['support']:>15d}{m_real['support']:>15d}{total:>15d}")
    print("-" * 72)
    print(f"{'Accuracy':<20}: {acc:.2f}%")
    print(f"{'Balanced Accuracy':<20}: {balanced_acc_pct:.2f}%")
    print(f"{'Weighted Precision':<20}: {weighted_precision:.4f}")
    print(f"{'Weighted Recall':<20}: {weighted_recall:.4f}")
    print(f"{'Weighted F1':<20}: {weighted_f1:.4f}")
    if roc_auc is not None:
        print(f"{'ROC-AUC':<20}: {roc_auc:.4f}")
    print("=" * 72)

    # ---------- ذخیره ----------
    with open(os.path.join(save_dir, "fixed_mean_weights_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    save_dict = {
        "y_true": y_true,
        "y_score": y_score,
        "y_pred": y_pred,
    }
    if roc_curve_data is not None:
        save_dict["fpr"] = np.array(roc_curve_data["fpr"])
        save_dict["tpr"] = np.array(roc_curve_data["tpr"])
    np.savez(os.path.join(save_dir, "roc_data_fixed.npz"), **save_dict)

    print(f"\nنتایج ذخیره شد در: {save_dir}")
    return results


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Fixed Ensemble with mean HFDE weights")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./ablation_fixed_mean_weights')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--fixed_weights', type=float, nargs='+',
                        default=[0.130522, 0.775484, 0.093994],
                        help="Mean weights from HFDE final evaluation")

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    DEFAULT_MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    DEFAULT_STDS  = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]

    num_models = len(args.model_paths)
    MEANS = DEFAULT_MEANS[:num_models]
    STDS  = DEFAULT_STDS[:num_models]

    base_models = load_pruned_models(args.model_paths, device)
    MODEL_NAMES = args.model_names[:len(base_models)]

    fixed_ensemble = FixedWeightEnsemble(
        models=base_models,
        means=MEANS,
        stds=STDS,
        fixed_weights=args.fixed_weights,
        device=device
    ).to(device)

    print(f"\nFixed weights being used: {args.fixed_weights}")

    _, _, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=False, seed=args.seed, is_main=True
    )

    results = evaluate_fixed(
        ensemble=fixed_ensemble,
        test_loader=test_loader,
        device=device,
        save_dir=args.save_dir,
        model_names=MODEL_NAMES
    )

if __name__ == "__main__":
    main()
