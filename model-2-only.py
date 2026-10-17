import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
import argparse
import json
import random
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from old_dataset_utils import create_dataloaders, get_sample_info
from efficiency_utils import Stopwatch, benchmark_efficiency, print_headline

try:
    from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
except ImportError as e:
    raise ImportError(
        "Cannot import ResNet_50_pruned_hardfakevsreal from model.ResNet_pruned"
    ) from e


# ================== UTILS ==================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available, using CPU")
    return device


def load_single_pruned_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ResNet_50_pruned_hardfakevsreal(masks=ckpt["masks"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model


def normalize(x, mean, std, device):
    mean_t = torch.tensor(mean).view(1, 3, 1, 1).to(device)
    std_t = torch.tensor(std).view(1, 3, 1, 1).to(device)
    return (x - mean_t) / std_t


# ================== METRIC HELPERS ==================
def _safe_div(a, b):
    return float(a) / float(b) if b > 0 else 0.0


def _class_metrics(tp, fp, fn, tn):
    """
    متریک‌های یک کلاس را وقتی همان کلاس 'positive' است حساب می‌کند.
    - Precision = TP / (TP + FP)
    - Recall (Sensitivity/TPR) = TP / (TP + FN)
    - Specificity (TNR)         = TN / (TN + FP)
    - F1                        = 2PR / (P+R)
    """
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    s = _safe_div(tn, tn + fp)
    f = _safe_div(2 * p * r, p + r)
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "support": int(tp + fn),
        "precision": p,
        "recall": r,
        "specificity": s,
        "f1": f,
    }


def compute_all_metrics(y_true, y_score, y_pred, threshold=0.5):
    """
    همه متریک‌ها را با قرارداد:
        0 = fake (negative in code)   |   1 = real (positive in code)
    خروجی: یک دیکشنری جامع که هم برای چاپ و هم برای JSON مناسب است.
    """
    y_true  = np.asarray(y_true,  dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float32)
    y_pred  = np.asarray(y_pred,  dtype=np.int64)

    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    total = TP + TN + FP + FN

    # ---- per-class ----
    real_m = _class_metrics(TP, FP, FN, TN)          # positive = real
    fake_m = _class_metrics(TN, FN, FP, TP)          # positive = fake (TN↔TP, FN↔FP)

    # ---- overall ----
    accuracy = _safe_div(TP + TN, total)

    # ---- macro ----
    macro_p  = (real_m["precision"] + fake_m["precision"]) / 2
    macro_r  = (real_m["recall"]    + fake_m["recall"])    / 2
    macro_f1 = (real_m["f1"]        + fake_m["f1"])        / 2

    # ---- weighted (by support) ----
    s_real = real_m["support"]
    s_fake = fake_m["support"]
    weighted_p  = _safe_div(real_m["precision"] * s_real + fake_m["precision"] * s_fake, total)
    weighted_r  = _safe_div(real_m["recall"]    * s_real + fake_m["recall"]    * s_fake, total)
    weighted_f1 = _safe_div(real_m["f1"]        * s_real + fake_m["f1"]        * s_fake, total)

    # ---- Balanced Accuracy = (recall_real + recall_fake) / 2 ----
    balanced_acc = (real_m["recall"] + fake_m["recall"]) / 2

    # ---- ROC-AUC (اختیاری) ----
    roc_auc = None
    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = float(roc_auc_score(y_true, y_score))
    except ImportError:
        pass

    return {
        "confusion": {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
        "num_samples": total,
        "num_per_class": {"fake": s_fake, "real": s_real},

        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),

        "class_0_fake": fake_m,
        "class_1_real": real_m,

        "macro_precision": float(macro_p),
        "macro_recall":    float(macro_r),
        "macro_f1":        float(macro_f1),

        "weighted_precision": float(weighted_p),
        "weighted_recall":    float(weighted_r),
        "weighted_f1":        float(weighted_f1),

        "roc_auc": roc_auc,
        "threshold": float(threshold),
    }


def _print_metric_table(metrics, title):
    cm = metrics["confusion"]
    fr = metrics["class_0_fake"]
    rl = metrics["class_1_real"]

    print(f"\n{'='*76}")
    print(title)
    print('='*76)
    print(f"Samples: {metrics['num_samples']}  "
          f"(fake={metrics['num_per_class']['fake']}, real={metrics['num_per_class']['real']})")
    print('-'*76)
    print(f"{'Metric':<20}{'Fake (0)':>15}{'Real (1)':>15}{'Macro Avg':>15}")
    print('-'*76)
    print(f"{'Precision':<20}{fr['precision']:>15.4f}{rl['precision']:>15.4f}{metrics['macro_precision']:>15.4f}")
    print(f"{'Recall (TPR)':<20}{fr['recall']:>15.4f}{rl['recall']:>15.4f}{metrics['macro_recall']:>15.4f}")
    print(f"{'Specificity (TNR)':<20}{fr['specificity']:>15.4f}{rl['specificity']:>15.4f}{'-':>15}")
    print(f"{'F1-Score':<20}{fr['f1']:>15.4f}{rl['f1']:>15.4f}{metrics['macro_f1']:>15.4f}")
    print(f"{'Support':<20}{fr['support']:>15d}{rl['support']:>15d}{metrics['num_samples']:>15d}")
    print('-'*76)
    print(f"{'Accuracy':<20}: {metrics['accuracy']*100:.2f}%")
    print(f"{'Balanced Accuracy':<20}: {metrics['balanced_accuracy']*100:.2f}%")
    print(f"{'Weighted Precision':<20}: {metrics['weighted_precision']:.4f}")
    print(f"{'Weighted Recall':<20}: {metrics['weighted_recall']:.4f}")
    print(f"{'Weighted F1':<20}: {metrics['weighted_f1']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"{'ROC-AUC':<20}: {metrics['roc_auc']:.4f}")
    print('='*76)
    print(f"Confusion: TP={cm['TP']}  TN={cm['TN']}  FP={cm['FP']}  FN={cm['FN']}")
    print('='*76)


# ================== AUTO-DETECT DATASET ==================
def find_dataset_dir(preferred_name="hard_fake_real",
                     search_roots=("/kaggle/input", "."),
                     max_depth=6):
    candidates = []
    for search_root in search_roots:
        if not os.path.exists(search_root):
            continue
        for root, dirs, _ in os.walk(search_root):
            depth = root[len(search_root):].count(os.sep)
            if depth > max_depth:
                dirs[:] = []
                continue
            dirs_lower = [d.lower() for d in dirs]
            if "fake" in dirs_lower and "real" in dirs_lower:
                candidates.append(root)
            if preferred_name in dirs_lower:
                sub = os.path.join(root, preferred_name)
                try:
                    sub_dirs = [d.lower() for d in os.listdir(sub)
                                if os.path.isdir(os.path.join(sub, d))]
                except PermissionError:
                    continue
                if "fake" in sub_dirs and "real" in sub_dirs:
                    candidates.append(root)
    return list(dict.fromkeys(candidates))


# ================== FULL EVALUATION (Model 2 only) ==================
def evaluate_model2_single(model, test_loader, device, mean, std, save_dir, model_name,
                           seed=42, dataset_name="unknown"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    test_dataset = test_loader.dataset
    num_samples = len(test_dataset)

    all_y_true, all_y_score, all_y_pred = [], [], []
    lines = []
    lines.append("=" * 100)
    lines.append(f"MODEL 2 ONLY — SAMPLE-BY-SAMPLE PREDICTIONS ({model_name})")
    lines.append("=" * 100)
    lines.append(f"{'ID':<10} {'Filename':<60} {'True':<10} {'Pred':<10} {'Correct':<10}")
    lines.append("-" * 100)

    print(f"\nRunning Model-2-only evaluation on {num_samples} samples ...")
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Model-2 Eval"):
            try:
                image, label = test_dataset[i]
                original_idx = test_dataset.indices[i]
                path, _ = get_sample_info(test_dataset.dataset, original_idx)
            except Exception as e:
                print(f"Error loading sample {i}: {e}")
                continue

            image = image.unsqueeze(0).to(device)
            label_int = int(label)

            image_n = normalize(image, mean, std, device)
            out = model(image_n)
            if isinstance(out, (tuple, list)):
                out = out[0]

            logit = out.squeeze().item()
            prob = float(torch.sigmoid(torch.tensor(logit)).item())
            pred_int = int(prob > 0.5)

            all_y_true.append(label_int)
            all_y_score.append(prob)
            all_y_pred.append(pred_int)

            correct_str = "Yes" if pred_int == label_int else "No"
            filename = os.path.basename(path)
            if len(filename) > 55:
                filename = filename[:25] + "..." + filename[-27:]
            lines.append(f"{i+1:<10} {filename:<60} {label_int:<10} {pred_int:<10} {correct_str:<10}")

    y_true_np  = np.array(all_y_true,  dtype=np.int64)
    y_score_np = np.array(all_y_score, dtype=np.float32)
    y_pred_np  = np.array(all_y_pred,  dtype=np.int64)

    # ============ همه متریک‌ها در یک جا ============
    metrics = compute_all_metrics(y_true_np, y_score_np, y_pred_np, threshold=0.5)

    acc            = metrics["accuracy"] * 100.0
    balanced_acc   = metrics["balanced_accuracy"] * 100.0
    macro_f1       = metrics["macro_f1"]
    macro_precision = metrics["macro_precision"]
    macro_recall    = metrics["macro_recall"]
    precision_fake = metrics["class_0_fake"]["precision"]
    recall_fake    = metrics["class_0_fake"]["recall"]
    f1_fake        = metrics["class_0_fake"]["f1"]
    precision_real = metrics["class_1_real"]["precision"]
    recall_real    = metrics["class_1_real"]["recall"]
    f1_real        = metrics["class_1_real"]["f1"]
    TP = metrics["confusion"]["TP"]
    TN = metrics["confusion"]["TN"]
    FP = metrics["confusion"]["FP"]
    FN = metrics["confusion"]["FN"]
    total = metrics["num_samples"]
    roc_auc = metrics["roc_auc"]

    # ---------- Confusion Matrix Figure ----------
    cm = np.array([[TN, FP], [FN, TP]])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Fake", "Predicted Real"],
        yticklabels=["Actual Fake", "Actual Real"],
        cbar=True, annot_kws={"size": 16},
    )
    title = (
        f"Model 2 Only ({model_name})\n"
        f"Acc: {acc:.2f}% | Bal Acc: {balanced_acc:.2f}% | Macro-F1: {macro_f1:.4f}"
    )
    if roc_auc is not None:
        title += f" | AUC: {roc_auc:.4f}"
    plt.title(title, fontsize=12, fontweight="bold")
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Confusion Matrix saved: {cm_path}")

    # ---------- Console summary ----------
    _print_metric_table(metrics, f"MODEL 2 ONLY — FINAL RESULTS ({model_name})")

    # ---------- results JSON ----------
    results = {
        "method": "Model-2-only",
        "model_name": model_name,
        "dataset": dataset_name,
        "seed": seed,

        # overall
        "accuracy": metrics["accuracy"] * 100.0,
        "balanced_accuracy": metrics["balanced_accuracy"] * 100.0,

        # macro
        "macro_precision": macro_precision,
        "macro_recall":    macro_recall,
        "macro_f1":        macro_f1,

        # weighted
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall":    metrics["weighted_recall"],
        "weighted_f1":        metrics["weighted_f1"],

        # per-class
        "precision_fake": precision_fake,
        "recall_fake":    recall_fake,
        "specificity_fake": metrics["class_0_fake"]["specificity"],
        "f1_fake":        f1_fake,
        "support_fake":   metrics["class_0_fake"]["support"],

        "precision_real": precision_real,
        "recall_real":    recall_real,
        "specificity_real": metrics["class_1_real"]["specificity"],
        "f1_real":        f1_real,
        "support_real":   metrics["class_1_real"]["support"],

        # raw confusion
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
        "num_samples": int(total),

        # AUC
        "roc_auc": roc_auc,
        "decision_threshold": 0.5,
    }
    with open(os.path.join(save_dir, "model2_only_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("✅ results JSON saved")

    # ---------- prediction log ----------
    summary_lines = [
        "-" * 100, "SUMMARY STATISTICS:", "-" * 100,
        f"Accuracy          : {acc:.2f}%",
        f"Balanced Accuracy : {balanced_acc:.2f}%",
        f"Macro Precision   : {macro_precision:.4f}",
        f"Macro Recall      : {macro_recall:.4f}",
        f"Macro F1          : {macro_f1:.4f}",
        f"Real  P/R/F1 : {precision_real:.4f} / {recall_real:.4f} / {f1_real:.4f} "
        f"(support={metrics['class_1_real']['support']})",
        f"Fake  P/R/F1 : {precision_fake:.4f} / {recall_fake:.4f} / {f1_fake:.4f} "
        f"(support={metrics['class_0_fake']['support']})",
    ]
    if roc_auc is not None:
        summary_lines.append(f"ROC-AUC     : {roc_auc:.4f}")
    summary_lines.extend(lines)
    log_path = os.path.join(save_dir, "prediction_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"✅ Prediction log saved: {log_path}")

    # ---------- ROC data ----------
    roc_json_path = os.path.join(save_dir, "roc_data_test.json")
    with open(roc_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "model": "Model-2-only",
                "model_name": model_name,
                "dataset": dataset_name,
                "seed": seed,
                "num_samples": int(total),
                "positive_count": int(y_true_np.sum()),
                "negative_count": int(total - y_true_np.sum()),
                "roc_auc": roc_auc,
            },
            "y_true": y_true_np.tolist(),
            "y_score": y_score_np.tolist(),
            "y_pred": y_pred_np.tolist(),
        }, f, indent=2, ensure_ascii=False)
    print(f"✅ ROC data (JSON) saved: {roc_json_path}")

    roc_txt_path = os.path.join(save_dir, "roc_data_test.txt")
    with open(roc_txt_path, "w", encoding="utf-8") as f:
        f.write("y_true\ty_score\ty_pred\n")
        for t, s, p in zip(y_true_np, y_score_np, y_pred_np):
            f.write(f"{int(t)}\t{s:.6f}\t{int(p)}\n")
    print(f"✅ ROC data (TXT) saved: {roc_txt_path}")

    return results


# ================== EFFICIENCY BENCHMARK ==================
def benchmark_single_model(model, test_loader, device, mean, std):
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 256, 256).to(device)
        _ = model(normalize(dummy, mean, std, device))
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    n_samples, n_batches = 0, 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            _ = model(normalize(images, mean, std, device))
            n_samples += images.size(0)
            n_batches += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    report = {
        "model": "Model-2-only",
        "device": str(device),
        "num_samples": int(n_samples),
        "num_batches": int(n_batches),
        "total_time_s": float(elapsed),
        "latency_per_sample_ms": float(1000.0 * elapsed / max(n_samples, 1)),
        "throughput_samples_per_s": float(n_samples / max(elapsed, 1e-9)),
    }
    print(f"\n[Efficiency] Latency: {report['latency_per_sample_ms']:.3f} ms/sample | "
          f"Throughput: {report['throughput_samples_per_s']:.2f} samples/s")
    return report


# ================== MAIN ==================
def main():
    parser = argparse.ArgumentParser(
        description="Model-2-only ablation evaluation (single model from the ensemble)")
    parser.add_argument("--model_paths", type=str, nargs="+", required=True,
                        help="لیست همه چک‌پوینت‌ها (فقط ایندکس --model_index استفاده می‌شود)")
    parser.add_argument("--model_names", type=str, nargs="+", required=True,
                        help="لیست نام مدل‌ها به همان ترتیب model_paths")
    parser.add_argument("--model_index", type=int, default=1,
                        help="ایندکس مدل مورد ارزیابی (پیش‌فرض: 1 = Model 2)")
    parser.add_argument("--dataset", type=str, default="hard_fake_real",
                        choices=["wild", "artiface", "hard_fake_real", "deepfake_lab",
                                 "uadfV", "custom_genai", "custom_genai_v2",
                                 "real_fake_dataset", "20k"])
    parser.add_argument("--data_dir", type=str, default=None,
                        help="اگر داده نشود، خودکار در /kaggle/input جست‌وجو می‌شود")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="./ablation_model2_only")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0 <= args.model_index < len(args.model_paths)):
        raise ValueError(f"--model_index={args.model_index} خارج از محدوده است")
    if len(args.model_names) != len(args.model_paths):
        print("[Warn] تعداد model_names با model_paths برابر نیست؛ از پیش‌فرض استفاده می‌شود.")
        while len(args.model_names) < len(args.model_paths):
            args.model_names.append(f"Model_{len(args.model_names)+1}")

    set_seed(args.seed)
    device = setup_device()
    os.makedirs(args.save_dir, exist_ok=True)

    DEFAULT_MEANS = [(0.5207, 0.4258, 0.3806),
                     (0.4460, 0.3622, 0.3416),
                     (0.4668, 0.3816, 0.3414)]
    DEFAULT_STDS = [(0.2490, 0.2239, 0.2212),
                    (0.2057, 0.1849, 0.1761),
                    (0.2410, 0.2161, 0.2081)]
    idx = args.model_index
    mean = DEFAULT_MEANS[idx] if idx < len(DEFAULT_MEANS) else DEFAULT_MEANS[-1]
    std = DEFAULT_STDS[idx] if idx < len(DEFAULT_STDS) else DEFAULT_STDS[-1]
    model_name = args.model_names[idx]

    print("\n" + "=" * 70)
    print("MODEL-2-ONLY ABLATION STUDY")
    print("=" * 70)
    print(f"Model index : {idx}  →  {model_name}")
    print(f"Checkpoint  : {args.model_paths[idx]}")
    print(f"Dataset     : {args.dataset}")
    print(f"Mean / Std  : {mean} / {std}")
    print("=" * 70 + "\n")

    data_dir = args.data_dir
    if data_dir is None:
        candidates = find_dataset_dir(args.dataset)
        if not candidates:
            raise FileNotFoundError(
                "مسیر دیتاست پیدا نشد. با --data_dir مسیر را دستی بده."
            )
        data_dir = candidates[0]
        print(f"[Auto-detect] Using data_dir: {data_dir}")

    _, _, test_loader = create_dataloaders(
        data_dir, args.batch_size, num_workers=args.num_workers,
        dataset_type=args.dataset, is_distributed=False,
        seed=args.seed, is_main=True,
    )

    model2 = load_single_pruned_model(args.model_paths[idx], device)
    n_params = sum(p.numel() for p in model2.parameters())
    print(f"Model 2 parameters: {n_params:,}")

    eff = benchmark_single_model(model2, test_loader, device, mean, std)
    with open(os.path.join(args.save_dir, "efficiency_report.json"), "w") as f:
        json.dump(eff, f, indent=2)

    sw = Stopwatch(device).start()
    results = evaluate_model2_single(
        model2, test_loader, device, mean, std,
        save_dir=args.save_dir, model_name=model_name,
        seed=args.seed, dataset_name=args.dataset,
    )
    eval_time = sw.stop()
    results["final_evaluation_time_s"] = float(eval_time)
    with open(os.path.join(args.save_dir, "model2_only_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation time: {eval_time:.2f}s")
    print(f"[Done] All outputs saved under: {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()
