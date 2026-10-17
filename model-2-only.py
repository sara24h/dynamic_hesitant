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
    """دقیقاً مطابق load_pruned_models فایل اصلی، ولی فقط یک چک‌پوینت."""
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
    TP = TN = FP = FN = 0
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

            if label_int == 1:
                if pred_int == 1:
                    TP += 1
                else:
                    FN += 1
            else:
                if pred_int == 1:
                    FP += 1
                else:
                    TN += 1

            correct_str = "Yes" if pred_int == label_int else "No"
            filename = os.path.basename(path)
            if len(filename) > 55:
                filename = filename[:25] + "..." + filename[-27:]
            lines.append(f"{i+1:<10} {filename:<60} {label_int:<10} {pred_int:<10} {correct_str:<10}")

    total = len(all_y_true)
    correct_count = TP + TN
    acc = 100.0 * correct_count / total if total > 0 else 0.0

    precision_real = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall_real = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_real = (2 * precision_real * recall_real / (precision_real + recall_real)
               if (precision_real + recall_real) > 0 else 0.0)

    precision_fake = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    recall_fake = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1_fake = (2 * precision_fake * recall_fake / (precision_fake + recall_fake)
               if (precision_fake + recall_fake) > 0 else 0.0)

    macro_precision = (precision_real + precision_fake) / 2
    macro_recall = (recall_real + recall_fake) / 2
    macro_f1 = (f1_real + f1_fake) / 2
    balanced_acc = ((recall_real + recall_fake) / 2) * 100

    # ---------- Confusion Matrix ----------
    cm = np.array([[TN, FP], [FN, TP]])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Fake", "Predicted Real"],
        yticklabels=["Actual Fake", "Actual Real"],
        cbar=True, annot_kws={"size": 16},
    )
    plt.title(
        f"Model 2 Only ({model_name})\n"
        f"Acc: {acc:.2f}% | Bal Acc: {balanced_acc:.2f}% | Macro-F1: {macro_f1:.4f}",
        fontsize=13, fontweight="bold",
    )
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Confusion Matrix saved: {cm_path}")

    # ---------- Console summary ----------
    print(f"\n{'='*70}")
    print(f"MODEL 2 ONLY — FINAL RESULTS ({model_name})")
    print(f"{'='*70}")
    print(f"Accuracy          : {acc:.2f}%")
    print(f"Balanced Accuracy : {balanced_acc:.2f}%")
    print(f"Macro-F1          : {macro_f1:.4f}")
    print()
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 45)
    print(f"{'Fake (0)':<12} {precision_fake:10.4f} {recall_fake:10.4f} {f1_fake:10.4f}")
    print(f"{'Real (1)':<12} {precision_real:10.4f} {recall_real:10.4f} {f1_real:10.4f}")
    print("-" * 45)
    print(f"{'Macro Avg':<12} {macro_precision:10.4f} {macro_recall:10.4f} {macro_f1:10.4f}")
    print(f"\nTP={TP} TN={TN} FP={FP} FN={FN} | Total={total}")
    print("=" * 70)

    # ---------- results JSON ----------
    results = {
        "method": "Model-2-only",
        "model_name": model_name,
        "dataset": dataset_name,
        "seed": seed,
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "precision_fake": float(precision_fake),
        "recall_fake": float(recall_fake),
        "f1_fake": float(f1_fake),
        "precision_real": float(precision_real),
        "recall_real": float(recall_real),
        "f1_real": float(f1_real),
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
        "num_samples": int(total),
    }
    with open(os.path.join(save_dir, "model2_only_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("✅ results JSON saved")

    # ---------- prediction log ----------
    summary_lines = [
        "-" * 100, "SUMMARY STATISTICS:", "-" * 100,
        f"Accuracy          : {acc:.2f}%",
        f"Balanced Accuracy : {balanced_acc:.2f}%",
        f"Macro F1          : {macro_f1:.4f}",
        f"Real  P/R/F1 : {precision_real:.4f} / {recall_real:.4f} / {f1_real:.4f}",
        f"Fake  P/R/F1 : {precision_fake:.4f} / {recall_fake:.4f} / {f1_fake:.4f}",
    ]
    summary_lines.extend(lines)
    log_path = os.path.join(save_dir, "prediction_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"✅ Prediction log saved: {log_path}")

    # ---------- ROC data ----------
    y_true_np = np.array(all_y_true)
    y_score_np = np.array(all_y_score)
    y_pred_np = np.array(all_y_pred)

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


# ================== EFFICIENCY BENCHMARK (single model) ==================
def benchmark_single_model(model, test_loader, device, mean, std):
  
    model.eval()
    # warm-up
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

    # ---------- sanity ----------
    if not (0 <= args.model_index < len(args.model_paths)):
        raise ValueError(f"--model_index={args.model_index} خارج از محدوده است")
    if len(args.model_names) != len(args.model_paths):
        print("[Warn] تعداد model_names با model_paths برابر نیست؛ از پیش‌فرض استفاده می‌شود.")
        while len(args.model_names) < len(args.model_paths):
            args.model_names.append(f"Model_{len(args.model_names)+1}")

    set_seed(args.seed)
    device = setup_device()
    os.makedirs(args.save_dir, exist_ok=True)

    # ---------- normalization (همان DEFAULTهای فایل اصلی) ----------
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

    # ---------- data_dir ----------
    data_dir = args.data_dir
    if data_dir is None:
        candidates = find_dataset_dir(args.dataset)
        if not candidates:
            raise FileNotFoundError(
                "مسیر دیتاست پیدا نشد. با --data_dir مسیر را دستی بده."
            )
        data_dir = candidates[0]
        print(f"[Auto-detect] Using data_dir: {data_dir}")

    # ---------- dataloaders ----------
    _, _, test_loader = create_dataloaders(
        data_dir, args.batch_size, num_workers=args.num_workers,
        dataset_type=args.dataset, is_distributed=False,
        seed=args.seed, is_main=True,
    )

    # ---------- load Model 2 ----------
    model2 = load_single_pruned_model(args.model_paths[idx], device)
    n_params = sum(p.numel() for p in model2.parameters())
    print(f"Model 2 parameters: {n_params:,}")

    # ---------- efficiency ----------
    eff = benchmark_single_model(model2, test_loader, device, mean, std)
    with open(os.path.join(args.save_dir, "efficiency_report.json"), "w") as f:
        json.dump(eff, f, indent=2)

    # ---------- evaluation ----------
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
