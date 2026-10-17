import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import json

from old_dataset_utils import create_dataloaders
from efficiency_utils import Stopwatch, benchmark_efficiency  # اگر ندارید، این خط را حذف کنید


# =====================================================================
# تابع اصلی ارزیابی (بدون تغییر)
# =====================================================================
def evaluate_model2_only(
    model2: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    mean: tuple,
    std: tuple,
    save_dir: str = "./model2_only_results",
):
    """
    فقط Model 2 را ارزیابی می‌کند (بدون هیچ ensemble).
    """
    os.makedirs(save_dir, exist_ok=True)
    model2.eval()

    # نرمال‌سازی مخصوص Model 2
    mean_t = torch.tensor(mean).view(1, 3, 1, 1).to(device)
    std_t = torch.tensor(std).view(1, 3, 1, 1).to(device)

    all_y_true, all_y_score, all_y_pred = [], [], []
    TP = TN = FP = FN = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Model-2-only Eval"):
            images = images.to(device)
            labels = labels.to(device).long()

            # نرمال‌سازی
            images_norm = (images - mean_t) / std_t

            logits = model2(images_norm)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            probs = torch.sigmoid(logits.squeeze(1))
            preds = (probs > 0.5).long()

            all_y_true.extend(labels.cpu().numpy().tolist())
            all_y_score.extend(probs.cpu().numpy().tolist())
            all_y_pred.extend(preds.cpu().numpy().tolist())

            for t, p in zip(labels, preds):
                if t == 1:
                    if p == 1:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if p == 1:
                        FP += 1
                    else:
                        TN += 1

    total = len(all_y_true)
    acc = 100.0 * (TP + TN) / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        "method": "Model-2-only",
        "accuracy": float(acc),
        "precision_real": float(precision),
        "recall_real": float(recall),
        "f1_real": float(f1),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "num_samples": total,
    }

    print("\n" + "=" * 60)
    print("MODEL-2-ONLY RESULTS")
    print("=" * 60)
    print(f"Accuracy  : {acc:.2f}%")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("=" * 60)

    with open(os.path.join(save_dir, "model2_only_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ذخیره برای ROC
    np.savez(
        os.path.join(save_dir, "roc_data_model2.npz"),
        y_true=np.array(all_y_true),
        y_score=np.array(all_y_score),
        y_pred=np.array(all_y_pred),
    )

    return results


# =====================================================================
# ابزار پیدا کردن خودکار مسیر دیتاست
# =====================================================================
def find_dataset_dir(preferred_name="hard_fake_real",
                     search_roots=("/kaggle/input", "."),
                     max_depth=6):
    """
    دنبال پوشه‌ای می‌گردد که یکی از این دو ساختار را داشته باشد:
        root/fake  +  root/real
        root/<preferred_name>/fake  +  root/<preferred_name>/real
    لیستی از مسیرهای کاندید برمی‌گرداند (ممکن است خالی باشد).
    """
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

            # حالت ۱: خودِ root شامل fake و real
            if "fake" in dirs_lower and "real" in dirs_lower:
                candidates.append(root)

            # حالت ۲: root/<preferred_name>/{fake,real}
            if preferred_name in dirs_lower:
                sub = os.path.join(root, preferred_name)
                try:
                    sub_dirs = [
                        d.lower()
                        for d in os.listdir(sub)
                        if os.path.isdir(os.path.join(sub, d))
                    ]
                except PermissionError:
                    continue
                if "fake" in sub_dirs and "real" in sub_dirs:
                    candidates.append(root)

    # حذف تکراری‌ها با حفظ ترتیب
    return list(dict.fromkeys(candidates))


def _print_kaggle_input_tree():
    """در صورت شکست، ساختار /kaggle/input را برای دیباگ چاپ می‌کند."""
    print("\n[Debug] Listing /kaggle/input (depth ≤ 3):")
    if not os.path.exists("/kaggle/input"):
        print("  /kaggle/input does not exist.")
        return
    for p in sorted(os.listdir("/kaggle/input")):
        full = os.path.join("/kaggle/input", p)
        print(f"  - {p}")
        if os.path.isdir(full):
            for sub in sorted(os.listdir(full))[:8]:
                sub_full = os.path.join(full, sub)
                suffix = "/" if os.path.isdir(sub_full) else ""
                print(f"      └─ {sub}{suffix}")


# =====================================================================
# اجرای اصلی
# =====================================================================
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Using device: {device}")

    # ------------------------------------------------------------------
    # گام ۱: پیدا کردن مسیر دیتاست
    # اگر مسیر واقعی را می‌دانی، مستقیم اینجا مقدار بده و Auto-detect را رد کن.
    # ------------------------------------------------------------------
    DATASET_PATH = None  # ← مثلاً "/kaggle/input/my-dataset"

    if DATASET_PATH is None:
        candidates = find_dataset_dir("hard_fake_real")
        if candidates:
            print("\n[Auto-detect] Dataset candidate(s) found:")
            for i, c in enumerate(candidates):
                print(f"  [{i}] {c}")
            DATASET_PATH = candidates[0]
            print(f"[Auto-detect] Using: {DATASET_PATH}\n")
        else:
            _print_kaggle_input_tree()
            raise FileNotFoundError(
                "دیتاستی با ساختار fake/real پیدا نشد.\n"
                "1) مطمئن شو دیتاست به نوتبوک اضافه شده (Add Data).\n"
                "2) در صورت نیاز DATASET_PATH را در همین فایل دستی مقدار بده.\n"
                "3) ساختار پوشه‌ها باید به یکی از این دو شکل باشد:\n"
                "     base_dir/fake , base_dir/real\n"
                "     base_dir/hard_fake_real/fake , base_dir/hard_fake_real/real"
            )

    # ------------------------------------------------------------------
    # گام ۲: ساخت DataLoader
    # ------------------------------------------------------------------
    _, _, test_loader = create_dataloaders(
        base_dir=DATASET_PATH,
        batch_size=32,
        num_workers=2,                 # در Kaggle/Linux معمولاً 2 یا 4 مناسب است
        dataset_type="hard_fake_real",
        is_distributed=False,
        seed=42,
    )


    if "base_models" not in globals():
        raise NameError(
            "base_models تعریف نشده است. باید لیست مدل‌های بارگذاری‌شده را قبل از "
            "این بلاک بسازی. همچنین MEANS و STDS را تعریف کن."
        )
    if "MEANS" not in globals() or "STDS" not in globals():
        raise NameError(
            "MEANS و/یا STDS تعریف نشده‌اند. این‌ها باید لیست تاپل‌های mean/std "
            "برای هر مدل باشند (بر اساس ترتیب base_models)."
        )

    model2 = base_models[1]
    mean2 = MEANS[1]
    std2 = STDS[1]

    # ------------------------------------------------------------------
    # گام ۴: ارزیابی
    # ------------------------------------------------------------------
    results_m2 = evaluate_model2_only(
        model2=model2,
        test_loader=test_loader,
        device=device,
        mean=mean2,
        std=std2,
        save_dir="./ablation_model2_only",
    )

    print("\n[Done] Results saved to ./ablation_model2_only/")
    print(json.dumps(results_m2, indent=2))
