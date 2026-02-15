import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
import warnings
import argparse
import json
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from metrics_utils import plot_roc_and_f1
warnings.filterwarnings("ignore")

from dataset_utils import (
    UADFVDataset, CustomGenAIDataset, NewGenAIDataset,
    create_dataloaders, get_sample_info, worker_init_fn
)
from visualization_utils import GradCAM, generate_lime_explanation, generate_visualizations

# ================== UTILITY FUNCTIONS ==================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# ================== UNIFIED FINAL EVALUATION ==================
def final_evaluation_unified(model, test_loader_full, device, save_dir, model_names, is_main):
    """
    این تابع همه کارهای ارزیابی نهایی را در یک مرحله انجام می‌دهد تا اعداد قطعا یکسان باشند.
    1. پیش‌بینی روی کل دیتاست (تست اصلی).
    2. چاپ نتایج در کنسول.
    3. ذخیره لاگ مک‌نمار.
    """
    if not is_main: return 0.0

    model.eval()
    
    # استخراج دیتاست پایه برای نام فایل‌ها
    base_dataset = test_loader_full.dataset
    if hasattr(base_dataset, 'dataset'):
        base_dataset = base_dataset.dataset
    
    # استخراج اندیس‌های تست
    if hasattr(test_loader_full, 'sampler') and hasattr(test_loader_full.sampler, 'indices'):
        test_indices = test_loader_full.sampler.indices
    elif hasattr(test_loader_full.dataset, 'indices'):
        test_indices = test_loader_full.dataset.indices
    else:
        # اگر اندیس نبود، فرض بر این است که کل دیتاست تست است
        test_indices = list(range(len(base_dataset)))

    all_preds = []
    all_labels = []
    
    lines = []
    lines.append("="*100)
    lines.append("SAMPLE-BY-SAMPLE PREDICTIONS")
    lines.append("="*100)
    header = f"{'ID':<10} {'Filename':<60} {'True':<10} {'Pred':<10} {'Correct':<10}"
    lines.append(header)
    lines.append("-"*100)

    TP, TN, FP, FN = 0, 0, 0, 0
    
    print(f"\nRunning Unified Final Evaluation on {len(test_indices)} samples...")
    
    with torch.no_grad():
        for i, global_idx in enumerate(tqdm(test_indices, desc="Final Eval")):
            try:
                # گرفتن داده مستقیم از دیتاست پایه با ایندکس سراسری
                image, label = base_dataset[global_idx]
                path, _ = get_sample_info(base_dataset, global_idx)
            except Exception as e:
                print(f"Error loading idx {global_idx}: {e}")
                continue

            image = image.unsqueeze(0).to(device)
            label_int = int(label)
            
            # اینفرنس مدل
            output = model(image)
            if isinstance(output, (tuple, list)): output = output[0]
            prob = torch.sigmoid(output.squeeze()).item()
            pred_int = int(prob > 0.5)
            
            # جمع‌آوری برای متریک‌ها
            all_preds.append(prob)
            all_labels.append(label_int)
            
            # محاسبه ماتریس آشفتگی
            if label_int == 1: # Real
                if pred_int == 1: TP += 1
                else: FN += 1
            else: # Fake
                if pred_int == 1: FP += 1
                else: TN += 1
            
            # خط لاگ
            correct_str = "Yes" if pred_int == label_int else "No"
            filename = os.path.basename(path)
            if len(filename) > 55: filename = filename[:25] + "..." + filename[-27:]
            line = f"{i+1:<10} {filename:<60} {label_int:<10} {pred_int:<10} {correct_str:<10}"
            lines.append(line)

    # --- محاسبه آمار نهایی ---
    total_samples = len(all_labels)
    correct_count = TP + TN
    acc = 100.0 * correct_count / total_samples if total_samples > 0 else 0.0
    
    # --- چاپ نتایج در کنسول ---
    print(f"\n{'='*70}")
    print(f"FINAL TEST SET RESULTS (Unified)")
    print(f"{'='*70}")
    print(f" → Accuracy: {acc:.3f}%")
    print(f" → Total Samples: {total_samples}")
    print(f" → Correct: {correct_count}")
    print(f" → Confusion Matrix: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
    print(f"{'='*70}")

    # --- ذخیره فایل لاگ ---
    output_str = []
    output_str.append("-" * 100)
    output_str.append("SUMMARY STATISTICS:")
    output_str.append("-" * 100)
    output_str.append(f"Accuracy: {acc:.2f}%")
    output_str.append(f"Correct: {correct_count} / {total_samples}")
    output_str.append("\nConfusion Matrix:")
    output_str.append(f"                 {'Pred Real':<15} {'Pred Fake':<15}")
    output_str.append(f"    Actual Real   {TP:<15} {FN:<15}")
    output_str.append(f"    Actual Fake   {FP:<15} {TN:<15}")
    output_str.extend(lines)
    
    log_path = os.path.join(save_dir, 'prediction_log.txt')
    with open(log_path, 'w') as f:
        f.write("\n".join(output_str))
    print(f"✅ Prediction log saved to: {log_path}")

    # --- ذخیره نتایج برای ROC ---
    # بازگرداندن آرایه‌های numpy برای استفاده در تابع ROC
    return np.array(all_labels), np.array(all_preds), acc

# ================== MODEL CLASSES ==================
# (کلاس‌های مدل دقیقاً مانند قبل بدون تغییر)
class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')

class HesitantFuzzyMembership(nn.Module):
    def __init__(self, input_dim: int, num_models: int, num_memberships: int = 3, dropout: float = 0.3):
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
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]],
                 stds: List[Tuple[float]], num_memberships: int = 3, freeze_models: bool = True,
                 cum_weight_threshold: float = 0.9, hesitancy_threshold: float = 0.2):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.hesitant_fuzzy = HesitantFuzzyMembership(
            input_dim=128, num_models=self.num_models, num_memberships=num_memberships)
        self.cum_weight_threshold = cum_weight_threshold
        self.hesitancy_threshold = hesitancy_threshold

        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

    def _compute_mask_vectorized(self, final_weights: torch.Tensor, avg_hesitancy: torch.Tensor):
        batch_size = final_weights.size(0)
        sorted_weights, sorted_indices = torch.sort(final_weights, dim=1, descending=True)
        cum_weights = torch.cumsum(sorted_weights, dim=1)
        mask = (cum_weights <= self.cum_weight_threshold).float()
        mask[:, 0] = 1.0
        high_hesitancy_mask = (avg_hesitancy > self.hesitancy_threshold).unsqueeze(1)
        mask = torch.where(high_hesitancy_mask, torch.ones_like(mask), mask)
        final_mask = torch.zeros_like(final_weights)
        final_mask.scatter_(1, sorted_indices, mask)
        return final_mask

    def forward(self, x: torch.Tensor, return_details: bool = False):
        final_weights, all_memberships = self.hesitant_fuzzy(x)
        hesitancy = all_memberships.var(dim=2)
        avg_hesitancy = hesitancy.mean(dim=1)
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

# ================== TRAIN & EVAL FUNCTIONS ==================
# (توابع آموزش و ارزیابی حین آموزش برای اختصار حذف نشدند اما تغییری ندارند)
# ... (توابع load_pruned_models, evaluate_accuracy_ddp, train_hesitant_fuzzy دقیقاً مانند قبل هستند) ...

def load_pruned_models(model_paths: List[str], device: torch.device, is_main: bool) -> List[nn.Module]:
    try:
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        raise ImportError("Cannot import ResNet_50_pruned_hardfakevsreal")
    models = []
    if is_main: print(f"Loading {len(model_paths)} pruned models...")
    for i, path in enumerate(model_paths):
        if not os.path.exists(path): continue
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device).eval()
            models.append(model)
        except: continue
    if len(models) == 0: raise ValueError("No models loaded!")
    return models

def evaluate_accuracy_ddp(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs, _ = model(images)
        pred = (outputs.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()
    
    correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
    # نکته: در ارزیابی حین آموزش هم بهتر است از طول واقعی استفاده کرد
    # اما چون اینجا برای سلکت کردن بهترین مدل است، دقت تقریبی مشکلی ندارد.
    # برای صحت بیشتر:
    real_total = len(loader.dataset)
    
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    acc = 100. * correct_tensor.item() / real_total
    return acc

def train_hesitant_fuzzy(ensemble_model, train_loader, val_loader, num_epochs, lr,
                        device, save_dir, is_main, model_names):
    os.makedirs(save_dir, exist_ok=True)
    hesitant_net = ensemble_model.module.hesitant_fuzzy if hasattr(ensemble_model, 'module') else ensemble_model.hesitant_fuzzy
    optimizer = torch.optim.AdamW(hesitant_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        if hasattr(train_loader.sampler, 'set_epoch'): train_loader.sampler.set_epoch(epoch)
        ensemble_model.train()
        train_loss = 0.0; train_correct = 0; train_total = 0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}', disable=not is_main):
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs, _, _, _ = ensemble_model(images, return_details=True)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward(); optimizer.step()
            train_loss += loss.item() * images.size(0)
            pred = (outputs.squeeze(1) > 0).long()
            train_correct += pred.eq(labels.long()).sum().item()
            train_total += images.size(0)
        
        train_acc = 100. * train_correct / train_total
        val_acc = evaluate_accuracy_ddp(ensemble_model, val_loader, device)
        scheduler.step()
        
        if is_main:
            print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({'hesitant_state_dict': hesitant_net.state_dict()}, os.path.join(save_dir, 'best_hesitant_fuzzy.pt'))
                print(f"✓ Saved Best Model")
    return best_val_acc, history

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        return device, local_rank, rank, world_size
    else:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized(): dist.destroy_process_group()

# ================== MAIN FUNCTION ==================
def main():
    parser = argparse.ArgumentParser(description="Optimized Fuzzy Hesitant Ensemble Training")
    # آرگومان‌ها
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device, local_rank, rank, world_size = setup_distributed()
    is_main = rank == 0

    # نرمالیزیشن
    DEFAULT_MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    DEFAULT_STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    MEANS = DEFAULT_MEANS[:len(args.model_paths)]
    STDS = DEFAULT_STDS[:len(args.model_paths)]

    base_models = load_pruned_models(args.model_paths, device, is_main)
    MODEL_NAMES = args.model_names[:len(base_models)]

    ensemble = FuzzyHesitantEnsemble(base_models, MEANS, STDS).to(device)
    if world_size > 1:
        ensemble = DDP(ensemble, device_ids=[local_rank], output_device=local_rank)

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=(world_size > 1), seed=args.seed, is_main=is_main)

    best_val_acc, history = train_hesitant_fuzzy(
        ensemble, train_loader, val_loader, args.epochs, args.lr, device, args.save_dir, is_main, MODEL_NAMES)

    # لود بهترین مدل
    ensemble_module = ensemble.module if hasattr(ensemble, 'module') else ensemble
    ckpt_path = os.path.join(args.save_dir, 'best_hesitant_fuzzy.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        ensemble_module.hesitant_fuzzy.load_state_dict(ckpt['hesitant_state_dict'])

    if is_main:
        print("\nPreparing full test loader for final reporting...")
        # ساخت دیتالودر تست غیرتوزیع‌شده (حیاتی برای یکسان بودن نتایج)
        _, _, test_loader_full = create_dataloaders(
            args.data_dir, args.batch_size, dataset_type=args.dataset,
            is_distributed=False, seed=args.seed, is_main=True
        )

        # ===== اجرای ارزیابی یکپارچه =====
        # این تابع هم دقت را چاپ می‌کند، هم لاگ را ذخیره می‌کند
        y_true, y_scores, final_acc = final_evaluation_unified(
            ensemble_module, test_loader_full, device, args.save_dir, MODEL_NAMES, is_main
        )

        # ===== ذخیره JSON نهایی =====
        final_results = {
            'ensemble_accuracy': final_acc,
            'best_val_accuracy': float(best_val_acc)
        }
        with open(os.path.join(args.save_dir, 'final_results.json'), 'w') as f:
            json.dump(final_results, f, indent=4)

        # ===== ذخیره مدل نهایی =====
        torch.save({
            'ensemble_state_dict': ensemble_module.state_dict(),
            'model_names': MODEL_NAMES,
            'means': MEANS,
            'stds': STDS
        }, os.path.join(args.save_dir, 'final_ensemble_model.pt'))

        # ===== ویژوالایزیشن =====
        vis_dir = os.path.join(args.save_dir, 'visualizations')
        generate_visualizations(
            ensemble_module, test_loader_full, device, vis_dir, MODEL_NAMES,
            args.num_grad_cam_samples if hasattr(args, 'num_grad_cam_samples') else 5, 
            args.num_lime_samples if hasattr(args, 'num_lime_samples') else 5,
            args.dataset, is_main)

    cleanup_distributed()
    
    # رسم ROC بعد از Cleanup (چون تک‌پردازشی است)
    if is_main:
        plot_roc_and_f1(
            ensemble_module,
            test_loader_full, 
            device, 
            args.save_dir, 
            MODEL_NAMES,
            is_main
        )

if __name__ == "__main__":
    main()
