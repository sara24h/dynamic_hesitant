import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

# هشدارها را غیرفعال می‌کنیم
warnings.filterwarnings("ignore")

# ایمپورت کردن توابع کمکی از فایل‌های موجود
from dataset_utils import (
    create_dataloaders, 
    get_sample_info, 
    worker_init_fn
)
# فرض بر این است که این ماژول‌ها در مسیر پروژه شما موجود هستند
from metrics_utils import plot_roc_and_f1
from visualization_utils import generate_visualizations

# ================== UTILITY FUNCTIONS ==================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class MultiModelNormalization(nn.Module):
    """کلاس نرمال‌سازی برای هر مدل با میانگین و انحراف معیار مخصوص به خود"""
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


# ================== BMA MODEL CLASS ==================
class BayesianModelAveraging(nn.Module):
    """
    پیاده‌سازی Bayesian Model Averaging.
    وزن‌ها بر اساس Negative Log-Likelihood روی دیتای Validation محاسبه می‌شوند.
    """
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.normalization = MultiModelNormalization(means, stds)
        self.num_models = len(models)
        
        # ثبت وزن‌ها با مقدار اولیه یکنواخت
        self.register_buffer('bma_weights', torch.ones(self.num_models) / self.num_models)

        # فریز کردن پارامترهای مدل‌ها (اگر قبلاً نشده باشد)
        for model in self.models:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

    def compute_posterior_weights(self, val_loader: DataLoader, device: torch.device, is_main: bool):
        """
        محاسبه وزن‌های BMA:
        1. محاسبه مجموع Negative Log-Likelihood (NLL) برای هر مدل روی Validation Set.
        2. تبدیل NLL به Likelihood: L = exp(-NLL).
        3. نرمال‌سازی Likelihood‌ها برای به دست آوردن احتمال پسین (Posterior).
        """
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
        total_nll = torch.zeros(self.num_models, device=device)

        if is_main:
            print("="*70)
            print("BMA: Computing Posterior Weights (Validation NLL)")
            print("="*70)

        # محاسبه Loss برای هر مدل به صورت جداگانه
        for i, model in enumerate(self.models):
            model.eval()
            running_loss = 0.0
            
            # استفاده از no_grad چون مدل‌ها فریز هستند
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device).float()
                    
                    # نرمال‌سازی مخصوص مدل i
                    norm_images = self.normalization(images, i)
                    
                    outputs = model(norm_images)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                        
                    loss = criterion(outputs.squeeze(1), labels)
                    running_loss += loss.item()

            total_nll[i] = running_loss

        # همگام‌سازی Lossها در محیط Distributed (چون هر GPU ممکن است بخشی از دیتا را دیده باشد)
        if dist.is_initialized():
            dist.all_reduce(total_nll, op=dist.ReduceOp.SUM)
        
        # محاسبه وزن‌ها (فقط در Main Process برای جلوگیری از تکرار محاسبات عددی، سپس Broadcast)
        if is_main:
            # تبدیل NLL به Likelihood
            # برای پایداری عددی (Numerical Stability)، ماکسیمم را تفریق می‌کنیم
            max_nll = total_nll.max()
            likelihoods = torch.exp(-(total_nll - max_nll))
            
            # احتمال پسین (Posterior Weights)
            posterior_weights = likelihoods / likelihoods.sum()
            
            print(f"\nCalculated BMA Weights (Posterior Probabilities):")
            for i, w in enumerate(posterior_weights):
                print(f"  Model {i+1}: {w.item():.4f} ({w.item()*100:.2f}%)")
            print("="*70 + "\n")
            
            # ذخیره وزن‌ها
            self.bma_weights = posterior_weights.to(device)
        
        # پخش کردن وزن‌های محاسبه شده به سایر GPUها
        if dist.is_initialized():
            dist.broadcast(self.bma_weights, src=0)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        # ماتریس خروجی‌ها: (Batch, Num_Models, 1)
        all_outputs = torch.zeros(batch_size, self.num_models, 1, device=x.device)
        
        for i, model in enumerate(self.models):
            with torch.no_grad():
                x_norm = self.normalization(x, i)
                out = model(x_norm)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                all_outputs[:, i, 0] = out.squeeze(1)
        
        # میانگین‌گیری وزندار
        # w_expanded: (1, Num_Models, 1)
        w_expanded = self.bma_weights.view(1, -1, 1)
        
        # خروجی نهایی: (Batch, 1)
        final_output = (all_outputs * w_expanded).sum(dim=1)
        return final_output


# ================== MODEL LOADING ==================
def load_pruned_models(model_paths: List[str], device: torch.device, is_main: bool) -> List[nn.Module]:
    try:
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        raise ImportError("Cannot import ResNet_50_pruned_hardfakevsreal. Ensure 'model' directory exists.")

    models = []
    if is_main:
        print(f"Loading {len(model_paths)} pruned models...")

    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            if is_main:
                print(f" [WARNING] File not found: {path}")
            continue
        if is_main:
            print(f" [{i+1}/{len(model_paths)}] Loading: {os.path.basename(path)}")
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device).eval()
            param_count = sum(p.numel() for p in model.parameters())
            if is_main:
                print(f" → Parameters: {param_count:,}")
            models.append(model)
        except Exception as e:
            if is_main:
                print(f" [ERROR] Failed to load {path}: {e}")
            continue

    if len(models) == 0:
        raise ValueError("No models loaded!")
    if is_main:
        print(f"All {len(models)} models loaded!\n")
    return models


# ================== EVALUATION FUNCTIONS ==================
@torch.no_grad()
def evaluate_single_model_ddp(model: nn.Module, loader: DataLoader, device: torch.device,
                              name: str, mean: Tuple[float, float, float],
                              std: Tuple[float, float, float], is_main: bool) -> float:
    """ارزیابی تکی مدل‌ها برای مقایسه اولیه"""
    model.eval()
    normalizer = MultiModelNormalization([mean], [std]).to(device)
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc=f"Evaluating {name}", disable=not is_main):
        images, labels = images.to(device), labels.to(device).float()
        images = normalizer(images, 0)
        out = model(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = (out.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()

    correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
    total_tensor = torch.tensor(total, dtype=torch.long, device=device)
    if dist.is_initialized():
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
    acc = 100. * correct_tensor.item() / total_tensor.item()
    if is_main:
        print(f" {name}: {acc:.2f}%")
    return acc


@torch.no_grad()
def evaluate_bma_final(model, loader, device, name, model_names, is_main=True):
    """ارزیابی نهایی مدل BMA"""
    model.eval()
    total_correct = 0
    total_samples = 0

    if is_main:
        print(f"\nEvaluating BMA on {name} set...")
        
    for images, labels in tqdm(loader, desc=f"BMA Eval {name}", disable=not is_main):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        
        pred = (outputs.squeeze(1) > 0).long()
        total_correct += pred.eq(labels.long()).sum().item()
        total_samples += labels.size(0)
        
    # همگام‌سازی آمار در DDP
    stats = torch.tensor([total_correct, total_samples], dtype=torch.long, device=device)
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        
    if is_main:
        total_correct = stats[0].item()
        total_samples = stats[1].item()
        acc = 100. * total_correct / total_samples
        
        # دریافت وزن‌های نهایی
        final_weights = model.bma_weights.cpu().numpy()

        print(f"\n{'='*70}")
        print(f"BMA {name.upper()} SET RESULTS")
        print(f"{'='*70}")
        print(f" → Accuracy: {acc:.3f}%")
        print(f" → Total Samples: {total_samples:,}")
        print(f"\nBMA Weights (Posterior):")
        for i, (w, mname) in enumerate(zip(final_weights, model_names)):
            print(f" {i+1:2d}. {mname:<25}: {w:6.4f} ({w*100:5.2f}%)")
        print(f"{'='*70}")
        return acc, final_weights.tolist()
    
    return 0.0, [0.0]*len(model_names)


# ================== DISTRIBUTED SETUP ==================
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        if rank == 0:
            print(f"Distributed: rank {rank}/{world_size}, local_rank {local_rank}")
        return device, local_rank, rank, world_size
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# ================== MAIN FUNCTION ==================
def main():
    parser = argparse.ArgumentParser(description="Bayesian Model Averaging (BMA) for Frozen Models")
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['wild', 'real_fake', 'hard_fake_real', 'uadfV', 'custom_genai', 'custom_genai_v2', 'real_fake_dataset'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    
    # آرگومان‌های غیر فعال برای سازگاری (BMA نیاز به آموزش ندارد)
    parser.add_argument('--epochs', type=int, default=30, help='Ignored for BMA')
    parser.add_argument('--lr', type=float, default=0.0001, help='Ignored for BMA')
    
    args = parser.parse_args()

    if len(args.model_names) != len(args.model_paths):
        raise ValueError("Number of model_names must match model_paths")

    set_seed(args.seed)
    device, local_rank, rank, world_size = setup_distributed()
    is_main = rank == 0

    if is_main:
        print("="*70)
        print(f"BAYESIAN MODEL AVERAGING (BMA)")
        print(f"Distributed on {world_size} GPU(s) | Seed: {args.seed}")
        print("="*70)
        print(f"Dataset: {args.dataset}")
        print(f"Data directory: {args.data_dir}")
        print(f"Models: {len(args.model_paths)}")
        print("="*70 + "\n")

    # مدیریت MEANS و STDS
    DEFAULT_MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    DEFAULT_STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    
    num_models_to_load = len(args.model_paths)
    if num_models_to_load > len(DEFAULT_MEANS):
        last_mean = DEFAULT_MEANS[-1]
        last_std = DEFAULT_STDS[-1]
        MEANS = DEFAULT_MEANS + [last_mean] * (num_models_to_load - len(DEFAULT_MEANS))
        STDS = DEFAULT_STDS + [last_std] * (num_models_to_load - len(DEFAULT_STDS))
        if is_main:
            print(f"[Warning] Reusing last normalization stats for extra models.")
    else:
        MEANS = DEFAULT_MEANS[:num_models_to_load]
        STDS = DEFAULT_STDS[:num_models_to_load]

    # لود کردن مدل‌ها
    base_models = load_pruned_models(args.model_paths, device, is_main)
    MODEL_NAMES = args.model_names[:len(base_models)]

    # لود کردن دیتاست‌ها
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, num_workers=args.num_workers, dataset_type=args.dataset,
        is_distributed=(world_size > 1), seed=args.seed, is_main=is_main)

    if is_main:
        print("\n" + "="*70)
        print("INDIVIDUAL MODEL PERFORMANCE (Before BMA)")
        print("="*70)

    # ارزیابی تک‌تک مدل‌ها روی تست
    individual_accs = []
    for i, model in enumerate(base_models):
        acc = evaluate_single_model_ddp(
            model, test_loader, device,
            f"Model {i+1} ({MODEL_NAMES[i]})",
            MEANS[i], STDS[i], is_main)
        individual_accs.append(acc)

    best_single = max(individual_accs)
    best_idx = individual_accs.index(best_single)

    if is_main:
        print(f"\nBest Single: Model {best_idx+1} ({MODEL_NAMES[best_idx]}) → {best_single:.2f}%")
        print("="*70)

    # ================== BMA LOGIC ==================
    
    # 1. ایجاد مدل BMA
    bma_ensemble = BayesianModelAveraging(
        base_models, MEANS, STDS
    ).to(device)

    if world_size > 1:
        bma_ensemble = DDP(bma_ensemble, device_ids=[local_rank], output_device=local_rank)

    if is_main:
        total_params = sum(p.numel() for p in bma_ensemble.parameters())
        trainable_params = sum(p.numel() for p in bma_ensemble.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}\n")

    # 2. محاسبه وزن‌ها (جایگزین Training)
    # در این مرحله وزن‌ها بر اساس Validation Set محاسبه می‌شوند
    bma_module = bma_ensemble.module if hasattr(bma_ensemble, 'module') else bma_ensemble
    bma_module.compute_posterior_weights(val_loader, device, is_main)

    # 3. ارزیابی نهایی BMA روی Test Set
    if is_main:
        print("\n" + "="*70)
        print("FINAL BMA EVALUATION")
        print("="*70)

    ensemble_test_acc, ensemble_weights = evaluate_bma_final(
        bma_module, test_loader, device, "Test", MODEL_NAMES, is_main)

    if is_main:
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        print(f"Best Single Model: {best_single:.2f}%")
        print(f"BMA Accuracy:      {ensemble_test_acc:.2f}%")
        print(f"Improvement:       {ensemble_test_acc - best_single:+.2f}%")
        print("="*70)

        # ذخیره نتایج
        final_results = {
            'method': 'Bayesian Model Averaging (BMA)',
            'best_single_model': {
                'name': MODEL_NAMES[best_idx],
                'accuracy': float(best_single)
            },
            'bma_ensemble': {
                'test_accuracy': float(ensemble_test_acc),
                'model_weights': {name: float(w) for name, w in zip(MODEL_NAMES, ensemble_weights)},
            },
            'improvement': float(ensemble_test_acc - best_single)
        }

        results_path = os.path.join(args.save_dir, 'bma_final_results.json')
        os.makedirs(args.save_dir, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"\nResults saved: {results_path}")

        final_model_path = os.path.join(args.save_dir, 'bma_ensemble_model.pt')
        torch.save({
            'ensemble_state_dict': bma_module.state_dict(),
            'test_accuracy': ensemble_test_acc,
            'model_names': MODEL_NAMES,
            'means': MEANS,
            'stds': STDS,
            'bma_weights': ensemble_weights
        }, final_model_path)
        print(f"Model saved: {final_model_path}")

        # ویژوالایز کردن (اختیاری)
        vis_dir = os.path.join(args.save_dir, 'visualizations')
        try:
            generate_visualizations(
                bma_module, test_loader, device, vis_dir, MODEL_NAMES,
                5, 5, args.dataset, is_main)
            
            plot_roc_and_f1(
                bma_module,
                test_loader, 
                device, 
                args.save_dir, 
                MODEL_NAMES,
                is_main
            )
        except Exception as e:
            print(f"Visualization skipped due to error: {e}")

    cleanup_distributed()

if __name__ == "__main__":
    main()
