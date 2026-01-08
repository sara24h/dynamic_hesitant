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
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP
from metrics_utils import plot_roc_and_f1

# Import LIME visualization utilities
from visualization_utils import generate_lime_explanation

# Imports for manual LIME saving (fallback)
import matplotlib.pyplot as plt
try:
    from skimage.segmentation import mark_boundaries
except ImportError:
    mark_boundaries = None

warnings.filterwarnings("ignore")

from dataset_utils import (
    UADFVDataset, 
    create_dataloaders, 
    get_sample_info, 
    worker_init_fn,
    DFDDataset
)

# Imports for Grad-CAM
try:
    from pytorch_grad_cam import GradCAM as PytorchGradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    USE_PYTORCH_GRADCAM = True
except ImportError:
    USE_PYTORCH_GRADCAM = False
    print("Warning: pytorch-grad-cam not available, skipping Grad-CAM visualizations")

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
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
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


# ================== MODEL LOADING ==================
def load_pruned_models(model_paths: List[str], device: torch.device, is_main: bool) -> List[nn.Module]:
    try:
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        raise ImportError("Cannot import ResNet_50_pruned_hardfakevsreal")

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

    if dist.is_initialized():
        correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
        total_tensor = torch.tensor(total, dtype=torch.long, device=device)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        correct = correct_tensor.item()
        total = total_tensor.item()
    
    acc = 100. * correct / total
    if is_main:
        print(f" {name}: {acc:.2f}%")
    return acc


@torch.no_grad()
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

    if dist.is_initialized():
        correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
        total_tensor = torch.tensor(total, dtype=torch.long, device=device)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        correct = correct_tensor.item()
        total = total_tensor.item()
    
    acc = 100. * correct / total
    return acc


@torch.no_grad()
def evaluate_ensemble_final_ddp(model, loader, device, name, model_names, is_main=True):
    model.eval()
    total_correct = 0
    total_samples = 0
    sum_weights = torch.zeros(len(model_names), device=device)
    sum_activation = torch.zeros(len(model_names), device=device)
    sum_membership_vals = torch.zeros(len(model_names), 3, device=device)
    sum_hesitancy = torch.zeros(len(model_names), device=device)

    if is_main:
        print(f"\nEvaluating {name} set...")
    for images, labels in tqdm(loader, desc=f"Evaluating {name}", disable=not is_main):
        images, labels = images.to(device), labels.to(device)
        outputs, weights, memberships, _ = model(images, return_details=True)
        pred = (outputs.squeeze(1) > 0).long()
        total_correct += pred.eq(labels.long()).sum().item()
        total_samples += labels.size(0)
        sum_weights += weights.sum(dim=0)
        sum_activation += (weights > 1e-4).sum(dim=0).float()
        sum_membership_vals += memberships.sum(dim=0)
        sum_hesitancy += memberships.var(dim=2).sum(dim=0)

    if dist.is_initialized():
        stats = torch.tensor([total_correct, total_samples], dtype=torch.long, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_weights, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_activation, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_membership_vals, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_hesitancy, op=dist.ReduceOp.SUM)
        total_correct = stats[0].item()
        total_samples = stats[1].item()

    if is_main:
        acc = 100. * total_correct / total_samples
        avg_weights = (sum_weights / total_samples).cpu().numpy()
        activation_percentages = (sum_activation / total_samples * 100).cpu().numpy()
        avg_membership = (sum_membership_vals / total_samples).cpu().numpy()
        avg_hesitancy = (sum_hesitancy / total_samples).cpu().numpy()

        print(f"\n{'='*70}")
        print(f"{name.upper()} SET RESULTS")
        print(f"{'='*70}")
        print(f" → Accuracy: {acc:.3f}%")
        print(f" → Total Samples: {total_samples:,}")
        print(f"\nAverage Model Weights:")
        for i, (w, mname) in enumerate(zip(avg_weights, model_names)):
            print(f" {i+1:2d}. {mname:<25}: {w:6.4f} ({w*100:5.2f}%)")
        print(f"\nActivation Frequency:")
        for i, (perc, mname) in enumerate(zip(activation_percentages, model_names)):
            print(f" {i+1:2d}. {mname:<25}: {perc:6.2f}% active")
        print(f"\nHesitant Membership Values:")
        for i, mname in enumerate(model_names):
            mu_str = ", ".join([f"{v:.3f}" for v in avg_membership[i]])
            print(f" {i+1:2d}. {mname:<25}: μ = [{mu_str}] | Hesitancy = {avg_hesitancy[i]:.4f}")
        print(f"{'='*70}")
        return acc, avg_weights.tolist(), activation_percentages.tolist()
    return 0.0, [0.0]*len(model_names), [0.0]*len(model_names)


def train_hesitant_fuzzy(ensemble_model, train_loader, val_loader, num_epochs, lr,
                        device, save_dir, is_main, model_names):
    """
    تابع اصلاح‌شده training با نمایش جزئیات کامل
    """
    os.makedirs(save_dir, exist_ok=True)
    hesitant_net = ensemble_model.module.hesitant_fuzzy if hasattr(ensemble_model, 'module') else ensemble_model.hesitant_fuzzy
    optimizer = torch.optim.AdamW(hesitant_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    history = {
        'train_loss': [], 
        'train_acc': [], 
        'val_acc': [], 
        'membership_variance': [],
        'per_model_hesitancy': [], 
        'cumsum_usage': [], 
        'model_activation': [] 
    }

    if is_main:
        print("="*70)
        print("Training Fuzzy Hesitant Network")
        print("="*70)
        print(f"Trainable params: {sum(p.numel() for p in hesitant_net.parameters()):,}")
        print(f"Epochs: {num_epochs} | Initial LR: {lr}")
        print(f"Hesitant memberships per model: {hesitant_net.num_memberships}")
        print(f"Number of models: {len(model_names)}\n")

    for epoch in range(num_epochs):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        ensemble_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        sum_per_model_hesitancy = torch.zeros(len(model_names), device=device)
        sum_cumsum_used = 0
        sum_active_models = torch.zeros(len(model_names), device=device)
        num_batches = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', disable=not is_main):
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            
            outputs, weights, memberships, _ = ensemble_model(images, return_details=True)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            pred = (outputs.squeeze(1) > 0).long()
            train_correct += pred.eq(labels.long()).sum().item()
            train_total += batch_size
            
            per_model_hesitancy = memberships.var(dim=2)
            sum_per_model_hesitancy += per_model_hesitancy.sum(dim=0)
            
            active_mask = (weights > 1e-4).float()
            num_active_per_sample = active_mask.sum(dim=1)
            cumsum_used_samples = (num_active_per_sample < len(model_names)).sum().item()
            sum_cumsum_used += cumsum_used_samples
            
            sum_active_models += active_mask.sum(dim=0)
            num_batches += 1

        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / train_total
        
        avg_per_model_hesitancy = sum_per_model_hesitancy / train_total
        avg_cumsum_usage = (sum_cumsum_used / train_total) * 100
        avg_model_activation = (sum_active_models / train_total) * 100
        overall_mean_hesitancy = avg_per_model_hesitancy.mean().item()
        
        val_acc = evaluate_accuracy_ddp(ensemble_model, val_loader, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['membership_variance'].append(overall_mean_hesitancy)
        history['per_model_hesitancy'].append(avg_per_model_hesitancy.cpu().tolist())
        history['cumsum_usage'].append(avg_cumsum_usage)
        history['model_activation'].append(avg_model_activation.cpu().tolist())

        if is_main:
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*70}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            print(f"\n{'Hesitancy (Variance) per Model:':^70}")
            print(f"{'-'*70}")
            for i, name in enumerate(model_names):
                hesitancy_val = avg_per_model_hesitancy[i].item()
                print(f"  {i+1:2d}. {name:<30}: {hesitancy_val:.6f}")
            print(f"{'-'*70}")
            print(f"  {'Overall Mean Hesitancy:':<30}  {overall_mean_hesitancy:.6f}")
            
            print(f"\n{'Cumulative Weight Threshold (Cumsum) Analysis:':^70}")
            print(f"{'-'*70}")
            print(f"  {'Cumsum activated in:':<30}  {avg_cumsum_usage:.2f}% of samples")
            if avg_cumsum_usage > 50:
                print(f"  {'Status:':<30}  ✓ Efficient")
            elif avg_cumsum_usage > 20:
                print(f"  {'Status:':<30}  ≈ Moderate")
            else:
                print(f"  {'Status:':<30}  ✗ Low efficiency")
            print(f"  {'All models used in:':<30}  {100-avg_cumsum_usage:.2f}% of samples")
            
            print(f"\n{'Model Activation Frequency:':^70}")
            print(f"{'-'*70}")
            for i, name in enumerate(model_names):
                activation_pct = avg_model_activation[i].item()
                bar_length = int(activation_pct / 2)
                bar = '█' * bar_length
                print(f"  {i+1:2d}. {name:<30}: {activation_pct:5.1f}% {bar}")

        if is_main and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, 'best_hesitant_fuzzy.pt')
            torch.save({
                'epoch': epoch + 1,
                'hesitant_state_dict': hesitant_net.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, save_path)
            print(f"\n✓ Best model saved → {val_acc:.2f}%")

        if is_main:
            print()

    if is_main:
        print(f"\n{'='*70}")
        print(f"Training Completed!")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"{'='*70}\n")
    
    return best_val_acc, history


# ================== VISUALIZATION FUNCTIONS ==================
def generate_visualizations_for_all_models(ensemble_module, base_models, test_loader, device, 
                                          vis_dir, MODEL_NAMES, MEANS, STDS,
                                          num_gradcam_samples, num_lime_samples, 
                                          dataset_type, is_main):
    """
    تابع جدید برای تولید visualizations برای ensemble و همه مدل‌های تکی
    """
    if not is_main:
        return
    
    os.makedirs(vis_dir, exist_ok=True)
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS FOR ALL MODELS")
    print("="*70)
    
    # انتخاب نمونه‌های مشترک
    all_images = []
    all_labels = []
    all_indices = []
    
    for idx, (images, labels) in enumerate(test_loader):
        all_images.append(images)
        all_labels.append(labels)
        all_indices.extend(range(idx * test_loader.batch_size, 
                                idx * test_loader.batch_size + len(images)))
        if len(all_images) * test_loader.batch_size >= max(num_gradcam_samples, num_lime_samples):
            break
    
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # تعیین نمونه‌ها برای Grad-CAM و LIME
    gradcam_indices = list(range(min(num_gradcam_samples, len(all_images))))
    lime_indices = list(range(min(num_lime_samples, len(all_images))))
    
    selected_images_gradcam = all_images[gradcam_indices]
    selected_labels_gradcam = all_labels[gradcam_indices]
    
    selected_images_lime = all_images[lime_indices]
    selected_labels_lime = all_labels[lime_indices]
    
    print(f"\nSelected {len(gradcam_indices)} samples for Grad-CAM")
    print(f"Selected {len(lime_indices)} samples for LIME\n")
    
    # 1. Visualizations برای Ensemble
    print("Generating visualizations for ENSEMBLE model...")
    ensemble_vis_dir = os.path.join(vis_dir, 'ensemble')
    os.makedirs(ensemble_vis_dir, exist_ok=True)
    
    generate_ensemble_gradcam(ensemble_module, selected_images_gradcam, 
                            selected_labels_gradcam, device, 
                            ensemble_vis_dir, MODEL_NAMES)
    
    generate_ensemble_lime(ensemble_module, selected_images_lime,
                          selected_labels_lime, device,
                          ensemble_vis_dir)
    
    # 2. Visualizations برای هر مدل تکی
    for model_idx, (model, model_name) in enumerate(zip(base_models, MODEL_NAMES)):
        print(f"\nGenerating visualizations for {model_name}...")
        model_vis_dir = os.path.join(vis_dir, f'model_{model_idx+1}_{model_name}')
        os.makedirs(model_vis_dir, exist_ok=True)
        
        # Normalize کردن تصاویر برای این مدل
        normalizer = MultiModelNormalization([MEANS[model_idx]], [STDS[model_idx]]).to(device)
        
        # Grad-CAM برای مدل تکی
        generate_single_model_gradcam(model, normalizer, selected_images_gradcam,
                                     selected_labels_gradcam, device,
                                     model_vis_dir, model_name)
        
        # LIME برای مدل تکی
        generate_single_model_lime(model, normalizer, selected_images_lime,
                                  selected_labels_lime, device,
                                  model_vis_dir, model_name)
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETED!")
    print(f"Saved to: {vis_dir}")
    print("="*70 + "\n")


def generate_ensemble_gradcam(ensemble_module, images, labels, device, save_dir, model_names):
    """تولید Grad-CAM برای مدل ensemble"""
    if not USE_PYTORCH_GRADCAM:
        print("Skipping Grad-CAM: pytorch-grad-cam not available")
        return
        
    import matplotlib.pyplot as plt
    import cv2
    
    gradcam_dir = os.path.join(save_dir, 'gradcam')
    os.makedirs(gradcam_dir, exist_ok=True)
    
    # پیدا کردن لایه target - از hesitant_fuzzy استفاده می‌کنیم
    target_layers = []
    for module in ensemble_module.hesitant_fuzzy.feature_net.modules():
        if isinstance(module, nn.Conv2d):
            target_layers.append(module)
    
    if not target_layers:
        print("Warning: No Conv2d layer found for Grad-CAM")
        return
    
    # استفاده از آخرین لایه
    target_layer = [target_layers[-1]]
    
    # ایجاد wrapper برای مدل
    class EnsembleWrapper(nn.Module):
        def __init__(self, ensemble):
            super().__init__()
            self.ensemble = ensemble
        
        def forward(self, x):
            output, _ = self.ensemble(x)
            return output.squeeze(1)
    
    wrapped_model = EnsembleWrapper(ensemble_module).eval()
    
    try:
        cam = PytorchGradCAM(model=wrapped_model, target_layers=target_layer)
        
        for idx in range(len(images)):
            img = images[idx:idx+1].to(device)
            label = labels[idx].item()
            
            # Fix: Pass None as targets to use highest scoring class, 
            # or create a ClassifierTarget if specific class is needed.
            targets = None 
            
            # تولید CAM
            grayscale_cam = cam(input_tensor=img, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # نرمال‌سازی تصویر برای نمایش
            img_np = img[0].cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            
            # ایجاد visualization
            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            
            # ذخیره تصویر
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_np)
            axes[0].set_title(f'Original (Label: {label})')
            axes[0].axis('off')
            
            axes[1].imshow(grayscale_cam, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')
            
            axes[2].imshow(visualization)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(gradcam_dir, f'ensemble_gradcam_sample_{idx}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"Error generating ensemble Grad-CAM: {e}")
    finally:
        del cam


def generate_ensemble_lime(ensemble_module, images, labels, device, save_dir):
    """تولید LIME برای مدل ensemble"""
    lime_dir = os.path.join(save_dir, 'lime')
    os.makedirs(lime_dir, exist_ok=True)
    
    for idx in range(len(images)):
        img = images[idx:idx+1].to(device)
        label = labels[idx].item()
        
        # تابع predict برای LIME
        def predict_fn(x):
            x_tensor = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(device)
            with torch.no_grad():
                outputs, _ = ensemble_module(x_tensor)
                probs = torch.sigmoid(outputs).cpu().numpy()
            return np.concatenate([1 - probs, probs], axis=1)
        
        # تولید توضیح LIME
        img_np = img[0].cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        # Fix: Removed dataset_type argument, passed label (integer) as 3rd arg
        try:
            explanation = generate_lime_explanation(
                img_np, predict_fn, label
            )
            
            # ذخیره سازی دستی تصویر
            if explanation is not None and mark_boundaries is not None:
                try:
                    temp, mask = explanation.get_image_and_mask(
                        explanation.top_labels[0], 
                        positive_only=True, 
                        num_features=5, 
                        hide_rest=False
                    )
                    plt.imshow(mark_boundaries(temp / 255.0, mask))
                    plt.title(f"Ensemble LIME - Label: {label}")
                    plt.savefig(os.path.join(lime_dir, f'ensemble_lime_sample_{idx}.png'))
                    plt.close()
                except Exception as save_e:
                    print(f"Could not save LIME image for sample {idx}: {save_e}")
        except Exception as te:
             # Fallback if arguments still don't match
             print(f"LIME Warning: {te}. Skipping visualization for sample {idx}.")


def generate_single_model_gradcam(model, normalizer, images, labels, device, 
                                 save_dir, model_name):
    """تولید Grad-CAM برای یک مدل تکی"""
    if not USE_PYTORCH_GRADCAM:
        print("Skipping Grad-CAM: pytorch-grad-cam not available")
        return
        
    import matplotlib.pyplot as plt
    
    gradcam_dir = os.path.join(save_dir, 'gradcam')
    os.makedirs(gradcam_dir, exist_ok=True)
    
    # پیدا کردن لایه‌های Conv2d
    target_layers = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            target_layers.append(module)
    
    if not target_layers:
        print(f"Warning: No Conv2d layer found for {model_name}")
        return
    
    # استفاده از آخرین لایه
    target_layer = [target_layers[-1]]
    
    # wrapper برای مدل که normalization را انجام دهد
    class NormalizedModel(nn.Module):
        def __init__(self, model, normalizer):
            super().__init__()
            self.model = model
            self.normalizer = normalizer
        
        def forward(self, x):
            x = self.normalizer(x, 0)
            out = self.model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out.squeeze(1)
    
    wrapped_model = NormalizedModel(model, normalizer).to(device).eval()
    
    try:
        cam = PytorchGradCAM(model=wrapped_model, target_layers=target_layer)
        
        for idx in range(len(images)):
            img = images[idx:idx+1].to(device)
            label = labels[idx].item()
            
            # Fix: Targets handling
            targets = None
            
            # تولید CAM
            grayscale_cam = cam(input_tensor=img, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # نرمال‌سازی برای نمایش
            img_np = img[0].cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            
            # ایجاد visualization
            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            
            # ذخیره
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_np)
            axes[0].set_title(f'Original (Label: {label})')
            axes[0].axis('off')
            
            axes[1].imshow(grayscale_cam, cmap='jet')
            axes[1].set_title(f'{model_name} - Grad-CAM')
            axes[1].axis('off')
            
            axes[2].imshow(visualization)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(gradcam_dir, f'{model_name}_gradcam_sample_{idx}.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"Error generating Grad-CAM for {model_name}: {e}")
    finally:
        del cam


def generate_single_model_lime(model, normalizer, images, labels, device,
                              save_dir, model_name):
    """تولید LIME برای یک مدل تکی"""
    lime_dir = os.path.join(save_dir, 'lime')
    os.makedirs(lime_dir, exist_ok=True)
    
    for idx in range(len(images)):
        img = images[idx:idx+1].to(device)
        label = labels[idx].item()
        
        # تابع predict
        def predict_fn(x):
            x_tensor = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(device)
            x_norm = normalizer(x_tensor, 0)
            with torch.no_grad():
                outputs = model(x_norm)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                probs = torch.sigmoid(outputs).cpu().numpy()
            return np.concatenate([1 - probs, probs], axis=1)
        
        # تولید توضیح
        img_np = img[0].cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        try:
            # Fix: Passed label (integer) as 3rd arg
            explanation = generate_lime_explanation(
                img_np, predict_fn, label
            )
            
            # ذخیره سازی دستی
            if explanation is not None and mark_boundaries is not None:
                 temp, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0], 
                    positive_only=True, 
                    num_features=5, 
                    hide_rest=False
                )
                 plt.imshow(mark_boundaries(temp / 255.0, mask))
                 plt.title(f"{model_name} LIME - Label: {label}")
                 plt.savefig(os.path.join(lime_dir, f'{model_name}_lime_sample_{idx}.png'))
                 plt.close()

        except Exception as e:
            print(f"Error generating LIME for {model_name} sample {idx}: {e}")


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
    parser = argparse.ArgumentParser(description="Optimized Fuzzy Hesitant Ensemble Training")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_memberships', type=int, default=3)
    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['wild', 'real_fake', 'hard_fake_real', 'deepflux', 'uadfV', 'dfd'])
    parser.add_argument('--cum_weight_threshold', type=float, default=0.9)
    parser.add_argument('--hesitancy_threshold', type=float, default=0.2)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if len(args.model_names) != len(args.model_paths):
        raise ValueError("Number of model_names must match model_paths")

    set_seed(args.seed)
    device, local_rank, rank, world_size = setup_distributed()
    is_main = rank == 0

    if is_main:
        print("="*70)
        print(f"OPTIMIZED FUZZY HESITANT ENSEMBLE TRAINING")
        print(f"Distributed on {world_size} GPU(s) | Seed: {args.seed}")
        print("="*70)
        print(f"Dataset: {args.dataset}")
        print(f"Data directory: {args.data_dir}")
        print(f"Batch size: {args.batch_size}")
        print(f"Models: {len(args.model_paths)}")
        print("="*70 + "\n")

    MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    
    MEANS = MEANS[:len(args.model_paths)]
    STDS = STDS[:len(args.model_paths)]

    base_models = load_pruned_models(args.model_paths, device, is_main)
    MODEL_NAMES = args.model_names[:len(base_models)]

    ensemble = FuzzyHesitantEnsemble(
        base_models, MEANS, STDS,
        num_memberships=args.num_memberships,
        freeze_models=True,
        cum_weight_threshold=args.cum_weight_threshold,
        hesitancy_threshold=args.hesitancy_threshold
    ).to(device)

    if world_size > 1:
        ensemble = DDP(ensemble, device_ids=[local_rank], output_device=local_rank)

    if is_main:
        trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
        total = sum(p.numel() for p in ensemble.parameters())
        print(f"Total params: {total:,} | Trainable: {trainable:,} | Frozen: {total-trainable:,}\n")

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=(world_size > 1), seed=args.seed, is_main=is_main)

    if is_main:
        print("\n" + "="*70)
        print("INDIVIDUAL MODEL PERFORMANCE (Before Training)")
        print("="*70)

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

    best_val_acc, history = train_hesitant_fuzzy(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_dir, is_main, MODEL_NAMES)

    ckpt_path = os.path.join(args.save_dir, 'best_hesitant_fuzzy.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        hesitant_net = ensemble.module.hesitant_fuzzy if hasattr(ensemble, 'module') else ensemble.hesitant_fuzzy
        hesitant_net.load_state_dict(ckpt['hesitant_state_dict'])
        if is_main:
            print("Best model loaded.\n")

    if is_main:
        print("\n" + "="*70)
        print("FINAL ENSEMBLE EVALUATION")
        print("="*70)

    ensemble_module = ensemble.module if hasattr(ensemble, 'module') else ensemble
    ensemble_test_acc, ensemble_weights, activation_percentages = evaluate_ensemble_final_ddp(
        ensemble_module, test_loader, device, "Test", MODEL_NAMES, is_main)

    if is_main:
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        print(f"Best Single Model: {best_single:.2f}%")
        print(f"Ensemble Accuracy: {ensemble_test_acc:.2f}%")
        print(f"Improvement: {ensemble_test_acc - best_single:+.2f}%")
        print("="*70)

        final_results = {
            'best_single_model': {
                'name': MODEL_NAMES[best_idx],
                'accuracy': float(best_single)
            },
            'ensemble': {
                'test_accuracy': float(ensemble_test_acc),
                'model_weights': {name: float(w) for name, w in zip(MODEL_NAMES, ensemble_weights)},
                'activation_percentages': {name: float(p) for name, p in zip(MODEL_NAMES, activation_percentages)}
            },
            'improvement': float(ensemble_test_acc - best_single),
            'training_history': history
        }

        results_path = os.path.join(args.save_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"\nResults saved: {results_path}")

        final_model_path = os.path.join(args.save_dir, 'final_ensemble_model.pt')
        torch.save({
            'ensemble_state_dict': ensemble_module.state_dict(),
            'hesitant_fuzzy_state_dict': ensemble_module.hesitant_fuzzy.state_dict(),
            'test_accuracy': ensemble_test_acc,
            'model_names': MODEL_NAMES,
            'means': MEANS,
            'stds': STDS
        }, final_model_path)
        print(f"Model saved: {final_model_path}")

        # تولید visualizations برای همه مدل‌ها
        vis_dir = os.path.join(args.save_dir, 'visualizations')
        generate_visualizations_for_all_models(
            ensemble_module, base_models, test_loader, device, vis_dir,
            MODEL_NAMES, MEANS, STDS,
            args.num_grad_cam_samples, args.num_lime_samples,
            args.dataset, is_main
        )

    cleanup_distributed()

    if is_main:
        plot_roc_and_f1(
            ensemble_module,
            test_loader, 
            device, 
            args.save_dir, 
            MODEL_NAMES,
            is_main
        )

if __name__ == "__main__":
    main()
