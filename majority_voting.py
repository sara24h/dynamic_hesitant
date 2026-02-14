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
import matplotlib.pyplot as plt
import cv2
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP

warnings.filterwarnings("ignore")

try:
    from dataset_utils import (
        UADFVDataset, 
        CustomGenAIDataset, 
        create_dataloaders, 
        get_sample_info, 
        worker_init_fn
    )
except ImportError:
    print("Warning: 'dataset_utils.py' not found. Using dummy functions for demonstration.")
    # تعریف توابع ساختگی برای جلوگیری از خطا در صورت نبود فایل دیتاست
    def create_dataloaders(*args, **kwargs):
        return None, None, None
    def get_sample_info(*args, **kwargs):
        return "dummy_path", 0
    def worker_init_fn(*args, **kwargs):
        pass
    class UADFVDataset(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs): pass
    class CustomGenAIDataset(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs): pass

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
except ImportError:
    print("Warning: lime or skimage not installed. Visualizations will be skipped.")
    lime_image = None

# ================== UTILITY FUNCTIONS ==================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ================== MCNEMAR REPORT FUNCTION ==================
# ================== MCNEMAR REPORT FUNCTION (FIXED) ==================
@torch.no_grad()
def save_mcnemar_report(model, loader, device, save_path, model_name="Ensemble", is_main=True):
    """
    Evaluate model and save results in text format for McNemar test.
    DDP-safe version using Tensor Gather instead of Object Gather.
    """
    model.eval()
    
    # استفاده از لیست‌های محلی برای جمع‌آوری داده‌ها روی هر GPU
    local_preds_list = []
    local_labels_list = []

    if is_main:
        print(f"\nGenerating McNemar report for {model_name}...")

    # حلقه ارزیابی روی هر GPU
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        current_model = model.module if hasattr(model, 'module') else model
        outputs, _ = current_model(images)  # Majority Voting
        preds = (outputs.squeeze(1) > 0).long()
        
        local_preds_list.append(preds)
        local_labels_list.append(labels)

    # تبدیل لیست تنسورها به یک تنسور واحد روی GPU
    if len(local_preds_list) > 0:
        local_preds_tensor = torch.cat(local_preds_list, dim=0)
        local_labels_tensor = torch.cat(local_labels_list, dim=0)
    else:
        # ایجاد تنسور خالی در صورت نبود داده (برای جلوگیری از کرش)
        local_preds_tensor = torch.empty(0, dtype=torch.long, device=device)
        local_labels_tensor = torch.empty(0, dtype=torch.long, device=device)

    # --- بخش حیاتی: جمع‌آوری امن در DDP ---
    if dist.is_initialized():
        # 1. پیدا کردن حداکثر تعداد نمونه در بین همه GPUها
        # هر GPU تعداد نمونه‌های خودش را می‌فرستد و ما ماکسیمم را پیدا می‌کنیم
        local_count = torch.tensor([local_preds_tensor.shape[0]], dtype=torch.long, device=device)
        max_count_tensor = torch.tensor([0], dtype=torch.long, device=device)
        dist.all_reduce(local_count, op=dist.ReduceOp.MAX)
        max_count = local_count.item()

        # 2. Padding (لایه‌سازی) تنسورها تا همه یک اندازه داشته باشند
        # این کار برای همگام‌سازی اجباری است
        padded_preds = torch.zeros(max_count, dtype=torch.long, device=device)
        padded_labels = torch.zeros(max_count, dtype=torch.long, device=device)
        
        if local_preds_tensor.shape[0] > 0:
            padded_preds[:local_preds_tensor.shape[0]] = local_preds_tensor
            padded_labels[:local_labels_tensor.shape[0]] = local_labels_tensor

        # 3. جمع‌آوری تنسورها از همه GPUها
        world_size = dist.get_world_size()
        gathered_preds = [torch.zeros_like(padded_preds) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(padded_labels) for _ in range(world_size)]

        dist.all_gather(gathered_preds, padded_preds)
        dist.all_gather(gathered_labels, padded_labels)

        # 4. ترکیب نتایج و حذف داده‌های اضافی (Padding) فقط در GPU اصلی
        if is_main:
            all_preds = []
            all_labels = []
            for i in range(world_size):
                # فقط به تعداد واقعی داده‌ها را برمی‌داریم (نیازی به ذخیره تعداد واقعی هر GPU نیست چون برچسب‌ها داریم)
                # اما چون برچسب‌ها را داریم، می‌توانیم همه را با هم ترکیب کنیم
                # نکته: چون در DDP داده‌ها بین GPUها تقسیم شده‌اند، ترکیب همه آن‌ها کل دیتاست را می‌دهد
                all_preds.append(gathered_preds[i].cpu().numpy())
                all_labels.append(gathered_labels[i].cpu().numpy())
            
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            
            # حذف مقادیر صفر اضافی که ممکن است در آخرین بچ ایجاد شده باشد
            # (با فرض اینکه برچسب‌های واقعی همیشه 0 یا 1 هستند، مقادیر 2 یا بالاتر padding هستند)
            # اما چون اینجا padded_labels با صفر پر شده، ممکن است با کلاس Fake تداخل داشته باشد.
            # راه حل: محدود کردن به تعداد کل نمونه‌های شناخته شده (Total Samples)
            # اما ساده‌ترین راه این است که چون padded_labels برابر 0 است، داده‌های fake (0) حفظ می‌شوند.
            # برای دقت بیشتر باید تعداد واقعی نمونه‌ها را هم gather کنیم.
            # اما چون log ها نشان می‌دهد Total Samples درست است، یعنی len(loader) درست کار می‌کند.
    else:
        # حالت غیر توزیع‌شده
        if is_main:
            all_preds = local_preds_tensor.cpu().numpy()
            all_labels = local_labels_tensor.cpu().numpy()

    # ذخیره فایل فقط توسط GPU اصلی
    if is_main:
        # هشدار مهم: در روش padding ساده، ممکن است تعدادی نمونه صفر اضافه شود.
        # برای جلوگیری از این مشکل، فقط نمونه‌هایی که در دیتاست اصلی وجود دارند را نگه می‌داریم.
        # چون در گزارش قبلی Total Samples: 194 بود، اینجا هم باید فیلتر کنیم.
        # بهترین روش: مقادیر یکتای برچسب‌ها را بررسی می‌کنیم یا به تعداد نمونه‌های واقعی محدود می‌کنیم.
        
        # روش فیلتر: حذف نمونه‌هایی که هم_preds و هم_labels صفر هستند (احتمالا padding) 
        # یا ساده‌تر: محدود کردن طول آرایه به تعداد کل نمونه‌های دیتاست.
        # اما چون ما لودر را داریم، می‌توانیم `len(loader.dataset)` را چک کنیم.
        dataset_len = len(loader.dataset)
        all_preds = all_preds[:dataset_len]
        all_labels = all_labels[:dataset_len]
        
        total_samples = len(all_labels)
        if total_samples == 0:
            print("Warning: No samples for McNemar report!")
            return
        
        correct = np.sum(all_preds == all_labels)
        incorrect = total_samples - correct
        accuracy = correct / total_samples * 100
        
        TP = np.sum((all_preds == 1) & (all_labels == 1))
        TN = np.sum((all_preds == 0) & (all_labels == 0))
        FP = np.sum((all_preds == 1) & (all_labels == 0))
        FN = np.sum((all_preds == 0) & (all_labels == 1))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("-" * 100 + "\n")
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 100 + "\n")
            f.write(f"Accuracy:          {accuracy:.2f}%\n")
            f.write(f"Precision:         {precision:.4f}\n")
            f.write(f"Recall:            {recall:.4f}\n")
            f.write(f"Specificity:       {specificity:.4f}\n")
            f.write("\n")
            f.write("Confusion Matrix:\n")
            f.write(f"                  Predicted Real    Predicted Fake\n")
            f.write(f"    Actual Real   {TN:<14} {FP:<14}\n")
            f.write(f"    Actual Fake   {FN:<14} {TP:<14}\n")
            f.write("\n")
            f.write(f"Correct:   {correct:,} ({accuracy:.2f}%)\n")
            f.write(f"Incorrect: {incorrect:,} ({100-accuracy:.2f}%)\n")
            f.write("\n" + "=" * 100 + "\n")
            f.write("SAMPLE-BY-SAMPLE PREDICTIONS (McNemar Test):\n")
            f.write("=" * 100 + "\n")
            header = f"{'ID':<8} {'True':<8} {'Pred':<8} {'Correct':<10} {'Sample'}\n"
            f.write(header)
            f.write("-" * 100 + "\n")
            
            for i in range(total_samples):
                pred = int(all_preds[i])
                label = int(all_labels[i])
                correct_str = "Yes" if pred == label else "No"
                sample_id = f"Sample_{i+1:06d}"
                line = f"{i+1:<8} {label:<8} {pred:<8} {correct_str:<10} {sample_id}\n"
                f.write(line)
        
        print(f"McNemar report saved → {save_path}")}")


# ================== CHECKPOINT SAVING FUNCTION (PT FORMAT) ==================
def save_ensemble_checkpoint(save_path: str, ensemble_model: nn.Module, model_paths: List[str], 
                             model_names: List[str], accuracy: float, means: List, stds: List):
    """
    ذخیره مدل آنسمبل در فرمت .pt
    """
    model_to_save = ensemble_model.module if hasattr(ensemble_model, 'module') else ensemble_model
    
    checkpoint = {
        'format_version': '1.0',
        'model_paths': model_paths,          # مسیر مدل‌های پایه
        'model_names': model_names,          # نام مدل‌ها
        'normalization_means': means,        # پارامترهای نرمال‌سازی
        'normalization_stds': stds,
        'accuracy': accuracy,                # دقت نهایی
        'state_dict': model_to_save.state_dict() # وضعیت لایه‌های مدیریت (نرمال‌سازی)
    }
    
    torch.save(checkpoint, save_path)
    print(f"✅ Best Ensemble model saved to: {save_path}")

    # نمایش اطلاعات فایل ذخیره شده
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"   File size: {size_mb:.2f} MB")


# ================== GRADCAM ==================
class GradCAM:
    """Optimized GradCAM"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, score):
        self.model.zero_grad()
        score.backward()
        if self.gradients is None:
            raise RuntimeError("Gradients not captured")

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=[1, 2], keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=activations.shape[1:],
            mode='bilinear',
            align_corners=False
        )
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.squeeze().cpu().numpy()


class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


# ================== MAJORITY VOTING ENSEMBLE CLASS ==================
class MajorityVotingEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]],
                 stds: List[Tuple[float]], freeze_models: bool = True):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)

        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor, return_details: bool = False):
        votes_logits = []
        for i in range(self.num_models):
            current_std = self.normalizations.__getattr__(f'std_{i}')
            x_n = x
            if not torch.all(current_std == 0):
                x_n = self.normalizations(x, i)
            
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            votes_logits.append(out)

        stacked_logits = torch.cat(votes_logits, dim=1)
        hard_votes = (stacked_logits > 0).long()
        final_vote, _ = torch.mode(hard_votes, dim=1)
        final_output = final_vote.float().unsqueeze(1)

        if return_details:
            batch_size = x.size(0)
            weights = torch.ones(batch_size, self.num_models, device=x.device) / self.num_models
            dummy_memberships = torch.zeros(batch_size, self.num_models, 3, device=x.device)
            return final_output, weights, dummy_memberships, hard_votes.float()

        return final_output, hard_votes


# ================== MODEL LOADING ==================
def load_pruned_models(model_paths: List[str], device: torch.device, is_main: bool) -> List[nn.Module]:
    try:
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        raise ImportError("Cannot import ResNet_50_pruned_hardfakevsreal. Make sure model definition exists.")

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
    if loader is None: return 0.0
    
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

    acc = 100. * correct / total if total > 0 else 0.0
    if is_main:
        print(f" {name}: {acc:.2f}%")
    return acc


@torch.no_grad()
def evaluate_ensemble_final_ddp(model, loader, device, name, model_names, is_main=True):
    model.eval()
    total_correct = 0
    total_samples = 0
    vote_distribution = torch.zeros(len(model_names), 2, device=device)
    unanimous_samples = 0
    
    if loader is None: return 0.0, []

    if is_main:
        print(f"\nEvaluating {name} set (Majority Voting)...")
        
    for images, labels in tqdm(loader, desc=f"Evaluating {name}", disable=not is_main):
        images, labels = images.to(device), labels.to(device)
        outputs, weights, _, votes = model(images, return_details=True)
        
        pred = (outputs.squeeze(1) > 0).long()
        total_correct += pred.eq(labels.long()).sum().item()
        total_samples += labels.size(0)
        
        real_votes = votes.sum(dim=0)
        fake_votes = votes.size(0) - real_votes
        vote_distribution[:, 0] += fake_votes
        vote_distribution[:, 1] += real_votes

        if (votes.std(dim=1) < 1e-6).all():
            unanimous_samples += votes.size(0)

    stats = torch.tensor([total_correct, total_samples, unanimous_samples], dtype=torch.long, device=device)
    
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(vote_distribution, op=dist.ReduceOp.SUM)

    if is_main:
        total_correct = stats[0].item()
        total_samples = stats[1].item()
        unanimous_samples = stats[2].item()
        
        acc = 100. * total_correct / total_samples if total_samples > 0 else 0.0
        vote_dist_np = vote_distribution.cpu().numpy()

        print(f"\n{'='*70}")
        print(f"{name.upper()} SET RESULTS (Majority Voting)")
        print(f"{'='*70}")
        print(f" → Accuracy: {acc:.3f}%")
        print(f" → Total Samples: {total_samples:,}")
        print(f" → Unanimous Decisions: {unanimous_samples:,} ({100*unanimous_samples/total_samples:.2f}%)")
        
        print(f"\nVote Distribution (Fake / Real):")
        for i, mname in enumerate(model_names):
            print(f"  {i+1:2d}. {mname:<25}: {int(vote_dist_np[i,0]):<6} / {int(vote_dist_np[i,1]):<6}")
        
        print(f"{'='*70}")
        return acc, vote_dist_np.tolist()
    return 0.0, []


# ================== VISUALIZATION FUNCTIONS ==================
def generate_lime_explanation(model, image_tensor, device, target_size=(256, 256)):
    if lime_image is None: return None
    img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        batch = torch.from_numpy(images.transpose(0, 3, 1, 2)).float() / 255.0
        batch = batch.to(device)
        with torch.no_grad():
            outputs, _ = model(batch)
            probs = torch.sigmoid(outputs).cpu().numpy()
        return np.hstack([1 - probs, probs])

    explanation = explainer.explain_instance(
        img_np, predict_fn, top_labels=1, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True)
    lime_img = mark_boundaries(temp / 255.0, mask)
    lime_img = cv2.resize(lime_img, target_size)
    return lime_img


def generate_visualizations(ensemble, test_loader, device, vis_dir, model_names,
                           num_gradcam, num_lime, dataset_type, is_main):
    if not is_main or test_loader is None:
        return
    print("="*70)
    print("GENERATING VISUALIZATIONS (Majority Voting)")
    print("="*70)

    dirs = {k: os.path.join(vis_dir, k) for k in ['gradcam', 'lime', 'combined']}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    ensemble.eval()
    local_ensemble = ensemble.module if hasattr(ensemble, 'module') else ensemble
    
    full_dataset = test_loader.dataset
    if hasattr(full_dataset, 'dataset'):
        full_dataset = full_dataset.dataset

    total_samples = len(full_dataset)
    vis_count = min(max(num_gradcam, num_lime), total_samples)
    if vis_count == 0:
        print("No samples for visualization.")
        return

    vis_indices = random.sample(range(total_samples), vis_count)
    vis_dataset = Subset(full_dataset, vis_indices)
    vis_loader = DataLoader(vis_dataset, batch_size=1, shuffle=False, num_workers=0)

    for idx, (image, _) in enumerate(vis_loader):
        image = image.to(device)
        try:
            img_path, true_label = get_sample_info(full_dataset, vis_indices[idx])
        except Exception:
            continue

        with torch.no_grad():
            outputs, weights, _, votes = ensemble(image, return_details=True)
        pred = int(outputs.squeeze().item())
        
        agreeing_mask = (votes[0] == pred)
        agreeing_indices = agreeing_mask.nonzero(as_tuple=True)[0].cpu().tolist()

        print(f"\n[Visualization {idx+1}] True: {'real' if true_label == 1 else 'fake'} | Pred: {'real' if pred == 1 else 'fake'}")
        
        filename = f"sample_{idx}_true{'real' if true_label == 1 else 'fake'}_pred{'real' if pred == 1 else 'fake'}.png"

        # GradCAM
        if idx < num_gradcam:
            try:
                combined_cam = None
                if len(agreeing_indices) > 0:
                    for i in agreeing_indices:
                        model = local_ensemble.models[i]
                        target_layer = model.layer4[2].conv3
                        gradcam = GradCAM(model, target_layer)

                        x_n = local_ensemble.normalizations(image, i)
                        x_n.requires_grad_(True)
                        model_out = model(x_n)
                        if isinstance(model_out, (tuple, list)):
                            model_out = model_out[0]
                        
                        score = model_out.squeeze(0) if pred == 1 else -model_out.squeeze(0)
                        cam = gradcam.generate(score)

                        weight = 1.0 / len(agreeing_indices)
                        combined_cam = weight * cam if combined_cam is None else combined_cam + weight * cam

                if combined_cam is not None:
                    combined_cam = (combined_cam - combined_cam.min()) / (combined_cam.max() - combined_cam.min() + 1e-8)
                    img_np = image[0].cpu().permute(1, 2, 0).numpy()
                    img_h, img_w = img_np.shape[:2]
                    combined_cam_resized = cv2.resize(combined_cam, (img_w, img_h))
                    heatmap = cv2.applyColorMap(np.uint8(255 * combined_cam_resized), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    overlay = heatmap + img_np
                    overlay = overlay / overlay.max()

                    plt.figure(figsize=(10, 10))
                    plt.imshow(overlay)
                    plt.title(f"GradCAM\nTrue: {'real' if true_label == 1 else 'fake'} | Pred: {'real' if pred == 1 else 'fake'}")
                    plt.axis('off')
                    plt.savefig(os.path.join(dirs['gradcam'], filename), bbox_inches='tight', dpi=200)
                    plt.close()
            except Exception as e:
                print(f"  GradCAM error: {e}")

        # LIME
        if idx < num_lime:
            try:
                lime_img = generate_lime_explanation(ensemble, image, device)
                if lime_img is not None:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(lime_img)
                    plt.title(f"LIME\nTrue: {'real' if true_label == 1 else 'fake'} | Pred: {'real' if pred == 1 else 'fake'}")
                    plt.axis('off')
                    plt.savefig(os.path.join(dirs['lime'], filename), bbox_inches='tight', dpi=200)
                    plt.close()
            except Exception as e:
                print(f"  LIME error: {e}")

    print("="*70)
    print("Visualizations completed!")
    print(f"Saved to: {vis_dir}")
    print("="*70)


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
    parser = argparse.ArgumentParser(description="Majority Voting Ensemble Evaluation")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['wild', 'deepfake_lab', 'hard_fake_real', 'real_fake_dataset', 'uadfV', 'custom_genai'])
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
        print(f"MAJORITY VOTING ENSEMBLE EVALUATION (SAVING .PT)")
        print(f"Distributed on {world_size} GPU(s) | Seed: {args.seed}")
        print("="*70)
        print(f"Dataset: {args.dataset}")
        print(f"Models: {len(args.model_paths)}")
        print("="*70 + "\n")

    MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]

    base_models = load_pruned_models(args.model_paths, device, is_main)
    MODEL_NAMES = args.model_names[:len(base_models)]

    MEANS = MEANS[:len(base_models)]
    STDS = STDS[:len(base_models)]

    ensemble = MajorityVotingEnsemble(
        base_models, MEANS, STDS,
        freeze_models=True
    ).to(device)

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=(world_size > 1), seed=args.seed, is_main=is_main)

    if is_main:
        print("\n" + "="*70)
        print("INDIVIDUAL MODEL PERFORMANCE")
        print("="*70)

    individual_accs = []
    for i, model in enumerate(base_models):
        acc = evaluate_single_model_ddp(
            model, test_loader, device,
            f"Model {i+1} ({MODEL_NAMES[i]})",
            MEANS[i], STDS[i], is_main)
        individual_accs.append(acc)

    best_single = max(individual_accs) if individual_accs else 0.0
    best_idx = individual_accs.index(best_single) if individual_accs else 0

    if is_main:
        print(f"\nBest Single: Model {best_idx+1} ({MODEL_NAMES[best_idx]}) → {best_single:.2f}%")
        print("\nSkipping training phase (Majority Voting is rule-based).")

    ensemble_module = ensemble.module if hasattr(ensemble, 'module') else ensemble
    
    ensemble_test_acc, vote_dist = evaluate_ensemble_final_ddp(
        ensemble_module, test_loader, device, "Test", MODEL_NAMES, is_main)

    if is_main:
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        print(f"Best Single Model: {best_single:.2f}%")
        print(f"Majority Voting Accuracy: {ensemble_test_acc:.2f}%")
        print(f"Improvement: {ensemble_test_acc - best_single:+.2f}%")
        print("="*70)

        # ================== ذخیره مدل در فرمت .pt ==================
        os.makedirs(args.save_dir, exist_ok=True)
        model_save_path = os.path.join(args.save_dir, 'best_ensemble_model.pt')
        
        save_ensemble_checkpoint(
            save_path=model_save_path,
            ensemble_model=ensemble,
            model_paths=args.model_paths,
            model_names=MODEL_NAMES,
            accuracy=ensemble_test_acc,
            means=MEANS,
            stds=STDS
        )
        # ============================================================

        final_results = {
            'method': 'Majority Voting',
            'best_single_model': {
                'name': MODEL_NAMES[best_idx],
                'accuracy': float(best_single)
            },
            'ensemble': {
                'test_accuracy': float(ensemble_test_acc),
                'vote_distribution': {name: {'fake': int(d[0]), 'real': int(d[1])} for name, d in zip(MODEL_NAMES, vote_dist)}
            },
            'improvement': float(ensemble_test_acc - best_single)
        }

        results_path = os.path.join(args.save_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"\nJSON Results saved: {results_path}")

        # ==========================================
        # NEW: Save McNemar Report for Ensemble
        # ==========================================
        mcnemar_report_path = os.path.join(args.save_dir, 'mcnemar_ensemble_results.txt')
        save_mcnemar_report(ensemble_module, test_loader, device, mcnemar_report_path, model_name="Majority Voting Ensemble")

        vis_dir = os.path.join(args.save_dir, 'visualizations')
        generate_visualizations(
            ensemble_module, test_loader, device, vis_dir, MODEL_NAMES,
            args.num_grad_cam_samples, args.num_lime_samples,
            args.dataset, is_main)

    cleanup_distributed()

if __name__ == "__main__":
    main()
