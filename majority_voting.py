import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, SequentialSampler
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

# =================== بخش ایمپورت دیتاست (شبیه‌سازی) ===================
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


# ================== CHECKPOINT SAVING FUNCTION (PT FORMAT) ==================
def save_ensemble_checkpoint(save_path: str, ensemble_model: nn.Module, model_paths: List[str], 
                             model_names: List[str], accuracy: float, means: List, stds: List):
    model_to_save = ensemble_model.module if hasattr(ensemble_model, 'module') else ensemble_model
    
    checkpoint = {
        'format_version': '1.0',
        'model_paths': model_paths,
        'model_names': model_names,
        'normalization_means': means,
        'normalization_stds': stds,
        'accuracy': accuracy,
        'state_dict': model_to_save.state_dict()
    }
    
    torch.save(checkpoint, save_path)
    print(f"✅ Best Ensemble model saved to: {save_path}")
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"   File size: {size_mb:.2f} MB")


# ================== GRADCAM ==================
class GradCAM:
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
    # Local stats: TP, TN, FP, FN, Total
    local_stats = torch.zeros(5, device=device)
    vote_distribution = torch.zeros(len(model_names), 2, device=device)
    
    if loader is None: return 0.0, []

    if is_main:
        print(f"\nEvaluating {name} set (Majority Voting)...")
        
    for images, labels in tqdm(loader, desc=f"Evaluating {name}", disable=not is_main):
        images, labels = images.to(device), labels.to(device)
        outputs, weights, _, votes = model(images, return_details=True)
        
        pred = (outputs.squeeze(1) > 0).long()
        
        # Calculate TP, TN, FP, FN
        # Labels: 1=Real, 0=Fake
        # Preds: 1=Real, 0=Fake
        is_tp = ((pred == 1) & (labels.long() == 1)).sum()
        is_tn = ((pred == 0) & (labels.long() == 0)).sum()
        is_fp = ((pred == 1) & (labels.long() == 0)).sum()
        is_fn = ((pred == 0) & (labels.long() == 1)).sum()
        
        local_stats[0] += is_tp
        local_stats[1] += is_tn
        local_stats[2] += is_fp
        local_stats[3] += is_fn
        local_stats[4] += labels.size(0)
        
        real_votes = votes.sum(dim=0)
        fake_votes = votes.size(0) - real_votes
        vote_distribution[:, 0] += fake_votes
        vote_distribution[:, 1] += real_votes

    # Aggregate stats
    if dist.is_initialized():
        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(vote_distribution, op=dist.ReduceOp.SUM)

    if is_main:
        tp = local_stats[0].item()
        tn = local_stats[1].item()
        fp = local_stats[2].item()
        fn = local_stats[3].item()
        total = local_stats[4].item()
        
        acc = 100. * (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        vote_dist_np = vote_distribution.cpu().numpy()

        print(f"\n{'='*70}")
        print(f"{name.upper()} SET RESULTS (Majority Voting)")
        print(f"{'='*70}")
        print(f" → Accuracy: {acc:.3f}%")
        print(f" → Precision: {precision:.4f}")
        print(f" → Recall: {recall:.4f}")
        print(f" → Specificity: {specificity:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted Real  Predicted Fake")
        print(f"    Actual Real      {int(tp):<15} {int(fn):<15}")
        print(f"    Actual Fake      {int(fp):<15} {int(tn):<15}")
        
        print(f"\nVote Distribution (Fake / Real):")
        for i, mname in enumerate(model_names):
            print(f"  {i+1:2d}. {mname:<25}: {int(vote_dist_np[i,0]):<6} / {int(vote_dist_np[i,1]):<6}")
        
        print(f"{'='*70}")
        return acc, vote_dist_np.tolist(), local_stats.cpu().tolist()
    return 0.0, [], []


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


# ================== SAVE PREDICTION LOG FUNCTION (FIXED) ==================
def save_prediction_log(ensemble, test_loader, device, save_path, is_main):
    """
    ذخیره لیست پیش‌بینی‌ها و آمار کامل در یک فایل متنی.
    فقط روی مجموعه تست اجرا می‌شود.
    """
    if not is_main or test_loader is None:
        return

    print("\n" + "="*70)
    print("GENERATING PREDICTION LOG FILE")
    print("="*70)

    # استخراج مدل (حتی اگر داخل DDP باشد)
    model = ensemble.module if hasattr(ensemble, 'module') else ensemble
    model.eval()

    # استخراج دیتاست پایه از داخل DataLoader
    # ممکن است test_loader شامل DistributedSampler یا Subset باشد
    base_dataset = test_loader.dataset
    if hasattr(base_dataset, 'dataset'):
        base_dataset = base_dataset.dataset

    # ایجاد یک DataLoader جدید که کل دیتاست تست را ترتیبی پیمایش کند
    # این کار برای اطمینان از ترتیب صحیح در خروجی است
    log_loader = DataLoader(
        base_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0, 
        sampler=SequentialSampler(base_dataset)
    )

    lines = []
    
    # متغیرهای آمار
    TP, TN, FP, FN = 0, 0, 0, 0
    total_samples = 0
    correct_count = 0
    sample_id = 0

    # هدر جدول نمونه‌ها
    lines.append("="*100)
    lines.append("SAMPLE-BY-SAMPLE PREDICTIONS (For McNemar Test Comparison):")
    lines.append("="*100)
    header = f"{'Sample_ID':<10} {'Sample_Path':<60} {'True_Label':<12} {'Predicted_Label':<15} {'Correct':<10}"
    lines.append(header)
    lines.append("-"*100)

    for image, label in tqdm(log_loader, desc="Logging predictions", total=len(base_dataset)):
        image = image.to(device)
        # label معمولاً شکل [1] دارد
        label_int = int(label.item())
        
        with torch.no_grad():
            output, _ = model(image)
        
        # خروجی مدل: logits هستند. اگر > 0 باشد کلاس 1 (Real) است
        pred_int = int(output.squeeze().item() > 0)
        
        is_correct = (pred_int == label_int)
        if is_correct:
            correct_count += 1

        # محاسبه ماتریس آشفتگی
        if label_int == 1:  # Real
            if pred_int == 1: TP += 1
            else: FN += 1
        else:  # Fake
            if pred_int == 1: FP += 1
            else: TN += 1
        
        total_samples += 1
        sample_id += 1

        # دریافت مسیر فایل (اگر ممکن باشد)
        try:
            # اندیس واقعی در دیتاست پایه (چون ترتیبی پیمایش می‌کنیم همان sample_id-1 است)
            path, _ = get_sample_info(base_dataset, sample_id - 1)
            filename = os.path.basename(path)
        except Exception:
            filename = f"Sample_{sample_id-1}"
        
        # کوتاه کردن نام فایل اگر خیلی طولانی باشد
        if len(filename) > 55:
            filename = filename[:25] + "..." + filename[-27:]
            
        line = f"{sample_id:<10} {filename:<60} {label_int:<12} {pred_int:<15} {'Yes' if is_correct else 'No':<10}"
        lines.append(line)

    # محاسبه آمار نهایی
    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total > 0 else 0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0

    # ساخت رشته خروجی نهایی (اول آمار، بعد لیست)
    output_str = []
    output_str.append("-" * 100)
    output_str.append("SUMMARY STATISTICS:")
    output_str.append("-" * 100)
    output_str.append(f"Accuracy: {acc*100:.2f}%")
    output_str.append(f"Precision: {prec:.4f}")
    output_str.append(f"Recall: {rec:.4f}")
    output_str.append(f"Specificity: {spec:.4f}")
    output_str.append("\nConfusion Matrix:")
    output_str.append(f"                 {'Predicted Real':<15} {'Predicted Fake':<15}")
    output_str.append(f"    Actual Real   {TP:<15} {FN:<15}")
    output_str.append(f"    Actual Fake   {FP:<15} {TN:<15}")
    output_str.append(f"\nCorrect Predictions: {correct_count} ({acc*100:.2f}%)")
    output_str.append(f"Incorrect Predictions: {total - correct_count} ({(1-acc)*100:.2f}%)")
    
    # اضافه کردن لیست نمونه‌ها
    output_str.extend(lines)

    # ذخیره فایل
    with open(save_path, 'w') as f:
        f.write("\n".join(output_str))
    
    print(f"✅ Prediction log saved to: {save_path}")
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
                       choices=['wild', 'real_fake', 'hard_fake_real', 'real_fake_dataset', 'uadfV', 'custom_genai'])
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
    
    # ارزیابی اصلی (سریع و توزیع شده)
    ensemble_test_acc, vote_dist, stats = evaluate_ensemble_final_ddp(
        ensemble_module, test_loader, device, "Test", MODEL_NAMES, is_main)

    if is_main:
        os.makedirs(args.save_dir, exist_ok=True)
        
        # ================== ذخیره مدل در فرمت .pt ==================
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
        
        # ================== ذخیره فایل متنی پیش‌بینی‌ها ==================
        log_path = os.path.join(args.save_dir, 'prediction_log.txt')
        save_prediction_log(
            ensemble=ensemble,
            test_loader=test_loader,
            device=device,
            save_path=log_path,
            is_main=is_main
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

        vis_dir = os.path.join(args.save_dir, 'visualizations')
        generate_visualizations(
            ensemble_module, test_loader, device, vis_dir, MODEL_NAMES,
            args.num_grad_cam_samples, args.num_lime_samples,
            args.dataset, is_main)

    cleanup_distributed()

if __name__ == "__main__":
    main()
