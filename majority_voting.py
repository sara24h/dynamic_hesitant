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
import matplotlib.pyplot as plt
import cv2

from metrics_utils import plot_roc_and_f1

warnings.filterwarnings("ignore")

# =================== بخش ایمپورت دیتاست ===================
try:
    from dataset_utils import (
        UADFVDataset, 
        CustomGenAIDataset, 
        create_dataloaders, 
        create_adabn_dataloader, 
        get_sample_info, 
        worker_init_fn
    )
except ImportError:
    print("Warning: 'dataset_utils.py' not found. Using dummy functions.")
    def create_dataloaders(*args, **kwargs): return None, None, None
    def create_adabn_dataloader(*args, **kwargs): return None
    def get_sample_info(*args, **kwargs): return "dummy_path", 0
    def worker_init_fn(*args, **kwargs): pass
    class UADFVDataset(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs): pass
    class CustomGenAIDataset(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs): pass

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
except ImportError:
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

def save_ensemble_checkpoint(save_path: str, ensemble_model: nn.Module, model_paths: List[str], 
                             model_names: List[str], accuracy: float, means: List, stds: List):
    # وقتی از DataParallel استفاده می‌کنیم، باید state_dict خودِ مدل اصلی را ذخیره کنیم
    model_to_save = ensemble_model.module if hasattr(ensemble_model, 'module') else ensemble_model
    
    checkpoint = {
        'format_version': '1.0', 'model_paths': model_paths, 'model_names': model_names,
        'normalization_means': means, 'normalization_stds': stds, 'accuracy': accuracy,
        'state_dict': model_to_save.state_dict()
    }
    torch.save(checkpoint, save_path)
    print(f"✅ Best Ensemble model saved to: {save_path}")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output): self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out): self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, score):
        self.model.zero_grad()
        score.backward()
        if self.gradients is None: raise RuntimeError("Gradients not captured")
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=[1, 2], keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=activations.shape[1:], mode='bilinear', align_corners=False)
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


# =====================================================================
# ================== BATCHNORM ADAPTATION (AdaBN) ====================
# =====================================================================
@torch.no_grad()
def adapt_batchnorm_for_new_dataset(models, means, stds, adabn_loader, device, is_main=True):
    if is_main:
        print("\n" + "="*70)
        print("STARTING BATCHNORM ADAPTATION (AdaBN) - CLEAN DATA (NO AUGMENTATION)")
        print("="*70)
        
    normalizer = MultiModelNormalization(means, stds).to(device)
    
    for model in models:
        # اگر مدل با DataParallel پوشانده شده باشد، باید به مدل درونی دسترسی پیدا کنیم
        actual_model = model.module if hasattr(model, 'module') else model
        actual_model.eval()
        for m in actual_model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.reset_running_stats()
                m.momentum = None
                m.train()
                
    for images, _ in tqdm(adabn_loader, desc="Adapting BN", disable=not is_main):
        images = images.to(device)
        for i, model in enumerate(models):
            x_n = normalizer(images, i)
            model(x_n)
            
    for model in models:
        actual_model = model.module if hasattr(model, 'module') else model
        for m in actual_model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.momentum = 0.1
        actual_model.eval()
        
    if is_main:
        print("✅ BatchNorm Adaptation Completed!\n")
# =====================================================================


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
            # برای جلوگیری از ارور در DataParallel، مستقیما از خودِ کلاس می‌خوانیم
            current_std = getattr(self.normalizations, f'std_{i}')
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
        sum_votes = hard_votes.sum(dim=1, keepdim=True)
        threshold = self.num_models / 2.0
        final_decision = (sum_votes >= threshold).long().float()
        
        avg_probs = torch.sigmoid(stacked_logits).mean(dim=1, keepdim=True)
        
        if return_details:
            batch_size = x.size(0)
            weights = torch.ones(batch_size, self.num_models, device=x.device) / self.num_models
            return final_decision, weights, avg_probs, stacked_logits
            
        return final_decision, hard_votes

# ================== MODEL LOADING ==================
def load_pruned_models(model_paths: List[str], device: torch.device, world_size: int) -> List[nn.Module]:
    try:
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        raise ImportError("Cannot import ResNet_50_pruned_hardfakevsreal.")
    models = []
    print(f"Loading {len(model_paths)} pruned models on {device}...")
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
            
            # 🔴 پوشاندن مدل با DataParallel برای 2 GPU
            if world_size > 1:
                model = nn.DataParallel(model)
                
            models.append(model)
        except Exception as e:
            print(f" [ERROR] Failed to load {path}: {e}")
    if len(models) == 0: raise ValueError("No models loaded!")
    print(f"All {len(models)} models loaded!\n")
    return models

# ================== EVALUATION FUNCTIONS ==================
@torch.no_grad()
def evaluate_single_model(model: nn.Module, loader: DataLoader, device: torch.device,
                          name: str, mean: Tuple[float, float, float],
                          std: Tuple[float, float, float]) -> float:
    model.eval()
    normalizer = MultiModelNormalization([mean], [std]).to(device)
    correct = 0
    total = 0
    if loader is None: return 0.0
    for images, labels in tqdm(loader, desc=f"Evaluating {name}"):
        images, labels = images.to(device), labels.to(device).float()
        images = normalizer(images, 0)
        out = model(images)
        if isinstance(out, (tuple, list)): out = out[0]
        pred = (out.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()
        
    acc = 100. * correct / total if total > 0 else 0.0
    print(f" {name}: {acc:.2f}%")
    return acc

@torch.no_grad()
def evaluate_ensemble_final(model, loader, device, name, model_names):
    model.eval()
    local_stats = torch.zeros(5, device=device)
    vote_distribution = torch.zeros(len(model_names), 2, device=device)
    
    if loader is None: return 0.0, [], []
    print(f"\nEvaluating {name} set (Majority Voting - Hard)...")
    
    for images, labels in tqdm(loader, desc=f"Evaluating {name}"):
        images, labels = images.to(device), labels.to(device)
        outputs, weights, _, stacked_logits = model(images, return_details=True)
        
        pred = outputs.squeeze(1).long()
        
        is_tp = ((pred == 1) & (labels.long() == 1)).sum()
        is_tn = ((pred == 0) & (labels.long() == 0)).sum()
        is_fp = ((pred == 1) & (labels.long() == 0)).sum()
        is_fn = ((pred == 0) & (labels.long() == 1)).sum()
        local_stats[0] += is_tp
        local_stats[1] += is_tn
        local_stats[2] += is_fp
        local_stats[3] += is_fn
        local_stats[4] += labels.size(0)
        
        current_votes = (stacked_logits > 0).long() 
        for i in range(len(model_names)):
            fake_count = (current_votes[:, i] == 0).sum()
            real_count = (current_votes[:, i] == 1).sum()
            vote_distribution[i, 0] += fake_count
            vote_distribution[i, 1] += real_count

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
    print(f"{name.upper()} SET RESULTS (Majority Voting - Hard)")
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

# ================== UNIFIED FINAL EVALUATION ==================
def final_evaluation_unified(model, test_loader_full, device, save_dir, model_names, args):
    model.eval()
    subset_dataset = test_loader_full.dataset 

    lines = []
    lines.append("="*100)
    lines.append("SAMPLE-BY-SAMPLE PREDICTIONS (Hard Voting Logic):")
    lines.append("="*100)
    header = f"{'ID':<10} {'Filename':<60} {'True':<10} {'Pred':<10} {'Correct':<10}"
    lines.append(header)
    lines.append("-"*100)

    TP, TN, FP, FN = 0, 0, 0, 0
    all_y_true = []
    all_y_score = [] 
    all_y_pred = []
    
    print(f"\nRunning Unified Final Evaluation on {len(subset_dataset)} samples...")
    
    with torch.no_grad():
        for i in tqdm(range(len(subset_dataset)), desc="Final Eval"):
            try:
                image, label = subset_dataset[i]
                original_idx = subset_dataset.indices[i]
                path, _ = get_sample_info(subset_dataset, original_idx)
            except Exception as e:
                continue

            image = image.unsqueeze(0).to(device)
            label_int = int(label)
            
            decision, _, avg_probs, stacked_logits = model(image, return_details=True)
            pred_int = int(decision.squeeze().item())
            score_for_roc = avg_probs.squeeze().item()
            
            all_y_true.append(label_int)
            all_y_score.append(score_for_roc)
            all_y_pred.append(pred_int)
            
            if label_int == 1:
                if pred_int == 1: TP += 1
                else: FN += 1
            else:
                if pred_int == 1: FP += 1
                else: TN += 1
            
            correct_str = "Yes" if pred_int == label_int else "No"
            filename = os.path.basename(path)
            if len(filename) > 55: filename = filename[:25] + "..." + filename[-27:]
            line = f"{i+1:<10} {filename:<60} {label_int:<10} {pred_int:<10} {correct_str:<10}"
            lines.append(line)

    total_samples = len(all_y_true)
    correct_count = TP + TN
    acc = 100.0 * correct_count / total_samples if total_samples > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    print(f"\n{'='*70}")
    print("FINAL RESULTS (HARD VOTING)")
    print(f"{'='*70}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted Real  Predicted Fake")
    print(f"    Actual Real      {TP:<15} {FN:<15}")
    print(f"    Actual Fake      {FP:<15} {TN:<15}")
    print(f"\nCorrect Predictions: {correct_count} ({acc:.2f}%)")
    print(f"Incorrect Predictions: {total_samples - correct_count} ({(1-acc)*100:.2f}%)")
    print("="*70)

    output_str = []
    output_str.append("-" * 100)
    output_str.append("SUMMARY STATISTICS:")
    output_str.append("-" * 100)
    output_str.append(f"Accuracy: {acc:.2f}%")
    output_str.append(f"Precision: {precision:.4f}")
    output_str.append(f"Recall: {recall:.4f}")
    output_str.append(f"Specificity: {specificity:.4f}")
    output_str.append("\nConfusion Matrix:")
    output_str.append(f"                 {'Predicted Real':<15} {'Predicted Fake':<15}")
    output_str.append(f"    Actual Real   {TP:<15} {FN:<15}")
    output_str.append(f"    Actual Fake   {FP:<15} {TN:<15}")
    output_str.append(f"\nCorrect Predictions: {correct_count} ({acc:.2f}%)")
    output_str.append(f"Incorrect Predictions: {total_samples - correct_count} ({(1-acc)*100:.2f}%)")
    output_str.extend(lines)
    
    log_path = os.path.join(save_dir, 'prediction_log.txt')
    with open(log_path, 'w') as f:
        f.write("\n".join(output_str))
    print(f"✅ Prediction log saved to: {log_path}")

    print("\nCollecting ROC data (y_true & y_score) ...")
    
    y_true_np = np.array(all_y_true)
    y_score_np = np.array(all_y_score)
    y_pred_np = np.array(all_y_pred)

    roc_json_path = os.path.join(save_dir, "roc_data_test.json")
    roc_data_json = {
        "metadata": {
            "seed": args.seed,
            "dataset": args.dataset,
            "num_samples": int(total_samples),
            "positive_count": int(np.sum(y_true_np)),
            "negative_count": int(total_samples - np.sum(y_true_np)),
            "model": "majority_voting_ensemble_HARD_AdaBN"
        },
        "y_true": y_true_np.tolist(),
        "y_score": y_score_np.tolist(),
        "y_pred": y_pred_np.tolist()
    }
    with open(roc_json_path, 'w', encoding='utf-8') as f:
        json.dump(roc_data_json, f, indent=2, ensure_ascii=False)
    print(f"ROC data saved (JSON): {roc_json_path}")

    roc_txt_path = os.path.join(save_dir, "roc_data_test.txt")
    with open(roc_txt_path, 'w', encoding='utf-8') as f:
        f.write("y_true\ty_score\ty_pred\n")
        for t, s, p in zip(y_true_np, y_score_np, y_pred_np):
            f.write(f"{int(t)}\t{s:.6f}\t{int(p)}\n")
    print(f"ROC data saved (TXT):  {roc_txt_path}")

    try:
        from visualization_utils import generate_visualizations
        vis_dir = os.path.join(save_dir, 'visualizations')
        generate_visualizations(
            model, test_loader_full, device, vis_dir, model_names,
            args.num_grad_cam_samples, args.num_lime_samples,
            args.dataset, is_main=True)
    except:
        pass

    return acc, y_true_np, y_score_np

# ================== MAIN FUNCTION ==================
def main():
    parser = argparse.ArgumentParser(description="Majority Voting Ensemble Evaluation (HARD VOTING) - Multi GPU + AdaBN")
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
    
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available! Please run this code on a machine with a GPU.")
    
    # 🔴 تشخیص تعداد GPU های موجود
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"Warning: {world_size} GPU(s) detected. For full performance, 2 or more GPUs are recommended.")
    
    # استفاده از همه GPU های موجود
    device_ids = list(range(world_size))
    device = torch.device(f'cuda:{device_ids[0]}')  # cuda:0 به عنوان دستگاه اصلی
    
    print(f"Using {world_size} GPU(s): {device_ids}")
    for idx in device_ids:
        print(f" - GPU {idx}: {torch.cuda.get_device_name(idx)}")

    print("="*70)
    print(f"MAJORITY VOTING ENSEMBLE EVALUATION (HARD VOTING) - Multi GPU")
    print(f"Seed: {args.seed}")
    print("="*70)

    MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]

    # ارسال world_size برای پوشاندن مدل‌ها با DataParallel
    base_models = load_pruned_models(args.model_paths, device, world_size)
    MODEL_NAMES = args.model_names[:len(base_models)]
    MEANS = MEANS[:len(base_models)]
    STDS = STDS[:len(base_models)]

    # =================== فراخوانی دقیق تابع AdaBN ===================
    adabn_loader = create_adabn_dataloader(
        args.data_dir, args.batch_size * world_size, num_workers=4 * world_size, # ضرب در تعداد GPU برای استفاده بهینه
        dataset_type=args.dataset, seed=args.seed, is_main=True
    )
    adapt_batchnorm_for_new_dataset(base_models, MEANS, STDS, adabn_loader, device, is_main=True)
    
    del adabn_loader
    torch.cuda.empty_cache()
    # ===============================================================

    # ساخت انسامبل
    ensemble = MajorityVotingEnsemble(base_models, MEANS, STDS, freeze_models=True).to(device)

    # 🔴 پوشاندن مدل انسامبل با DataParallel
    if world_size > 1:
        ensemble = nn.DataParallel(ensemble, device_ids=device_ids)

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            args.data_dir, args.batch_size * world_size, dataset_type=args.dataset,
            is_distributed=False, seed=args.seed, is_main=True)
    except FileNotFoundError as e:
        print(f"\n[ERROR] Dataset loading failed: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

    print("\n" + "="*70)
    print("INDIVIDUAL MODEL PERFORMANCE (After AdaBN)")
    print("="*70)
    individual_accs = []
    for i, model in enumerate(base_models):
        acc = evaluate_single_model(model, test_loader, device, f"Model {i+1} ({MODEL_NAMES[i]})", MEANS[i], STDS[i])
        individual_accs.append(acc)

    best_single = max(individual_accs) if individual_accs else 0.0
    best_idx = individual_accs.index(best_single) if individual_accs else 0
    print(f"\nBest Single: Model {best_idx+1} ({MODEL_NAMES[best_idx]}) → {best_single:.2f}%")

    ensemble_test_acc, vote_dist, stats = evaluate_ensemble_final(ensemble, test_loader, device, "Test", MODEL_NAMES)

    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "="*70)
    print("FINAL ENSEMBLE EVALUATION (Unified - Hard Voting)")
    print("="*70)

    _, _, test_loader_full = create_dataloaders(
        args.data_dir, args.batch_size * world_size, dataset_type=args.dataset,
        is_distributed=False, seed=args.seed, is_main=True
    )

    final_acc, y_true, y_scores = final_evaluation_unified(
        ensemble, test_loader_full, device, args.save_dir, MODEL_NAMES, args
    )

    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"Best Single Model: {best_single:.2f}%")
    print(f"Ensemble Accuracy (Hard Voting): {final_acc:.2f}%")
    print(f"Improvement: {final_acc - best_single:+.2f}%")
    print("="*70)

    save_ensemble_checkpoint(os.path.join(args.save_dir, 'best_ensemble_model.pt'), ensemble, args.model_paths, MODEL_NAMES, final_acc, MEANS, STDS)

    final_results = {
        'seed': args.seed,
        'method': 'Majority Voting (Hard) + AdaBN',
        'best_single_model': {'name': MODEL_NAMES[best_idx], 'accuracy': float(best_single)},
        'ensemble': {'test_accuracy': float(final_acc), 'vote_distribution': {name: {'fake': int(d[0]), 'real': int(d[1])} for name, d in zip(MODEL_NAMES, vote_dist)}},
        'improvement': float(final_acc - best_single)
    }
    with open(os.path.join(args.save_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)

    plot_roc_and_f1(
        ensemble,
        test_loader_full, 
        device, 
        args.save_dir, 
        MODEL_NAMES,
        is_main=True
    )

if __name__ == "__main__":
    main()
