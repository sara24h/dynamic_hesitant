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

from dataset_utils import (
    UADFVDataset, 
    create_dataloaders, 
    get_sample_info, 
    worker_init_fn
)

from dataset_utils import (
    DFDDataset, 
    create_dataloaders, 
    get_sample_info, 
    worker_init_fn
)


from lime import lime_image
from skimage.segmentation import mark_boundaries

# ================== UTILITY FUNCTIONS ==================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


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
        # Collect votes (logits) from all models
        votes_logits = []
        for i in range(self.num_models):
            # Check if normalization is needed (if std is not 0)
            current_std = self.normalizations.__getattr__(f'std_{i}')
            
            x_n = x
            # Only normalize if std is not all zeros
            if not torch.all(current_std == 0):
                x_n = self.normalizations(x, i)
            
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            votes_logits.append(out)

        # Stack: (Batch, NumModels)
        stacked_logits = torch.cat(votes_logits, dim=1)
        
        # Convert logits to hard votes (0 or 1)
        hard_votes = (stacked_logits > 0).long()

        # Majority Voting
        final_vote, _ = torch.mode(hard_votes, dim=1)
        final_output = final_vote.float().unsqueeze(1) # Shape: (Batch, 1)

        if return_details:
            batch_size = x.size(0)
            # Return dummy weights (uniform) to match signature
            weights = torch.ones(batch_size, self.num_models, device=x.device) / self.num_models
            
            # Dummy memberships to match original signature
            dummy_memberships = torch.zeros(batch_size, self.num_models,3, device=x.device)
            
            return final_output, weights, dummy_memberships, hard_votes.float()

        return final_output, hard_votes

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

    correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
    total_tensor = torch.tensor(total, dtype=torch.long, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    acc = 100. * correct_tensor.item() / total_tensor.item()
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

    correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
    total_tensor = torch.tensor(total, dtype=torch.long, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    acc = 100. * correct_tensor.item() / total_tensor.item()
    return acc


@torch.no_grad()
def evaluate_ensemble_final_ddp(model, loader, device, name, model_names, is_main=True):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    # For majority voting stats
    # count_fake, count_real per model
    vote_distribution = torch.zeros(len(model_names), 2, device=device)
    unanimous_samples = 0

    if is_main:
        print(f"\nEvaluating {name} set (Majority Voting)...")
        
    for images, labels in tqdm(loader, desc=f"Evaluating {name}", disable=not is_main):
        images, labels = images.to(device), labels.to(device)
        # outputs: final majority vote (0 or 1)
        # votes: individual votes (0 or 1) [Batch, NumModels]
        outputs, weights, _, votes = model(images, return_details=True)
        
        pred = (outputs.squeeze(1) > 0).long()
        total_correct += pred.eq(labels.long()).sum().item()
        total_samples += labels.size(0)
        
        # Update Vote Distribution
        # Count Real votes (1s) and Fake votes (0s)
        real_votes = votes.sum(dim=0) # Sum over batch -> shape (NumModels)
        fake_votes = votes.size(0) - real_votes
        vote_distribution[:, 0] += fake_votes
        vote_distribution[:, 1] += real_votes

        # Check Unanimity: all models voted the same
        # If variance of votes is 0, they are unanimous
        if (votes.std(dim=1) < 1e-6).all():
            unanimous_samples += votes.size(0)

    stats = torch.tensor([total_correct, total_samples, unanimous_samples], dtype=torch.long, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    dist.all_reduce(vote_distribution, op=dist.ReduceOp.SUM)

    if is_main:
        total_correct = stats[0].item()
        total_samples = stats[1].item()
        unanimous_samples = stats[2].item()
        
        acc = 100. * total_correct / total_samples
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


# ================== LIME EXPLANATION ==================
def generate_lime_explanation(model, image_tensor, device, target_size=(256, 256)):
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


# ================== VISUALIZATION FUNCTIONS ==================
def generate_visualizations(ensemble, test_loader, device, vis_dir, model_names,
                           num_gradcam, num_lime, dataset_type, is_main):
    if not is_main:
        return
    print("="*70)
    print("GENERATING VISUALIZATIONS (Majority Voting)")
    print("="*70)

    dirs = {k: os.path.join(vis_dir, k) for k in ['gradcam', 'lime', 'combined']}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    ensemble.eval()
    local_ensemble = ensemble.module if hasattr(ensemble, 'module') else ensemble
    for model in local_ensemble.models:
        model.eval()

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

        # Get prediction
        with torch.no_grad():
            outputs, weights, _, votes = ensemble(image, return_details=True)
        pred = int(outputs.squeeze().item())
        
        # Identify models agreeing with majority
        # votes is (1, num_models)
        agreeing_mask = (votes[0] == pred)
        agreeing_indices = agreeing_mask.nonzero(as_tuple=True)[0].cpu().tolist()

        print(f"\n[Visualization {idx+1}] True: {'real' if true_label == 1 else 'fake'} | Pred: {'real' if pred == 1 else 'fake'}")
        print(f"  Models agreeing with majority: {len(agreeing_indices)}/{len(model_names)}")
        
        filename = f"sample_{idx}_true{'real' if true_label == 1 else 'fake'}_pred{'real' if pred == 1 else 'fake'}.png"

        # ---------- GradCAM ----------
        if idx < num_gradcam:
            try:
                combined_cam = None
                # Average GradCAM of models that agreed with the majority vote
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
                    plt.title(f"GradCAM (Agreeing Models)\nTrue: {'real' if true_label == 1 else 'fake'} | Pred: {'real' if pred == 1 else 'fake'}")
                    plt.axis('off')
                    plt.savefig(os.path.join(dirs['gradcam'], filename), bbox_inches='tight', dpi=200)
                    plt.close()
                    print(f"  GradCAM saved")
            except Exception as e:
                print(f"  GradCAM error: {e}")

        # ---------- LIME ----------
        if idx < num_lime:
            try:
                lime_img = generate_lime_explanation(ensemble, image, device)
                plt.figure(figsize=(10, 10))
                plt.imshow(lime_img)
                plt.title(f"LIME\nTrue: {'real' if true_label == 1 else 'fake'} | Pred: {'real' if pred == 1 else 'fake'}")
                plt.axis('off')
                plt.savefig(os.path.join(dirs['lime'], filename), bbox_inches='tight', dpi=200)
                plt.close()
                print(f"  LIME saved")

                if idx < num_gradcam and 'overlay' in locals():
                    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                    axes[0].imshow(img_np)
                    axes[0].set_title("Original")
                    axes[0].axis('off')
                    axes[1].imshow(overlay)
                    axes[1].set_title("GradCAM")
                    axes[1].axis('off')
                    axes[2].imshow(lime_img)
                    axes[2].set_title("LIME")
                    axes[2].axis('off')
                    plt.suptitle(f"True: {'real' if true_label == 1 else 'fake'} | Pred: {'real' if pred == 1 else 'fake'}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(dirs['combined'], filename), bbox_inches='tight', dpi=200)
                    plt.close()
                    print(f"  Combined saved")
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
                       choices=['wild', 'real_fake', 'hard_fake_real', 'deepflux', 'uadfV'])
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
        print(f"MAJORITY VOTING ENSEMBLE EVALUATION")
        print(f"Distributed on {world_size} GPU(s) | Seed: {args.seed}")
        print("="*70)
        print(f"Dataset: {args.dataset}")
        print(f"Data directory: {args.data_dir}")
        print(f"Batch size: {args.batch_size}")
        print(f"Models: {len(args.model_paths)}")
        print("="*70 + "\n")

    MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]

    base_models = load_pruned_models(args.model_paths, device, is_main)
    MODEL_NAMES = args.model_names[:len(base_models)]

    # Adjust MEANS/STDS based on loaded models
    MEANS = MEANS[:len(base_models)]
    STDS = STDS[:len(base_models)]

    ensemble = MajorityVotingEnsemble(
        base_models, MEANS, STDS,
        freeze_models=True
    ).to(device)

    # Handle Distributed Wrapping
    # We won't wrap in DDP if there are no parameters to optimize (avoids error)
    # But we use DP for speed if needed, or just raw model.
    # Here we stick to raw model + distributed samplers.
    
    if is_main:
        trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
        total = sum(p.numel() for p in ensemble.parameters())
        print(f"Total params: {total:,} | Trainable: {trainable:,} | Frozen: {total-trainable:,}\n")

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

    best_single = max(individual_accs)
    best_idx = individual_accs.index(best_single)

    if is_main:
        print(f"\nBest Single: Model {best_idx+1} ({MODEL_NAMES[best_idx]}) → {best_single:.2f}%")
        print("="*70)
        print("\nSkipping training phase (Majority Voting is rule-based).")
        print("Proceeding directly to evaluation...\n")

    # Directly evaluate without training
    ensemble_module = ensemble.module if hasattr(ensemble, 'module') else ensemble
    
    if is_main:
        print("\n" + "="*70)
        print("FINAL ENSEMBLE EVALUATION")
        print("="*70)

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

        os.makedirs(args.save_dir, exist_ok=True)
        results_path = os.path.join(args.save_dir, 'final_results_majority_voting.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"\nResults saved: {results_path}")

        final_model_path = os.path.join(args.save_dir, 'majority_voting_model_info.json')
        with open(final_model_path, 'w') as f:
            json.dump({
                'model_names': MODEL_NAMES,
                'model_paths': args.model_paths
            }, f, indent=4)
        print(f"Model info saved: {final_model_path}")

        vis_dir = os.path.join(args.save_dir, 'visualizations')
        generate_visualizations(
            ensemble_module, test_loader, device, vis_dir, MODEL_NAMES,
            args.num_grad_cam_samples, args.num_lime_samples,
            args.dataset, is_main)

    cleanup_distributed()

if __name__ == "__main__":
    main()
