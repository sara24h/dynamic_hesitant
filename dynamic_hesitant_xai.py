import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict, Any
import warnings
import argparse
import shutil
import json
import random
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import cv2

warnings.filterwarnings("ignore")

# ====================== GRAD-CAM IMPLEMENTATION ======================
class GradCAM:
    """Grad-CAM implementation for CNN models"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """Generate CAM for input image"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        
        # Backward pass
        self.model.zero_grad()
        if target_class is None:
            target_class = (output > 0).long()
        
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target_class.view(-1, 1), 1.0)
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Calculate weights
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Generate CAM
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def __call__(self, input_tensor, target_class=None):
        return self.generate_cam(input_tensor, target_class)


class HybridGradCAM:
    """Hybrid Grad-CAM for ensemble models with fuzzy weights"""
    def __init__(self, ensemble_model, target_layers):
        """
        Args:
            ensemble_model: The fuzzy hesitant ensemble model
            target_layers: List of target layers for each base model
        """
        self.ensemble_model = ensemble_model
        self.num_models = len(target_layers)
        
        # Create GradCAM for each model
        self.gradcams = []
        for i, layer in enumerate(target_layers):
            if hasattr(ensemble_model, 'module'):
                model = ensemble_model.module.models[i]
            else:
                model = ensemble_model.models[i]
            self.gradcams.append(GradCAM(model, layer))
    
    def generate_hybrid_cam(self, input_tensor, return_individual=False):
        """
        Generate hybrid CAM weighted by fuzzy membership values
        
        Args:
            input_tensor: Input image tensor
            return_individual: If True, return individual CAMs as well
            
        Returns:
            hybrid_cam: Weighted combination of all CAMs
            individual_cams: List of individual CAMs (if return_individual=True)
            weights: Fuzzy weights for each model
            memberships: Hesitant membership values
        """
        self.ensemble_model.eval()
        
        # Get fuzzy weights and memberships
        if hasattr(self.ensemble_model, 'module'):
            ensemble = self.ensemble_model.module
        else:
            ensemble = self.ensemble_model
            
        with torch.no_grad():
            _, fuzzy_weights, memberships, _ = ensemble(input_tensor, return_details=True)
        
        # Generate individual CAMs
        individual_cams = []
        batch_size = input_tensor.size(0)
        
        for i, gradcam in enumerate(self.gradcams):
            # Normalize input for this model
            normalized_input = ensemble.normalizations(input_tensor, i)
            
            # Generate CAM
            cam = gradcam(normalized_input)
            individual_cams.append(cam)
        
        # Stack all CAMs
        all_cams = torch.stack(individual_cams, dim=1)  # [B, num_models, 1, H, W]
        
        # Weight by fuzzy weights
        fuzzy_weights = fuzzy_weights.view(batch_size, self.num_models, 1, 1, 1)
        hybrid_cam = (all_cams * fuzzy_weights).sum(dim=1)  # [B, 1, H, W]
        
        if return_individual:
            return hybrid_cam, individual_cams, fuzzy_weights.squeeze(), memberships
        
        return hybrid_cam, fuzzy_weights.squeeze(), memberships


def visualize_cam(image, cam, alpha=0.5):
    """
    Overlay CAM on original image
    
    Args:
        image: Original image tensor [C, H, W] or numpy array [H, W, C]
        cam: CAM tensor [1, H, W] or [H, W]
        alpha: Transparency for overlay
        
    Returns:
        Visualization as numpy array [H, W, C]
    """
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.shape[0] == 3:  # [C, H, W] -> [H, W, C]
            image = np.transpose(image, (1, 2, 0))
    
    if isinstance(cam, torch.Tensor):
        cam = cam.cpu().numpy()
        if len(cam.shape) == 3 and cam.shape[0] == 1:  # [1, H, W]
            cam = cam[0]
    
    # Normalize image to [0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Resize CAM to match image size
    h, w = image.shape[:2]
    cam = cv2.resize(cam, (w, h))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255.0
    
    # Overlay
    overlay = alpha * heatmap + (1 - alpha) * image
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def save_xai_visualization(images, labels, predictions, hybrid_cams, individual_cams, 
                          fuzzy_weights, memberships, model_names, save_path, 
                          num_samples=4):
    """
    Save comprehensive XAI visualization
    
    Args:
        images: Batch of images
        labels: True labels
        predictions: Model predictions
        hybrid_cams: Hybrid CAMs from ensemble
        individual_cams: Individual CAMs from each model
        fuzzy_weights: Fuzzy weights for each model
        memberships: Hesitant membership values
        model_names: Names of base models
        save_path: Path to save visualization
        num_samples: Number of samples to visualize
    """
    num_models = len(individual_cams)
    num_samples = min(num_samples, images.size(0))
    
    # Create figure
    fig = plt.figure(figsize=(20, 5 * num_samples))
    
    for idx in range(num_samples):
        img = images[idx]
        label = labels[idx].item()
        pred = predictions[idx].item()
        
        # Original image
        ax = plt.subplot(num_samples, num_models + 3, idx * (num_models + 3) + 1)
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        ax.imshow(img_np)
        ax.set_title(f'Original\nTrue: {label}, Pred: {pred}', fontsize=10)
        ax.axis('off')
        
        # Individual model CAMs
        for model_idx in range(num_models):
            ax = plt.subplot(num_samples, num_models + 3, idx * (num_models + 3) + 2 + model_idx)
            cam = individual_cams[model_idx][idx]
            overlay = visualize_cam(img, cam)
            ax.imshow(overlay)
            
            weight = fuzzy_weights[idx, model_idx].item()
            hesitancy = memberships[idx, model_idx].var().item()
            
            title = f'{model_names[model_idx]}\nW: {weight:.3f}\nH: {hesitancy:.3f}'
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        
        # Hybrid CAM
        ax = plt.subplot(num_samples, num_models + 3, idx * (num_models + 3) + num_models + 2)
        hybrid_cam = hybrid_cams[idx]
        overlay = visualize_cam(img, hybrid_cam)
        ax.imshow(overlay)
        ax.set_title('Hybrid Ensemble\n(Fuzzy Weighted)', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Weights distribution
        ax = plt.subplot(num_samples, num_models + 3, idx * (num_models + 3) + num_models + 3)
        weights = fuzzy_weights[idx].cpu().numpy()
        colors = plt.cm.viridis(np.linspace(0, 1, num_models))
        bars = ax.barh(range(num_models), weights, color=colors)
        ax.set_yticks(range(num_models))
        ax.set_yticklabels([f'M{i+1}' for i in range(num_models)], fontsize=8)
        ax.set_xlabel('Fuzzy Weight', fontsize=8)
        ax.set_title('Model Weights', fontsize=10)
        ax.set_xlim([0, 1])
        
        for i, (bar, w) in enumerate(zip(bars, weights)):
            ax.text(w, i, f'{w:.3f}', va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"XAI visualization saved: {save_path}")


def get_target_layer(model, model_name='resnet'):
    """Get the target layer for Grad-CAM based on model architecture"""
    if 'resnet' in model_name.lower():
        # For ResNet, use the last conv layer before avgpool
        if hasattr(model, 'layer4'):
            return model.layer4[-1].conv3 if hasattr(model.layer4[-1], 'conv3') else model.layer4[-1].conv2
        elif hasattr(model, 'conv5'):
            return model.conv5
    
    # Fallback: find last Conv2d layer
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    
    if last_conv is None:
        raise ValueError(f"Could not find suitable target layer for {model_name}")
    
    return last_conv


@torch.no_grad()
def evaluate_single_model_with_xai(model, loader, device, model_name, normalizer, 
                                   save_dir, split_name='test', num_visualizations=8):
    """
    Evaluate a single model with XAI (Grad-CAM)
    
    Args:
        model: Single base model
        loader: DataLoader
        device: Device
        model_name: Name of the model
        normalizer: Function to normalize input (mean, std)
        save_dir: Directory to save visualizations
        split_name: Data split name
        num_visualizations: Number of samples to visualize
    """
    model.eval()
    
    # Get target layer
    try:
        target_layer = get_target_layer(model, model_name)
        print(f"  Using layer: {target_layer.__class__.__name__}")
    except Exception as e:
        print(f"  Warning: Could not find target layer: {e}")
        return None
    
    # Create GradCAM
    gradcam = GradCAM(model, target_layer)
    
    # Collect samples
    all_images = []
    all_labels = []
    all_predictions = []
    all_cams = []
    
    correct_total = 0
    total_samples = 0
    
    print(f"  Generating Grad-CAM for {model_name}...")
    
    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"  XAI-{model_name}", leave=False)):
        images = images.to(device)
        labels = labels.to(device)
        
        # Normalize input
        if normalizer is not None:
            normalized_images = normalizer(images)
        else:
            normalized_images = images
        
        # Get predictions
        outputs = model(normalized_images)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        predictions = (outputs.squeeze(1) > 0).long()
        
        correct_total += predictions.eq(labels.long()).sum().item()
        total_samples += images.size(0)
        
        # Generate CAMs
        cam = gradcam(normalized_images)
        
        # Store results
        all_images.append(images.cpu())
        all_labels.append(labels.cpu())
        all_predictions.append(predictions.cpu())
        all_cams.append(cam.cpu())
        
        if total_samples >= num_visualizations:
            break
    
    # Concatenate results
    all_images = torch.cat(all_images)[:num_visualizations]
    all_labels = torch.cat(all_labels)[:num_visualizations]
    all_predictions = torch.cat(all_predictions)[:num_visualizations]
    all_cams = torch.cat(all_cams)[:num_visualizations]
    
    accuracy = 100.0 * correct_total / total_samples
    
    # Visualize
    num_cols = 4
    num_rows = (num_visualizations + num_cols - 1) // num_cols
    
    fig = plt.figure(figsize=(16, 4 * num_rows))
    
    for idx in range(num_visualizations):
        # Original image
        ax = plt.subplot(num_rows, num_cols * 2, idx * 2 + 1)
        img = all_images[idx]
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        ax.imshow(img_np)
        label = all_labels[idx].item()
        pred = all_predictions[idx].item()
        ax.set_title(f'Original\nTrue: {label}, Pred: {pred}', fontsize=10)
        ax.axis('off')
        
        # CAM overlay
        ax = plt.subplot(num_rows, num_cols * 2, idx * 2 + 2)
        cam = all_cams[idx]
        overlay = visualize_cam(img, cam)
        ax.imshow(overlay)
        ax.set_title(f'Grad-CAM\nAcc: {accuracy:.1f}%', fontsize=10)
        ax.axis('off')
    
    plt.suptitle(f'{model_name} - {split_name.upper()} Set Grad-CAM Visualization', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    viz_path = os.path.join(save_dir, f'xai_single_{model_name}_{split_name}.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {viz_path}")
    print(f"  Accuracy: {accuracy:.2f}%\n")
    
    return {
        'model_name': model_name,
        'split': split_name,
        'accuracy': accuracy,
        'num_samples': total_samples,
        'visualization_path': viz_path
    }


@torch.no_grad()
def evaluate_with_xai(ensemble_model, loader, device, model_names, save_dir, 
                     split_name='test', num_visualizations=8, rank=0):
    """
    Evaluate ensemble with XAI visualizations
    
    Args:
        ensemble_model: The fuzzy hesitant ensemble
        loader: DataLoader
        device: Device to use
        model_names: Names of base models
        save_dir: Directory to save visualizations
        split_name: Name of data split (train/val/test)
        num_visualizations: Number of samples to visualize
        rank: Process rank for DDP
    """
    if rank != 0:
        return None, None
    
    ensemble_model.eval()
    
    # Get target layers for each model
    if hasattr(ensemble_model, 'module'):
        base_models = ensemble_model.module.models
    else:
        base_models = ensemble_model.models
    
    target_layers = []
    for i, model in enumerate(base_models):
        try:
            layer = get_target_layer(model, model_names[i])
            target_layers.append(layer)
            print(f"Model {i+1} ({model_names[i]}): Using layer {layer.__class__.__name__}")
        except Exception as e:
            print(f"Warning: Could not find target layer for {model_names[i]}: {e}")
            return None, None
    
    # Create Hybrid GradCAM
    hybrid_gradcam = HybridGradCAM(ensemble_model, target_layers)
    
    # Collect samples for visualization
    all_images = []
    all_labels = []
    all_predictions = []
    all_hybrid_cams = []
    all_individual_cams = []
    all_fuzzy_weights = []
    all_memberships = []
    
    correct_total = 0
    total_samples = 0
    
    print(f"\nGenerating XAI explanations for {split_name} set...")
    
    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"XAI-{split_name}")):
        images = images.to(device)
        labels = labels.to(device)
        
        # Get predictions
        outputs, _ = ensemble_model(images)
        predictions = (outputs.squeeze(1) > 0).long()
        
        correct_total += predictions.eq(labels.long()).sum().item()
        total_samples += images.size(0)
        
        # Generate CAMs
        hybrid_cam, individual_cams, fuzzy_weights, memberships = hybrid_gradcam.generate_hybrid_cam(
            images, return_individual=True
        )
        
        # Store results
        all_images.append(images.cpu())
        all_labels.append(labels.cpu())
        all_predictions.append(predictions.cpu())
        all_hybrid_cams.append(hybrid_cam.cpu())
        all_fuzzy_weights.append(fuzzy_weights.cpu())
        all_memberships.append(memberships.cpu())
        
        # Store individual CAMs
        if batch_idx == 0:
            all_individual_cams = [[] for _ in range(len(individual_cams))]
        for i, cam in enumerate(individual_cams):
            all_individual_cams[i].append(cam.cpu())
        
        # Stop after collecting enough samples
        if total_samples >= num_visualizations:
            break
    
    # Concatenate results
    all_images = torch.cat(all_images)[:num_visualizations]
    all_labels = torch.cat(all_labels)[:num_visualizations]
    all_predictions = torch.cat(all_predictions)[:num_visualizations]
    all_hybrid_cams = torch.cat(all_hybrid_cams)[:num_visualizations]
    all_fuzzy_weights = torch.cat(all_fuzzy_weights)[:num_visualizations]
    all_memberships = torch.cat(all_memberships)[:num_visualizations]
    
    for i in range(len(all_individual_cams)):
        all_individual_cams[i] = torch.cat(all_individual_cams[i])[:num_visualizations]
    
    # Save visualizations
    os.makedirs(save_dir, exist_ok=True)
    viz_path = os.path.join(save_dir, f'xai_{split_name}_visualization.png')
    
    save_xai_visualization(
        all_images, all_labels, all_predictions,
        all_hybrid_cams, all_individual_cams,
        all_fuzzy_weights, all_memberships,
        model_names, viz_path,
        num_samples=min(num_visualizations, 8)
    )
    
    # Calculate and save statistics
    accuracy = 100.0 * correct_total / total_samples
    
    xai_stats = {
        'split': split_name,
        'accuracy': accuracy,
        'num_samples': total_samples,
        'avg_fuzzy_weights': all_fuzzy_weights.mean(dim=0).tolist(),
        'std_fuzzy_weights': all_fuzzy_weights.std(dim=0).tolist(),
        'avg_hesitancy': all_memberships.var(dim=2).mean(dim=0).tolist(),
        'model_names': model_names
    }
    
    stats_path = os.path.join(save_dir, f'xai_{split_name}_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(xai_stats, f, indent=2)
    
    print(f"\nXAI Statistics for {split_name}:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Average Fuzzy Weights:")
    for i, (name, w, s) in enumerate(zip(model_names, xai_stats['avg_fuzzy_weights'], 
                                         xai_stats['std_fuzzy_weights'])):
        print(f"    {name}: {w:.4f} ± {s:.4f}")
    
    return xai_stats, viz_path


# ====================== REST OF THE ORIGINAL CODE ======================
# [Keep all the existing classes and functions: UADFVDataset, set_seed, worker_init_fn,
#  create_reproducible_split, prepare_*_dataset functions, setup_ddp, cleanup_ddp,
#  HesitantFuzzyMembership, MultiModelNormalization, FuzzyHesitantEnsemble,
#  load_pruned_models, TransformSubset, create_dataloaders_ddp, evaluate_single_model,
#  train_hesitant_fuzzy_ddp, evaluate_ensemble_final_ddp, evaluate_accuracy_ddp]

# [Previous code remains exactly the same - I'm not repeating it to save space]
# Copy all the code from your original file here, from UADFVDataset down to evaluate_accuracy_ddp

# ====================== UPDATED MAIN FUNCTION ======================
def main():
    SEED = 42
    set_seed(SEED)
    
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
    is_main = (rank == 0)
    
    parser = argparse.ArgumentParser(description="Train Fuzzy Hesitant Ensemble with XAI")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_memberships', type=int, default=3)
    
    parser.add_argument('--dataset', type=str, 
                       choices=['wild', 'real_fake', 'hard_fake_real', 'deepflux', 'uadfV'], 
                       required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/checkpoints')
    parser.add_argument('--seed', type=int, default=SEED)
    
    # XAI arguments
    parser.add_argument('--enable_xai', action='store_true', 
                       help='Enable XAI visualization with Grad-CAM')
    parser.add_argument('--xai_samples', type=int, default=16,
                       help='Number of samples for XAI visualization')
    
    args = parser.parse_args()
    
    if len(args.model_names) != len(args.model_paths):
        raise ValueError(f"Number of model_names must match model_paths")
    
    # [Previous initialization code remains the same]
    # MEANS, STDS, seed setting, model loading, ensemble creation, DDP wrapping
    
    # ... [Copy the initialization code from your original main function] ...
    
    # Training
    best_val_acc, history = train_hesitant_fuzzy_ddp(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_dir, rank, world_size
    )
    
    # Load best model
    ckpt_path = os.path.join(args.save_dir, 'best_hesitant_fuzzy.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ensemble.module.hesitant_fuzzy.load_state_dict(ckpt['hesitant_state_dict'])
        if is_main:
            print("Best hesitant fuzzy network loaded.\n")
    
    dist.barrier()
    
    # Evaluation with XAI
    if is_main:
        print("\n" + "="*70)
        print("EVALUATING INDIVIDUAL MODELS WITH XAI (Before Ensemble)")
        print("="*70)
        
        individual_xai_stats = []
        
        if args.enable_xai:
            xai_dir = os.path.join(args.save_dir, 'xai_individual_models')
            os.makedirs(xai_dir, exist_ok=True)
            
            # Evaluate each model with XAI
            for i, (model, name) in enumerate(zip(base_models, MODEL_NAMES)):
                print(f"\n[{i+1}/{len(base_models)}] Evaluating {name} with XAI...")
                
                # Create normalizer for this model
                mean = torch.tensor(MEANS[i]).view(1, 3, 1, 1).to(device)
                std = torch.tensor(STDS[i]).view(1, 3, 1, 1).to(device)
                normalizer = lambda x: (x - mean) / std
                
                xai_stat = evaluate_single_model_with_xai(
                    model, test_loader, device, name, 
                    normalizer, xai_dir, 
                    split_name='test',
                    num_visualizations=args.xai_samples
                )
                
                if xai_stat:
                    individual_xai_stats.append(xai_stat)
            
            print(f"\n{'='*70}")
            print("Individual Model XAI Summary:")
            print(f"{'='*70}")
            for stat in individual_xai_stats:
                print(f"  {stat['model_name']:<20}: Acc = {stat['accuracy']:.2f}%")
            print(f"{'='*70}\n")
        
        # Now evaluate ensemble
        print("\n" + "="*70)
        print("EVALUATING FUZZY HESITANT ENSEMBLE")
        print("="*70)
        
        ensemble_test_acc, ensemble_weights, membership_values, activation_percentages = \
            evaluate_ensemble_final_ddp(ensemble, test_loader, device, "Test", MODEL_NAMES, rank)
        
        # XAI Visualization for Ensemble
        if args.enable_xai:
            print("\n" + "="*70)
            print("GENERATING ENSEMBLE XAI EXPLANATIONS (Hybrid Explainability)")
            print("="*70)
            
            xai_ensemble_dir = os.path.join(args.save_dir, 'xai_ensemble')
            os.makedirs(xai_ensemble_dir, exist_ok=True)
            
            # Generate XAI for test set
            ensemble_xai_stats, viz_path = evaluate_with_xai(
                ensemble, test_loader, device, MODEL_NAMES,
                xai_ensemble_dir, split_name='test',
                num_visualizations=args.xai_samples,
                rank=rank
            )
            
            if ensemble_xai_stats:
                print(f"\nEnsemble XAI visualizations saved to: {xai_ensemble_dir}")
        
        # Save results
        results = {
            "method": "Fuzzy Hesitant Sets with Complete XAI",
            "dataset": args.dataset,
            "xai_enabled": args.enable_xai,
            "individual_models": {
                "accuracies": {MODEL_NAMES[i]: acc for i, acc in enumerate(individual_accs)},
                "best_single": {"name": MODEL_NAMES[best_idx], "acc": best_single}
            },
            "ensemble": {
                "acc": ensemble_test_acc,
                "weights": ensemble_weights,
                "membership_values": membership_values,
                "activation_percentages": activation_percentages
            },
            "improvement": improvement,
            "training_history": history
        }
        
        # Add XAI statistics
        if args.enable_xai:
            results['xai_individual_models'] = individual_xai_stats if individual_xai_stats else []
            if ensemble_xai_stats:
                results['xai_ensemble'] = ensemble_xai_stats
        
        result_path = os.path.join(args.save_dir, 'hesitant_fuzzy_xai_results.json')
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {result_path}")
        print("="*70)
        print("All done!")
    
    cleanup_ddp()

if __name__ == "__main__":
    main()
