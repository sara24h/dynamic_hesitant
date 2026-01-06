import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
import warnings
import argparse
import json
from typing import List

warnings.filterwarnings("ignore")

# Import Dataset Utilities
from dataset_utils import create_dataloaders

# Import Visualization Utilities
from visualization_utils import GradCAM, generate_lime_explanation, generate_visualizations
#from metrics_utils import plot_roc_and_f1

# ================== WRAPPER FOR NORMALIZATION ==================
class NormalizationWrapper(nn.Module):
    
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        # Convert mean and std to tensors and reshape for (batch, channel, h, w)
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x, return_details=False):
        # Normalize input using the FIXED values passed during init
        x_norm = (x - self.mean) / self.std
        
        # Pass through model
        out = self.model(x_norm)
        
        # Handle models that return tuples (e.g., (logits, features))
        if isinstance(out, (tuple, list)):
            out = out[0]
        
        # Compatibility fix: If caller asks for details, just return logits 
        # (since single models don't have ensemble weights/memberships to return)
        return out

# ================== MODEL LOADING ==================
def load_pruned_models(model_paths: List[str], device: torch.device) -> List[nn.Module]:
    try:
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        raise ImportError("Cannot import ResNet_50_pruned_hardfakevsreal. Ensure 'model' directory exists.")

    models = []
    print(f"Loading {len(model_paths)} pruned models...")

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
            param_count = sum(p.numel() for p in model.parameters())
            print(f" → Parameters: {param_count:,}")
            models.append(model)
        except Exception as e:
            print(f" [ERROR] Failed to load {path}: {e}")
            continue

    if len(models) == 0:
        raise ValueError("No models loaded!")
    print(f"All {len(models)} models loaded!\n")
    return models

# ================== EVALUATION & VISUALIZATION ==================
@torch.no_grad()
def evaluate_single_model(model, loader, device, name):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    print(f"\nEvaluating {name}...")
    for images, labels in tqdm(loader, desc=f"Evaluating {name}", disable=False):
        images, labels = images.to(device), labels.to(device).float()
        
        outputs = model(images)
        preds_prob = torch.sigmoid(outputs.squeeze(1))
        preds = (preds_prob > 0.5).long()
        
        total += labels.size(0)
        correct += preds.eq(labels.long()).sum().item()
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds_prob.cpu().numpy())

    acc = 100. * correct / total
    print(f" {name} Accuracy: {acc:.2f}%")
    return acc, np.array(all_labels), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Single Model Evaluation & Visualization")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['wild', 'real_fake', 'hard_fake_real', 'deepflux', 'uadfV', 'dfd'])
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, help='Paths to .pt model files')
    parser.add_argument('--model_names', type=str, nargs='+', required=True, help='Names for the models')
    parser.add_argument('--save_dir', type=str, default='./output_single_eval', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)
    args = parser.parse_args()

    if len(args.model_names) != len(args.model_paths):
        raise ValueError("Number of model_names must match model_paths")

    # Setup Device (No Distributed / DDP logic)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create Save Directory
    os.makedirs(args.save_dir, exist_ok=True)

    # ================== FIXED NORMALIZATION VALUES ==================
    MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    
    MEANS = MEANS[:len(args.model_paths)]
    STDS = STDS[:len(args.model_paths)]

    # Load Data
    print("Loading Data...")
    _, _, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=False, seed=42, is_main=True
    )

    # Load Base Models
    base_models = load_pruned_models(args.model_paths, device)

    results_summary = {}

    # ================== LOOP OVER MODELS ==================
    for i, model in enumerate(base_models):
        model_name = args.model_names[i]
        print("="*70)
        print(f"PROCESSING MODEL: {model_name}")
        print("="*70)

        # 1. Wrap Model with Normalization (Compatible Wrapper)
        wrapped_model = NormalizationWrapper(model, MEANS[i], STDS[i]).to(device)
        
        # 2. Evaluate
        acc, labels, preds = evaluate_single_model(wrapped_model, test_loader, device, model_name)
        
        # Save metrics for this model
        model_results_dir = os.path.join(args.save_dir, model_name.replace(" ", "_"))
        os.makedirs(model_results_dir, exist_ok=True)
        
        # 3. Plot ROC and F1
        # plot_roc_and_f1 likely calls model(x, return_details=True) internally.
        # Our wrapper now handles that argument safely.
        plot_roc_and_f1(
            wrapped_model, 
            test_loader, 
            device, 
            model_results_dir, 
            [model_name], 
            is_main=True
        )

        # 4. Generate Visualizations (GradCAM & LIME)
        vis_dir = os.path.join(model_results_dir, 'visualizations')
        print(f"\nGenerating Visualizations for {model_name} in {vis_dir}...")
        
        try:
            generate_visualizations(
                model=wrapped_model, 
                data_loader=test_loader, 
                device=device, 
                save_dir=vis_dir, 
                model_names=[model_name],
                num_grad_cam=args.num_grad_cam_samples,
                num_lime=args.num_lime_samples,
                dataset_name=args.dataset,
                is_main=True
            )
            print(f"Visualizations saved successfully.")
        except Exception as e:
            print(f"Error generating visualizations: {e}")

        results_summary[model_name] = {
            'accuracy': acc,
            'mean': MEANS[i],
            'std': STDS[i]
        }

    # Save Final Summary
    summary_path = os.path.join(args.save_dir, 'evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print("\n" + "="*70)
    print("ALL MODELS PROCESSED.")
    print(f"Summary saved to: {summary_path}")
    print("="*70)

if __name__ == "__main__":
    main()
