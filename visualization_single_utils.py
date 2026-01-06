import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME is not installed. Skipping LIME visualizations.")

def get_sample_info(dataset, index):
    """Helper to get file path from dataset."""
    if hasattr(dataset, 'samples'):
        return dataset.samples[index]
    elif hasattr(dataset, 'dataset'):
        return get_sample_info(dataset.dataset, index)
    else:
        # Fallback for simple datasets
        return f"index_{index}", 0

# ================== GRADCAM CLASS ==================
class GradCAM:
    """Optimized GradCAM for Single Models"""
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
        score.backward(retain_graph=True)
        if self.gradients is None:
            raise RuntimeError("Gradients not captured – forgot requires_grad_(True)?")

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=[1, 2], keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=activations.shape[1:], # Resize to original feature map size
            mode='bilinear',
            align_corners=False
        )
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.squeeze().cpu().numpy()

# ================== LIME EXPLANATION ==================
def generate_lime_explanation(model, image_tensor, device, target_size=(256, 256)):
    """
    Generate LIME explanation for a single model.
    Assumes model takes (0-1) tensors and returns logits.
    """
    if not LIME_AVAILABLE:
        raise ImportError("LIME library is not installed.")

    # Convert tensor to numpy image (0-255)
    img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()
    # If image is normalized (0-1), convert to 0-255. 
    # Assuming the wrapper handles raw inputs or we de-normalize if needed.
    # Here we assume input is 0-1 as passed from dataloader.
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        # images are numpy (0-255), convert to tensor (0-1)
        batch = torch.from_numpy(images.transpose(0, 3, 1, 2)).float() / 255.0
        batch = batch.to(device)
        
        with torch.no_grad():
            # Pass through the wrapped model (which handles normalization internally)
            outputs = model(batch)
            probs = torch.sigmoid(outputs).cpu().numpy()
        
        # LIME expects probabilities for all classes (Fake, Real)
        # Class 0: Fake, Class 1: Real
        return np.hstack([1 - probs, probs])

    explanation = explainer.explain_instance(
        img_np, predict_fn, top_labels=1, hide_color=0, num_samples=1000)
    
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
    
    lime_img = mark_boundaries(temp / 255.0, mask)
    lime_img = cv2.resize(lime_img, target_size)
    return lime_img

# ================== VISUALIZATION FUNCTIONS ==================
def generate_visualizations(model, test_loader, device, vis_dir, model_names,
                           num_gradcam, num_lime, dataset_type, is_main):
    """
    Main function to generate GradCAM and LIME for Single Models.
    Compatible with the VisualizationWrapper used in single_models.py
    """
    if not is_main:
        return
    
    print("="*70)
    print("GENERATING VISUALIZATIONS (Single Model)")
    print("="*70)

    # Create directories
    dirs = {k: os.path.join(vis_dir, k) for k in ['gradcam', 'lime', 'combined']}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    model.eval()
    
    # Access the underlying raw model for GradCAM hooks if needed, 
    # but our VisualizationWrapper handles forward passes.
    # We assume 'model' is the VisualizationWrapper instance.
    
    # Get the base model to find the target layer
    if hasattr(model, 'base_model'):
        base_model = model.base_model
    elif hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model # Fallback

    # Determine target layer (usually the last conv block)
    # Try common ResNet structure
    if hasattr(base_model, 'layer4'):
        target_layer = base_model.layer4[-1] 
    elif hasattr(base_model, 'features') and hasattr(base_model.features, '24'): # VGG/DenseNet like
        target_layer = list(base_model.features.children())[-1]
    else:
        print("Warning: Could not auto-detect target layer. Using last module.")
        target_layer = list(base_model.children())[-1]

    # دریافت دیتاست اصلی
    full_dataset = test_loader.dataset
    if hasattr(full_dataset, 'dataset'):
        full_dataset = full_dataset.dataset

    total_samples = len(full_dataset)
    vis_count = min(max(num_gradcam, num_lime), total_samples)
    
    if vis_count == 0:
        print("No samples requested for visualization.")
        return

    # Select random indices
    vis_indices = random.sample(range(total_samples), vis_count)
    vis_dataset = Subset(full_dataset, vis_indices)
    
    # Use num_workers=0 to avoid multiprocessing issues with file paths
    vis_loader = DataLoader(vis_dataset, batch_size=1, shuffle=False, num_workers=0)

    for idx, (image, label_int) in enumerate(vis_loader):
        image = image.to(device)
        true_label = int(label_int.item())
        
        # --- Get Path Info ---
        try:
            info = get_sample_info(full_dataset, vis_indices[idx])
            img_path = info[0] if isinstance(info, (list, tuple)) else str(info)
            img_name = os.path.basename(img_path)
        except:
            img_name = f"sample_{idx}"

        print(f"\n[Visualization {idx+1}/{len(vis_loader)}] {img_name}")
        print(f"  True Label: {'Real' if true_label == 1 else 'Fake'}")

        with torch.no_grad():
            output = model(image)
        
        pred_prob = torch.sigmoid(output).item()
        pred_label = 1 if pred_prob > 0.5 else 0
        print(f"  Pred Label: {'Real' if pred_label == 1 else 'Fake'} ({pred_prob:.4f})")

        # Filename for saving
        filename = f"sample_{idx}_true{true_label}_pred{pred_label}.png"

        # Prepare raw image for overlays
        # Note: DataLoader usually returns 0-1 tensors. 
        img_np = image[0].cpu().permute(1, 2, 0).numpy()
        
        # ---------- GradCAM ----------
        if idx < num_gradcam:
            try:
                # We pass the WRAPPED model to GradCAM, but hook into the BASE model's layer
                gradcam = GradCAM(model, target_layer)
                
                # To get gradients, we need requires_grad=True
                image_input = image.clone().detach().requires_grad_(True)
                
                # Forward pass through wrapper
                model_out = model(image_input)
                if isinstance(model_out, tuple):
                    model_out = model_out[0]
                
                # Calculate score for backward
                # If prediction is Real (1), maximize it. If Fake (0), minimize it (or maximize 1 - pred)
                if pred_label == 1:
                    score = model_out.squeeze(0)
                else:
                    score = -model_out.squeeze(0)
                
                cam = gradcam.generate(score)
                
                # Resize CAM to image size
                img_h, img_w = img_np.shape[:2]
                cam_resized = cv2.resize(cam, (img_w, img_h))
                cam_resized = np.uint8(255 * cam_resized)
                
                # Apply colormap
                heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                overlay = heatmap + img_np
                overlay = overlay / np.max(overlay)

                plt.figure(figsize=(10, 10))
                plt.imshow(overlay)
                plt.title(f"GradCAM - {img_name}\nTrue: {true_label} | Pred: {pred_label} ({pred_prob:.2f})")
                plt.axis('off')
                plt.savefig(os.path.join(dirs['gradcam'], filename), bbox_inches='tight', dpi=200)
                plt.close()
                print(f"  [OK] GradCAM saved")
            except Exception as e:
                print(f"  [ERR] GradCAM failed: {e}")

        # ---------- LIME ----------
        if idx < num_lime and LIME_AVAILABLE:
            try:
                lime_img = generate_lime_explanation(model, image, device)
                plt.figure(figsize=(10, 10))
                plt.imshow(lime_img)
                plt.title(f"LIME - {img_name}\nTrue: {true_label} | Pred: {pred_label}")
                plt.axis('off')
                plt.savefig(os.path.join(dirs['lime'], filename), bbox_inches='tight', dpi=200)
                plt.close()
                print(f"  [OK] LIME saved")

                # Combined View
                if idx < num_gradcam:
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
                    
                    plt.suptitle(f"{img_name} (True: {true_label}, Pred: {pred_label})", fontsize=16)
                    plt.tight_layout()
                    plt.savefig(os.path.join(dirs['combined'], filename), bbox_inches='tight', dpi=200)
                    plt.close()
                    print(f"  [OK] Combined saved")
            except Exception as e:
                print(f"  [ERR] LIME failed: {e}")

    print("="*70)
    print("Visualizations completed!")
    print(f"Saved to: {vis_dir}")
    print("="*70)
