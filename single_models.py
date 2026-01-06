import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import os
from tqdm import tqdm
import numpy as np
import warnings
import argparse
import json
import random
import matplotlib.pyplot as plt
import cv2

# Libraries for LIME and GradCAM
try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

warnings.filterwarnings("ignore")

# ================== IMPORT UTILITIES ==================
# ایمپورت از فایل‌های موجود در گیت‌هاب
from dataset_utils import create_dataloaders, get_sample_info
# این تابع را خودمان می‌نویسیم تا با ارگومان‌ها تداخل نداشته باشد
# from metrics_utils import plot_roc_and_f1 

# ================== SIMPLE AVERAGING ENSEMBLE ==================
class MultiModelNormalization(nn.Module):
    def __init__(self, means, stds):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x, idx):
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')

class SimpleAveragingEnsemble(nn.Module):
    """
    انسامبل ساده با میانگین‌گیری.
    """
    def __init__(self, models, means, stds):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)

    def forward(self, x, return_details=False):
        outputs = []
        for i, model in enumerate(self.models):
            x_norm = self.normalizations(x, i)
            out = model(x_norm)
            if isinstance(out, (tuple, list)):
                out = out[0]
            outputs.append(out)
        
        stacked_out = torch.stack(outputs, dim=1) # (Batch, Num_Models, 1)
        final_output = stacked_out.mean(dim=1) # Average
        
        if return_details:
            # Return format compatible with utils: output, weights, memberships, raw_outputs
            weights = torch.ones(x.size(0), len(self.models), device=x.device) / len(self.models)
            dummy_memberships = None
            return final_output, weights, dummy_memberships, stacked_out
        
        return final_output

# ================== INTERNAL VISUALIZATION UTILS ==================
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
        score.backward(retain_graph=True)
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=[1, 2], keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        # Resize
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=activations.shape[1:], mode='bilinear', align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.squeeze().cpu().numpy()

def generate_lime_explanation(model, image_tensor, device):
    if not LIME_AVAILABLE: return None
    img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    
    explainer = lime_image.LimeImageExplainer()
    def predict_fn(images):
        batch = torch.from_numpy(images.transpose(0, 3, 1, 2)).float() / 255.0
        batch = batch.to(device)
        with torch.no_grad():
            outputs, _, _, _ = model(batch, return_details=True) 
            probs = torch.sigmoid(outputs).cpu().numpy()
        return np.hstack([1 - probs, probs])

    explanation = explainer.explain_instance(img_np, predict_fn, top_labels=1, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
    lime_img = mark_boundaries(temp / 255.0, mask)
    return cv2.resize(lime_img, (256, 256))

def generate_visualizations_internal(model, test_loader, device, vis_dir, model_names, num_gradcam, num_lime):
    print("="*70)
    print("GENERATING VISUALIZATIONS (Simple Averaging Ensemble)")
    print("="*70)
    
    dirs = {k: os.path.join(vis_dir, k) for k in ['gradcam', 'lime', 'combined']}
    for d in dirs.values(): os.makedirs(d, exist_ok=True)

    full_dataset = test_loader.dataset
    if hasattr(full_dataset, 'dataset'): full_dataset = full_dataset.dataset

    total_samples = len(full_dataset)
    vis_count = min(max(num_gradcam, num_lime), total_samples)
    vis_indices = random.sample(range(total_samples), vis_count)
    vis_dataset = Subset(full_dataset, vis_indices)
    vis_loader = DataLoader(vis_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Get target layer from the first model in the ensemble
    base_model = model.models[0]
    if hasattr(base_model, 'layer4'):
        target_layer = base_model.layer4[-1]
    else:
        target_layer = list(base_model.children())[-1]

    for idx, (image, label_int) in enumerate(vis_loader):
        image = image.to(device)
        true_label = int(label_int.item())
        
        # --- Get Filename ---
        try:
            info = get_sample_info(full_dataset, vis_indices[idx])
            img_path = info[0] if isinstance(info, (list, tuple)) else str(info)
            img_name = os.path.basename(img_path)
        except:
            img_name = f"sample_{idx}"

        print(f"  Processing {img_name} ({idx+1}/{len(vis_loader)})")

        with torch.no_grad():
            output, weights, _, _ = model(image, return_details=True)
        
        pred_prob = torch.sigmoid(output).item()
        pred_label = 1 if pred_prob > 0.5 else 0
        
        filename = f"sample_{idx}_t{true_label}_p{pred_label}.png"
        img_np = image[0].cpu().permute(1, 2, 0).numpy()

        # --- GradCAM ---
        overlay = None
        if idx < num_gradcam:
            try:
                gradcam = GradCAM(model, target_layer)
                image_grad = image.clone().detach().requires_grad_(True)
                
                out_val = model(image_grad)[0]
                score = out_val if pred_label == 1 else -out_val
                
                cam = gradcam.generate(score)
                cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
                cam_uint8 = np.uint8(255 * cam_resized)
                heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET) / 255.0
                overlay = (heatmap + img_np) / np.max(heatmap + img_np)

                plt.imsave(os.path.join(dirs['gradcam'], filename), overlay)
            except Exception as e:
                print(f"    GradCAM Error: {e}")

        # --- LIME ---
        if idx < num_lime and LIME_AVAILABLE:
            try:
                lime_img = generate_lime_explanation(model, image, device)
                plt.imsave(os.path.join(dirs['lime'], filename), lime_img)
                
                # Combined
                if idx < num_gradcam and overlay is not None:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis('off')
                    axes[1].imshow(overlay); axes[1].set_title("GradCAM"); axes[1].axis('off')
                    axes[2].imshow(lime_img); axes[2].set_title("LIME"); axes[2].axis('off')
                    plt.suptitle(img_name)
                    plt.savefig(os.path.join(dirs['combined'], filename), bbox_inches='tight')
                    plt.close()
            except Exception as e:
                print(f"    LIME Error: {e}")

# ================== MAIN SCRIPT ==================
def main():
    parser = argparse.ArgumentParser(description="Simple Averaging Ensemble with Visualization")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['wild', 'real_fake', 'hard_fake_real', 'deepflux', 'uadfV', 'dfd'])
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./output_simple_ensemble')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Normalization Stats
    MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    MEANS = MEANS[:len(args.model_paths)]
    STDS = STDS[:len(args.model_paths)]

    # Load Models
    print("Loading models...")
    from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    models = []
    for i, path in enumerate(args.model_paths):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
        model.load_state_dict(ckpt['model_state_dict'])
        models.append(model.to(device).eval())
        print(f"  Loaded {args.model_names[i]}")

    # Create Ensemble
    ensemble = SimpleAveragingEnsemble(models, MEANS, STDS).to(device)

    # Load Data
    print("Loading Data...")
    # آرگومان‌ها را دقیقاً مطابق با تعریف تابع در dataset_utils پاس می‌دهیم
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, 
        args.batch_size, 
        args.dataset,       # dataset_type
        False,              # is_distributed
        42,                # seed
        True                # is_main
    )

    # Evaluate Ensemble
    print("\nEvaluating Ensemble...")
    ensemble.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = ensemble(images)
            preds = (torch.sigmoid(outputs.squeeze(1)) > 0).long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    acc = 100 * correct / total
    print(f"Ensemble Accuracy: {acc:.2f}%")

    # Visualizations
    vis_dir = os.path.join(args.save_dir, 'visualizations')
    os.makedirs(args.save_dir, exist_ok=True)
    
    generate_visualizations_internal(
        ensemble, test_loader, device, vis_dir, args.model_names,
        args.num_grad_cam_samples, args.num_lime_samples
    )
    
    # Simple manual ROC plotting to avoid external utils issues
    try:
        print("Plotting ROC...")
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = ensemble(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_labels.extend(labels.numpy())
                all_preds.extend(probs)
        
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(args.save_dir, 'roc_curve.png'))
        plt.close()
        print(f"ROC curve saved to {args.save_dir}")
    except Exception as e:
        print(f"Could not plot ROC: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
