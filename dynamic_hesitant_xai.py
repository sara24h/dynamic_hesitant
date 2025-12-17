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
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap
        Args:
            input_image: Input tensor [1, C, H, W]
            target_class: Target class for CAM (None = predicted class)
        Returns:
            cam: Grad-CAM heatmap [H, W]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        if isinstance(output, (tuple, list)):
            output = output[0]
        
        # Get predicted class if not specified
        if target_class is None:
            target_class = (output.squeeze() > 0).long().item()
        
        # Backward pass
        self.model.zero_grad()
        output.squeeze().backward()
        
        # Get gradients and activations
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]
        
        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1).squeeze()  # [H, W]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()

def get_target_layer(model):
    """
    Get the last convolutional layer for Grad-CAM
    """
    # For ResNet architecture
    if hasattr(model, 'layer4'):
        return model.layer4[-1].conv2
    elif hasattr(model, 'features'):
        # For other architectures with features
        for layer in reversed(list(model.features.children())):
            if isinstance(layer, nn.Conv2d):
                return layer
    else:
        # Try to find the last Conv2d layer
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if conv_layers:
            return conv_layers[-1]
    
    raise ValueError("Could not find suitable target layer for Grad-CAM")

def visualize_gradcam(image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image
    Args:
        image: Original image tensor [C, H, W] or numpy array [H, W, C]
        cam: Grad-CAM heatmap [H, W]
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap
    Returns:
        visualization: Overlayed image
    """
    # Convert image to numpy if needed
    if torch.is_tensor(image):
        image = image.cpu().numpy()
        if image.shape[0] == 3:  # [C, H, W] -> [H, W, C]
            image = np.transpose(image, (1, 2, 0))
    
    # Normalize image to [0, 255]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image = (image * 255).astype(np.uint8)
    
    # Resize CAM to match image size
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Apply colormap
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    visualization = (heatmap * alpha + image * (1 - alpha)).astype(np.uint8)
    
    return visualization, heatmap

def save_gradcam_comparison(images, labels, model_names, base_models, normalizations, 
                           save_dir, device, num_samples=5, rank=0):
    """
    Generate and save Grad-CAM visualizations for individual models
    """
    if rank != 0:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create Grad-CAM objects for each model
    gradcams = []
    for model in base_models:
        try:
            target_layer = get_target_layer(model)
            gradcam = GradCAM(model, target_layer)
            gradcams.append(gradcam)
        except Exception as e:
            print(f"Warning: Could not create Grad-CAM for model: {e}")
            gradcams.append(None)
    
    # Process samples
    for idx in range(min(num_samples, len(images))):
        image = images[idx:idx+1].to(device)
        label = labels[idx].item()
        
        fig, axes = plt.subplots(2, len(base_models) + 1, figsize=(4 * (len(base_models) + 1), 8))
        
        # Original image
        orig_img = images[idx].cpu().numpy().transpose(1, 2, 0)
        orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
        
        axes[0, 0].imshow(orig_img)
        axes[0, 0].set_title(f'Original\nLabel: {"Real" if label == 1 else "Fake"}', fontsize=10)
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # Generate Grad-CAM for each model
        for i, (model, gradcam, name) in enumerate(zip(base_models, gradcams, model_names)):
            if gradcam is None:
                continue
            
            try:
                # Normalize image for this model
                normalized_img = normalizations(image, i)
                
                # Generate Grad-CAM
                cam = gradcam.generate_cam(normalized_img)
                
                # Get prediction
                with torch.no_grad():
                    output = model(normalized_img)
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    pred = (output.squeeze() > 0).long().item()
                    conf = torch.sigmoid(output).item()
                
                # Visualize
                overlay, heatmap = visualize_gradcam(images[idx], cam)
                
                # Plot overlay
                axes[0, i+1].imshow(overlay)
                axes[0, i+1].set_title(
                    f'{name}\nPred: {"Real" if pred == 1 else "Fake"} ({conf:.2f})',
                    fontsize=9
                )
                axes[0, i+1].axis('off')
                
                # Plot heatmap only
                axes[1, i+1].imshow(heatmap)
                axes[1, i+1].set_title('Heatmap', fontsize=9)
                axes[1, i+1].axis('off')
                
            except Exception as e:
                print(f"Error generating Grad-CAM for model {name}: {e}")
                axes[0, i+1].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[0, i+1].axis('off')
                axes[1, i+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'gradcam_sample_{idx}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\nGrad-CAM visualizations saved to: {save_dir}")

# ====================== ORIGINAL CODE (UADFVDataset, etc.) ======================
class UADFVDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'fake': 0, 'real': 1}
        self.classes = list(self.class_to_idx.keys())

        fake_frames_dir = os.path.join(self.root_dir, 'fake', 'frames')
        if os.path.exists(fake_frames_dir):
            for subdir in os.listdir(fake_frames_dir):
                subdir_path = os.path.join(fake_frames_dir, subdir)
                if os.path.isdir(subdir_path):
                    for img_file in os.listdir(subdir_path):
                        img_path = os.path.join(subdir_path, img_file)
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((img_path, self.class_to_idx['fake']))
        
        real_frames_dir = os.path.join(self.root_dir, 'real', 'frames')
        if os.path.exists(real_frames_dir):
            for subdir in os.listdir(real_frames_dir):
                subdir_path = os.path.join(real_frames_dir, subdir)
                if os.path.isdir(subdir_path):
                    for img_file in os.listdir(subdir_path):
                        img_path = os.path.join(subdir_path, img_file)
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((img_path, self.class_to_idx['real']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[SEED] All random seeds set to: {seed}")
 
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_reproducible_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    num_samples = len(dataset)
    indices = list(range(num_samples))
    labels = [dataset.samples[i][1] for i in indices]
    
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_ratio, random_state=seed, stratify=labels
    )
    
    train_val_labels = [labels[i] for i in train_val_indices]
    val_size = val_ratio / (train_ratio + val_ratio)
    
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size, random_state=seed, stratify=train_val_labels
    )
    
    return train_indices, val_indices, test_indices

def prepare_real_fake_dataset(base_dir, seed=42):
    if os.path.exists(os.path.join(base_dir, 'training_fake')) and \
       os.path.exists(os.path.join(base_dir, 'training_real')):
        real_fake_dir = base_dir
    elif os.path.exists(os.path.join(base_dir, 'real_and_fake_face')):
        real_fake_dir = os.path.join(base_dir, 'real_and_fake_face')
    else:
        raise FileNotFoundError(f"Could not find training_fake/training_real in: {base_dir}")
    
    temp_transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.ImageFolder(real_fake_dir, transform=temp_transform)
    
    print(f"\n[Dataset Info - Real/Fake]")
    print(f"Total samples: {len(full_dataset)}")
    print(f"Classes: {full_dataset.classes}")
    
    train_indices, val_indices, test_indices = create_reproducible_split(
        full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=seed
    )
    
    print(f"\n[Split Statistics]")
    print(f"Train: {len(train_indices)} | Valid: {len(val_indices)} | Test: {len(test_indices)}")
    
    return full_dataset, train_indices, val_indices, test_indices

def prepare_hard_fake_real_dataset(base_dir, seed=42):
    if os.path.exists(os.path.join(base_dir, 'fake')) and os.path.exists(os.path.join(base_dir, 'real')):
        dataset_dir = base_dir
    elif os.path.exists(os.path.join(base_dir, 'hardfakevsrealfaces')):
        dataset_dir = os.path.join(base_dir, 'hardfakevsrealfaces')
    else:
        raise FileNotFoundError(f"Could not find fake/real folders in: {base_dir}")
    
    temp_transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.ImageFolder(dataset_dir, transform=temp_transform)
    
    print(f"\n[Dataset Info - HardFakeVsReal]")
    print(f"Total samples: {len(full_dataset)}")
    
    train_indices, val_indices, test_indices = create_reproducible_split(full_dataset, seed=seed)
    return full_dataset, train_indices, val_indices, test_indices

def prepare_deepflux_dataset(base_dir, seed=42):
    if os.path.exists(os.path.join(base_dir, 'Fake')) and os.path.exists(os.path.join(base_dir, 'Real')):
        dataset_dir = base_dir
    elif os.path.exists(os.path.join(base_dir, 'DeepFLUX')):
        dataset_dir = os.path.join(base_dir, 'DeepFLUX')
    else:
        raise FileNotFoundError(f"Could not find Fake/Real folders in: {base_dir}")
    
    temp_transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.ImageFolder(dataset_dir, transform=temp_transform)
    
    print(f"\n[Dataset Info - DeepFLUX]")
    print(f"Total samples: {len(full_dataset)}")
    
    train_indices, val_indices, test_indices = create_reproducible_split(full_dataset, seed=seed)
    return full_dataset, train_indices, val_indices, test_indices

def prepare_uadfV_dataset(base_dir, seed=42):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"UADFV dataset directory not found: {base_dir}")

    temp_transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = UADFVDataset(base_dir, transform=temp_transform)

    print(f"\n[Dataset Info - UADFV]")
    print(f"Total samples: {len(full_dataset)}")

    train_indices, val_indices, test_indices = create_reproducible_split(full_dataset, seed=seed)
    return full_dataset, train_indices, val_indices, test_indices

def setup_ddp():
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size
 
def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_net(x).flatten(1)
        memberships = self.membership_generator(features)
        memberships = memberships.view(-1, self.num_models, self.num_memberships)
        memberships = torch.sigmoid(memberships)
        agg_weights = F.softmax(self.aggregation_weights, dim=0)
        final_weights = (memberships * agg_weights.view(1, 1, -1)).sum(dim=2)
        final_weights = F.softmax(final_weights, dim=1)
        return final_weights, memberships

class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))
    
    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')

class FuzzyHesitantEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]],
                 stds: List[Tuple[float]], num_memberships: int = 3, freeze_models: bool = True,
                 cum_weight_threshold: float = 0.9, hesitancy_threshold: float = 0.2):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.hesitant_fuzzy = HesitantFuzzyMembership(
            input_dim=128, num_models=self.num_models, num_memberships=num_memberships
        )
        self.cum_weight_threshold = cum_weight_threshold
        self.hesitancy_threshold = hesitancy_threshold
        
        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
    
    def forward(self, x: torch.Tensor, return_details: bool = False):
        final_weights, all_memberships = self.hesitant_fuzzy(x)
        
        hesitancy = all_memberships.var(dim=2)
        avg_hesitancy = hesitancy.mean(dim=1)
        
        mask = torch.ones_like(final_weights)
        high_hesitancy_mask = (avg_hesitancy > self.hesitancy_threshold).unsqueeze(1)
        
        sorted_weights, sorted_indices = torch.sort(final_weights, dim=1, descending=True)
        cum_weights = torch.cumsum(sorted_weights, dim=1)
        
        for b in range(x.size(0)):
            if high_hesitancy_mask[b]:
                continue
            active_count = torch.sum(cum_weights[b] < self.cum_weight_threshold) + 1
            top_indices = sorted_indices[b, :active_count]
            sample_mask = torch.zeros(self.num_models, device=x.device)
            sample_mask[top_indices] = 1.0
            mask[b] = sample_mask
        
        final_weights = final_weights * mask
        final_weights = final_weights / (final_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
        
        active_model_indices = set()
        for b in range(x.size(0)):
            active_model_indices.update(torch.nonzero(final_weights[b] > 0).squeeze(-1).cpu().tolist())
        
        for i in list(active_model_indices):
            x_n = self.normalizations(x, i)
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            outputs[:, i] = out
        
        final_output = (outputs * final_weights.unsqueeze(-1)).sum(dim=1)
        
        if return_details:
            return final_output, final_weights, all_memberships, outputs
        return final_output, final_weights

def load_pruned_models(model_paths: List[str], device: torch.device, rank: int) -> List[nn.Module]:
    try:
        from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        raise ImportError("Cannot import ResNet_50_pruned_hardfakevsreal")
    
    models = []
    if rank == 0:
        print(f"Loading {len(model_paths)} pruned models...")
    
    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            if rank == 0:
                print(f" [WARNING] File not found: {path}")
            continue
        
        if rank == 0:
            print(f" [{i+1}/{len(model_paths)}] Loading: {os.path.basename(path)}")
        
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device).eval()
            
            if rank == 0:
                param_count = sum(p.numel() for p in model.parameters())
                print(f" → Parameters: {param_count:,}")
            
            models.append(model)
        except Exception as e:
            if rank == 0:
                print(f" [ERROR] Failed to load {path}: {e}")
            continue
    
    if len(models) == 0:
        raise ValueError("No models loaded!")
    
    if rank == 0:
        print(f"All {len(models)} models loaded!\n")
    
    return models

class TransformSubset(Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform
    
    def __getitem__(self, idx):
        img, label = self.dataset.samples[self.indices[idx]]
        img = self.dataset.loader(img)
        if self.transform:
            img = self.transform(img)
        return img, label

def create_dataloaders_ddp(base_dir: str, batch_size: int, rank: int, world_size: int,
                          num_workers: int = 2, dataset_type: str = 'wild'):
    
    if rank == 0:
        print("="*70)
        print(f"Creating DataLoaders with DDP (Dataset: {dataset_type})")
        print("="*70)
    
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    
    if dataset_type == 'wild':
        splits = ['train', 'valid', 'test']
        datasets_dict = {}
        
        for split in splits:
            path = os.path.join(base_dir, split)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Folder not found: {path}")
            
            if rank == 0:
                print(f"{split.capitalize():5}: {path}")
            
            transform = train_transform if split == 'train' else val_test_transform
            datasets_dict[split] = datasets.ImageFolder(path, transform=transform)
        
        if rank == 0:
            print(f"\nDataset Stats:")
            for split, ds in datasets_dict.items():
                print(f" {split.capitalize():5}: {len(ds):,} images")
        
        loaders = {}
        for split, ds in datasets_dict.items():
            if split == 'train':
                sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
                loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True, drop_last=True,
                                  worker_init_fn=worker_init_fn)
            else:
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True, drop_last=False,
                                  worker_init_fn=worker_init_fn)
            loaders[split] = loader
        
        train_loader = loaders['train']
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True, drop_last=False,
                               worker_init_fn=worker_init_fn)
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=worker_init_fn)

    elif dataset_type == 'hard_fake_real':
        if rank == 0:
            print(f"Processing hardfakevsrealfaces dataset from: {base_dir}")
            full_dataset, train_indices, val_indices, test_indices = prepare_hard_fake_real_dataset(base_dir, seed=42)
        
        dist.barrier()
        
        if os.path.exists(os.path.join(base_dir, 'fake')) and os.path.exists(os.path.join(base_dir, 'real')):
            dataset_dir = base_dir
        elif os.path.exists(os.path.join(base_dir, 'hardfakevsrealfaces')):
            dataset_dir = os.path.join(base_dir, 'hardfakevsrealfaces')
        else:
            raise FileNotFoundError(f"Could not find fake/real folders in {base_dir}")
        
        temp_transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = datasets.ImageFolder(dataset_dir, transform=temp_transform)
        
        train_indices, val_indices, test_indices = create_reproducible_split(full_dataset, seed=42)
        
        train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
        val_dataset = TransformSubset(full_dataset, val_indices, val_test_transform)
        test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 worker_init_fn=worker_init_fn)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True, drop_last=False,
                               worker_init_fn=worker_init_fn)
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=worker_init_fn)

    elif dataset_type == 'deepflux':
        if rank == 0:
            print(f"Processing DeepFLUX dataset from: {base_dir}")
            full_dataset, train_indices, val_indices, test_indices = prepare_deepflux_dataset(base_dir, seed=42)
        
        dist.barrier()
        
        if os.path.exists(os.path.join(base_dir, 'Fake')) and os.path.exists(os.path.join(base_dir, 'Real')):
            dataset_dir = base_dir
        elif os.path.exists(os.path.join(base_dir, 'DeepFLUX')):
            dataset_dir = os.path.join(base_dir, 'DeepFLUX')
        else:
            raise FileNotFoundError(f"Could not find Fake/Real folders in {base_dir}")
        
        temp_transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = datasets.ImageFolder(dataset_dir, transform=temp_transform)
        
        train_indices, val_indices, test_indices = create_reproducible_split(full_dataset, seed=42)
        
        train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
        val_dataset = TransformSubset(full_dataset, val_indices, val_test_transform)
        test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 worker_init_fn=worker_init_fn)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True, drop_last=False,
                               worker_init_fn=worker_init_fn)
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=worker_init_fn)

    elif dataset_type == 'uadfV':
        if rank == 0:
            print(f"Processing UADFV dataset from: {base_dir}")
            full_dataset, train_indices, val_indices, test_indices = prepare_uadfV_dataset(base_dir, seed=42)

        dist.barrier()

        temp_transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = UADFVDataset(base_dir, transform=temp_transform)

        train_indices, val_indices, test_indices = create_reproducible_split(full_dataset, seed=42)

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)

        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_test_transform
        test_dataset.dataset.transform = val_test_transform

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 worker_init_fn=worker_init_fn)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True, drop_last=False,
                               worker_init_fn=worker_init_fn)
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=worker_init_fn)

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    if rank == 0:
        print(f"DataLoaders ready! Batch size per GPU: {batch_size}")
        print("="*70 + "\n")
    
    return train_loader, val_loader, test_loader

@torch.no_grad()
def evaluate_single_model(model: nn.Module, loader: DataLoader, device: torch.device, name: str, rank: int) -> float:
    model.eval()
    correct = total = 0
    
    iterator = tqdm(loader, desc=f"Evaluating {name}") if rank == 0 else loader
    
    for images, labels in iterator:
        images, labels = images.to(device), labels.to(device).float()
        out = model(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = (out.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()
    
    acc = 100. * correct / total
    
    if rank == 0:
        print(f" {name}: {acc:.2f}%")
    
    return acc

def train_hesitant_fuzzy_ddp(ensemble_model, train_loader, val_loader, num_epochs, lr, device, save_dir, rank, world_size):
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    
    hesitant_net = ensemble_model.module.hesitant_fuzzy
    optimizer = torch.optim.AdamW(hesitant_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'membership_variance': []}
    
    if rank == 0:
        print("="*70)
        print("Training Fuzzy Hesitant Network (DDP)")
        print("="*70)
        print(f"Trainable params: {sum(p.numel() for p in hesitant_net.parameters()):,}")
        print(f"Epochs: {num_epochs} | Initial LR: {lr}\n")
    
    for epoch in range(num_epochs):
        ensemble_model.train()
        train_loader.sampler.set_epoch(epoch)
        
        train_loss = train_correct = train_total = 0.0
        membership_vars = []
        
        iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') if rank == 0 else train_loader
        
        for images, labels in iterator:
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
            
            membership_vars.append(memberships.var(dim=2).mean().item())
            
            if rank == 0:
                current_acc = 100. * train_correct / train_total
                avg_loss = train_loss / train_total
                iterator.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
        
        train_loss_tensor = torch.tensor(train_loss).to(device)
        train_correct_tensor = torch.tensor(train_correct).to(device)
        train_total_tensor = torch.tensor(train_total).to(device)
        
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_total_tensor, op=dist.ReduceOp.SUM)
        
        train_acc = 100. * train_correct_tensor.item() / train_total_tensor.item()
        train_loss = train_loss_tensor.item() / train_total_tensor.item()
        avg_membership_var = np.mean(membership_vars)
        
        val_acc = evaluate_accuracy_ddp(ensemble_model, val_loader, device, rank)
        scheduler.step()
        
        if rank == 0:
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['membership_variance'].append(avg_membership_var)
            
            print(f"\nEpoch {epoch+1}:")
            print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f" Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                tmp_path = os.path.join(save_dir, 'best_tmp.pt')
                final_path = os.path.join(save_dir, 'best_hesitant_fuzzy.pt')
                
                try:
                    torch.save({
                        'epoch': epoch + 1,
                        'hesitant_state_dict': hesitant_net.state_dict(),
                        'val_acc': val_acc,
                        'history': history
                    }, tmp_path)
                    
                    if os.path.exists(tmp_path):
                        shutil.move(tmp_path, final_path)
                        print(f" Best model saved → {val_acc:.2f}%")
                except Exception as e:
                    print(f" [ERROR] Failed to save model: {e}")
            
            print("-" * 70)
        
        dist.barrier()
    
    if rank == 0:
        print(f"\nTraining completed! Best Val Acc: {best_val_acc:.2f}%")
    
    return best_val_acc, history

@torch.no_grad()
def evaluate_ensemble_final_ddp(model, loader, device, name, model_names, rank):
    model.eval()
    
    all_preds = []
    all_labels = []
    all_weights = []
    all_memberships = []
    
    activation_counts = torch.zeros(len(model_names), device=device)
    total_samples = 0

    iterator = tqdm(loader, desc=f"Evaluating {name}") if rank == 0 else loader

    for images, labels in iterator:
        images = images.to(device)
        labels = labels.to(device)

        outputs, weights, memberships, _ = model(images, return_details=True)
        pred = (outputs.squeeze(1) > 0).long()

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_weights.append(weights.cpu())
        all_memberships.append(memberships.cpu())

        active_per_model = (weights > 1e-4).sum(dim=0).float()
        activation_counts += active_per_model
        total_samples += images.size(0)

    dist.all_reduce(activation_counts, op=dist.ReduceOp.SUM)
    
    total_samples_tensor = torch.tensor(total_samples, device=device)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    total_samples = total_samples_tensor.item()

    all_weights = torch.cat(all_weights).cpu().numpy()
    all_memberships = torch.cat(all_memberships).cpu().numpy()
    activation_counts = activation_counts.cpu().numpy()

    acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    avg_weights = all_weights.mean(axis=0)
    activation_percentages = (activation_counts / total_samples) * 100

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"{name.upper()} SET RESULTS")
        print(f"{'='*70}")
        print(f" → Accuracy: {acc:.3f}%")
        print(f" → Total Samples: {total_samples:,}")

        print(f"\nAverage Model Weights:")
        for i, (w, name) in enumerate(zip(avg_weights, model_names)):
            print(f"   {i+1:2d}. {name:<25}: {w:6.4f} ({w*100:5.2f}%)")

        print(f"\nActivation Frequency:")
        for i, (perc, count, name) in enumerate(zip(activation_percentages, activation_counts, model_names)):
            print(f"   {i+1:2d}. {name:<25}: {perc:6.2f}% ({int(count):,}/{total_samples:,})")

        print(f"{'='*70}")

    return acc, avg_weights.tolist(), all_memberships.mean(axis=0).tolist(), activation_percentages.tolist()

@torch.no_grad()
def evaluate_accuracy_ddp(model, loader, device, rank):
    model.eval()
    correct = total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs, _ = model(images)
        pred = (outputs.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()
    
    acc = 100. * correct / total
    return acc

def main():
    SEED = 42
    set_seed(SEED)
    
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
    is_main = (rank == 0)
    
    parser = argparse.ArgumentParser(description="Train Fuzzy Hesitant Ensemble with Grad-CAM")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_memberships', type=int, default=3)
    
    parser.add_argument('--dataset', type=str, choices=['wild', 'real_fake', 'hard_fake_real', 'deepflux', 'uadfV'], 
                       required=True, help='Dataset type')
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory of dataset')
    
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, help='Paths to pruned models')
    parser.add_argument('--model_names', type=str, nargs='+', required=True, help='Names for each model')
    
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/checkpoints')
    parser.add_argument('--gradcam_dir', type=str, default='/kaggle/working/gradcam_visualizations',
                       help='Directory to save Grad-CAM visualizations')
    parser.add_argument('--num_gradcam_samples', type=int, default=10,
                       help='Number of samples to visualize with Grad-CAM')
    parser.add_argument('--seed', type=int, default=SEED)
    
    args = parser.parse_args()
    
    if len(args.model_names) != len(args.model_paths):
        raise ValueError(f"Number of model_names must match model_paths")
    
    MEANS = [(0.5207, 0.4258, 0.3806), (0.4868, 0.3972, 0.3624), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2296, 0.2066, 0.2009), (0.2410, 0.2161, 0.2081)]
    
    MEANS = MEANS[:len(args.model_paths)]
    STDS = STDS[:len(args.model_paths)]
    
    if args.seed != SEED:
        set_seed(args.seed)
    
    if is_main:
        print(f"="*70)
        print(f"Multi-GPU Training with DDP + Grad-CAM | SEED: {args.seed}")
        print(f"="*70)
        print(f"World Size: {world_size} GPUs")
        print(f"Dataset: {args.dataset}")
        print(f"Data directory: {args.data_dir}")
        print(f"\nModels to load:")
        for i, (path, name) in enumerate(zip(args.model_paths, args.model_names)):
            print(f" {i+1}. {name}: {path}")
        print(f"="*70 + "\n")
    
    base_models = load_pruned_models(args.model_paths, device, rank)
    
    if len(base_models) != len(args.model_paths):
        if is_main:
            print(f"[WARNING] Only {len(base_models)}/{len(args.model_paths)} models loaded")
        MEANS = MEANS[:len(base_models)]
        STDS = STDS[:len(base_models)]
        MODEL_NAMES = args.model_names[:len(base_models)]
    else:
        MODEL_NAMES = args.model_names
    
    ensemble = FuzzyHesitantEnsemble(
        base_models, MEANS, STDS,
        num_memberships=args.num_memberships,
        freeze_models=True,
        cum_weight_threshold=0.9,
        hesitancy_threshold=0.2
    ).to(device)
    
    ensemble = DDP(ensemble, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    train_loader, val_loader, test_loader = create_dataloaders_ddp(
        args.data_dir, args.batch_size, rank, world_size, dataset_type=args.dataset
    )
    
    # ==================== GRAD-CAM VISUALIZATION (Before Training) ====================
    if is_main:
        print("\n" + "="*70)
        print("GENERATING GRAD-CAM VISUALIZATIONS (Before Training)")
        print("="*70)
        
        # Get sample batch from test loader
        test_iter = iter(test_loader)
        sample_images, sample_labels = next(test_iter)
        
        gradcam_save_dir = os.path.join(args.gradcam_dir, 'before_training')
        save_gradcam_comparison(
            sample_images, 
            sample_labels,
            MODEL_NAMES,
            base_models,
            ensemble.module.normalizations,
            gradcam_save_dir,
            device,
            num_samples=min(args.num_gradcam_samples, len(sample_images)),
            rank=rank
        )
    
    dist.barrier()
    
    # ==================== EVALUATE INDIVIDUAL MODELS ====================
    if is_main:
        print("\n" + "="*70)
        print("EVALUATING INDIVIDUAL MODELS ON TEST SET")
        print("="*70)
        individual_accs = []
        for i, model in enumerate(base_models):
            acc = evaluate_single_model(model, test_loader, device, f"Model {i+1} ({MODEL_NAMES[i]})", rank)
            individual_accs.append(acc)
        best_single = max(individual_accs)
        best_idx = individual_accs.index(best_single)
        print(f"\nBest Single Model: {MODEL_NAMES[best_idx]} → {best_single:.2f}%")
    
    dist.barrier()
    
    # ==================== TRAIN ENSEMBLE ====================
    best_val_acc, history = train_hesitant_fuzzy_ddp(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_dir, rank, world_size
    )
    
    # ==================== LOAD BEST MODEL ====================
    ckpt_path = os.path.join(args.save_dir, 'best_hesitant_fuzzy.pt')
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ensemble.module.hesitant_fuzzy.load_state_dict(ckpt['hesitant_state_dict'])
        if is_main:
            print("\nBest model loaded.\n")
    
    dist.barrier()
    
    # ==================== FINAL EVALUATION ====================
    if is_main:
        print("\n" + "="*70)
        print("EVALUATING FUZZY HESITANT ENSEMBLE")
        print("="*70)
        ensemble_test_acc, ensemble_weights, membership_values, activation_percentages = evaluate_ensemble_final_ddp(
            ensemble, test_loader, device, "Test", MODEL_NAMES, rank
        )
        
        # ==================== GRAD-CAM VISUALIZATION (After Training) ====================
        print("\n" + "="*70)
        print("GENERATING GRAD-CAM VISUALIZATIONS (After Training)")
        print("="*70)
        
        test_iter = iter(test_loader)
        sample_images, sample_labels = next(test_iter)
        
        gradcam_save_dir = os.path.join(args.gradcam_dir, 'after_training')
        save_gradcam_comparison(
            sample_images,
            sample_labels,
            MODEL_NAMES,
            base_models,
            ensemble.module.normalizations,
            gradcam_save_dir,
            device,
            num_samples=min(args.num_gradcam_samples, len(sample_images)),
            rank=rank
        )
        
        # ==================== FINAL RESULTS ====================
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        print(f"Best Single Model Acc: {best_single:.2f}%")
        print(f"Hesitant Ensemble Acc: {ensemble_test_acc:.2f}%")
        improvement = ensemble_test_acc - best_single
        print(f"Improvement: {improvement:+.2f}%")
        
        results = {
            "method": "Fuzzy Hesitant Sets with Grad-CAM",
            "dataset": args.dataset,
            "data_dir": args.data_dir,
            "seed": args.seed,
            "num_gpus": world_size,
            "num_memberships": args.num_memberships,
            "model_paths": args.model_paths,
            "model_names": MODEL_NAMES,
            "normalization": {
                "means": [list(m) for m in MEANS],
                "stds": [list(s) for s in STDS]
            },
            "individual_accuracies": {MODEL_NAMES[i]: acc for i, acc in enumerate(individual_accs)},
            "best_single": {"name": MODEL_NAMES[best_idx], "acc": best_single},
            "ensemble": {
                "acc": ensemble_test_acc,
                "weights": ensemble_weights,
                "membership_values": membership_values,
                "activation_percentages": activation_percentages
            },
            "improvement": improvement,
            "training_history": history,
            "gradcam_dir": args.gradcam_dir
        }
        
        result_path = os.path.join(args.save_dir, 'results_with_gradcam.json')
        
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {result_path}")
        
        final_model_path = os.path.join(args.save_dir, 'final_model.pt')
        torch.save({
            'hesitant_state_dict': ensemble.module.hesitant_fuzzy.state_dict(),
            'results': results,
            'args': vars(args)
        }, final_model_path)
        
        print(f"Final model saved: {final_model_path}")
        print("="*70)
        print("All done!")
    
    cleanup_ddp()

if __name__ == "__main__":
    main() = loaders['valid']
        test_loader = loaders['test']
    
    elif dataset_type == 'real_fake':
        if rank == 0:
            print(f"Processing real-fake dataset from: {base_dir}")
            full_dataset, train_indices, val_indices, test_indices = prepare_real_fake_dataset(base_dir, seed=42)
        
        dist.barrier()
        
        if os.path.exists(os.path.join(base_dir, 'training_fake')) and \
           os.path.exists(os.path.join(base_dir, 'training_real')):
            dataset_dir = base_dir
        elif os.path.exists(os.path.join(base_dir, 'real_and_fake_face')):
            dataset_dir = os.path.join(base_dir, 'real_and_fake_face')
        else:
            raise FileNotFoundError(f"Could not find training folders in {base_dir}")
        
        temp_transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = datasets.ImageFolder(dataset_dir, transform=temp_transform)
        
        train_indices, val_indices, test_indices = create_reproducible_split(full_dataset, seed=42)
        
        train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
        val_dataset = TransformSubset(full_dataset, val_indices, val_test_transform)
        test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 worker_init_fn=worker_init_fn)
        
        val_loader
