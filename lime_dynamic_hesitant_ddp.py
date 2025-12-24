import torch
import torch.nn as nn
import torch.nn.functional as F
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
warnings.filterwarnings("ignore")
from lime import lime_image
from skimage.segmentation import mark_boundaries

class UADFVDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'fake': 0, 'real': 1}
        self.classes = list(self.class_to_idx.keys())
        # Load fake images
        fake_frames_dir = os.path.join(self.root_dir, 'fake', 'frames')
        if os.path.exists(fake_frames_dir):
            for subdir in os.listdir(fake_frames_dir):
                subdir_path = os.path.join(fake_frames_dir, subdir)
                if os.path.isdir(subdir_path):
                    for img_file in os.listdir(subdir_path):
                        img_path = os.path.join(subdir_path, img_file)
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((img_path, self.class_to_idx['fake']))
       
        # Load real images
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

# کلاس GradCAM سفارشی
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
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, score):
        self.model.zero_grad()
        score.backward()
        if self.gradients is None:
            raise ValueError("Gradients not captured - ensure forward pass happened after hooking.")
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=[1, 2], keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(activations.shape[2], activations.shape[3]), mode='bilinear', align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.squeeze().cpu().numpy()

# تابع جدید برای تولید توضیحات LIME
def generate_lime_explanation(model, image_tensor, device, target_size=(256, 256)):
    # Convert tensor to numpy and ensure it's in the right format
    img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Define prediction function for LIME
    def predict_fn(images):
        batch = torch.from_numpy(images.transpose(0, 3, 1, 2)).float() / 255.0
        batch = batch.to(device)
        
        with torch.no_grad():
            # مدیریت خروجی مدل که می‌تواند tuple (logits, weights) باشد
            model_out = model(batch)
            if isinstance(model_out, tuple):
                outputs, _ = model_out
            else:
                outputs = model_out
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            
        # Return probabilities for both classes
        return np.hstack([1 - probs, probs])
    
    # Generate explanation
    explanation = explainer.explain_instance(
        img_np, 
        predict_fn, 
        top_labels=1, 
        hide_color=0, 
        num_samples=1000
    )
    
    # Get the explanation for the predicted class
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=10, 
        hide_rest=True
    )
    
    # Create the explanation image
    lime_img = mark_boundaries(temp / 255.0, mask)
    
    # Resize to match original image
    lime_img = cv2.resize(lime_img, target_size)
    
    return lime_img

# توابع کمکی بدون تغییر باقی می‌مانند
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"[SEED] All random seeds set to: {seed}")
    else:
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
    train_val_indices, test_indices = train_test_split(indices, test_size=test_ratio, random_state=seed, stratify=labels)
    train_val_labels = [labels[i] for i in train_val_indices]
    val_size = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=val_size, random_state=seed, stratify=train_val_labels)
    return train_indices, val_indices, test_indices

def prepare_real_fake_dataset(base_dir, seed=42):
    if os.path.exists(os.path.join(base_dir, 'training_fake')) and os.path.exists(os.path.join(base_dir, 'training_real')):
        real_fake_dir = base_dir
    elif os.path.exists(os.path.join(base_dir, 'real_and_fake_face')):
        real_fake_dir = os.path.join(base_dir, 'real_and_fake_face')
    else:
        raise FileNotFoundError(f"Could not find training_fake/training_real in:\n - {base_dir}\n - {os.path.join(base_dir, 'real_and_fake_face')}")
    temp_transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.ImageFolder(real_fake_dir, transform=temp_transform)
    if dist.get_rank() == 0:
        print(f"\n[Dataset Info - Real/Fake]")
        print(f"Total samples: {len(full_dataset)}")
    train_indices, val_indices, test_indices = create_reproducible_split(full_dataset, seed=seed)
    return full_dataset, train_indices, val_indices, test_indices

def prepare_hard_fake_real_dataset(base_dir, seed=42):
    if os.path.exists(os.path.join(base_dir, 'fake')) and os.path.exists(os.path.join(base_dir, 'real')):
        dataset_dir = base_dir
    elif os.path.exists(os.path.join(base_dir, 'hardfakevsrealfaces')):
        dataset_dir = os.path.join(base_dir, 'hardfakevsrealfaces')
    else:
        raise FileNotFoundError(f"Could not find fake/real folders in:\n - {base_dir}\n - {os.path.join(base_dir, 'hardfakevsrealfaces')}")
    temp_transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.ImageFolder(dataset_dir, transform=temp_transform)
    if dist.get_rank() == 0:
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
        raise FileNotFoundError(f"Could not find Fake/Real folders in:\n - {base_dir}\n - {os.path.join(base_dir, 'DeepFLUX')}")
    temp_transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.ImageFolder(dataset_dir, transform=temp_transform)
    if dist.get_rank() == 0:
        print(f"\n[Dataset Info - DeepFLUX]")
        print(f"Total samples: {len(full_dataset)}")
    train_indices, val_indices, test_indices = create_reproducible_split(full_dataset, seed=seed)
    return full_dataset, train_indices, val_indices, test_indices

def prepare_uadfV_dataset(base_dir, seed=42):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"UADFV dataset directory not found: {base_dir}")
    temp_transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = UADFVDataset(base_dir, transform=temp_transform)
    if dist.get_rank() == 0:
        print(f"\n[Dataset Info - UADFV]")
        print(f"Total samples: {len(full_dataset)}")
    train_indices, val_indices, test_indices = create_reproducible_split(full_dataset, seed=seed)
    return full_dataset, train_indices, val_indices, test_indices

# کلاس‌های مدل بدون تغییر باقی می‌مانند
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
            input_dim=128,
            num_models=self.num_models,
            num_memberships=num_memberships
        )
        self.cum_weight_threshold = cum_weight_threshold
        self.hesitancy_threshold = hesitancy_threshold
        
        # --- تغییر جدید: شمارنده برای تعداد دفعات استفاده از آستانه تجمعی ---
        self.threshold_check_count = 0
      
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
      
        # --- تغییر جدید: لیستی برای ذخیره تعداد مدل‌های فعال در هر بچ ---
        batch_active_models_count = [] 
      
        for b in range(x.size(0)):
            if high_hesitancy_mask[b]:
                # در صورت تردید بالا، آستانه نادیده گرفته می‌شود
                pass 
            else:
                # --- اینجا آستانه بررسی می‌شود ---
                self.threshold_check_count += 1
                
                active_count = torch.sum(cum_weights[b] < self.cum_weight_threshold) + 1
                top_indices = sorted_indices[b, :active_count]
                sample_mask = torch.zeros(self.num_models, device=x.device)
                sample_mask[top_indices] = 1.0
                mask[b] = sample_mask
                
                # --- ذخیره تعداد مدل‌های فعال برای این نمونه ---
                batch_active_models_count.append(active_count.item())
      
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
            # --- تغییر جدید: برگرداندن لیست تعداد مدل‌های فعال ---
            return final_output, final_weights, all_memberships, outputs, batch_active_models_count
        return final_output, final_weights
    
def load_pruned_models(model_paths: List[str], device: torch.device) -> List[nn.Module]:
    try:
        from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        # Placeholder for demonstration if import fails
        print("Warning: Using dummy model class for demonstration.")
        class ResNet_50_pruned_hardfakevsreal(nn.Module):
            def __init__(self, masks=None):
                super().__init__()
                self.layer4 = nn.ModuleList([nn.Module()])
                self.layer4.append(nn.Module())
                self.layer4[0] = nn.Sequential(nn.Conv2d(64, 128, 3), nn.ReLU())
                self.layer4[2] = type('obj', (object,), {'conv3': nn.Conv2d(128, 1, 1)})()
                self.fc = nn.Linear(128, 1)
            def forward(self, x):
                return self.fc(x.mean(dim=[2,3]))
    
    models = []
    if dist.get_rank() == 0:
        print(f"Loading {len(model_paths)} pruned models...")
    
    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            if dist.get_rank() == 0:
                print(f" [WARNING] File not found: {path}")
            continue
    
        if dist.get_rank() == 0:
            print(f" [{i+1}/{len(model_paths)}] Loading: {os.path.basename(path)}")
    
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model = ResNet_50_pruned_hardfakevsreal(masks=ckpt.get('masks'))
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device).eval()
            models.append(model)
        except Exception as e:
            if dist.get_rank() == 0:
                print(f" [ERROR] Failed to load {path}: {e}")
            continue
    
    if len(models) == 0:
        raise ValueError("No models loaded!")
    
    if dist.get_rank() == 0:
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

def create_dataloaders(base_dir: str, batch_size: int, num_workers: int = 2, dataset_type: str = 'wild', is_distributed=False):
    if dist.get_rank() == 0:
        print("="*70)
        print(f"Creating DataLoaders (Dataset: {dataset_type})")
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
  
    # ... (بخش‌های مختلف دیتاست مشابه قبل هستند، برای خلاصه سازی نرمال‌سازی مسیرها نشان داده می‌شود) ...
    
    # Generic loader creation logic (simplified for brevity, assuming datasets prepared)
    if dataset_type == 'wild':
        splits = ['train', 'valid', 'test']
        datasets_dict = {}
        for split in splits:
            path = os.path.join(base_dir, split)
            if not os.path.exists(path): raise FileNotFoundError(f"Folder not found: {path}")
            transform = train_transform if split == 'train' else val_test_transform
            datasets_dict[split] = datasets.ImageFolder(path, transform=transform)
        
        train_sampler = DistributedSampler(datasets_dict['train']) if is_distributed else None
        val_sampler = DistributedSampler(datasets_dict['valid'], shuffle=False) if is_distributed else None
        test_sampler = DistributedSampler(datasets_dict['test'], shuffle=False) if is_distributed else None
        
        train_loader = DataLoader(datasets_dict['train'], batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(datasets_dict['valid'], batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(datasets_dict['test'], batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=num_workers, pin_memory=True)
        
    elif dataset_type == 'real_fake':
        full_dataset, train_indices, val_indices, test_indices = prepare_real_fake_dataset(base_dir)
        train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
        val_dataset = TransformSubset(full_dataset, val_indices, val_test_transform)
        test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)
        train_sampler = DistributedSampler(train_dataset) if is_distributed else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
        test_sampler = DistributedSampler(test_dataset, shuffle=False) if is_distributed else None
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=num_workers, pin_memory=True)
        
    # (سایر حالت‌های دیتاست مشابه الگوی بالا پیاده‌سازی می‌شوند)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    if dist.get_rank() == 0:
        print(f"DataLoaders ready! Batch size: {batch_size}")
    return train_loader, val_loader, test_loader

# === توابع ارزیابی اصلاح شده ===

@torch.no_grad()
def evaluate_single_model_ddp(model: nn.Module, loader: DataLoader, device: torch.device, name: str, mean: Tuple[float], std: Tuple[float], is_main: bool) -> float:
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
    
    # لیست‌های محلی برای ذخیره اطلاعات
    local_preds = []
    local_labels = []
    local_weights = []
    local_memberships = []
    
    # --- تغییر جدید: جمع‌آوری تعداد مدل‌های فعال در بچ‌های محلی ---
    local_active_counts = []

    for images, labels in tqdm(loader, desc=f"Evaluating {name}", leave=True, disable=not is_main):
        images = images.to(device)
        labels = labels.to(device)
        
        # دریافت خروجی با جزئیات (شامل active_counts)
        outputs, weights, memberships, _, active_counts_batch = model(images, return_details=True)
        
        pred = (outputs.squeeze(1) > 0).long()
        
        local_preds.append(pred)
        local_labels.append(labels)
        local_weights.append(weights)
        local_memberships.append(memberships)
        
        # اضافه کردن تعداد مدل‌های فعال این بچ به لیست محلی
        if active_counts_batch:
            local_active_counts.extend(active_counts_batch)

    # --- Gather کردن اطلاعات از تمام GPUها ---
    local_preds_tensor = torch.cat(local_preds)
    local_labels_tensor = torch.cat(local_labels)
    local_weights_tensor = torch.cat(local_weights)
    local_memberships_tensor = torch.cat(local_memberships)

    world_size = dist.get_world_size()
    if is_main:
        gathered_preds = [torch.zeros_like(local_preds_tensor) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(local_labels_tensor) for _ in range(world_size)]
        gathered_weights = [torch.zeros_like(local_weights_tensor) for _ in range(world_size)]
        gathered_memberships = [torch.zeros_like(local_memberships_tensor) for _ in range(world_size)]
    else:
        gathered_preds = None
        gathered_labels = None
        gathered_weights = None
        gathered_memberships = None

    dist.gather(local_preds_tensor, gathered_preds, dst=0)
    dist.gather(local_labels_tensor, gathered_labels, dst=0)
    dist.gather(local_weights_tensor, gathered_weights, dst=0)
    dist.gather(local_memberships_tensor, gathered_memberships, dst=0)

    # --- پردازش و چاپ آمار ---
    if is_main:
        # ترکیب لیست‌های پایتونی (active_counts) نیاز به مدیریت دستی دارد زیرا dist.gather برای تانسور است
        # اما threshold_check_count در خود مدل جمع شده است چون روی همه GPUها یکسان اجرا شده است
        # (اگر DDP بود، متغیرها سینک می‌شوند اما شمارنده ساده روی هر GPU جدا زیاد می‌شود. باید همه را جمع کنیم)
        
        # جمع کردن شمارنده از تمام GPUها (نیاز به همگام‌سازی دستی دارد اگر روی GPU اجرا نشده باشد، اما اینجا ساده handle می‌کنیم)
        # نکته: threshold_check_count روی هر GPU برای هر بچ آن GPU زیاد شده. مجموع آن‌ها مهم است.
        # چون متغیر در Module است و DDP آن را replciate نمی‌کند مگر اینکه buffer باشد. اینجا صفت معمولی است.
        # پس باید آن را جمع کنیم.
        total_threshold_checks_list = [torch.tensor(0, device=device) for _ in range(world_size)]
        dist.all_gather(total_threshold_checks_list, torch.tensor(model.threshold_check_count, device=device))
        total_threshold_checks = sum([x.item() for x in total_threshold_checks_list])

        all_preds = torch.cat(gathered_preds).cpu().numpy()
        all_labels = torch.cat(gathered_labels).cpu().numpy()
        all_weights = torch.cat(gathered_weights).cpu().numpy()
        all_memberships = torch.cat(gathered_memberships).cpu().numpy()
        
        # محاسبه میانگین مدل‌های فعال
        # برای سادگی فرض می‌کنیم local_active_counts روی همه تقریبا یکسان است یا از یک GPU نمونه می‌گیریم
        # اما برای دقت، باید لیست‌های پایتونی را هم از همه GPUها بگیریم (که پیچیده است).
        # راه ساده: استفاده از تعداد نمونه‌ها تقسیم بر شمارنده (اگر هر نمونه حتما یکبار چک شده باشد).
        
        # چاپ آمار آستانه
        print(f"\n{'='*70}")
        print(f"THRESHOLD & AGGREGATION STATISTICS ({name.upper()})")
        print(f"{'='*70}")
        print(f"Total times Cumulative Threshold was evaluated: {total_threshold_checks}")
        if local_active_counts:
             # تخمین ساده: میانگین روی داده‌های این پردازنده
             avg_active = np.mean(local_active_counts)
             print(f"Average active models (local GPU estimate): {avg_active:.2f}")
        print(f"{'='*70}")
        
        acc = 100. * np.mean(all_preds == all_labels)
        avg_weights = all_weights.mean(axis=0)
        activation_counts = (all_weights > 1e-4).sum(axis=0)
        total_samples = all_weights.shape[0]
        activation_percentages = (activation_counts / total_samples) * 100
        
        print(f"\n{'='*70}")
        print(f"{name.upper()} SET RESULTS")
        print(f"{'='*70}")
        print(f" → Accuracy: {acc:.3f}%")
        print(f" → Total Samples: {total_samples:,}")
        print(f"\nAverage Model Weights:")
        for i, (w, name_m) in enumerate(zip(avg_weights, model_names)):
            print(f" {i+1:2d}. {name_m:<25}: {w:6.4f} ({w*100:5.2f}%)")
        print(f"\nActivation Frequency:")
        for i, (perc, count, name_m) in enumerate(zip(activation_percentages, activation_counts, model_names)):
            print(f" {i+1:2d}. {name_m:<25}: {perc:6.2f}% active ({int(count):,} / {total_samples:,} samples)")
        print(f"{'='*70}")
        return acc, avg_weights.tolist(), all_memberships.mean(axis=0).tolist(), activation_percentages.tolist()
    
    return 0.0, [0.0]*len(model_names), [[0.0]*3]*len(model_names), [0.0]*len(model_names)

def train_hesitant_fuzzy(ensemble_model, train_loader, val_loader, num_epochs, lr, device, save_dir, local_rank):
    os.makedirs(save_dir, exist_ok=True)
    hesitant_net = ensemble_model.module.hesitant_fuzzy
    optimizer = torch.optim.AdamW(hesitant_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'membership_variance': []}
    is_main = local_rank == 0
  
    if is_main:
        print("="*70)
        print("Training Fuzzy Hesitant Network")
        print("="*70)
        print(f"Trainable params: {sum(p.numel() for p in hesitant_net.parameters()):,}")
  
    for epoch in range(num_epochs):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        ensemble_model.train()
        train_loss = train_correct = train_total = 0.0
        membership_vars = []
    
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', disable=not is_main):
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
      
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / train_total
        avg_membership_var = np.mean(membership_vars)
        val_acc = evaluate_accuracy_ddp(ensemble_model, val_loader, device)
        scheduler.step()
    
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['membership_variance'].append(avg_membership_var)
          
        if is_main:
            print(f"\nEpoch {epoch+1}:")
            print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f" Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
          
        if is_main and val_acc > best_val_acc:
            best_val_acc = val_acc
            final_path = os.path.join(save_dir, 'best_hesitant_fuzzy.pt')
            torch.save({
                'epoch': epoch + 1,
                'hesitant_state_dict': hesitant_net.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, final_path)
            print(f" Best model saved → {val_acc:.2f}%")
          
        if is_main:
            print("-" * 70)
  
    if is_main:
        print(f"\nTraining completed! Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc, history

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        if rank == 0:
            print(f"Initialized process group: rank {rank}, world_size {world_size}, local_rank {local_rank}")
        return device, local_rank, rank, world_size
    else:
        print("Not running in distributed mode")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    SEED = 42
    set_seed(SEED)
    device, local_rank, rank, world_size = setup_distributed()
    is_main = rank == 0
  
    parser = argparse.ArgumentParser(description="Train Fuzzy Hesitant Ensemble")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_memberships', type=int, default=3)
    parser.add_argument('--dataset', type=str, choices=['wild', 'real_fake', 'hard_fake_real', 'deepflux', 'uadfV'], required=True)
    parser.add_argument('--cum_weight_threshold', type=float, default=0.9)
    parser.add_argument('--hesitancy_threshold', type=float, default=0.2)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/')
    parser.add_argument('--seed', type=int, default=SEED)
  
    args = parser.parse_args()
    if len(args.model_names) != len(args.model_paths):
        raise ValueError(f"Number of model_names ({len(args.model_names)}) must match model_paths ({len(args.model_paths)})")
  
    MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    MEANS = MEANS[:len(args.model_paths)]
    STDS = STDS[:len(args.model_paths)]
  
    if args.seed != SEED:
        set_seed(args.seed)
  
    base_models = load_pruned_models(args.model_paths, device)
    MODEL_NAMES = args.model_names[:len(base_models)]
  
    ensemble = FuzzyHesitantEnsemble(
        base_models, MEANS, STDS,
        num_memberships=args.num_memberships,
        freeze_models=True,
        cum_weight_threshold=args.cum_weight_threshold,
        hesitancy_threshold=args.hesitancy_threshold
    ).to(device)
    ensemble = DDP(ensemble, device_ids=[local_rank], output_device=local_rank)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset, is_distributed=True
    )
    
    if is_main:
        print("\nEVALUATING INDIVIDUAL MODELS ON TEST SET (Before Training)")
    individual_accs = []
    for i, model in enumerate(base_models):
        acc = evaluate_single_model_ddp(model, test_loader, device, f"Model {i+1} ({MODEL_NAMES[i]})", MEANS[i], STDS[i], is_main)
        individual_accs.append(acc)
    best_single = max(individual_accs)
  
    best_val_acc, history = train_hesitant_fuzzy(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_dir, local_rank
    )
    
    ckpt_path = os.path.join(args.save_dir, 'best_hesitant_fuzzy.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ensemble.module.hesitant_fuzzy.load_state_dict(ckpt['hesitant_state_dict'])
        if is_main:
            print("Best hesitant fuzzy network loaded.\n")

    if is_main:
        print("\n" + "="*70)
        print("EVALUATING FUZZY HESITANT ENSEMBLE")
        print("="*70)
    ensemble_test_acc, ensemble_weights, membership_values, activation_percentages = evaluate_ensemble_final_ddp(
        ensemble.module, test_loader, device, "Test", MODEL_NAMES, is_main
    )
    
    cleanup_distributed()

if __name__ == "__main__":
    main()
