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
from sklearn.metrics import cohen_kappa_score
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
warnings.filterwarnings("ignore")


# ============================================================================
# Kappa Diversity Calculator
# ============================================================================

class KappaDiversityCalculator:
    """محاسبه diversity با Cohen's Kappa"""
    
    @staticmethod
    def calculate_pairwise_kappa(predictions: torch.Tensor) -> torch.Tensor:
        batch_size, num_models = predictions.shape
        predictions_np = predictions.cpu().numpy()
        kappa_matrix = np.ones((num_models, num_models))
        
        for i in range(num_models):
            for j in range(i + 1, num_models):
                try:
                    kappa = cohen_kappa_score(predictions_np[:, i], predictions_np[:, j])
                    kappa_matrix[i, j] = kappa
                    kappa_matrix[j, i] = kappa
                except:
                    kappa_matrix[i, j] = 1.0
                    kappa_matrix[j, i] = 1.0
        
        return torch.tensor(kappa_matrix, dtype=torch.float32)
    
    @staticmethod
    def calculate_diversity_score(predictions: torch.Tensor) -> torch.Tensor:
        """
        محاسبه diversity score برای هر نمونه با استفاده از Kappa
        
        برای هر sample، میانگین Kappa بین همه جفت مدل‌ها را محاسبه می‌کنیم
        diversity = 1 - kappa (کم‌تر Kappa = بیشتر diversity)
        
        Returns:
            diversity_scores: (batch_size,) - بالاتر = diversity بیشتر
        """
        batch_size, num_models = predictions.shape
        diversity_scores = []
        predictions_np = predictions.cpu().numpy()
        
        for b in range(batch_size):
            sample_preds = predictions_np[b, :]
            
            # اگر همه مدل‌ها یک جواب دادند، diversity = 0
            if len(np.unique(sample_preds)) == 1:
                diversity_scores.append(0.0)
                continue
            
            # محاسبه Kappa-based agreement بین همه جفت مدل‌ها
            # برای یک sample، agreement ساده است: موافق یا مخالف
            agreements = []
            for i in range(num_models):
                for j in range(i + 1, num_models):
                    if sample_preds[i] == sample_preds[j]:
                        agreements.append(1.0)  # موافقت
                    else:
                        agreements.append(0.0)  # عدم موافقت
            
            # میانگین agreement (شبیه به Kappa ساده‌شده)
            avg_agreement = np.mean(agreements) if agreements else 1.0
            
            # تبدیل به diversity: agreement بالا → diversity پایین
            diversity = 1.0 - avg_agreement
            diversity_scores.append(diversity)
        
        return torch.tensor(diversity_scores, dtype=torch.float32)
    
    @staticmethod
    def calculate_average_kappa(predictions: torch.Tensor) -> float:
        kappa_matrix = KappaDiversityCalculator.calculate_pairwise_kappa(predictions)
        num_models = kappa_matrix.shape[0]
        mask = torch.triu(torch.ones(num_models, num_models), diagonal=1).bool()
        avg_kappa = kappa_matrix[mask].mean().item()
        return avg_kappa


# ============================================================================
# Dataset Classes
# ============================================================================

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


class TransformSubset(Subset):
    """✅ کلاس اضافه شده"""
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform
  
    def __getitem__(self, idx):
        img, label = self.dataset.samples[self.indices[idx]]
        img = self.dataset.loader(img)
        if self.transform:
            img = self.transform(img)
        return img, label


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
            raise ValueError("Gradients not captured")
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=[1, 2], keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), activations.shape[1:], 
                           mode='bilinear', align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.squeeze().cpu().numpy()


# ============================================================================
# Helper Functions
# ============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    if dist.is_initialized() and dist.get_rank() == 0:
        print(f"[SEED] All random seeds set to: {seed}")

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_reproducible_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
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


# Dataset preparation functions
def prepare_real_fake_dataset(base_dir, seed=42):
    if os.path.exists(os.path.join(base_dir, 'training_fake')) and \
       os.path.exists(os.path.join(base_dir, 'training_real')):
        real_fake_dir = base_dir
    elif os.path.exists(os.path.join(base_dir, 'real_and_fake_face')):
        real_fake_dir = os.path.join(base_dir, 'real_and_fake_face')
    else:
        raise FileNotFoundError(f"Could not find training_fake/training_real in {base_dir}")
  
    temp_transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.ImageFolder(real_fake_dir, transform=temp_transform)
  
    if dist.get_rank() == 0:
        print(f"\n[Dataset Info - Real/Fake]")
        print(f"Total samples: {len(full_dataset)}")
        print(f"Classes: {full_dataset.classes}")
  
    train_indices, val_indices, test_indices = create_reproducible_split(
        full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=seed
    )
  
    if dist.get_rank() == 0:
        print(f"\n[Split Statistics]")
        print(f"Train: {len(train_indices)} | Valid: {len(val_indices)} | Test: {len(test_indices)}")
  
    return full_dataset, train_indices, val_indices, test_indices

def prepare_hard_fake_real_dataset(base_dir, seed=42):
    if os.path.exists(os.path.join(base_dir, 'fake')) and \
       os.path.exists(os.path.join(base_dir, 'real')):
        dataset_dir = base_dir
    elif os.path.exists(os.path.join(base_dir, 'hardfakevsrealfaces')):
        dataset_dir = os.path.join(base_dir, 'hardfakevsrealfaces')
    else:
        raise FileNotFoundError(f"Could not find fake/real folders in {base_dir}")
  
    temp_transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.ImageFolder(dataset_dir, transform=temp_transform)
  
    if dist.get_rank() == 0:
        print(f"\n[Dataset Info - HardFakeVsReal]")
        print(f"Total samples: {len(full_dataset)}")
  
    train_indices, val_indices, test_indices = create_reproducible_split(
        full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=seed
    )
    return full_dataset, train_indices, val_indices, test_indices

def prepare_deepflux_dataset(base_dir, seed=42):
    if os.path.exists(os.path.join(base_dir, 'Fake')) and \
       os.path.exists(os.path.join(base_dir, 'Real')):
        dataset_dir = base_dir
    elif os.path.exists(os.path.join(base_dir, 'DeepFLUX')):
        dataset_dir = os.path.join(base_dir, 'DeepFLUX')
    else:
        raise FileNotFoundError(f"Could not find Fake/Real folders in {base_dir}")
  
    temp_transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.ImageFolder(dataset_dir, transform=temp_transform)
  
    if dist.get_rank() == 0:
        print(f"\n[Dataset Info - DeepFLUX]")
        print(f"Total samples: {len(full_dataset)}")
  
    train_indices, val_indices, test_indices = create_reproducible_split(
        full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=seed
    )
    return full_dataset, train_indices, val_indices, test_indices

def prepare_uadfV_dataset(base_dir, seed=42):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"UADFV dataset directory not found: {base_dir}")
    temp_transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = UADFVDataset(base_dir, transform=temp_transform)
    
    if dist.get_rank() == 0:
        print(f"\n[Dataset Info - UADFV]")
        print(f"Total samples: {len(full_dataset)}")
    
    train_indices, val_indices, test_indices = create_reproducible_split(
        full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=seed
    )
    return full_dataset, train_indices, val_indices, test_indices


# ============================================================================
# ✅ تابع create_dataloaders کامل
# ============================================================================

def create_dataloaders(base_dir: str, batch_size: int, num_workers: int = 2, 
                      dataset_type: str = 'wild', is_distributed=False):
    """✅ تابع کامل برای ساخت DataLoader"""
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
  
    if dataset_type == 'wild':
        splits = ['train', 'valid', 'test']
        datasets_dict = {}
      
        for split in splits:
            path = os.path.join(base_dir, split)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Folder not found: {path}")
            transform = train_transform if split == 'train' else val_test_transform
            datasets_dict[split] = datasets.ImageFolder(path, transform=transform)
      
        train_sampler = DistributedSampler(datasets_dict['train']) if is_distributed else None
        val_sampler = DistributedSampler(datasets_dict['valid'], shuffle=False) if is_distributed else None
        test_sampler = DistributedSampler(datasets_dict['test'], shuffle=False) if is_distributed else None
        
        train_loader = DataLoader(datasets_dict['train'], batch_size=batch_size, 
                                 shuffle=(train_sampler is None), sampler=train_sampler,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 worker_init_fn=worker_init_fn)
        val_loader = DataLoader(datasets_dict['valid'], batch_size=batch_size, 
                               shuffle=False, sampler=val_sampler,
                               num_workers=num_workers, pin_memory=True, drop_last=False,
                               worker_init_fn=worker_init_fn)
        test_loader = DataLoader(datasets_dict['test'], batch_size=batch_size, 
                                shuffle=False, sampler=test_sampler,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=worker_init_fn)
  
    elif dataset_type == 'real_fake':
        full_dataset, train_indices, val_indices, test_indices = prepare_real_fake_dataset(base_dir, seed=42)
        
        train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
        val_dataset = TransformSubset(full_dataset, val_indices, val_test_transform)
        test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)
      
        train_sampler = DistributedSampler(train_dataset) if is_distributed else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
        test_sampler = DistributedSampler(test_dataset, shuffle=False) if is_distributed else None
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=(train_sampler is None), sampler=train_sampler,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, sampler=val_sampler,
                               num_workers=num_workers, pin_memory=True, drop_last=False,
                               worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, sampler=test_sampler,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=worker_init_fn)
                                
    elif dataset_type == 'hard_fake_real':
        full_dataset, train_indices, val_indices, test_indices = prepare_hard_fake_real_dataset(base_dir, seed=42)
        
        train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
        val_dataset = TransformSubset(full_dataset, val_indices, val_test_transform)
        test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)
      
        train_sampler = DistributedSampler(train_dataset) if is_distributed else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
        test_sampler = DistributedSampler(test_dataset, shuffle=False) if is_distributed else None
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=(train_sampler is None), sampler=train_sampler,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, sampler=val_sampler,
                               num_workers=num_workers, pin_memory=True, drop_last=False,
                               worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, sampler=test_sampler,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=worker_init_fn)
                                
    elif dataset_type == 'deepflux':
        full_dataset, train_indices, val_indices, test_indices = prepare_deepflux_dataset(base_dir, seed=42)
        
        train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
        val_dataset = TransformSubset(full_dataset, val_indices, val_test_transform)
        test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)
      
        train_sampler = DistributedSampler(train_dataset) if is_distributed else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
        test_sampler = DistributedSampler(test_dataset, shuffle=False) if is_distributed else None
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=(train_sampler is None), sampler=train_sampler,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, sampler=val_sampler,
                               num_workers=num_workers, pin_memory=True, drop_last=False,
                               worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, sampler=test_sampler,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=worker_init_fn)
                                
    elif dataset_type == 'uadfV':
        full_dataset, train_indices, val_indices, test_indices = prepare_uadfV_dataset(base_dir, seed=42)
        
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)
        
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_test_transform
        test_dataset.dataset.transform = val_test_transform
        
        train_sampler = DistributedSampler(train_dataset) if is_distributed else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
        test_sampler = DistributedSampler(test_dataset, shuffle=False) if is_distributed else None
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=(train_sampler is None), sampler=train_sampler,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, sampler=val_sampler,
                               num_workers=num_workers, pin_memory=True, drop_last=False,
                               worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, sampler=test_sampler,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=worker_init_fn)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
  
    if dist.get_rank() == 0:
        print(f"DataLoaders ready! Batch size: {batch_size}")
        print(f"Batches → Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
        print("="*70 + "\n")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Model Classes
# ============================================================================

class KappaAwareHesitantFuzzy(nn.Module):
    def __init__(self, input_dim: int, num_models: int, num_memberships: int = 3, 
                 dropout: float = 0.3, diversity_weight: float = 0.3):
        super().__init__()
        self.num_models = num_models
        self.num_memberships = num_memberships
        self.diversity_weight = diversity_weight
        
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
        
        self.diversity_modulator = nn.Sequential(
            nn.Linear(128 + num_models, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_models),
            nn.Sigmoid()
        )
        
        self.aggregation_weights = nn.Parameter(torch.ones(num_memberships) / num_memberships)
    
    def forward(self, x: torch.Tensor, model_outputs: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_net(x).flatten(1)
        
        memberships = self.membership_generator(features)
        memberships = memberships.view(-1, self.num_models, self.num_memberships)
        memberships = torch.sigmoid(memberships)
        
        agg_weights = F.softmax(self.aggregation_weights, dim=0)
        base_weights = (memberships * agg_weights.view(1, 1, -1)).sum(dim=2)
        base_weights = F.softmax(base_weights, dim=1)
        
        if model_outputs is not None:
            model_preds = (model_outputs > 0).float()
            diversity_scores = KappaDiversityCalculator.calculate_diversity_score(model_preds)
            diversity_scores = diversity_scores.unsqueeze(1).to(x.device)
            
            combined_features = torch.cat([features, model_preds], dim=1)
            diversity_modulation = self.diversity_modulator(combined_features)
            
            final_weights = base_weights * (1 - self.diversity_weight) + \
                           diversity_modulation * self.diversity_weight
            
            final_weights = final_weights / (final_weights.sum(dim=1, keepdim=True) + 1e-8)
        else:
            final_weights = base_weights
            diversity_scores = torch.zeros(x.size(0), 1, device=x.device)
        
        return final_weights, memberships, diversity_scores


class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))
  
    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


class KappaAwareFuzzyEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]],
                 stds: List[Tuple[float]], num_memberships: int = 3, 
                 freeze_models: bool = True, diversity_weight: float = 0.3,
                 min_diversity_threshold: float = 0.2):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        
        self.hesitant_fuzzy = KappaAwareHesitantFuzzy(
            input_dim=128,
            num_models=self.num_models,
            num_memberships=num_memberships,
            diversity_weight=diversity_weight
        )
        
        self.min_diversity_threshold = min_diversity_threshold
        self.kappa_calculator = KappaDiversityCalculator()
        
        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
    
    def forward(self, x: torch.Tensor, return_details: bool = False):
        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
        
        for i in range(self.num_models):
            x_n = self.normalizations(x, i)
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            outputs[:, i] = out
        
        final_weights, all_memberships, diversity_scores = self.hesitant_fuzzy(
            x, outputs.squeeze(-1)
        )
        
        mask = torch.ones_like(final_weights)
        
        for b in range(x.size(0)):
            if diversity_scores[b] < self.min_diversity_threshold:
                top_k = max(2, self.num_models // 2)
                _, top_indices = torch.topk(final_weights[b], top_k)
                sample_mask = torch.zeros(self.num_models, device=x.device)
                sample_mask[top_indices] = 1.0
                mask[b] = sample_mask
        
        final_weights = final_weights * mask
        final_weights = final_weights / (final_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        final_output = (outputs * final_weights.unsqueeze(-1)).sum(dim=1)
        
        if return_details:
            return final_output, final_weights, all_memberships, outputs, diversity_scores
        return final_output, final_weights


def load_pruned_models(model_paths: List[str], device: torch.device) -> List[nn.Module]:
    try:
        from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        raise ImportError("Cannot import ResNet_50_pruned_hardfakevsreal")
  
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
            model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device).eval()
            models.append(model)
        except Exception as e:
            if dist.get_rank() == 0:
                print(f" [ERROR] Failed to load {path}: {e}")
            continue
    
    if len(models) == 0:
        raise ValueError("No models loaded!")
    
    return models


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_kappa_aware_ensemble(ensemble_model, train_loader, val_loader, num_epochs, 
                               lr, device, save_dir, local_rank):
    os.makedirs(save_dir, exist_ok=True)
    hesitant_net = ensemble_model.module.hesitant_fuzzy if hasattr(ensemble_model, 'module') else ensemble_model.hesitant_fuzzy
    
    optimizer = torch.optim.AdamW(hesitant_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [], 'val_acc': [], 
        'membership_variance': [], 'avg_diversity': [], 'avg_kappa': []
    }
    
    is_main = local_rank == 0
    
    if is_main:
        print("="*70)
        print("🆕 Training Kappa-Aware Fuzzy Hesitant Ensemble")
        print("="*70)
        print(f"Trainable params: {sum(p.numel() for p in hesitant_net.parameters()):,}")
        print(f"Diversity weight: {hesitant_net.diversity_weight}")
        print("="*70 + "\n")
    
    for epoch in range(num_epochs):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        ensemble_model.train()
        train_loss = train_correct = train_total = 0.0
        membership_vars = []
        diversity_scores_epoch = []
        kappa_scores_epoch = []
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', disable=not is_main):
            images, labels = images.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            outputs, weights, memberships, all_outputs, diversity_scores = ensemble_model(
                images, return_details=True
            )
            
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()
            
            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            pred = (outputs.squeeze(1) > 0).long()
            train_correct += pred.eq(labels.long()).sum().item()
            train_total += batch_size
            
            membership_vars.append(memberships.var(dim=2).mean().item())
            diversity_scores_epoch.append(diversity_scores.mean().item())
            
            with torch.no_grad():
                model_preds = (all_outputs.squeeze(-1) > 0).long()
                avg_kappa = KappaDiversityCalculator.calculate_average_kappa(model_preds)
                kappa_scores_epoch.append(avg_kappa)
        
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / train_total
        avg_membership_var = np.mean(membership_vars)
        avg_diversity = np.mean(diversity_scores_epoch)
        avg_kappa = np.mean(kappa_scores_epoch)
        
        val_acc = evaluate_kappa_aware(ensemble_model, val_loader, device, is_distributed=dist.is_initialized())
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['membership_variance'].append(avg_membership_var)
        history['avg_diversity'].append(avg_diversity)
        history['avg_kappa'].append(avg_kappa)
        
        if is_main:
            print(f"\nEpoch {epoch+1}:")
            print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f" Val Acc: {val_acc:.2f}%")
            print(f" 🆕 Diversity Score: {avg_diversity:.4f} | Avg Kappa: {avg_kappa:.4f}")
            print(f" (Lower Kappa = Higher Diversity = Better Ensemble)")
            print("-" * 70)
        
        if is_main and val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(save_dir, 'best_kappa_aware_ensemble.pt')
            torch.save({
                'epoch': epoch + 1,
                'hesitant_state_dict': hesitant_net.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, ckpt_path)
            print(f" ✓ Best model saved → {val_acc:.2f}%")
    
    return best_val_acc, history


@torch.no_grad()
def evaluate_kappa_aware(model, loader, device, is_distributed=False):
    """✅ ارزیابی با DDP support"""
    model.eval()
    correct = total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs, _ = model(images)
        pred = (outputs.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()
    
    if is_distributed:
        correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
        total_tensor = torch.tensor(total, dtype=torch.long, device=device)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        correct = correct_tensor.item()
        total = total_tensor.item()
    
    acc = 100. * correct / total
    return acc


@torch.no_grad()
def evaluate_ensemble_final_kappa(model, loader, device, name, model_names, is_main=True):
    model.eval()
    local_preds = []
    local_labels = []
    local_weights = []
    local_diversity = []
    all_model_outputs = []
    
    for images, labels in tqdm(loader, desc=f"Evaluating {name}", disable=not is_main):
        images = images.to(device)
        labels = labels.to(device)
        outputs, weights, memberships, model_outs, diversity_scores = model(images, return_details=True)
        pred = (outputs.squeeze(1) > 0).long()
        
        local_preds.append(pred)
        local_labels.append(labels)
        local_weights.append(weights)
        local_diversity.append(diversity_scores)
        all_model_outputs.append(model_outs)
    
    local_preds_tensor = torch.cat(local_preds)
    local_labels_tensor = torch.cat(local_labels)
    local_weights_tensor = torch.cat(local_weights)
    local_diversity_tensor = torch.cat(local_diversity)
    all_model_outputs_tensor = torch.cat(all_model_outputs)
    
    if is_main:
        all_preds = local_preds_tensor.cpu().numpy()
        all_labels = local_labels_tensor.cpu().numpy()
        all_weights = local_weights_tensor.cpu().numpy()
        all_diversity = local_diversity_tensor.cpu().numpy()
        
        model_predictions = (all_model_outputs_tensor.squeeze(-1) > 0).long()
        avg_kappa = KappaDiversityCalculator.calculate_average_kappa(model_predictions)
        
        acc = 100. * np.mean(all_preds == all_labels)
        avg_weights = all_weights.mean(axis=0)
        activation_counts = (all_weights > 1e-4).sum(axis=0)
        total_samples = all_weights.shape[0]
        activation_percentages = (activation_counts / total_samples) * 100
        avg_diversity_score = all_diversity.mean()
        
        print(f"\n{'='*70}")
        print(f"{name.upper()} SET RESULTS (🆕 WITH KAPPA DIVERSITY)")
        print(f"{'='*70}")
        print(f" → Accuracy: {acc:.3f}%")
        print(f" → Total Samples: {total_samples:,}")
        print(f" 🆕 → Avg Diversity Score: {avg_diversity_score:.4f}")
        print(f" 🆕 → Avg Kappa: {avg_kappa:.4f} (Lower = More Diverse)")
        print(f"\nAverage Model Weights:")
        for i, (w, name) in enumerate(zip(avg_weights, model_names)):
            print(f" {i+1:2d}. {name:<25}: {w:6.4f} ({w*100:5.2f}%)")
        print(f"\nActivation Frequency:")
        for i, (perc, count, name) in enumerate(zip(activation_percentages, activation_counts, model_names)):
            print(f" {i+1:2d}. {name:<25}: {perc:6.2f}% active ({int(count):,} / {total_samples:,})")
        print(f"{'='*70}")
        
        return acc, avg_weights.tolist(), avg_diversity_score, avg_kappa, activation_percentages.tolist()
    
    return 0.0, [0.0]*len(model_names), 0.0, 0.0, [0.0]*len(model_names)


# ============================================================================
# DDP Setup
# ============================================================================

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        if rank == 0:
            print(f"✅ Initialized DDP: rank {rank}, world_size {world_size}")
        return device, local_rank, rank, world_size
    else:
        print("⚠️ Not running in distributed mode")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# Main Function
# ============================================================================

def main():
    SEED = 42
    set_seed(SEED)
    
    device, local_rank, rank, world_size = setup_distributed()
    is_main = rank == 0
    
    parser = argparse.ArgumentParser(description="🆕 Train Kappa-Aware Fuzzy Hesitant Ensemble")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_memberships', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--dataset', type=str, 
                       choices=['wild', 'real_fake', 'hard_fake_real', 'deepflux', 'uadfV'], 
                       required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./results_kappa/')
    parser.add_argument('--diversity_weight', type=float, default=0.3)
    parser.add_argument('--min_diversity_threshold', type=float, default=0.2)
    
    args = parser.parse_args()
    
    if len(args.model_names) != len(args.model_paths):
        raise ValueError(f"Model names and paths count mismatch")
    
    MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    
    MEANS = MEANS[:len(args.model_paths)]
    STDS = STDS[:len(args.model_paths)]
    
    if is_main:
        print("="*70)
        print("🆕 KAPPA-AWARE DIVERSITY ENSEMBLE TRAINING")
        print("="*70)
        print(f"Diversity Weight: {args.diversity_weight}")
        print(f"Min Diversity Threshold: {args.min_diversity_threshold}")
        print("="*70 + "\n")
    
    base_models = load_pruned_models(args.model_paths, device)
    MODEL_NAMES = args.model_names[:len(base_models)]
    
    ensemble = KappaAwareFuzzyEnsemble(
        base_models, MEANS, STDS,
        num_memberships=args.num_memberships,
        freeze_models=True,
        diversity_weight=args.diversity_weight,
        min_diversity_threshold=args.min_diversity_threshold
    ).to(device)
    
    if world_size > 1:
        ensemble = DDP(ensemble, device_ids=[local_rank], output_device=local_rank)
    
    # ✅ ساخت DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, 
        num_workers=args.num_workers,
        dataset_type=args.dataset, 
        is_distributed=(world_size > 1)
    )
    
    # آموزش
    best_val_acc, history = train_kappa_aware_ensemble(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_dir, local_rank
    )
    
    # ارزیابی نهایی
    if is_main:
        ckpt_path = os.path.join(args.save_dir, 'best_kappa_aware_ensemble.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model_to_load = ensemble.module if hasattr(ensemble, 'module') else ensemble
            model_to_load.hesitant_fuzzy.load_state_dict(ckpt['hesitant_state_dict'])
        
        model_for_eval = ensemble.module if hasattr(ensemble, 'module') else ensemble
        test_acc, ensemble_weights, diversity_score, avg_kappa, activation_pct = \
            evaluate_ensemble_final_kappa(model_for_eval, test_loader, device, "Test", MODEL_NAMES, is_main)
        
        # ذخیره نتایج
        final_results = {
            'test_accuracy': test_acc,
            'diversity_score': diversity_score,
            'average_kappa': avg_kappa,
            'model_weights': {name: float(w) for name, w in zip(MODEL_NAMES, ensemble_weights)},
            'activation_percentages': {name: float(p) for name, p in zip(MODEL_NAMES, activation_pct)},
            'training_history': history
        }
        
        with open(os.path.join(args.save_dir, 'kappa_results.json'), 'w') as f:
            json.dump(final_results, f, indent=4)
        
        print("\n✅ Kappa-Aware Ensemble Training Complete!")
        print(f"Results saved to: {args.save_dir}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
