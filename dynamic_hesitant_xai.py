import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
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

# ====================== SEED SETUP ======================
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
    """
    Create reproducible train/val/test splits from a dataset
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
  
    num_samples = len(dataset)
    indices = list(range(num_samples))
  
    # Get labels for stratified split
    labels = [dataset.samples[i][1] for i in indices]
  
    # First split: separate test set
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels
    )
  
    # Second split: separate train and val
    train_val_labels = [labels[i] for i in train_val_indices]
    val_size = val_ratio / (train_ratio + val_ratio)
  
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size,
        random_state=seed,
        stratify=train_val_labels
    )
  
    return train_indices, val_indices, test_indices

def prepare_real_fake_dataset(base_dir, seed=42):
    if os.path.exists(os.path.join(base_dir, 'training_fake')) and \
       os.path.exists(os.path.join(base_dir, 'training_real')):
        real_fake_dir = base_dir
    elif os.path.exists(os.path.join(base_dir, 'real_and_fake_face')):
        real_fake_dir = os.path.join(base_dir, 'real_and_fake_face')
    else:
        raise FileNotFoundError(
            f"Could not find training_fake/training_real in:\n"
            f" - {base_dir}\n"
            f" - {os.path.join(base_dir, 'real_and_fake_face')}"
        )
  
    temp_transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.ImageFolder(real_fake_dir, transform=temp_transform)
  
    print(f"\n[Dataset Info - Real/Fake]")
    print(f"Total samples: {len(full_dataset)}")
    print(f"Classes: {full_dataset.classes}")
    print(f"Class to index: {full_dataset.class_to_idx}")
  
    train_indices, val_indices, test_indices = create_reproducible_split(
        full_dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed
    )
  
    print(f"\n[Split Statistics]")
    print(f"Train: {len(train_indices)} samples ({len(train_indices)/len(full_dataset)*100:.1f}%)")
    print(f"Valid: {len(val_indices)} samples ({len(val_indices)/len(full_dataset)*100:.1f}%)")
    print(f"Test: {len(test_indices)} samples ({len(test_indices)/len(full_dataset)*100:.1f}%)")
  
    return full_dataset, train_indices, val_indices, test_indices

def prepare_hard_fake_real_dataset(base_dir, seed=42):
    if os.path.exists(os.path.join(base_dir, 'fake')) and \
       os.path.exists(os.path.join(base_dir, 'real')):
        dataset_dir = base_dir
    elif os.path.exists(os.path.join(base_dir, 'hardfakevsrealfaces')):
        dataset_dir = os.path.join(base_dir, 'hardfakevsrealfaces')
    else:
        raise FileNotFoundError(
            f"Could not find fake/real folders in:\n"
            f" - {base_dir}\n"
            f" - {os.path.join(base_dir, 'hardfakevsrealfaces')}"
        )
  
    temp_transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.ImageFolder(dataset_dir, transform=temp_transform)
  
    print(f"\n[Dataset Info - HardFakeVsReal]")
    print(f"Total samples: {len(full_dataset)}")
    print(f"Classes: {full_dataset.classes}")
    print(f"Class to index: {full_dataset.class_to_idx}")
  
    train_indices, val_indices, test_indices = create_reproducible_split(
        full_dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed
    )
  
    print(f"\n[Split Statistics]")
    print(f"Train: {len(train_indices)} samples ({len(train_indices)/len(full_dataset)*100:.1f}%)")
    print(f"Valid: {len(val_indices)} samples ({len(val_indices)/len(full_dataset)*100:.1f}%)")
    print(f"Test: {len(test_indices)} samples ({len(test_indices)/len(full_dataset)*100:.1f}%)")
  
    return full_dataset, train_indices, val_indices, test_indices

def prepare_deepflux_dataset(base_dir, seed=42):
    if os.path.exists(os.path.join(base_dir, 'Fake')) and \
       os.path.exists(os.path.join(base_dir, 'Real')):
        dataset_dir = base_dir
    elif os.path.exists(os.path.join(base_dir, 'DeepFLUX')):
        dataset_dir = os.path.join(base_dir, 'DeepFLUX')
    else:
        raise FileNotFoundError(
            f"Could not find Fake/Real folders in:\n"
            f" - {base_dir}\n"
            f" - {os.path.join(base_dir, 'DeepFLUX')}"
        )
  
    temp_transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.ImageFolder(dataset_dir, transform=temp_transform)
  
    print(f"\n[Dataset Info - DeepFLUX]")
    print(f"Total samples: {len(full_dataset)}")
    print(f"Classes: {full_dataset.classes}")
    print(f"Class to index: {full_dataset.class_to_idx}")
  
    train_indices, val_indices, test_indices = create_reproducible_split(
        full_dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed
    )
  
    print(f"\n[Split Statistics]")
    print(f"Train: {len(train_indices)} samples ({len(train_indices)/len(full_dataset)*100:.1f}%)")
    print(f"Valid: {len(val_indices)} samples ({len(val_indices)/len(full_dataset)*100:.1f}%)")
    print(f"Test: {len(test_indices)} samples ({len(test_indices)/len(full_dataset)*100:.1f}%)")
  
    return full_dataset, train_indices, val_indices, test_indices

def prepare_uadfV_dataset(base_dir, seed=42):
    """
    Prepares the UADFV dataset for splitting.
    Assumes the base_dir is the root of the UADFV folder.
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"UADFV dataset directory not found: {base_dir}")
    temp_transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = UADFVDataset(base_dir, transform=temp_transform)
    print(f"\n[Dataset Info - UADFV]")
    print(f"Total samples: {len(full_dataset)}")
    print(f"Classes: {full_dataset.classes}")
    print(f"Class to index: {full_dataset.class_to_idx}")
    train_indices, val_indices, test_indices = create_reproducible_split(
        full_dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed
    )
    print(f"\n[Split Statistics]")
    print(f"Train: {len(train_indices)} samples ({len(train_indices)/len(full_dataset)*100:.1f}%)")
    print(f"Valid: {len(val_indices)} samples ({len(val_indices)/len(full_dataset)*100:.1f}%)")
    print(f"Test: {len(test_indices)} samples ({len(test_indices)/len(full_dataset)*100:.1f}%)")
    return full_dataset, train_indices, val_indices, test_indices

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
    
def load_pruned_models(model_paths: List[str], device: torch.device) -> List[nn.Module]:
    try:
        from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        raise ImportError("Cannot import ResNet_50_pruned_hardfakevsreal. Ensure model.pruned_model.ResNet_pruned is available.")
  
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

def create_dataloaders(base_dir: str, batch_size: int, num_workers: int = 2, dataset_type: str = 'wild'):
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
        
            print(f"{split.capitalize():5}: {path}")
        
            transform = train_transform if split == 'train' else val_test_transform
            datasets_dict[split] = datasets.ImageFolder(path, transform=transform)
      
        print(f"\nDataset Stats:")
        for split, ds in datasets_dict.items():
            print(f" {split.capitalize():5}: {len(ds):,} images | Classes: {ds.classes}")
        print(f" Class → Index: {datasets_dict['train'].class_to_idx}\n")
      
        train_loader = DataLoader(datasets_dict['train'], batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 worker_init_fn=worker_init_fn)
        val_loader = DataLoader(datasets_dict['valid'], batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True, drop_last=False,
                               worker_init_fn=worker_init_fn)
        test_loader = DataLoader(datasets_dict['test'], batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=worker_init_fn)
  
    elif dataset_type == 'real_fake':
        print(f"Processing real-fake dataset from: {base_dir}")
        full_dataset, train_indices, val_indices, test_indices = prepare_real_fake_dataset(base_dir, seed=42)
      
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
      
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True, drop_last=False,
                               worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=worker_init_fn)
                                
    elif dataset_type == 'hard_fake_real':
        print(f"Processing hardfakevsrealfaces dataset from: {base_dir}")
        full_dataset, train_indices, val_indices, test_indices = prepare_hard_fake_real_dataset(base_dir, seed=42)
      
        if os.path.exists(os.path.join(base_dir, 'fake')) and \
           os.path.exists(os.path.join(base_dir, 'real')):
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
      
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True, drop_last=False,
                               worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=worker_init_fn)
                                
    elif dataset_type == 'deepflux':
        print(f"Processing DeepFLUX dataset from: {base_dir}")
        full_dataset, train_indices, val_indices, test_indices = prepare_deepflux_dataset(base_dir, seed=42)
      
        if os.path.exists(os.path.join(base_dir, 'Fake')) and \
           os.path.exists(os.path.join(base_dir, 'Real')):
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
      
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True, drop_last=False,
                               worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=worker_init_fn)
                                
    elif dataset_type == 'uadfV':
        print(f"Processing UADFV dataset from: {base_dir}")
        full_dataset, train_indices, val_indices, test_indices = prepare_uadfV_dataset(base_dir, seed=42)
        
        temp_transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = UADFVDataset(base_dir, transform=temp_transform)
        train_indices, val_indices, test_indices = create_reproducible_split(full_dataset, seed=42)
        
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)
        
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_test_transform
        test_dataset.dataset.transform = val_test_transform
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True, drop_last=False,
                               worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=worker_init_fn)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Use 'wild', 'real_fake', 'hard_fake_real', 'deepflux', or 'uadfV'")
  
    print(f"DataLoaders ready! Batch size: {batch_size}")
    print(f" Batches → Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    print("="*70 + "\n")
    return train_loader, val_loader, test_loader

@torch.no_grad()
def evaluate_single_model(model: nn.Module, loader: DataLoader, device: torch.device, name: str) -> float:
    model.eval()
    correct = total = 0
    for images, labels in tqdm(loader, desc=f"Evaluating {name}"):
        images, labels = images.to(device), labels.to(device).float()
        out = model(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = (out.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()
    acc = 100. * correct / total
    print(f" {name}: {acc:.2f}%")
    return acc

def train_hesitant_fuzzy(ensemble_model, train_loader, val_loader, num_epochs, lr, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    hesitant_net = ensemble_model.hesitant_fuzzy
    optimizer = torch.optim.AdamW(hesitant_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()
  
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'membership_variance': []}
  
    print("="*70)
    print("Training Fuzzy Hesitant Network")
    print("="*70)
    print(f"Trainable params: {sum(p.numel() for p in hesitant_net.parameters()):,}")
    print(f"Epochs: {num_epochs} | Initial LR: {lr}")
    print(f"Hesitant memberships per model: {hesitant_net.num_memberships}\n")
  
    for epoch in range(num_epochs):
        ensemble_model.train()
        train_loss = train_correct = train_total = 0.0
        membership_vars = []
    
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
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
        val_acc = evaluate_accuracy(ensemble_model, val_loader, device)
        scheduler.step()
    
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['membership_variance'].append(avg_membership_var)
          
        print(f"\nEpoch {epoch+1}:")
        print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f" Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f" Membership Variance (Hesitancy): {avg_membership_var:.4f}")
          
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
  
    print(f"\nTraining completed! Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc, history

@torch.no_grad()
def evaluate_ensemble_final(model, loader, device, name, model_names):
    model.eval()
   
    all_preds = []
    all_labels = []
    all_weights = []
    all_memberships = []
   
    activation_counts = torch.zeros(len(model_names), device=device)
    total_samples = 0
    
    for images, labels in tqdm(loader, desc=f"Evaluating {name}", leave=True):
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
    
    all_weights = torch.cat(all_weights).cpu().numpy()
    all_memberships = torch.cat(all_memberships).cpu().numpy()
    activation_counts = activation_counts.cpu().numpy()
    acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    avg_weights = all_weights.mean(axis=0)
    activation_percentages = (activation_counts / total_samples) * 100
    
    print(f"\n{'='*70}")
    print(f"{name.upper()} SET RESULTS")
    print(f"{'='*70}")
    print(f" → Accuracy: {acc:.3f}%")
    print(f" → Total Samples: {total_samples:,}")
    print(f"\nAverage Model Weights:")
    for i, (w, name) in enumerate(zip(avg_weights, model_names)):
        print(f" {i+1:2d}. {name:<25}: {w:6.4f} ({w*100:5.2f}%)")
    print(f"\nActivation Frequency:")
    for i, (perc, count, name) in enumerate(zip(activation_percentages, activation_counts, model_names)):
        print(f" {i+1:2d}. {name:<25}: {perc:6.2f}% active ({int(count):,} / {total_samples:,} samples)")
    print(f"\nHesitant Membership Values:")
    for i, name in enumerate(model_names):
        mems = all_memberships[:, i].mean(axis=0)
        var = all_memberships[:, i].var(axis=0).mean()
        print(f" {i+1:2d}. {name:<25}: μ = [{', '.join([f'{m:.3f}' for m in mems])}] | Hesitancy = {var:.4f}")
    print(f"{'='*70}")
    return acc, avg_weights.tolist(), all_memberships.mean(axis=0).tolist(), activation_percentages.tolist()

@torch.no_grad()
def evaluate_accuracy(model, loader, device):
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
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), activations.shape[1:], mode='bilinear', align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.squeeze().cpu().numpy()

def main():
    SEED = 42
    set_seed(SEED)
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
  
    parser = argparse.ArgumentParser(description="Train Fuzzy Hesitant Ensemble")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_memberships', type=int, default=3, help='Number of membership values per model')
    parser.add_argument('--num_grad_cam_samples', type=int, default=5, help='Number of samples for GradCAM visualization')
    parser.add_argument('--dataset', type=str, choices=['wild', 'real_fake', 'hard_fake_real', 'deepflux', 'uadfV'], required=True,
                       help='Dataset type')
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory of dataset')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, help='Paths to pruned model checkpoints')
    parser.add_argument('--model_names', type=str, nargs='+', required=True, help='Names for each model')
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed for reproducibility')
  
    args = parser.parse_args()
  
    if len(args.model_names) != len(args.model_paths):
        raise ValueError(f"Number of model_names ({len(args.model_names)}) must match model_paths ({len(args.model_paths)})")
  
    MEANS = [(0.5207, 0.4258, 0.3806), (0.4868, 0.3972, 0.3624), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2296, 0.2066, 0.2009), (0.2410, 0.2161, 0.2081)]
  
    MEANS = MEANS[:len(args.model_paths)]
    STDS = STDS[:len(args.model_paths)]
  
    if args.seed != SEED:
        set_seed(args.seed)
  
    print(f"="*70)
    print(f"Single GPU Training | SEED: {args.seed}")
    print(f"="*70)
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"\nUsing normalization parameters:")
    print(f" MEANS: {MEANS}")
    print(f" STDS: {STDS}")
    print(f"\nModels to load:")
    for i, (path, name) in enumerate(zip(args.model_paths, args.model_names)):
        print(f" {i+1}. {name}: {path}")
    print(f"="*70 + "\n")
  
    base_models = load_pruned_models(args.model_paths, device)
    if len(base_models) != len(args.model_paths):
        print(f"[WARNING] Only {len(base_models)}/{len(args.model_paths)} models loaded. Adjusting parameters.")
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
    
    hesitant_net = ensemble.hesitant_fuzzy
    trainable = sum(p.numel() for p in hesitant_net.parameters())
    total_params = sum(p.numel() for p in ensemble.parameters())
    print(f"Total params: {total_params:,} | Trainable: {trainable:,} | Frozen: {total_params - trainable:,}\n")
  
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset
    )
    
    print("\n" + "="*70)
    print("EVALUATING INDIVIDUAL MODELS ON TEST SET (Before Training)")
    print("="*70)
    individual_accs = []
    for i, model in enumerate(base_models):
        acc = evaluate_single_model(model, test_loader, device, f"Model {i+1} ({MODEL_NAMES[i]})")
        individual_accs.append(acc)
    best_single = max(individual_accs)
    best_idx = individual_accs.index(best_single)
    print(f"\nBest Single Model: Model {best_idx+1} ({MODEL_NAMES[best_idx]}) → {best_single:.2f}%")
  
    best_val_acc, history = train_hesitant_fuzzy(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_dir
    )
    
    ckpt_path = os.path.join(args.save_dir, 'best_hesitant_fuzzy.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ensemble.hesitant_fuzzy.load_state_dict(ckpt['hesitant_state_dict'])
        print("Best hesitant fuzzy network loaded.\n")

    print("\n" + "="*70)
    print("EVALUATING FUZZY HESITANT ENSEMBLE")
    print("="*70)
    ensemble_test_acc, ensemble_weights, membership_values, activation_percentages = evaluate_ensemble_final(
        ensemble, test_loader, device, "Test", MODEL_NAMES
    )
    
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"Best Single Model Acc : {best_single:.2f}%")
    print(f"Hesitant Ensemble Acc : {ensemble_test_acc:.2f}%")
    improvement = ensemble_test_acc - best_single
    print(f"Improvement : {improvement:+.2f}%")
    
    # Save final results
    final_results = {
        'best_single_model': {
            'name': MODEL_NAMES[best_idx],
            'accuracy': best_single
        },
        'ensemble': {
            'test_accuracy': ensemble_test_acc,
            'model_weights': {name: float(w) for name, w in zip(MODEL_NAMES, ensemble_weights)},
            'activation_percentages': {name: float(p) for name, p in zip(MODEL_NAMES, activation_percentages)}
        },
        'improvement': float(improvement),
        'training_history': history
    }
    
    results_path = os.path.join(args.save_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\nResults saved to: {results_path}")
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, 'final_ensemble_model.pt')
    torch.save({
        'ensemble_state_dict': ensemble.state_dict(),
        'hesitant_fuzzy_state_dict': ensemble.hesitant_fuzzy.state_dict(),
        'test_accuracy': ensemble_test_acc,
        'model_names': MODEL_NAMES,
        'means': MEANS,
        'stds': STDS
    }, final_model_path)
    print(f"Final model saved: {final_model_path}")

        # GradCAM Visualization
    print("="*70)
    print("GENERATING GRADCAM VISUALIZATIONS FOR ENSEMBLE OUTPUT")
    print("="*70)

    ensemble.eval()
    vis_dir = os.path.join(args.save_dir, 'gradcam_vis')
    os.makedirs(vis_dir, exist_ok=True)

    # 1. شاخص‌های مجموعه تست را دریافت کنید
    # test_loader.dataset یک شیء Subset است که شامل شاخص‌های مجموعه تست است
    test_indices = test_loader.dataset.indices
    
    # 2. این شاخص‌ها را برای نمونه‌برداری تصادفی به هم بریزید
    # از یک کپی استفاده می‌کنیم تا ترتیب اصلی شاخص‌ها حفظ شود
    vis_indices = test_indices.copy()
    random.shuffle(vis_indices)
    
    # 3. فقط به تعداد مورد نیاز نمونه برداری کنید
    vis_indices = vis_indices[:args.num_grad_cam_samples]

    # 4. یک Subset و DataLoader جدید برای مصورسازی ایجاد کنید
    # این DataLoader دیگر نیازی به shuffle ندارد چون ما خودمان شاخص‌ها را به هم ریخته‌ایم
    vis_dataset = Subset(test_loader.dataset.dataset, vis_indices)
    vis_loader = DataLoader(vis_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    # 5. در حلقه، از شاخص صحیح برای استخراج مسیر و لیبل استفاده کنید
    for idx, (image, label_from_loader) in enumerate(vis_loader):
        image = image.to(device)
        
        # شاخص واقعی این نمونه در مجموعه داده اصلی
        original_full_dataset_index = vis_indices[idx]
        
        # مسیر و لیبل واقعی را با استفاده از شاخص صحیح استخراج کنید
        img_path, true_label = test_loader.dataset.dataset.samples[original_full_dataset_index]
        
        # برای اطمینان، می‌توانید بررسی کنید که لیبل دریافتی از DataLoader با لیبل استخراج شده برابر است
        assert label_from_loader == true_label, "Mismatch between DataLoader label and dataset label!"

        print(f"\n[GradCAM {idx+1}/{len(vis_loader)}]")
        print(f"  Original image path: {img_path}")
        print(f"  True label: {'real' if true_label == 1 else 'fake'} (label={true_label})")

        # پیش‌بینی انسامبل
        with torch.no_grad():
            output, weights, _, _ = ensemble(image, return_details=True)
        pred = 1 if output.squeeze().item() > 0 else 0
        print(f"  Predicted: {'real' if pred == 1 else 'fake'}")

        # بقیه کد GradCAM بدون تغییر باقی می‌ماند...
        active_models = torch.where(weights[0] > 1e-4)[0].cpu().tolist()
        combined_cam = None
        for i in active_models:
            model = ensemble.models[i]
            for p in model.parameters():
                p.requires_grad_(True)
            target_layer = model.layer4[2].conv3
            gradcam = GradCAM(model, target_layer)
            with torch.enable_grad():
                x_n = ensemble.normalizations(image, i)
                model_out = model(x_n)
                if isinstance(model_out, (tuple, list)):
                    model_out = model_out[0]
                model_out = model_out.squeeze()
                score = model_out if pred == 1 else -model_out
                cam = gradcam.generate(score)
            weight = weights[0, i].item()
            if combined_cam is None:
                combined_cam = weight * cam
            else:
                combined_cam += weight * cam
            for p in model.parameters():
                p.requires_grad_(False)

        if combined_cam is not None:
            combined_cam = (combined_cam - combined_cam.min()) / (combined_cam.max() - combined_cam.min() + 1e-8)
            img_np = image[0].cpu().permute(1, 2, 0).numpy()
            img_h, img_w = img_np.shape[:2]
            combined_cam_resized = cv2.resize(combined_cam, (img_w, img_h))

            heatmap = cv2.applyColorMap(np.uint8(255 * combined_cam_resized), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            overlay = heatmap + img_np
            overlay = overlay / overlay.max()

            save_path = os.path.join(vis_dir, f"sample_{idx}_true{'real' if true_label==1 else 'fake'}_pred{'real' if pred==1 else 'fake'}.png")
            
            plt.figure(figsize=(10, 10))
            plt.imshow(overlay)
            plt.title(f"True: {'real' if true_label == 1 else 'fake'} | Pred: {'real' if pred == 1 else 'fake'}\n{os.path.basename(img_path)}")
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            plt.close()
            
            print(f"  GradCAM saved: {save_path}")

    print("="*70)
    print("GradCAM visualizations completed!")
    print("="*70)

if __name__ == "__main__":
    main()
