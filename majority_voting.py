import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
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

# ================== DATASET CLASSES ==================
class UADFVDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'fake': 0, 'real': 1}
        self.classes = list(self.class_to_idx.keys())

        for class_name in ['fake', 'real']:
            frames_dir = os.path.join(self.root_dir, class_name, 'frames')
            if os.path.exists(frames_dir):
                for subdir in os.listdir(frames_dir):
                    subdir_path = os.path.join(frames_dir, subdir)
                    if os.path.isdir(subdir_path):
                        for img_file in os.listdir(subdir_path):
                            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                img_path = os.path.join(subdir_path, img_file)
                                self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class TransformSubset(Subset):
    """Subset with custom transform – FIXED loader"""
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[self.indices[idx]]
        img = Image.open(img_path).convert('RGB')   # FIXED
        if self.transform:
            img = self.transform(img)
        return img, label


# ================== UTILITY FUNCTIONS ==================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_video_level_uadfV_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
   
    all_video_ids = set()
    
    for img_path, label in dataset.samples:
       
        dir_name = os.path.basename(os.path.dirname(img_path))
        
        if '_fake' in dir_name:
            video_id = dir_name.replace('_fake', '')
        else:
            video_id = dir_name
            
        all_video_ids.add(video_id)

    all_video_ids = sorted(list(all_video_ids))
    print(f"[Split] Found {len(all_video_ids)} unique video pairs (Real+Fake).")

    train_val_ids, test_ids = train_test_split(
        all_video_ids,
        test_size=test_ratio,
        random_state=seed
    )
    
    # سپس Train را از Val جدا می‌کنیم
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size_adjusted,
        random_state=seed
    )
    
    print(f"[Split] Videos -> Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # 3. نسبت دادن تصاویر (فریم‌ها) به لیست‌ها بر اساس نام پوشه والد
    train_indices = []
    val_indices = []
    test_indices = []
    
    for idx, (img_path, label) in enumerate(dataset.samples):
        dir_name = os.path.basename(os.path.dirname(img_path))
        
        if '_fake' in dir_name:
            vid_id = dir_name.replace('_fake', '')
        else:
            vid_id = dir_name
            
        if vid_id in test_ids:
            test_indices.append(idx)
        elif vid_id in val_ids:
            val_indices.append(idx)
        elif vid_id in train_ids:
            train_indices.append(idx)
        else:
            # این بخش نباید اجرا شود مگر خطایی در لاجیک داشته باشیم
            print(f"[Warning] Video ID {vid_id} not found in splits!")
            
    print(f"[Split] Frames -> Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    return train_indices, val_indices, test_indices


def get_sample_info(dataset, index):
    if hasattr(dataset, 'samples'):
        return dataset.samples[index]
    elif hasattr(dataset, 'dataset'):
        return get_sample_info(dataset.dataset, index)
    else:
        raise AttributeError("Cannot find samples in dataset")


# ================== HELPER SPLIT FUNCTIONS ==================

def create_standard_reproducible_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    تابع اسپلیت استاندارد برای دیتاست‌هایی که ساختار ویدئویی ندارند (کلاس/تصویر.jpg).
    فقط روی شاخص‌ها کار می‌کند و کلاس‌ها را حفظ می‌کند (Stratified).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    num_samples = len(dataset)
    indices = list(range(num_samples))
    
    # دریافت لیبل‌ها برای Stratify
    labels = [dataset.samples[i][1] for i in indices]
    
    # 1. جدا کردن Train+Val از Test
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels
    )
    
    # 2. جدا کردن Train از Val
    train_val_labels = [labels[i] for i in train_val_indices]
    val_size = val_ratio / (train_ratio + val_ratio)
    
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size,
        random_state=seed,
        stratify=train_val_labels
    )
    
    print(f"[Standard Split] Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    return train_indices, val_indices, test_indices


# ================== MAIN PREPARE FUNCTION ==================

def prepare_dataset(base_dir: str, dataset_type: str, seed: int = 42):
  
    dataset_paths = {
        'real_fake': ['training_fake', 'training_real'],
        'hard_fake_real': ['fake', 'real'],
        'deepflux': ['Fake', 'Real'],
    }

    print(f"\n[Dataset Loading] Processing: {dataset_type}")
    
    # 1. لود کردن دیتاست (Dataset Loading)
    if dataset_type == 'uadfV':
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"UADFV dataset directory not found: {base_dir}")
        temp_transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = UADFVDataset(base_dir, transform=temp_transform)
        print("[Dataset Loading] UADFVDataset loaded.")
        
    elif dataset_type in dataset_paths:
        folders = dataset_paths[dataset_type]
        if all(os.path.exists(os.path.join(base_dir, f)) for f in folders):
            dataset_dir = base_dir
        else:
            alt_names = {
                'real_fake': 'real_and_fake_face',
                'hard_fake_real': 'hardfakevsrealfaces',
                'deepflux': 'DeepFLUX'
            }
            dataset_dir = os.path.join(base_dir, alt_names[dataset_type])
            if not os.path.exists(dataset_dir):
                raise FileNotFoundError(f"Could not find dataset folders in {base_dir}")
        
        temp_transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = datasets.ImageFolder(dataset_dir, transform=temp_transform)
        print(f"[Dataset Loading] ImageFolder loaded from: {dataset_dir}")
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # 2. تشخیص هوشمند ساختار (Structure Detection)
    sample_path = full_dataset.samples[0][0]
    immediate_parent = os.path.basename(os.path.dirname(sample_path))
    
    is_video_structure = immediate_parent not in full_dataset.classes

    print(f"[Structure Detection] Parent of image: '{immediate_parent}' | Classes: {full_dataset.classes}")
    
    if is_video_structure:
        print("[Structure Decision] >> Detected VIDEO/NESTED structure. Using Video-Level Split.")
        train_indices, val_indices, test_indices = create_video_level_uadfV_split(
            full_dataset, 
            train_ratio=0.7, 
            val_ratio=0.15, 
            test_ratio=0.15, 
            seed=seed
        )
    else:
        print("[Structure Decision] >> Detected FLAT IMAGE structure. Using Standard Stratified Split.")
        train_indices, val_indices, test_indices = create_standard_reproducible_split(
            full_dataset, 
            train_ratio=0.7, 
            val_ratio=0.15, 
            test_ratio=0.15, 
            seed=seed
        )
        
    return full_dataset, train_indices, val_indices, test_indices


class GradCAM:
    """Optimized GradCAM – FIXED gradient check & squeeze"""
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
            raise RuntimeError("Gradients not captured – forgot requires_grad_(True)?")

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


# ================== MAJORITY VOTING ENSEMBLE ==================

class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


class MajorityVotingEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]],
                 stds: List[Tuple[float]]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.num_models = len(models)
        
        # Freeze base models
        for model in self.models:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor, return_details: bool = False):
        # Collect votes from all models: shape (batch_size, num_models)
        votes = torch.zeros(x.size(0), self.num_models, device=x.device)

        with torch.no_grad():
            for i in range(self.num_models):
                x_n = self.normalizations(x, i)
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                # Convert logits to hard vote (0 or 1)
                pred = (out.squeeze() > 0).long()
                votes[:, i] = pred

        # Majority voting: 0 or 1
        # torch.mode returns values (voted class) and indices (counts usually)
        final_pred, _ = torch.mode(votes, dim=1)
        final_pred = final_pred.float().unsqueeze(1) # Match shape (batch, 1)

        if return_details:
            # Return final prediction, individual votes, dummy weights (uniform), dummy memberships
            # Returning dummy weights/outputs to keep signature compatible with evaluation functions
            return final_pred, votes, torch.ones_like(votes, dtype=torch.float), votes.unsqueeze(-1).float()
        return final_pred, votes


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


# ================== DATA LOADERS ==================
def create_dataloaders(base_dir: str, batch_size: int, num_workers: int = 2,
                       dataset_type: str = 'wild', is_distributed: bool = False,
                       seed: int = 42, is_main: bool = True):
    if is_main:
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
            if is_main:
                print(f"{split.capitalize():5}: {path}")
            transform = train_transform if split == 'train' else val_test_transform
            datasets_dict[split] = datasets.ImageFolder(path, transform=transform)

        if is_main:
            print(f"\nDataset Stats:")
            for split, ds in datasets_dict.items():
                print(f" {split.capitalize():5}: {len(ds):,} images | Classes: {ds.classes}")
            print(f" Class → Index: {datasets_dict['train'].class_to_idx}\n")

        train_sampler = DistributedSampler(datasets_dict['train'], shuffle=True) if is_distributed else None
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
    else:
        if is_main:
            print(f"Processing {dataset_type} dataset from: {base_dir}")
        full_dataset, train_indices, val_indices, test_indices = prepare_dataset(
            base_dir, dataset_type, seed=seed)
        if is_main:
            print(f"\nDataset Stats:")
            print(f" Total: {len(full_dataset):,} images")
            print(f" Train: {len(train_indices):,} ({len(train_indices)/len(full_dataset)*100:.1f}%)")
            print(f" Valid: {len(val_indices):,} ({len(val_indices)/len(full_dataset)*100:.1f}%)")
            print(f" Test: {len(test_indices):,} ({len(test_indices)/len(full_dataset)*100:.1f}%)\n")

        if dataset_type == 'uadfV':
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
            test_dataset = Subset(full_dataset, test_indices)
            train_dataset.dataset.transform = train_transform
            val_dataset.dataset.transform = val_test_transform
            test_dataset.dataset.transform = val_test_transform
        else:
            train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
            val_dataset = TransformSubset(full_dataset, val_indices, val_test_transform)
            test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)

        train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
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

    if is_main:
        print(f"DataLoaders ready! Batch size: {batch_size}")
        print(f" Batches → Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
        print("="*70 + "\n")
    return train_loader, val_loader, test_loader


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
    """
    Evaluation function adapted for Majority Voting.
    Analyzes individual model votes and unanimity.
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    
    # Stats: [Fake Votes, Real Votes] per model
    vote_counts = torch.zeros(len(model_names), 2, device=device)
    unanimous_samples = 0

    if is_main:
        print(f"\nEvaluating {name} set (Majority Voting)...")
        
    for images, labels in tqdm(loader, desc=f"Evaluating {name}", disable=not is_main):
        images, labels = images.to(device), labels.to(device)
        # outputs: final majority vote (0 or 1)
        # votes: individual votes (0 or 1)
        # _, _ (weights, memberships) are dummies from our forward pass
        outputs, votes, _, _ = model(images, return_details=True)
        
        pred = (outputs.squeeze(1) > 0).long()
        total_correct += pred.eq(labels.long()).sum().item()
        total_samples += labels.size(0)
        
        # Accumulate vote stats
        real_votes = votes.sum(dim=0) # Shape: (num_models)
        fake_votes = votes.size(0) - real_votes
        
        vote_counts[:, 1] += real_votes
        vote_counts[:, 0] += fake_votes
        
        # Check for unanimous decision (variance is 0)
        if (votes.std(dim=1) < 1e-6).all():
            unanimous_samples += votes.size(0)

    # Distributed Synchronization
    stats = torch.tensor([total_correct, total_samples, unanimous_samples], dtype=torch.long, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    dist.all_reduce(vote_counts, op=dist.ReduceOp.SUM)

    if is_main:
        total_correct = stats[0].item()
        total_samples = stats[1].item()
        unanimous_samples = stats[2].item()
        acc = 100. * total_correct / total_samples
        vote_counts_np = vote_counts.cpu().numpy()

        print(f"\n{'='*70}")
        print(f"{name.upper()} SET RESULTS (Majority Voting)")
        print(f"{'='*70}")
        print(f" → Accuracy: {acc:.3f}%")
        print(f" → Total Samples: {total_samples:,}")
        print(f" → Unanimous Decisions: {unanimous_samples:,} ({100*unanimous_samples/total_samples:.2f}%)")
        
        print(f"\nModel Voting Distribution (Fake / Real):")
        for i, name in enumerate(model_names):
            print(f"  {i+1:2d}. {name:<25}: {int(vote_counts_np[i, 0]):<6} / {int(vote_counts_np[i, 1]):<6}")
        print(f"{'='*70}")
        
        return acc, vote_counts_np.tolist()
    return 0.0, []


def train_majority_voting(ensemble_model, train_loader, val_loader, num_epochs, lr,
                        device, save_dir, is_main, model_names):
    """
    Placeholder for training loop. 
    Since Majority Voting has no trainable parameters in aggregation and models are frozen,
    we simply evaluate and save a dummy checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if is_main:
        print("="*70)
        print("Majority Voting Setup")
        print("="*70)
        print("No trainable parameters in aggregation (Fixed Logic).")
        print("Base models are frozen.")
        print("Running validation to ensure setup is correct...\n")

    # Run validation
    val_acc = evaluate_accuracy_ddp(ensemble_model, val_loader, device)
    
    # Save dummy checkpoint to maintain compatibility with pipeline
    if is_main:
        save_path = os.path.join(save_dir, 'best_majority_voting.pt')
        torch.save({
            'ensemble_state_dict': ensemble_model.state_dict(),
            'val_acc': val_acc
        }, save_path)
        print(f"\nModel saved (Majority Voting) → {save_dir}")
        print(f"Validation Accuracy: {val_acc:.2f}%\n")
    
    return val_acc, {}

# ================== LIME EXPLANATION ==================
def generate_lime_explanation(model, image_tensor, device, target_size=(256, 256)):
    img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        batch = torch.from_numpy(images.transpose(0, 3, 1, 2)).float() / 255.0
        batch = batch.to(device)
        with torch.no_grad():
            # Majority voting returns hard votes (0 or 1)
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

        with torch.no_grad():
            outputs, votes, _, _ = ensemble(image, return_details=True)
        pred = outputs.squeeze().long().item()
        
        # Find models that agreed with the majority vote
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
                # Average GradCAM of models that agreed with the final vote
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

                        # Equal weight for all agreeing models
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

                # Combined
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
    parser.add_argument('--epochs', type=int, default=30, help="Ignored for Majority Voting, kept for compatibility")
    parser.add_argument('--lr', type=float, default=0.0001, help="Ignored for Majority Voting")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_memberships', type=int, default=3, help="Ignored for Majority Voting")
    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['wild', 'real_fake', 'hard_fake_real', 'deepflux', 'uadfV'])
    parser.add_argument('--cum_weight_threshold', type=float, default=0.9, help="Ignored")
    parser.add_argument('--hesitancy_threshold', type=float, default=0.2, help="Ignored")
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
        print(f"MAJORITY VOTING ENSEMBLE")
        print(f"Distributed on {world_size} GPU(s) | Seed: {args.seed}")
        print("="*70)
        print(f"Dataset: {args.dataset}")
        print(f"Data directory: {args.data_dir}")
        print(f"Batch size: {args.batch_size}")
        print(f"Models: {len(args.model_paths)}")
        print("="*70 + "\n")

    MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    MEANS = MEANS[:len(args.model_paths)]
    STDS = STDS[:len(args.model_paths)]

    base_models = load_pruned_models(args.model_paths, device, is_main)
    MODEL_NAMES = args.model_names[:len(base_models)]

    # Using MajorityVotingEnsemble instead of FuzzyHesitantEnsemble
    ensemble = MajorityVotingEnsemble(
        base_models, MEANS, STDS
    ).to(device)

    if world_size > 1:
        ensemble = DDP(ensemble, device_ids=[local_rank], output_device=local_rank)

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

    # "Training" phase - just validation for Majority Voting
    val_acc, _ = train_majority_voting(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_dir, is_main, MODEL_NAMES)

    if is_main:
        print("\n" + "="*70)
        print("FINAL ENSEMBLE EVALUATION")
        print("="*70)

    ensemble_module = ensemble.module if hasattr(ensemble, 'module') else ensemble
    ensemble_test_acc, vote_stats = evaluate_ensemble_final_ddp(
        ensemble_module, test_loader, device, "Test", MODEL_NAMES, is_main)

    if is_main:
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        print(f"Best Single Model: {best_single:.2f}%")
        print(f"Ensemble Accuracy: {ensemble_test_acc:.2f}%")
        print(f"Improvement: {ensemble_test_acc - best_single:+.2f}%")
        print("="*70)

        final_results = {
            'best_single_model': {
                'name': MODEL_NAMES[best_idx],
                'accuracy': float(best_single)
            },
            'ensemble': {
                'type': 'MajorityVoting',
                'test_accuracy': float(ensemble_test_acc),
                'vote_distribution': {name: {'fake': int(v[0]), 'real': int(v[1])} for name, v in zip(MODEL_NAMES, vote_stats)}
            },
            'improvement': float(ensemble_test_acc - best_single)
        }

        results_path = os.path.join(args.save_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"\nResults saved: {results_path}")

        final_model_path = os.path.join(args.save_dir, 'final_ensemble_model.pt')
        torch.save({
            'ensemble_state_dict': ensemble_module.state_dict(),
            'test_accuracy': ensemble_test_acc,
            'model_names': MODEL_NAMES,
            'means': MEANS,
            'stds': STDS
        }, final_model_path)
        print(f"Model saved: {final_model_path}")

        vis_dir = os.path.join(args.save_dir, 'visualizations')
        generate_visualizations(
            ensemble_module, test_loader, device, vis_dir, MODEL_NAMES,
            args.num_grad_cam_samples, args.num_lime_samples,
            args.dataset, is_main)

    cleanup_distributed()

if __name__ == "__main__":
    main()
