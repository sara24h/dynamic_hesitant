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
import json
import random
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

warnings.filterwarnings("ignore")

# LIME library imports
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
    """Subset with custom transform"""
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[self.indices[idx]]
        img = Image.open(img_path).convert('RGB')
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


def create_reproducible_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    num_samples = len(dataset)
    indices = list(range(num_samples))
    labels = [dataset.samples[i][1] for i in indices]

    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_ratio, random_state=seed, stratify=labels)
    train_val_labels = [labels[i] for i in train_val_indices]
    val_size = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size, random_state=seed, stratify=train_val_labels)
    return train_indices, val_indices, test_indices


def get_sample_info(dataset, index):
    if hasattr(dataset, 'samples'):
        return dataset.samples[index]
    elif hasattr(dataset, 'dataset'):
        return get_sample_info(dataset.dataset, index)
    else:
        raise AttributeError("Cannot find samples in dataset")


def prepare_dataset(base_dir: str, dataset_type: str, seed: int = 42):
    dataset_paths = {
        'real_fake': ['training_fake', 'training_real'],
        'hard_fake_real': ['fake', 'real'],
        'deepflux': ['Fake', 'Real'],
    }

    if dataset_type == 'uadfV':
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"UADFV dataset directory not found: {base_dir}")
        temp_transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = UADFVDataset(base_dir, transform=temp_transform)
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
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    train_indices, val_indices, test_indices = create_reproducible_split(
        full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=seed)
    return full_dataset, train_indices, val_indices, test_indices


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

# ================== WEIGHTED ENSEMBLE CLASS ==================
class WeightedEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]],
                 stds: List[Tuple[float]], freeze_models: bool = True):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)

        # Initialize learnable weights
        self.raw_weights = nn.Parameter(torch.zeros(self.num_models))

        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
        
        # مهم: مطمئن میشویم که خود وزن‌ها قابل آموزش هستند
        self.raw_weights.requires_grad = True

    def forward(self, x: torch.Tensor, return_details: bool = False):
        outputs = []
        for i in range(self.num_models):
            x_n = self.normalizations(x, i)
            with torch.no_grad(): 
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            outputs.append(out)

        stacked_outputs = torch.cat(outputs, dim=1)
        weights = F.softmax(self.raw_weights, dim=0)
        final_output = (stacked_outputs * weights.unsqueeze(0)).sum(dim=1, keepdim=True)

        if return_details:
            batch_size = x.size(0)
            weights_matrix = weights.unsqueeze(0).expand(batch_size, -1)
            dummy_memberships = torch.zeros(batch_size, self.num_models, 3, device=x.device)
            return final_output, weights_matrix, dummy_memberships, stacked_outputs
        
        # اصلاح ارور UnboundLocalError: فقط در این حالت نرمال خروجی می‌دهیم
        return final_output, weights.unsqueeze(0).expand(x.size(0), -1)

# ================== MODEL LOADING ==================
def load_pruned_models(model_paths: List[str], device: torch.device, is_main: bool) -> List[nn.Module]:
    try:
        # تلاش برای ایمپورت مدل اصلی شما
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
        use_placeholder = False
    except ImportError:
        if is_main:
            print("[WARNING] model.ResNet_pruned not found. Using torchvision ResNet50 as placeholder.")
        use_placeholder = True
    
    if use_placeholder:
        try:
            from torchvision import models
            class MockPrunedResNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                    self.layer4 = self.resnet.layer4 
                
                def forward(self, x):
                    return self.resnet(x)
            
            models_list = []
            for _ in model_paths:
                m = MockPrunedResNet().to(device).eval()
                for p in m.parameters(): p.requires_grad = False
                models_list.append(m)
            return models_list
        except Exception as e:
            raise ImportError(f"Cannot import torchvision: {e}")

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
    model.eval()
    total_correct = 0
    total_samples = 0
    sum_weights = torch.zeros(len(model_names), device=device)
    sum_activation = torch.zeros(len(model_names), device=device)
    sum_membership_vals = torch.zeros(len(model_names), 3, device=device)
    sum_hesitancy = torch.zeros(len(model_names), device=device)

    if is_main:
        print(f"\nEvaluating {name} set...")
    for images, labels in tqdm(loader, desc=f"Evaluating {name}", disable=not is_main):
        images, labels = images.to(device), labels.to(device)
        outputs, weights, memberships, _ = model(images, return_details=True)
        pred = (outputs.squeeze(1) > 0).long()
        total_correct += pred.eq(labels.long()).sum().item()
        total_samples += labels.size(0)
        sum_weights += weights.sum(dim=0)
        sum_activation += (weights > 1e-4).sum(dim=0).float()
        sum_membership_vals += memberships.sum(dim=0)
        sum_hesitancy += memberships.var(dim=2).sum(dim=0)

    stats = torch.tensor([total_correct, total_samples], dtype=torch.long, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    dist.all_reduce(sum_weights, op=dist.ReduceOp.SUM)
    dist.all_reduce(sum_activation, op=dist.ReduceOp.SUM)
    dist.all_reduce(sum_membership_vals, op=dist.ReduceOp.SUM)
    dist.all_reduce(sum_hesitancy, op=dist.ReduceOp.SUM)

    if is_main:
        total_correct = stats[0].item()
        total_samples = stats[1].item()
        acc = 100. * total_correct / total_samples
        avg_weights = (sum_weights / total_samples).cpu().numpy()
        activation_percentages = (sum_activation / total_samples * 100).cpu().numpy()
        avg_membership = (sum_membership_vals / total_samples).cpu().numpy()
        avg_hesitancy = (sum_hesitancy / total_samples).cpu().numpy()

        print(f"\n{'='*70}")
        print(f"{name.upper()} SET RESULTS")
        print(f"{'='*70}")
        print(f" → Accuracy: {acc:.3f}%")
        print(f" → Total Samples: {total_samples:,}")
        print(f"\nAverage Model Weights (Learned/Fixed for Weighted Ensemble):")
        for i, (w, mname) in enumerate(zip(avg_weights, model_names)):
            print(f" {i+1:2d}. {mname:<25}: {w:6.4f} ({w*100:5.2f}%)")
        print(f"\nActivation Frequency:")
        for i, (perc, mname) in enumerate(zip(activation_percentages, model_names)):
            print(f" {i+1:2d}. {mname:<25}: {perc:6.2f}% active")
        print(f"{'='*70}")
        return acc, avg_weights.tolist(), activation_percentages.tolist()
    return 0.0, [0.0]*len(model_names), [0.0]*len(model_names)


# ================== TRAINING FUNCTION ==================
def train_ensemble_weights(model, train_loader, val_loader, device, epochs, lr, is_main=True):
    """
    فقط وزن‌های ensemble را آموزش می‌دهد (مدل‌های پایه فریز هستند).
    """
    if is_main:
        print("="*70)
        print("TRAINING ENSEMBLE WEIGHTS (Meta-Learning)")
        print("="*70)
        print(f"Epochs: {epochs} | Learning Rate: {lr}")
    
    # اگر مدل DDP است، باید از module استفاده کنیم
    if isinstance(model, DDP):
        raw_weights_param = model.module.raw_weights
    else:
        raw_weights_param = model.raw_weights

    # فقط پارامترهای raw_weights را بهOptimizer می‌دهیم
    optimizer = torch.optim.Adam([raw_weights_param], lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # اصلاح ارور: حذف پارامتر verbose برای سازگاری با PyTorch جدید
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        
        # مطمئن میشویم فقط وزن‌ها گرادیان می‌گیرند
        raw_weights_param.requires_grad = True
        for p in model.parameters():
            if p is not raw_weights_param:
                p.requires_grad = False

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Set sampler epoch for distributed training consistency
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # Training Loop
        if is_main:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        else:
            pbar = train_loader

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(images)
            
            # Calculate Loss
            loss = criterion(outputs, labels)
            
            # Backward pass & Optimize (فقط روی وزن‌ها اعمال می‌شود)
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            total_correct += (preds == labels.long()).sum().item()
            total_samples += images.size(0)

            if is_main:
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*total_correct/total_samples:.2f}%'})

        # Validation
        val_acc = evaluate_accuracy_ddp(model, val_loader, device)
        
        # Update Scheduler
        scheduler.step(val_acc)

        if is_main:
            avg_train_loss = total_loss / len(train_loader.dataset)
            avg_train_acc = 100. * total_correct / total_samples
            print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc

    if is_main:
        print("="*70)
        print("Weight Training Completed.")
        print(f"Best Validation Accuracy during training: {best_val_acc:.2f}%")
        # نمایش وزن‌های نهایی آموخته شده
        final_weights = F.softmax(raw_weights_param, dim=0).detach().cpu().numpy()
        print(f"Final Learned Weights: {final_weights}")
        print("="*70)

    return model


# ================== LIME EXPLANATION ==================
def generate_lime_explanation(model, image_tensor, device, target_size=(256, 256)):
    img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        batch = torch.from_numpy(images.transpose(0, 3, 1, 2)).float() / 255.0
        batch = batch.to(device)
        with torch.no_grad():
            out = model(batch)
            if isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out
            
            probs = torch.sigmoid(logits).cpu().numpy()
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
    print("GENERATING VISUALIZATIONS")
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
            print(f"Warning: Could not get info for sample {idx}")
            continue

        print(f"\n[Visualization {idx+1}/{len(vis_loader)}]")
        print(f"  Path: {os.path.basename(img_path)}")
        print(f"  True: {'real' if true_label == 1 else 'fake'}")

        with torch.no_grad():
            output, weights, _, _ = ensemble(image, return_details=True)
        pred = 1 if output.squeeze().item() > 0 else 0
        print(f"  Pred: {'real' if pred == 1 else 'fake'}")

        filename = f"sample_{idx}_true{'real' if true_label == 1 else 'fake'}_pred{'real' if pred == 1 else 'fake'}.png"

        # ---------- GradCAM ----------
        if idx < num_gradcam:
            try:
                active_models = list(range(len(local_ensemble.models))) 
                combined_cam = None
                for i in active_models:
                    model = local_ensemble.models[i]
                    try:
                        target_layer = model.layer4[2].conv3
                    except AttributeError:
                        target_layer = model.resnet.layer4[2].conv3

                    gradcam = GradCAM(model, target_layer)

                    x_n = local_ensemble.normalizations(image, i)
                    x_n.requires_grad_(True)
                    model_out = model(x_n)
                    if isinstance(model_out, (tuple, list)):
                        model_out = model_out[0]
                    score = model_out.squeeze(0) if pred == 1 else -model_out.squeeze(0)
                    cam = gradcam.generate(score)

                    weight = weights[0, i].item() 
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
                    plt.title(f"GradCAM\nTrue: {'real' if true_label == 1 else 'fake'} | Pred: {'real' if pred == 1 else 'fake'}")
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
    parser = argparse.ArgumentParser(description="Weighted Average Ensemble Evaluation with Training")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['wild', 'real_fake', 'hard_fake_real', 'deepflux', 'uadfV'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./output_weighted')
    parser.add_argument('--seed', type=int, default=42)
    
    # Weight Training Arguments
    parser.add_argument('--train_weights_epochs', type=int, default=10, help="Epochs to train ensemble weights")
    parser.add_argument('--train_weights_lr', type=float, default=0.01, help="Learning rate for weight optimization")
    
    args = parser.parse_args()

    if len(args.model_names) != len(args.model_paths):
        raise ValueError("Number of model_names must match model_paths")

    set_seed(args.seed)
    device, local_rank, rank, world_size = setup_distributed()
    is_main = rank == 0

    if is_main:
        print("="*70)
        print(f"WEIGHTED AVERAGE ENSEMBLE EVALUATION")
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

    ensemble = WeightedEnsemble(
        base_models, MEANS, STDS,
        freeze_models=True
    ).to(device)

    if is_main:
        trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
        total = sum(p.numel() for p in ensemble.parameters())
        print(f"Total params: {total:,} | Trainable (weights): {trainable:,} | Frozen: {total-trainable:,}\n")

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=(world_size > 1), seed=args.seed, is_main=is_main)

    # ============================================================
    #   WEIGHT TRAINING SECTION
    # ============================================================
    
    # تبدیل به DDP اگر توزیع شده است
    if world_size > 1:
        ensemble = DDP(ensemble, device_ids=[local_rank], find_unused_parameters=True)
    
    # شروع آموزش وزن‌ها
    ensemble = train_ensemble_weights(
        ensemble, train_loader, val_loader, device, 
        epochs=args.train_weights_epochs, 
        lr=args.train_weights_lr, 
        is_main=is_main
    )
    
    # ============================================================
    #   END WEIGHT TRAINING
    # ============================================================

    if is_main:
        print("\n" + "="*70)
        print("INDIVIDUAL MODEL PERFORMANCE (For Reference)")
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
        print("\nProceeding to final evaluation with learned weights...\n")

    # Evaluate
    ensemble_module = ensemble.module if hasattr(ensemble, 'module') else ensemble
    
    if is_main:
        print("\n" + "="*70)
        print("FINAL ENSEMBLE EVALUATION")
        print("="*70)

    ensemble_test_acc, ensemble_weights, activation_percentages = evaluate_ensemble_final_ddp(
        ensemble_module, test_loader, device, "Test", MODEL_NAMES, is_main)

    if is_main:
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        print(f"Best Single Model: {best_single:.2f}%")
        print(f"Weighted Ensemble Accuracy: {ensemble_test_acc:.2f}%")
        print(f"Improvement: {ensemble_test_acc - best_single:+.2f}%")
        print("="*70)

        final_results = {
            'method': 'Weighted Average Ensemble (Learned Weights)',
            'best_single_model': {
                'name': MODEL_NAMES[best_idx],
                'accuracy': float(best_single)
            },
            'ensemble': {
                'test_accuracy': float(ensemble_test_acc),
                'model_weights': {name: float(w) for name, w in zip(MODEL_NAMES, ensemble_weights)},
                'activation_percentages': {name: float(p) for name, p in zip(MODEL_NAMES, activation_percentages)}
            },
            'improvement': float(ensemble_test_acc - best_single)
        }

        os.makedirs(args.save_dir, exist_ok=True)
        results_path = os.path.join(args.save_dir, 'final_results_weighted_ensemble.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"\nResults saved: {results_path}")

        final_model_path = os.path.join(args.save_dir, 'weighted_ensemble_model_info.json')
        with open(final_model_path, 'w') as f:
            json.dump({
                'model_names': MODEL_NAMES,
                'model_paths': args.model_paths,
                'means': [list(m) for m in MEANS],
                'stds': [list(s) for s in STDS],
                'normalized_weights': {name: float(w) for name, w in zip(MODEL_NAMES, ensemble_weights)}
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
