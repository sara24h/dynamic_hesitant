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
from typing import List, Tuple
import warnings
import argparse
import shutil
import json
import random
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

warnings.filterwarnings("ignore")

# ====================== GRAD-CAM ======================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        output = self.model(input_image)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if target_class is None:
            target_class = (output.squeeze() > 0).long().item()
        self.model.zero_grad()
        output.squeeze().backward()
        gradients = self.gradients
        activations = self.activations
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

def get_target_layer(model):
    if hasattr(model, 'layer4'):
        return model.layer4[-1].conv2
    elif hasattr(model, 'features'):
        for layer in reversed(list(model.features.children())):
            if isinstance(layer, nn.Conv2d):
                return layer
    else:
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if conv_layers:
            return conv_layers[-1]
    raise ValueError("Could not find suitable target layer for Grad-CAM")

def visualize_gradcam(image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    if torch.is_tensor(image):
        image = image.cpu().numpy()
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image = (image * 255).astype(np.uint8)
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    visualization = (heatmap * alpha + image * (1 - alpha)).astype(np.uint8)
    return visualization, heatmap

def save_gradcam_comparison(images, labels, model_names, base_models, normalizations, 
                           save_dir, device, num_samples=5, rank=0):
    if rank != 0:
        return
    os.makedirs(save_dir, exist_ok=True)
    gradcams = []
    for model in base_models:
        try:
            target_layer = get_target_layer(model)
            gradcam = GradCAM(model, target_layer)
            gradcams.append(gradcam)
        except Exception as e:
            print(f"Warning: Could not create Grad-CAM: {e}")
            gradcams.append(None)
    
    for idx in range(min(num_samples, len(images))):
        image = images[idx:idx+1].to(device)
        label = labels[idx].item()
        fig, axes = plt.subplots(2, len(base_models) + 1, figsize=(4 * (len(base_models) + 1), 8))
        orig_img = images[idx].cpu().numpy().transpose(1, 2, 0)
        orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
        axes[0, 0].imshow(orig_img)
        axes[0, 0].set_title(f'Original\nLabel: {"Real" if label == 1 else "Fake"}', fontsize=10)
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        for i, (model, gradcam, name) in enumerate(zip(base_models, gradcams, model_names)):
            if gradcam is None:
                continue
            try:
                normalized_img = normalizations(image, i)
                cam = gradcam.generate_cam(normalized_img)
                with torch.no_grad():
                    output = model(normalized_img)
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    pred = (output.squeeze() > 0).long().item()
                    conf = torch.sigmoid(output).item()
                overlay, heatmap = visualize_gradcam(images[idx], cam)
                axes[0, i+1].imshow(overlay)
                axes[0, i+1].set_title(f'{name}\nPred: {"Real" if pred == 1 else "Fake"} ({conf:.2f})', fontsize=9)
                axes[0, i+1].axis('off')
                axes[1, i+1].imshow(heatmap)
                axes[1, i+1].set_title('Heatmap', fontsize=9)
                axes[1, i+1].axis('off')
            except Exception as e:
                print(f"Error generating Grad-CAM for {name}: {e}")
                axes[0, i+1].axis('off')
                axes[1, i+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'gradcam_sample_{idx}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    print(f"\nGrad-CAM visualizations saved to: {save_dir}")

# ====================== DATASET ======================
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
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    num_samples = len(dataset)
    indices = list(range(num_samples))
    labels = [dataset.samples[i][1] for i in indices]
    train_val_indices, test_indices = train_test_split(indices, test_size=test_ratio, random_state=seed, stratify=labels)
    train_val_labels = [labels[i] for i in train_val_indices]
    val_size = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=val_size, random_state=seed, stratify=train_val_labels)
    return train_indices, val_indices, test_indices

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

# ====================== MODELS ======================
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
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
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
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]], stds: List[Tuple[float]], 
                 num_memberships: int = 3, freeze_models: bool = True, cum_weight_threshold: float = 0.9, 
                 hesitancy_threshold: float = 0.2):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.hesitant_fuzzy = HesitantFuzzyMembership(128, self.num_models, num_memberships)
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
    from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    models = []
    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device).eval()
        models.append(model)
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

def create_dataloaders_ddp(base_dir: str, batch_size: int, rank: int, world_size: int, num_workers: int = 2, dataset_type: str = 'uadfV'):
    train_transform = transforms.Compose([
        transforms.Resize(256), transforms.RandomCrop(256), transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10), transforms.ColorJitter(0.2, 0.2), transforms.ToTensor(),
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(),
    ])
    
    if dataset_type == 'uadfV':
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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
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
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        ensemble_model.train()
        train_loader.sampler.set_epoch(epoch)
        train_loss = train_correct = train_total = 0.0
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
        
        train_loss_tensor = torch.tensor(train_loss).to(device)
        train_correct_tensor = torch.tensor(train_correct).to(device)
        train_total_tensor = torch.tensor(train_total).to(device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_total_tensor, op=dist.ReduceOp.SUM)
        train_acc = 100. * train_correct_tensor.item() / train_total_tensor.item()
        train_loss = train_loss_tensor.item() / train_total_tensor.item()
        
        val_acc = evaluate_accuracy_ddp(ensemble_model, val_loader, device, rank)
        scheduler.step()
        
        if rank == 0:
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            print(f"\nEpoch {epoch+1}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({'epoch': epoch + 1, 'hesitant_state_dict': hesitant_net.state_dict(), 'val_acc': val_acc, 'history': history}, 
                          os.path.join(save_dir, 'best_hesitant_fuzzy.pt'))
        dist.barrier()
    
    return best_val_acc, history

@torch.no_grad()
def evaluate_ensemble_final_ddp(model, loader, device, name, model_names, rank):
    model.eval()
    all_preds = []
    all_labels = []
    iterator = tqdm(loader, desc=f"Evaluating {name}") if rank == 0 else loader
    for images, labels in iterator:
        images = images.to(device)
        labels = labels.to(device)
        outputs, weights, memberships, _ = model(images, return_details=True)
        pred = (outputs.squeeze(1) > 0).long()
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"{name.upper()} Accuracy: {acc:.3f}%")
        print(f"{'='*70}")
    return acc, [], [], []

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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_memberships', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='uadfV')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--gradcam_dir', type=str, default='./gradcam')
    parser.add_argument('--num_gradcam_samples', type=int, default=10)
    args = parser.parse_args()
    
    MEANS = [(0.5207, 0.4258, 0.3806), (0.4868, 0.3972, 0.3624), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2296, 0.2066, 0.2009), (0.2410, 0.2161, 0.2081)]
    MEANS = MEANS[:len(args.model_paths)]
    STDS = STDS[:len(args.model_paths)]
    
    base_models = load_pruned_models(args.model_paths, device, rank)
    MODEL_NAMES = args.model_names[:len(base_models)]
    
    ensemble = FuzzyHesitantEnsemble(base_models, MEANS, STDS, num_memberships=args.num_memberships, freeze_models=True, cum_weight_threshold=0.9, hesitancy_threshold=0.2).to(device)
    ensemble = DDP(ensemble, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    train_loader, val_loader, test_loader = create_dataloaders_ddp(args.data_dir, args.batch_size, rank, world_size, dataset_type=args.dataset)
    
    # GRAD-CAM Before Training
    if is_main:
        print("\n" + "="*70)
        print("GENERATING GRAD-CAM (Before Training)")
        print("="*70)
        test_iter = iter(test_loader)
        sample_images, sample_labels = next(test_iter)
        save_gradcam_comparison(sample_images, sample_labels, MODEL_NAMES, base_models, ensemble.module.normalizations, 
                               os.path.join(args.gradcam_dir, 'before_training'), device, min(args.num_gradcam_samples, len(sample_images)), rank)
    dist.barrier()
    
    # Evaluate Individual Models
    if is_main:
        print("\n" + "="*70)
        print("EVALUATING INDIVIDUAL MODELS")
        print("="*70)
        individual_accs = []
        for i, model in enumerate(base_models):
            acc = evaluate_single_model(model, test_loader, device, f"{MODEL_NAMES[i]}", rank)
            individual_accs.append(acc)
        best_single = max(individual_accs)
        best_idx = individual_accs.index(best_single)
        print(f"\nBest Single Model: {MODEL_NAMES[best_idx]} → {best_single:.2f}%")
    dist.barrier()
    
    # Train Ensemble
    best_val_acc, history = train_hesitant_fuzzy_ddp(ensemble, train_loader, val_loader, args.epochs, args.lr, device, args.save_dir, rank, world_size)
    
    # Load Best Model
    ckpt_path = os.path.join(args.save_dir, 'best_hesitant_fuzzy.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ensemble.module.hesitant_fuzzy.load_state_dict(ckpt['hesitant_state_dict'])
    dist.barrier()
    
    # Final Evaluation
    if is_main:
        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)
        ensemble_test_acc, _, _, _ = evaluate_ensemble_final_ddp(ensemble, test_loader, device, "Test", MODEL_NAMES, rank)
        
        # GRAD-CAM After Training
        print("\n" + "="*70)
        print("GENERATING GRAD-CAM (After Training)")
        print("="*70)
        test_iter = iter(test_loader)
        sample_images, sample_labels = next(test_iter)
        save_gradcam_comparison(sample_images, sample_labels, MODEL_NAMES, base_models, ensemble.module.normalizations,
                               os.path.join(args.gradcam_dir, 'after_training'), device, min(args.num_gradcam_samples, len(sample_images)), rank)
        
        # Final Results
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        print(f"Best Single Model Acc: {best_single:.2f}%")
        print(f"Hesitant Ensemble Acc: {ensemble_test_acc:.2f}%")
        improvement = ensemble_test_acc - best_single
        print(f"Improvement: {improvement:+.2f}%")
        print("="*70)
        
        results = {
            "method": "Fuzzy Hesitant Sets with Grad-CAM",
            "dataset": args.dataset,
            "seed": SEED,
            "num_gpus": world_size,
            "model_names": MODEL_NAMES,
            "individual_accuracies": {MODEL_NAMES[i]: acc for i, acc in enumerate(individual_accs)},
            "best_single": {"name": MODEL_NAMES[best_idx], "acc": best_single},
            "ensemble": {"acc": ensemble_test_acc},
            "improvement": improvement,
            "training_history": history,
            "gradcam_dir": args.gradcam_dir
        }
        
        with open(os.path.join(args.save_dir, 'results_with_gradcam.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {os.path.join(args.save_dir, 'results_with_gradcam.json')}")
        print("All done!")
    
    cleanup_ddp()

if __name__ == "__main__":
    main()
