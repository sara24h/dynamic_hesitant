import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, datasets, transforms as T
import torch.distributed as dist
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
import warnings
import argparse
import json
import random
from sklearn.model_selection import train_test_split
from PIL import Image

warnings.filterwarnings("ignore")

# ================== DATASET CLASSES & UTILITIES ==================
class CustomGenAIDataset(Dataset):
    def __init__(self, root_dir, fake_classes, real_class, transform=None):
        self.root_dir, self.transform, self.samples = root_dir, transform, []
        self.label_map = {'fake': 0, 'real': 1}
        for class_name in fake_classes:
            class_path = os.path.join(root_dir, class_name)
            if os.path.exists(class_path):
                for f in os.listdir(class_path):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_path, f), self.label_map['fake']))
        real_path = os.path.join(root_dir, real_class)
        if os.path.exists(real_path):
            for f in os.listdir(real_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(real_path, f), self.label_map['real']))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label

class NewGenAIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir, self.transform, self.samples = root_dir, transform, []
        self.label_map = {'fake': 0, 'real': 1}
        for dirpath, _, filenames in os.walk(self.root_dir):
            if os.path.basename(dirpath) in self.label_map:
                for f in filenames:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(dirpath, f), self.label_map[os.path.basename(dirpath)]))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label

class UADFVDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir, self.transform, self.samples = root_dir, transform, []
        self.class_to_idx = {'fake': 0, 'real': 1}
        for class_name in ['fake', 'real']:
            frames_dir = os.path.join(self.root_dir, class_name, 'frames')
            if os.path.exists(frames_dir):
                for subdir in os.listdir(frames_dir):
                    subdir_path = os.path.join(frames_dir, subdir)
                    if os.path.isdir(subdir_path):
                        for f in os.listdir(subdir_path):
                            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.samples.append((os.path.join(subdir_path, f), self.class_to_idx[class_name]))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label

class TransformSubset(Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform
    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform: img = self.transform(img)
        return img, label

def get_sample_info(dataset, index):
    if hasattr(dataset, 'samples'): return dataset.samples[index]
    elif hasattr(dataset, 'dataset'): return get_sample_info(dataset.dataset, index)
    return "dummy_path", 0

def create_standard_reproducible_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    indices = list(range(len(dataset)))
    labels = [dataset.samples[i][1] for i in indices]
    train_val_idx, test_idx = train_test_split(indices, test_size=test_ratio, random_state=seed, stratify=labels)
    val_adj = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_adj, random_state=seed, stratify=[labels[i] for i in train_val_idx])
    return train_idx, val_idx, test_idx

def create_video_level_uadfV_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    vids = set()
    for p, _ in dataset.samples:
        d = os.path.basename(os.path.dirname(p))
        vids.add(d.replace('_fake', '') if '_fake' in d else d)
    vids = sorted(list(vids))
    tv_ids, te_ids = train_test_split(vids, test_size=test_ratio, random_state=seed)
    va_adj = val_ratio / (train_ratio + val_ratio)
    tr_ids, va_ids = train_test_split(tv_ids, test_size=va_adj, random_state=seed)
    
    tr, va, te = [], [], []
    for idx, (p, _) in enumerate(dataset.samples):
        d = os.path.basename(os.path.dirname(p))
        vid = d.replace('_fake', '') if '_fake' in d else d
        if vid in te_ids: te.append(idx)
        elif vid in va_ids: va.append(idx)
        elif vid in tr_ids: tr.append(idx)
    return tr, va, te

def prepare_dataset(base_dir: str, dataset_type: str, seed: int = 42):
    paths_map = {'real_fake': ['training_fake', 'training_real'], 'hard_fake_real': ['fake', 'real'], 'deepflux': ['Fake', 'Real'], 'real_fake_dataset': ['face_fake', 'face_real'], 'deepfake_lab': ['training_fake', 'training_real']}
    if dataset_type == 'custom_genai':
        ds = CustomGenAIDataset(base_dir, ['DALL-E', 'DeepFaceLab', 'Midjourney', 'StyleGAN'], 'Real')
    elif dataset_type == 'custom_genai_v2':
        ds = NewGenAIDataset(base_dir)
    elif dataset_type == 'uadfV':
        ds = UADFVDataset(base_dir)
        return ds, *create_video_level_uadfV_split(ds, seed)
    elif dataset_type in paths_map:
        folders = paths_map[dataset_type]
        d_dir = base_dir if all(os.path.exists(os.path.join(base_dir, f)) for f in folders) else os.path.join(base_dir, dataset_type)
        ds = datasets.ImageFolder(d_dir)
    else: raise ValueError(f"Unknown: {dataset_type}")
    return ds, *create_standard_reproducible_split(ds, seed)

def worker_init_fn(worker_id):
    w_seed = torch.initial_seed() % 2**32
    np.random.seed(w_seed); random.seed(w_seed)

def create_dataloaders(base_dir, batch_size, num_workers=0, dataset_type='wild', is_distributed=False, seed=42, is_main=True):
    t_train = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(256), transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(10), transforms.ColorJitter(0.2, 0.2), transforms.ToTensor()])
    t_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])
    
    if dataset_type == 'wild':
        d_dict = {s: datasets.ImageFolder(os.path.join(base_dir, s), transform=t_train if s=='train' else t_test) for s in ['train','valid','test']}
    else:
        full, tr_i, va_i, te_i = prepare_dataset(base_dir, dataset_type, seed)
        d_dict = {'train': TransformSubset(full, tr_i, t_train), 'valid': TransformSubset(full, va_i, t_test), 'test': TransformSubset(full, te_i, t_test)}
        
    s_dict = {s: (DistributedSampler(d_dict[s], shuffle=(s=='train')) if is_distributed else None) for s in ['train','valid','test']}
    loaders = {s: DataLoader(d_dict[s], batch_size=batch_size, shuffle=(s_dict[s] is None and s=='train'), sampler=s_dict[s], num_workers=num_workers, pin_memory=True, drop_last=(s=='train'), worker_init_fn=worker_init_fn) for s in ['train','valid','test']}
    return loaders['train'], loaders['valid'], loaders['test']

def create_adabn_dataloader(base_dir, batch_size, num_workers=0, dataset_type='wild', seed=42, is_main=True, is_distributed=False):
    t_adabn = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])
    if dataset_type == 'wild':
        ds = datasets.ImageFolder(os.path.join(base_dir, 'train'), transform=t_adabn)
    else:
        full, tr_i, _, _ = prepare_dataset(base_dir, dataset_type, seed)
        ds = TransformSubset(full, tr_i, t_adabn)
    sampler = DistributedSampler(ds, shuffle=False) if is_distributed else None
    return DataLoader(ds, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=False)

# ================== MODEL & EVALUATION UTILITIES ==================
try:
    from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
except ImportError:
    print("Warning: 'model.ResNet_pruned' not found.")
    ResNet_50_pruned_hardfakevsreal = None

class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))
    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')

class MajorityVotingEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]], stds: List[Tuple[float]], freeze_models: bool = True):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters(): p.requires_grad = False

    def forward(self, x: torch.Tensor, return_details: bool = False):
        votes_logits = []
        for i in range(self.num_models):
            x_n = x if torch.all(self.normalizations.__getattr__(f'std_{i}') == 0) else self.normalizations(x, i)
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)): out = out[0]
            votes_logits.append(out)

        stacked_logits = torch.cat(votes_logits, dim=1)
        hard_votes = (stacked_logits > 0).long()
        sum_votes = hard_votes.sum(dim=1, keepdim=True)
        final_decision = (sum_votes >= (self.num_models / 2.0)).long().float()
        avg_probs = torch.sigmoid(stacked_logits).mean(dim=1, keepdim=True)
        
        if return_details:
            weights = torch.ones(x.size(0), self.num_models, device=x.device) / self.num_models
            return final_decision, weights, avg_probs, stacked_logits
        return final_decision, hard_votes

def load_pruned_models(model_paths: List[str], device: torch.device, is_main: bool) -> List[nn.Module]:
    models = []
    for path in model_paths:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
            model.load_state_dict(ckpt['model_state_dict'])
            models.append(model.to(device).eval())
    if not models: raise ValueError("No models loaded!")
    return models

# ================== ADABN (مشابه کد شما، بهینه شده برای DDP) ==================
@torch.no_grad()
def adapt_batchnorm_for_new_dataset(models, means, stds, adabn_loader, device, is_main, is_distributed=False):
    if is_main:
        print("\n" + "="*70)
        print("STARTING BATCHNORM ADAPTATION (AdaBN) - CLEAN DATA (NO AUGMENTATION)")
        print("="*70)
        
    normalizer = MultiModelNormalization(means, stds).to(device)
    
    for model in models:
        model.eval()  
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.reset_running_stats()
                m.num_batches_tracked.zero_()  # ریست شمارنده برای محاسبه صحیح میانگین تجمعی
                m.train()          # فقط BN در حالت train باشد
                
    for images, _ in tqdm(adabn_loader, desc="Adapting BN", disable=not is_main):
        images = images.to(device)
        for i, model in enumerate(models):
            x_n = normalizer(images, i)
            model(x_n)  # فقط یک forward pass برای آپدیت آمار BN

    # سینک کردن آمارها در صورت استفاده از چند گرافیک
    if is_distributed and dist.is_initialized():
        for model in models:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    dist.all_reduce(m.running_mean, op=dist.ReduceOp.SUM)
                    m.running_mean /= dist.get_world_size()
                    dist.all_reduce(m.running_var, op=dist.ReduceOp.SUM)
                    m.running_var /= dist.get_world_size()
            
    for model in models:
        model.eval()  # برگرداندن کل مدل به حالت تست
        
    if is_main:
        print("✅ BatchNorm Adaptation Completed!\n")
# ========================================================================

@torch.no_grad()
def evaluate_single_model_ddp(model, loader, device, name, mean, std, is_main):
    model.eval()
    normalizer = MultiModelNormalization([mean], [std]).to(device)
    correct = total = 0
    if not loader: return 0.0
    for images, labels in tqdm(loader, desc=f"Eval {name}", disable=not is_main, leave=False):
        images, labels = images.to(device), labels.to(device).float()
        out = model(normalizer(images, 0))
        if isinstance(out, (tuple, list)): out = out[0]
        pred = (out.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()
        
    if dist.is_initialized():
        correct_t = torch.tensor(correct, device=device)
        total_t = torch.tensor(total, device=device)
        dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
        correct, total = correct_t.item(), total_t.item()
        
    acc = 100. * correct / total if total > 0 else 0.0
    if is_main: print(f" {name}: {acc:.2f}%")
    return acc

@torch.no_grad()
def evaluate_ensemble_final_ddp(model, loader, device, name, model_names, is_main=True):
    model.eval()
    local_stats = torch.zeros(5, device=device)
    if not loader: return 0.0, [], []
    
    for images, labels in tqdm(loader, desc=f"Eval {name}", disable=not is_main):
        images, labels = images.to(device), labels.to(device)
        outputs, _, _, stacked_logits = model(images, return_details=True)
        pred = outputs.squeeze(1).long()
        local_stats[0] += ((pred == 1) & (labels.long() == 1)).sum()
        local_stats[1] += ((pred == 0) & (labels.long() == 0)).sum()
        local_stats[2] += ((pred == 1) & (labels.long() == 0)).sum()
        local_stats[3] += ((pred == 0) & (labels.long() == 1)).sum()
        local_stats[4] += labels.size(0)

    if dist.is_initialized(): dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)

    if is_main:
        tp, tn, fp, fn, total = [s.item() for s in local_stats]
        acc = 100. * (tp + tn) / total if total > 0 else 0.0
        print(f"\n{'='*50}\n{name.upper()} RESULTS (Hard Voting)\n{'='*50}")
        print(f"Accuracy: {acc:.3f}% | TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}\n{'='*50}")
        return acc, [], local_stats.cpu().tolist()
    return 0.0, [], []

def final_evaluation_unified(model, test_loader_full, device, save_dir, model_names, args, is_main):
    if not is_main: return 0.0, None, None
    model.eval()
    base_dataset = test_loader_full.dataset
    if hasattr(base_dataset, 'dataset'): base_dataset = base_dataset.dataset
    test_indices = test_loader_full.sampler.indices if hasattr(test_loader_full, 'sampler') and hasattr(test_loader_full.sampler, 'indices') else list(range(len(base_dataset)))

    all_y_true, all_y_score, all_y_pred = [], [], []
    TP = TN = FP = FN = 0

    with torch.no_grad():
        for global_idx in tqdm(test_indices, desc="Final Eval"):
            try:
                image, label = base_dataset[global_idx]
                if not isinstance(image, torch.Tensor): image = T.ToTensor()(image)
                image = image.unsqueeze(0).to(device)
                label_int = int(label)
                decision, _, avg_probs, _ = model(image, return_details=True)
                pred_int = int(decision.squeeze().item())
                score_for_roc = avg_probs.squeeze().item()

                all_y_true.append(label_int)
                all_y_score.append(score_for_roc)
                all_y_pred.append(pred_int)

                if label_int == 1: TP += pred_int == 1; FN += pred_int == 0
                else: TN += pred_int == 0; FP += pred_int == 1
            except: continue

    total_samples = len(all_y_true)
    acc = 100.0 * (TP + TN) / total_samples if total_samples > 0 else 0.0
    print(f"\nFinal Accuracy (Hard Voting): {acc:.2f}%")

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "roc_data_test.json"), 'w') as f: 
        json.dump({"y_true": all_y_true, "y_score": all_y_score, "y_pred": all_y_pred}, f)
    
    try:
        from metrics_utils import plot_roc_and_f1
        plot_roc_and_f1(model, test_loader_full, device, save_dir, model_names, is_main)
    except: pass

    return acc, np.array(all_y_true), np.array(all_y_score)

# ================== DISTRIBUTED & MAIN ==================
# ================== DISTRIBUTED & MAIN ==================
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # ✅ تغییر nccl به gloo برای رفع خطای Kaggle
        dist.init_process_group(backend='gloo')
        
        # اگر GPU در دسترس بود، آن را تنظیم کن
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
            
        return device, local_rank, rank, world_size
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized(): dist.destroy_process_group()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    parser = argparse.ArgumentParser(description="Ensemble Evaluation with AdaBN")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dataset', type=str, required=True, choices=['wild', 'deepfake_lab', 'hard_fake_real', 'real_fake_dataset', 'uadfV', 'custom_genai'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--apply_adabn', action='store_true', help="Activate Adaptive Batch Normalization")
    
    args = parser.parse_args()
    if len(args.model_names) != len(args.model_paths): raise ValueError("model_names must match model_paths")

    set_seed(args.seed)
    device, local_rank, rank, world_size = setup_distributed()
    is_main = rank == 0
    is_distributed = world_size > 1

    MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]

    base_models = load_pruned_models(args.model_paths, device, is_main)
    MODEL_NAMES = args.model_names[:len(base_models)]
    MEANS = MEANS[:len(base_models)]
    STDS = STDS[:len(base_models)]

    # =================== فراخوانی دقیق منطق AdaBN شما ===================
    if args.apply_adabn:
        adabn_loader = create_adabn_dataloader(
            args.data_dir, args.batch_size, num_workers=args.num_workers,
            dataset_type=args.dataset, seed=args.seed, is_main=is_main,
            is_distributed=is_distributed
        )
        # استفاده از تابع مشابه با کد شما
        adapt_batchnorm_for_new_dataset(
            models=base_models, 
            means=MEANS, 
            stds=STDS, 
            adabn_loader=adabn_loader, 
            device=device, 
            is_main=is_main,
            is_distributed=is_distributed  # این پارامتر برای سینک روی چند GPU است
        )
    # =======================================================================

    ensemble = MajorityVotingEnsemble(base_models, MEANS, STDS, freeze_models=True).to(device)
    ensemble_module = ensemble.module if hasattr(ensemble, 'module') else ensemble

    try:
        train_loader, val_loader, test_loader = create_dataloaders(args.data_dir, args.batch_size, args.num_workers, args.dataset, is_distributed, args.seed, is_main)
    except FileNotFoundError as e:
        if is_main: print(f"[ERROR] {e}")
        cleanup_distributed()
        return

    if is_main: print("\n" + "="*70); print("INDIVIDUAL MODEL PERFORMANCE"); print("="*70)
    individual_accs = [evaluate_single_model_ddp(m, test_loader, device, f"Model {i+1} ({MODEL_NAMES[i]})", MEANS[i], STDS[i], is_main) for i, m in enumerate(base_models)]
    best_single = max(individual_accs) if individual_accs else 0.0

    ensemble_test_acc, vote_dist, stats = evaluate_ensemble_final_ddp(ensemble_module, test_loader, device, "Test", MODEL_NAMES, is_main)

    if is_main:
        os.makedirs(args.save_dir, exist_ok=True)
        _, _, test_loader_full = create_dataloaders(args.data_dir, args.batch_size, args.num_workers, args.dataset, False, args.seed, True)
        final_acc, _, _ = final_evaluation_unified(ensemble_module, test_loader_full, device, args.save_dir, MODEL_NAMES, args, is_main)
        
        print(f"\nBest Single Model: {best_single:.2f}%")
        print(f"Ensemble Accuracy: {final_acc:.2f}% (Improvement: {final_acc - best_single:+.2f}%)")

    cleanup_distributed()

if __name__ == "__main__":
    main()
