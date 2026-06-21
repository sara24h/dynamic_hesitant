import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
import os
import random
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from PIL import Image

# ================== DATASET CLASSES ==================

class CustomGenAIDataset(Dataset):
    def __init__(self, root_dir, fake_classes, real_class, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.label_map = {'fake': 0, 'real': 1}

        print(f"[Old CustomDataset] Loading Fake images from: {fake_classes}")
        for class_name in fake_classes:
            class_path = os.path.join(root_dir, class_name)
            if os.path.exists(class_path):
                files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img_file in files:
                    self.samples.append((os.path.join(class_path, img_file), self.label_map['fake']))

        print(f"[Old CustomDataset] Loading Real images from: {real_class}")
        real_path = os.path.join(root_dir, real_class)
        if os.path.exists(real_path):
            files = [f for f in os.listdir(real_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in files:
                self.samples.append((os.path.join(real_path, img_file), self.label_map['real']))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label


class NewGenAIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.label_map = {'fake': 0, 'real': 1}
        
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            current_folder_name = os.path.basename(dirpath)
            if current_folder_name in ['real', 'fake']:
                label = self.label_map[current_folder_name]
                valid_files = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img_file in valid_files:
                    self.samples.append((os.path.join(dirpath, img_file), label))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label


class UADFVDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'fake': 0, 'real': 1}

        for class_name in ['fake', 'real']:
            frames_dir = os.path.join(self.root_dir, class_name, 'frames')
            if os.path.exists(frames_dir):
                for subdir in os.listdir(frames_dir):
                    subdir_path = os.path.join(frames_dir, subdir)
                    if os.path.isdir(subdir_path):
                        for img_file in os.listdir(subdir_path):
                            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.samples.append((os.path.join(subdir_path, img_file), self.class_to_idx[class_name]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label


class TransformSubset(Subset):
    """Subset with custom transform that bypasses dataset's transform if necessary."""
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

    # اضافه کردن این متد برای رفع ارور نسخه‌های جدید PyTorch
    def __getitems__(self, indices):
        return [self.__getitem__(i) for i in indices]

# ================== UTILITY FUNCTIONS ==================

def get_sample_info(dataset, index):
    if hasattr(dataset, 'samples'):
        return dataset.samples[index]
    elif hasattr(dataset, 'dataset'):
        return get_sample_info(dataset.dataset, index)
    else:
        raise AttributeError("Cannot find samples in dataset")

def create_standard_reproducible_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    num_samples = len(dataset)
    indices = list(range(num_samples))
    labels = [dataset.samples[i][1] for i in indices]
    train_val_indices, test_indices = train_test_split(indices, test_size=test_ratio, random_state=seed, stratify=labels)
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=val_size_adjusted, random_state=seed, stratify=[labels[i] for i in train_val_indices])
    return train_indices, val_indices, test_indices

def create_video_level_uadfV_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    all_video_ids = set()
    for img_path, label in dataset.samples:
        dir_name = os.path.basename(os.path.dirname(img_path))
        video_id = dir_name.replace('_fake', '') if '_fake' in dir_name else dir_name
        all_video_ids.add(video_id)
    all_video_ids = sorted(list(all_video_ids))
    train_val_ids, test_ids = train_test_split(all_video_ids, test_size=test_ratio, random_state=seed)
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size_adjusted, random_state=seed)
    train_indices, val_indices, test_indices = [], [], []
    for idx, (img_path, label) in enumerate(dataset.samples):
        dir_name = os.path.basename(os.path.dirname(img_path))
        vid_id = dir_name.replace('_fake', '') if '_fake' in dir_name else dir_name
        if vid_id in test_ids: test_indices.append(idx)
        elif vid_id in val_ids: val_indices.append(idx)
        elif vid_id in train_ids: train_indices.append(idx)
    return train_indices, val_indices, test_indices

def prepare_dataset(base_dir: str, dataset_type: str, seed: int = 42):
    dataset_paths = {
        'real_fake': ['training_fake', 'training_real'],
        'hard_fake_real': ['fake', 'real'],
        'deepflux': ['Fake', 'Real'],
        'real_fake_dataset': ['face_fake', 'face_real'], 
        'deepfake_lab': ['training_fake', 'training_real'], 
    }
    temp_transform = None 

    if dataset_type == 'custom_genai':
        full_dataset = CustomGenAIDataset(base_dir, fake_classes=['DALL-E', 'DeepFaceLab', 'Midjourney', 'StyleGAN'], real_class='Real', transform=temp_transform)
        train_indices, val_indices, test_indices = create_standard_reproducible_split(full_dataset, seed=seed)
    elif dataset_type == 'custom_genai_v2':
        full_dataset = NewGenAIDataset(base_dir, transform=temp_transform)
        train_indices, val_indices, test_indices = create_standard_reproducible_split(full_dataset, seed=seed)
    elif dataset_type == 'uadfV':
        full_dataset = UADFVDataset(base_dir, transform=temp_transform)
        train_indices, val_indices, test_indices = create_video_level_uadfV_split(full_dataset, seed=seed)
    elif dataset_type in dataset_paths:
        folders = dataset_paths[dataset_type]
        dataset_dir = base_dir
        if not all(os.path.exists(os.path.join(dataset_dir, f)) for f in folders):
            possible_sub_dir = os.path.join(base_dir, dataset_type)
            if all(os.path.exists(os.path.join(possible_sub_dir, f)) for f in folders):
                dataset_dir = possible_sub_dir
            else:
                raise FileNotFoundError(f"Could not find dataset folders {folders} in {base_dir}")
        full_dataset = datasets.ImageFolder(dataset_dir, transform=temp_transform)
        train_indices, val_indices, test_indices = create_standard_reproducible_split(full_dataset, seed=seed)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    return full_dataset, train_indices, val_indices, test_indices

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloaders(base_dir: str, batch_size: int, num_workers: int = 0,
                       dataset_type: str = 'wild', is_distributed: bool = False,
                       seed: int = 42, is_main: bool = True):
    if is_main:
        print("="*70)
        print(f"Creating DataLoaders (Dataset: {dataset_type})")
        print("="*70)

    # --- کاهش شدت Augmentation ---
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10), # کاهش چرخش
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), # کاهش تغییر رنگ
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
            if not os.path.exists(path): raise FileNotFoundError(f"Folder not found: {path}")
            transform = train_transform if split == 'train' else val_test_transform
            datasets_dict[split] = datasets.ImageFolder(path, transform=transform)

        train_sampler = DistributedSampler(datasets_dict['train'], shuffle=True) if is_distributed else None
        val_sampler = DistributedSampler(datasets_dict['valid'], shuffle=False) if is_distributed else None
        test_sampler = DistributedSampler(datasets_dict['test'], shuffle=False) if is_distributed else None

        train_loader = DataLoader(datasets_dict['train'], batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(datasets_dict['valid'], batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
        test_loader = DataLoader(datasets_dict['test'], batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
        
        # ساخت لودر تمیز برای AdaBN
        clean_dataset = datasets.ImageFolder(os.path.join(base_dir, 'train'), transform=val_test_transform)
        train_loader_clean = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

    else:
        if is_main: print(f"Processing {dataset_type} dataset from: {base_dir}")
        full_dataset, train_indices, val_indices, test_indices = prepare_dataset(base_dir, dataset_type, seed=seed)

        train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
        val_dataset = TransformSubset(full_dataset, val_indices, val_test_transform)
        test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)

        train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
        sampler_test = DistributedSampler(test_dataset, shuffle=False) if is_distributed else None

        train_drop_last = (len(train_dataset) % batch_size == 1)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=train_drop_last, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=sampler_test, num_workers=num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

        # ===== ساخت دیتالودر تمیز فقط برای AdaBN =====
        train_dataset_clean = TransformSubset(full_dataset, train_indices, val_test_transform)
        train_loader_clean = DataLoader(train_dataset_clean, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
        # ================================================

    if is_main:
        print(f"DataLoaders ready! Batch size: {batch_size}")
        print(f" Batches → Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
        print("="*70 + "\n")
        
    return train_loader, val_loader, test_loader, train_loader_clean
