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

# ================== DATASETS ==================

class CustomGenAIDataset(Dataset):
    """
    کلاس اصلاح شده برای پشتیبانی از دو ساختار:
    1. Flat: class_name/folder (ساختار قبلی)
    2. Hierarchical: class_name/split/fake|real (ساختار جدید درختی)
    """
    def __init__(self, root_dir, fake_classes, real_class, transform=None, hierarchical=False):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.label_map = {'fake': 0, 'real': 1}

        if hierarchical:
            # ================= ساختار جدید (Tree: class_name/train|test/fake|real) =================
            # ما داده‌ها را یکجا بارگذاری می‌کنیم تا بعداً به نسبت 70-15-15 تقسیم کنیم.
            # در اینجا فقط "train" و "test" موجود هستند. فرض بر این است که کل دیتاست در این دو پوشه است.
            splits_to_load = ['train', 'test'] 
            
            print(f"[CustomDataset] Loading HIERARCHICAL structure from: {root_dir}")
            print(f" -> Splits detected: {splits_to_load}")

            for class_name in fake_classes + [real_class]:
                print(f"   Processing class: {class_name}")
                for split in splits_to_load:
                    split_path = os.path.join(root_dir, class_name, split)
                    if os.path.exists(split_path):
                        for label_name in ['fake', 'real']:
                            label_path = os.path.join(split_path, label_name)
                            if os.path.exists(label_path):
                                files = [f for f in os.listdir(label_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                                for img_file in files:
                                    img_path = os.path.join(label_path, img_file)
                                    self.samples.append((img_path, self.label_map[label_name]))
        else:
            # ================= ساختار قبلی (Flat: class_name/folder) =================
            print(f"[CustomDataset] Loading FLAT structure from: {root_dir}")
            
            print(f" -> Loading Fake images from: {fake_classes}")
            for class_name in fake_classes:
                class_path = os.path.join(root_dir, class_name)
                if os.path.exists(class_path):
                    files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    for img_file in files:
                        img_path = os.path.join(class_path, img_file)
                        self.samples.append((img_path, self.label_map['fake']))
                else:
                    print(f"[Warning] Path not found: {class_path}")

            print(f" -> Loading Real images from: {real_class}")
            real_path = os.path.join(root_dir, real_class)
            if os.path.exists(real_path):
                files = [f for f in os.listdir(real_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img_file in files:
                    img_path = os.path.join(real_path, img_file)
                    self.samples.append((img_path, self.label_map['real']))
            else:
                print(f"[Warning] Path not found: {real_path}")

        print(f"[CustomDataset] Total loaded images: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


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

def get_sample_info(dataset, index):
    if hasattr(dataset, 'samples'):
        return dataset.samples[index]
    elif hasattr(dataset, 'dataset'):
        return get_sample_info(dataset.dataset, index)
    else:
        raise AttributeError("Cannot find samples in dataset")


def create_standard_reproducible_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """تقسیم داده‌ها به نسبت 70-15-15"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    num_samples = len(dataset)
    indices = list(range(num_samples))
    
    labels = [dataset.samples[i][1] for i in indices]

    # تقسیم به Train+Val و Test
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_ratio, random_state=seed, stratify=labels)
    
    # محاسبه نسبت مجدد برای Val نسبت به باقی‌مانده
    train_val_labels = [labels[i] for i in train_val_indices]
    val_size = val_ratio / (train_ratio + val_ratio)
    
    # تقسیم به Train و Val
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size, random_state=seed, stratify=train_val_labels)
        
    return train_indices, val_indices, test_indices


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

    train_val_ids, test_ids = train_test_split(all_video_ids, test_size=test_ratio, random_state=seed)
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size_adjusted, random_state=seed)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for idx, (img_path, label) in enumerate(dataset.samples):
        dir_name = os.path.basename(os.path.dirname(img_path))
        if '_fake' in dir_name:
            vid_id = dir_name.replace('_fake', '')
        else:
            vid_id = dir_name
            
        if vid_id in test_ids: test_indices.append(idx)
        elif vid_id in val_ids: val_indices.append(idx)
        elif vid_id in train_ids: train_indices.append(idx)
            
    return train_indices, val_indices, test_indices


def create_video_level_dfd_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    all_video_ids = set()
    for img_path, label in dataset.samples:
        dir_name = os.path.basename(os.path.dirname(img_path))
        vid_id = dir_name
        if vid_id.startswith('fake_'): vid_id = vid_id.replace('fake_', '', 1)
        elif vid_id.startswith('real_'): vid_id = vid_id.replace('real_', '', 1)
        elif '_fake' in vid_id: vid_id = vid_id.replace('_fake', '_')
        elif '_real' in vid_id: vid_id = vid_id.replace('_real', '_')
        all_video_ids.add(vid_id)

    all_video_ids = sorted(list(all_video_ids))
    train_val_ids, test_ids = train_test_split(all_video_ids, test_size=test_ratio, random_state=seed)
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size_adjusted, random_state=seed)
    
    train_indices, val_indices, test_indices = [], [], []
    for idx, (img_path, label) in enumerate(dataset.samples):
        dir_name = os.path.basename(os.path.dirname(img_path))
        vid_id = dir_name
        if vid_id.startswith('fake_'): vid_id = vid_id.replace('fake_', '', 1)
        elif vid_id.startswith('real_'): vid_id = vid_id.replace('real_', '', 1)
        elif '_fake' in vid_id: vid_id = vid_id.replace('_fake', '_')
        elif '_real' in vid_id: vid_id = vid_id.replace('_real', '_')
            
        if vid_id in test_ids: test_indices.append(idx)
        elif vid_id in val_ids: val_indices.append(idx)
        elif vid_id in train_ids: train_indices.append(idx)
            
    return train_indices, val_indices, test_indices


def prepare_dataset(base_dir: str, dataset_type: str, seed: int = 42):
    dataset_paths = {
        'real_fake': ['training_fake', 'training_real'],
        'hard_fake_real': ['fake', 'real'],
        'deepflux': ['Fake', 'Real'],
    }

    print(f"\n[Dataset Loading] Processing: {dataset_type}")

    if dataset_type == 'custom_genai':
        # ساختار قبلی (Flat)
        fake_folders = ['DALL-E', 'DeepFaceLab', 'Midjourney','StyleGAN']
        real_folder = 'Real'
        
        temp_transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = CustomGenAIDataset(
            base_dir, 
            fake_classes=fake_folders, 
            real_class=real_folder, 
            transform=temp_transform,
            hierarchical=False
        )
        print("[Dataset Loading] Custom GenAI Dataset (Flat) loaded.")
        train_indices, val_indices, test_indices = create_standard_reproducible_split(
            full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=seed
        )
    
    elif dataset_type == 'custom_genai_tree':
        # ======================= ساختار جدید (Tree) =======================
        fake_folders = ['insightface', 'photoshop', 'stablediffusion v1.5', 'stylegan']
        real_folder = 'real' # توجه: پوشه real در روت است
        
        temp_transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = CustomGenAIDataset(
            base_dir, 
            fake_classes=fake_folders, 
            real_class=real_folder, 
            transform=temp_transform,
            hierarchical=True # فعال‌سازی حالت درختی
        )
        print("[Dataset Loading] Custom GenAI Dataset (Tree) loaded.")
        
        # ایجاد تقسیم 70-15-15
        train_indices, val_indices, test_indices = create_standard_reproducible_split(
            full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=seed
        )
        
    elif dataset_type == 'uadfV':
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"UADFV dataset directory not found: {base_dir}")
        temp_transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = UADFVDataset(base_dir, transform=temp_transform)
        train_indices, val_indices, test_indices = create_video_level_uadfV_split(full_dataset, seed=seed)
        
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
        train_indices, val_indices, test_indices = create_standard_reproducible_split(full_dataset, seed=seed)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    return full_dataset, train_indices, val_indices, test_indices


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
        # مدیریت همه دیتاست‌های دیگر (uadfV, dfd, custom_genai, custom_genai_tree)
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

        # اعمال Transform ها با استفاده از TransformSubset
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
                           
if __name__ == '__main__':
    path_to_dataset = '/path/to/your/root/dataset'
    
    train_loader, val_loader, test_loader = create_dataloaders(
        base_dir=path_to_dataset,
        batch_size=32,
        num_workers=4,
        dataset_type='custom_genai_tree',  # <-- اینجا را برای تست جدید تغییر دهید
        is_distributed=False,
        seed=42
    )
