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


class DFDDataset(Dataset):
    def __init__(self, root_dir, split=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'fake': 0, 'real': 1}

        # اگر split مشخص شده باشد (مثل 'train') فقط آن را لود می‌کند
        # اگر split=None باشد، کل پوشه‌ها (train, val, test) را لود می‌کند
        search_dirs = [split] if split else ['train', 'val', 'test']
        
        for s in search_dirs:
            split_path = os.path.join(root_dir, s)
            if not os.path.exists(split_path):
                continue
                
            for video_folder in os.listdir(split_path):
                video_path = os.path.join(split_path, video_folder)
                if not os.path.isdir(video_path):
                    continue
                
                # تشخیص لیبل
                if 'fake' in video_folder.lower():
                    label = 0
                elif 'real' in video_folder.lower():
                    label = 1
                else:
                    continue

                for img_file in os.listdir(video_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(video_path, img_file), label))

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


def create_video_level_uadfV_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    all_video_ids = set()
    
    for img_path, label in dataset.samples:
        dir_name = os.path.basename(os.path.dirname(img_path))
        
        # منطق UADFV: حذف _fake از انتهای نام
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
    
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size_adjusted,
        random_state=seed
    )
    
    print(f"[Split] Videos -> Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

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
            print(f"[Warning] Video ID {vid_id} not found in splits!")
            
    print(f"[Split] Frames -> Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    return train_indices, val_indices, test_indices


def create_video_level_dfd_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
   
    all_video_ids = set()
    
    for img_path, label in dataset.samples:
        dir_name = os.path.basename(os.path.dirname(img_path))
 
        
        vid_id = dir_name
        if vid_id.startswith('fake_'):
            vid_id = vid_id.replace('fake_', '', 1)
        elif vid_id.startswith('real_'):
            vid_id = vid_id.replace('real_', '', 1)
        elif '_fake' in vid_id: # پشتیبانی از حالت‌ دیگر
            vid_id = vid_id.replace('_fake', '_')
        elif '_real' in vid_id:
            vid_id = vid_id.replace('_real', '_')
            
        all_video_ids.add(vid_id)

    all_video_ids = sorted(list(all_video_ids))
    print(f"[Split] Found {len(all_video_ids)} unique video IDs.")

    train_val_ids, test_ids = train_test_split(
        all_video_ids,
        test_size=test_ratio,
        random_state=seed
    )
    
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size_adjusted,
        random_state=seed
    )
    
    print(f"[Split] Videos -> Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    train_indices = []
    val_indices = []
    test_indices = []
    
    for idx, (img_path, label) in enumerate(dataset.samples):
        dir_name = os.path.basename(os.path.dirname(img_path))
        
        vid_id = dir_name
        if vid_id.startswith('fake_'):
            vid_id = vid_id.replace('fake_', '', 1)
        elif vid_id.startswith('real_'):
            vid_id = vid_id.replace('real_', '', 1)
        elif '_fake' in vid_id:
            vid_id = vid_id.replace('_fake', '_')
        elif '_real' in vid_id:
            vid_id = vid_id.replace('_real', '_')
            
        if vid_id in test_ids:
            test_indices.append(idx)
        elif vid_id in val_ids:
            val_indices.append(idx)
        elif vid_id in train_ids:
            train_indices.append(idx)
        else:
            print(f"[Warning] Video ID {vid_id} not found in splits!")
            
    print(f"[Split] Frames -> Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    return train_indices, val_indices, test_indices


def create_standard_reproducible_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
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


def prepare_dataset(base_dir: str, dataset_type: str, seed: int = 42):
    dataset_paths = {
        'real_fake': ['training_fake', 'training_real'],
        'hard_fake_real': ['fake', 'real'],
        'deepflux': ['Fake', 'Real'],
    }

    print(f"\n[Dataset Loading] Processing: {dataset_type}")
    
    if dataset_type == 'uadfV':
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"UADFV dataset directory not found: {base_dir}")
        temp_transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = UADFVDataset(base_dir, transform=temp_transform)
        print("[Dataset Loading] UADFVDataset loaded.")

    elif dataset_type == 'dfd':
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"DFD dataset directory not found: {base_dir}")
        
        temp_transform = transforms.Compose([transforms.ToTensor()])
        # قرار دادن split=None باعث می‌شود کل ۸۵ هزار فریم لود شود
        full_dataset = DFDDataset(base_dir, split=None, transform=temp_transform)
        print(f"[Dataset Loading] DFDDataset loaded with {len(full_dataset)} images.")
        
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

    # 2. Structure Detection Logic
    # اگر دیتاست UADFV یا DFD است، ساختار ویدئویی فرض می‌شود
    is_video_structure = False
    if dataset_type in ['uadfV', 'dfd']:
        is_video_structure = True
    else:
        # برای دیتاست‌های دیگر تشخیص خودکار
        sample_path = full_dataset.samples[0][0]
        immediate_parent = os.path.basename(os.path.dirname(sample_path))
        is_video_structure = immediate_parent not in full_dataset.classes

    print(f"[Structure Detection] Video Level: {is_video_structure}")
    
    if is_video_structure:
        print("[Structure Decision] >> Detected VIDEO/NESTED structure. Using Video-Level Split.")
        if dataset_type == 'uadfV':
            train_indices, val_indices, test_indices = create_video_level_uadfV_split(
                full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=seed
            )
        elif dataset_type == 'dfd':
            train_indices, val_indices, test_indices = create_video_level_dfd_split(
                full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=seed
            )
        else:
            # Fallback for other video structures (like wild dataset if needed)
            # در اینجا می‌توان از متد استاندارد استفاده کرد یا یک متد عمومی ویدئویی نوشت
            # فعلاً برای اطمینان از استاندارد استفاده می‌کنیم
            train_indices, val_indices, test_indices = create_standard_reproducible_split(
                full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=seed
            )
    else:
        print("[Structure Decision] >> Detected FLAT IMAGE structure. Using Standard Stratified Split.")
        train_indices, val_indices, test_indices = create_standard_reproducible_split(
            full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=seed
        )
        
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

        # اعمال Transform ها
        if dataset_type == 'uadfV':
            # UADFV خودش کلاس Dataset سفارشی دارد و مستقیماً ترنسفرم را قبول می‌کند
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
            test_dataset = Subset(full_dataset, test_indices)
            # این جایگزین بخش اعمال Transform ها در کد شما شود
        # برای همه دیتاست‌های لیستی (UADFV, DFD, etc.)
            train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
            val_dataset = TransformSubset(full_dataset, val_indices, val_test_transform)
            test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)
        else:
            # برای DFD و سایر دیتاست‌ها از TransformSubset استفاده می‌کنیم
            # چون ImageFolder و DFDDataset در Subset معمولی ترنسفرم را به درستی هندل نمی‌کنند
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
