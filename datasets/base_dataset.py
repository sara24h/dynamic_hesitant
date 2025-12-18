import os
import warnings
import random
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

class UADFVDataset(Dataset):
    """
    Custom Dataset for UADFV dataset structure.
    Expects a directory with 'fake/frames' and 'real/frames' subdirectories.
    """
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

# ====================== DATASET SPLIT FUNCTIONS ======================
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

class TransformSubset(Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform
   
    def __getitem__(self, idx):
        # This handles ImageFolder datasets
        if hasattr(self.dataset, 'samples'):
            img_path, label = self.dataset.samples[self.indices[idx]]
            img = self.dataset.loader(img_path)
        # This handles custom datasets like UADFVDataset
        else:
            img_path, label = self.dataset.samples[self.indices[idx]]
            img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img, label

def create_dataloaders_ddp(base_dir: str, batch_size: int, rank: int, world_size: int,
                          num_workers: int = 2, dataset_type: str = 'wild'):
  
    if rank == 0:
        print("="*70)
        print(f"Creating DataLoaders with DDP (Dataset: {dataset_type})")
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
         
            if rank == 0:
                print(f"{split.capitalize():5}: {path}")
         
            transform = train_transform if split == 'train' else val_test_transform
            datasets_dict[split] = datasets.ImageFolder(path, transform=transform)
       
        if rank == 0:
            print(f"\nDataset Stats:")
            for split, ds in datasets_dict.items():
                print(f" {split.capitalize():5}: {len(ds):,} images | Classes: {ds.classes}")
            print(f" Class → Index: {datasets_dict['train'].class_to_idx}\n")
       
        loaders = {}
        for split, ds in datasets_dict.items():
            if split == 'train':
                sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
                loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True, drop_last=True,
                                  worker_init_fn=worker_init_fn)
            else:
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True, drop_last=False,
                                  worker_init_fn=worker_init_fn)
            loaders[split] = loader
       
        train_loader = loaders['train']
        val_loader = loaders['valid']
        test_loader = loaders['test']
   
    elif dataset_type == 'real_fake':
        if rank == 0:
            print(f"Processing real-fake dataset from: {base_dir}")
       
        # Only rank 0 does the split preparation to avoid race conditions
        if rank == 0:
            full_dataset, train_indices, val_indices, test_indices = prepare_real_fake_dataset(
                base_dir, seed=42
            )
       
        dist.barrier() # Wait for rank 0 to finish

        # All ranks load the dataset
        if os.path.exists(os.path.join(base_dir, 'training_fake')) and \
           os.path.exists(os.path.join(base_dir, 'training_real')):
            dataset_dir = base_dir
        elif os.path.exists(os.path.join(base_dir, 'real_and_fake_face')):
            dataset_dir = os.path.join(base_dir, 'real_and_fake_face')
        else:
            raise FileNotFoundError(f"Could not find training folders in {base_dir}")
        
        full_dataset = datasets.ImageFolder(dataset_dir)
        train_indices, val_indices, test_indices = create_reproducible_split(full_dataset, seed=42)
        
        train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
        val_dataset = TransformSubset(full_dataset, val_indices, val_test_transform)
        test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 worker_init_fn=worker_init_fn)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True, drop_last=False,
                               worker_init_fn=worker_init_fn)
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=worker_init_fn)

    elif dataset_type == 'hard_fake_real':
        if rank == 0:
            print(f"Processing hardfakevsrealfaces dataset from: {base_dir}")
        
        if rank == 0:
            full_dataset, train_indices, val_indices, test_indices = prepare_hard_fake_real_dataset(
                base_dir, seed=42
            )
        
        dist.barrier()

        if os.path.exists(os.path.join(base_dir, 'fake')) and \
           os.path.exists(os.path.join(base_dir, 'real')):
            dataset_dir = base_dir
        elif os.path.exists(os.path.join(base_dir, 'hardfakevsrealfaces')):
            dataset_dir = os.path.join(base_dir, 'hardfakevsrealfaces')
        else:
            raise FileNotFoundError(f"Could not find fake/real folders in {base_dir}")
        
        full_dataset = datasets.ImageFolder(dataset_dir)
        train_indices, val_indices, test_indices = create_reproducible_split(full_dataset, seed=42)
        
        train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
        val_dataset = TransformSubset(full_dataset, val_indices, val_test_transform)
        test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 worker_init_fn=worker_init_fn)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True, drop_last=False,
                               worker_init_fn=worker_init_fn)
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=worker_init_fn)

    elif dataset_type == 'deepflux':
        if rank == 0:
            print(f"Processing DeepFLUX dataset from: {base_dir}")
        
        if rank == 0:
            full_dataset, train_indices, val_indices, test_indices = prepare_deepflux_dataset(
                base_dir, seed=42
            )
        
        dist.barrier()

        if os.path.exists(os.path.join(base_dir, 'Fake')) and \
           os.path.exists(os.path.join(base_dir, 'Real')):
            dataset_dir = base_dir
        elif os.path.exists(os.path.join(base_dir, 'DeepFLUX')):
            dataset_dir = os.path.join(base_dir, 'DeepFLUX')
        else:
            raise FileNotFoundError(f"Could not find Fake/Real folders in {base_dir}")
        
        full_dataset = datasets.ImageFolder(dataset_dir)
        train_indices, val_indices, test_indices = create_reproducible_split(full_dataset, seed=42)
        
        train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
        val_dataset = TransformSubset(full_dataset, val_indices, val_test_transform)
        test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 worker_init_fn=worker_init_fn)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True, drop_last=False,
                               worker_init_fn=worker_init_fn)
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=worker_init_fn)

    elif dataset_type == 'uadfV':
        if rank == 0:
            print(f"Processing UADFV dataset from: {base_dir}")
        
        if rank == 0:
            full_dataset, train_indices, val_indices, test_indices = prepare_uadfV_dataset(
                base_dir, seed=42
            )

        dist.barrier()

        full_dataset = UADFVDataset(base_dir)
        train_indices, val_indices, test_indices = create_reproducible_split(full_dataset, seed=42)

        train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
        val_dataset = TransformSubset(full_dataset, val_indices, val_test_transform)
        test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
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
   
    if rank == 0:
        print(f"DataLoaders ready! Batch size per GPU: {batch_size}")
        print(f" Effective batch size: {batch_size * world_size}")
        print(f" Batches per GPU → Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
        print("="*70 + "\n")
 
    return train_loader, val_loader, test_loader
