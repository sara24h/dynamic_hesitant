# dynamic_hesitant/datasets/base_dataset.py

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import os
from tqdm import tqdm
import torch.distributed as dist

# واردات توابع و کلاس‌های مورد نیاز از فایل loaders.py
from .loaders import (
    UADFVDataset, 
    TransformSubset, 
    worker_init_fn, 
    create_reproducible_split,
    prepare_real_fake_dataset,
    prepare_hard_fake_real_dataset,
    prepare_deepflux_dataset,
    prepare_uadfV_dataset
)

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
