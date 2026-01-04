import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from PIL import Image
import sys

# ==========================================
# 1. CONFIGURATION & SEEDING
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==========================================
# 2. DATASET CLASSES (DEFINED HERE TO AVOID IMPORT ERROR)
# ==========================================

# کلاس UADFV را اضافه کردم تا اگر جایی صدا زده شد ارور ندهد
class UADFVDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        # تنظیمات پیش‌فرض برای UADFV
        for class_name, label in [('fake', 1), ('real', 0)]:
            path = os.path.join(root_dir, class_name)
            if os.path.exists(path):
                 for root, _, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith(('png', 'jpg', 'jpeg')):
                            self.samples.append((os.path.join(root, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class DFDDataset(Dataset):
    def __init__(self, root_dir, split=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # --- اصلاح جابجایی لیبل‌ها ---
        # اگر دقت کم است، یعنی جای 0 و 1 عوض شده. اینجا تنظیم کردیم: Fake=1, Real=0
        self.class_to_idx = {'fake': 1, 'real': 0} 

        # اسکن تمام پوشه‌ها (Train, Val, Test) برای رسیدن به 85 هزار عکس
        search_dirs = [split] if split else ['train', 'val', 'test']
        
        print(f"[Init] Scanning directories: {search_dirs} in {root_dir}")
        
        for s in search_dirs:
            split_path = os.path.join(root_dir, s)
            if not os.path.exists(split_path):
                continue
                
            for video_folder in os.listdir(split_path):
                video_path = os.path.join(split_path, video_folder)
                if not os.path.isdir(video_path):
                    continue
                
                # تشخیص لیبل
                folder_lower = video_folder.lower()
                if 'fake' in folder_lower:
                    label = self.class_to_idx['fake']
                elif 'real' in folder_lower:
                    label = self.class_to_idx['real']
                else:
                    continue

                for img_file in os.listdir(video_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(video_path, img_file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # جلوگیری از کرش کردن اگر یک عکس خراب بود
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros((3, 256, 256)), label

class TransformSubset(Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[self.indices[idx]]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# ==========================================
# 3. SPLITTING LOGIC
# ==========================================
def create_video_level_dfd_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    all_video_ids = set()
    
    # استخراج شناسه ویدیوها
    for img_path, _ in dataset.samples:
        dir_name = os.path.basename(os.path.dirname(img_path))
        vid_id = dir_name.replace('fake_', '').replace('real_', '').replace('_fake', '').replace('_real', '')
        all_video_ids.add(vid_id)

    all_video_ids = sorted(list(all_video_ids))
    
    train_val_ids, test_ids = train_test_split(all_video_ids, test_size=test_ratio, random_state=seed)
    val_size_adj = val_ratio / (train_ratio + val_ratio)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size_adj, random_state=seed)

    train_indices, val_indices, test_indices = [], [], []
    for idx, (img_path, _) in enumerate(dataset.samples):
        dir_name = os.path.basename(os.path.dirname(img_path))
        vid_id = dir_name.replace('fake_', '').replace('real_', '').replace('_fake', '').replace('_real', '')
        
        if vid_id in test_ids:
            test_indices.append(idx)
        elif vid_id in val_ids:
            val_indices.append(idx)
        else:
            train_indices.append(idx)
            
    return train_indices, val_indices, test_indices

# ==========================================
# 4. LOADERS
# ==========================================
def get_dataloaders(base_dir, batch_size, is_distributed=False):
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # نرمال‌سازی معمولا کمک می‌کند، اگر مدل با این نرمال‌سازی آموزش دیده:
        # transforms.Normalize([0.5]*3, [0.5]*3) 
    ])

    print("--- Loading Dataset ---")
    full_dataset = DFDDataset(base_dir, split=None, transform=None)
    print(f"Total Images Found: {len(full_dataset)}")

    print("--- Splitting (Video Level) ---")
    _, _, test_idx = create_video_level_dfd_split(full_dataset)
    print(f"Test Set Size: {len(test_idx)}")

    test_ds = TransformSubset(full_dataset, test_idx, test_transform)

    sampler = DistributedSampler(test_ds, shuffle=False) if is_distributed else None

    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True
    )
    
    return test_loader

# ==========================================
# 5. SAFE MODEL LOADING (FIXED)
# ==========================================
def load_models_safely(model_paths, device):
    models = []
    print("\n--- Loading Models ---")
    for path in model_paths:
        print(f"Attempting to load: {os.path.basename(path)} ...", end=" ")
        try:
            # map_location برای جلوگیری از مشکلات GPU
            # weights_only=False برای اطمینان از لود شدن کامل ساختار
            model = torch.load(path, map_location=device)
            
            if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
                
            model.to(device)
            model.eval()
            models.append(model)
            print("SUCCESS ✅")
        except Exception as e:
            print(f"\nFAILED ❌ Error loading {path}: {e}")
            print(f"Skipping this model and continuing...")
    
    return models

# ==========================================
# 6. EVALUATION
# ==========================================
def evaluate_ensemble(models, dataloader, device, is_master):
    all_preds = []
    all_labels = []
    
    if is_master:
        print("\n--- Starting Majority Voting Evaluation ---")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            batch_votes = torch.zeros(images.size(0)).to(device)
            
            for model in models:
                outputs = model(images)
                
                # تبدیل خروجی به 0 یا 1
                if outputs.shape[1] == 1:
                    preds = (torch.sigmoid(outputs) > 0.5).float().squeeze()
                else:
                    preds = torch.argmax(outputs, dim=1).float()
                
                batch_votes += preds
            
            # رای اکثریت
            final_preds = (batch_votes > (len(models) / 2)).long()
            
            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if is_master and batch_idx % 20 == 0:
                print(f"Processed batch {batch_idx}/{len(dataloader)}")

    return np.array(all_preds), np.array(all_labels)

# ==========================================
# 7. MAIN
# ==========================================
def main():
    # تنظیمات Distributed
    is_distributed = 'RANK' in os.environ
    if is_distributed:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        is_master = (int(os.environ['RANK']) == 0)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_master = True

    set_seed(42)

    # --- تنظیمات آدرس‌ها (لطفاً چک کنید) ---
    DATA_DIR = '/kaggle/input/extracted-deepfake-frames' 
    
    MODEL_PATHS = [
        '/kaggle/working/140k_pearson_pruned.pt',
        '/kaggle/working/200k_final.pt',
        '/kaggle/working/190k_pearson_pruned.pt'
    ]
    BATCH_SIZE = 32

    # اجرا
    test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE, is_distributed)
    models = load_models_safely(MODEL_PATHS, device)
    
    if len(models) == 0:
        print("CRITICAL: No models loaded. Exiting.")
        return

    preds, labels = evaluate_ensemble(models, test_loader, device, is_master)

    if is_master:
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        
        acc = np.mean(preds == labels) * 100
        print(f"Accuracy: {acc:.2f}%")
        
        cm = confusion_matrix(labels, preds)
        print("\nConfusion Matrix:")
        print(f"TN (Real Correct): {cm[0,0]} | FP (Real->Fake): {cm[0,1]}")
        print(f"FN (Fake->Real): {cm[1,0]} | TP (Fake Correct): {cm[1,1]}")

if __name__ == '__main__':
    main()
