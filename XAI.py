import torch
import torch.nn as nn
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

# کتابخانه‌های مورد نیاز برای Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- وارد کردن کلاس‌های اصلی پروژه خودتان ---
# مطمئن شوید که این مسیرها صحیح هستند
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
# از فایل اصلی خود، این کلاس‌ها و توابع را کپی کنید یا import کنید
# از آنجایی که این یک اسکریپت مستقل است، باید تعاریف آن‌ها را در دسترس داشته باشد
# برای سادگی، من تعاریف را اینجا کپی می‌کنم
class MultiModelNormalization(nn.Module):
    def __init__(self, means: list, stds: list):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))
   
    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')

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
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_models * num_memberships)
        )
        self.aggregation_weights = nn.Parameter(torch.ones(num_memberships) / num_memberships)
     
    def forward(self, x: torch.Tensor):
        features = self.feature_net(x).flatten(1)
        memberships = self.membership_generator(features)
        memberships = memberships.view(-1, self.num_models, self.num_memberships)
        memberships = torch.sigmoid(memberships)
        agg_weights = F.softmax(self.aggregation_weights, dim=0)
        final_weights = (memberships * agg_weights.view(1, 1, -1)).sum(dim=2)
        final_weights = F.softmax(final_weights, dim=1)
        return final_weights, memberships

class FuzzyHesitantEnsemble(nn.Module):
    def __init__(self, models: list, means: list, stds: list, num_memberships: int = 3, freeze_models: bool = True):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.hesitant_fuzzy = HesitantFuzzyMembership(
            input_dim=128,
            num_models=self.num_models,
            num_memberships=num_memberships
        )
       
        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
   
    def forward(self, x: torch.Tensor):
        final_weights, all_memberships = self.hesitant_fuzzy(x)
        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
       
        for i in range(self.num_models):
            x_n = self.normalizations(x, i)
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            outputs[:, i] = out
       
        final_output = (outputs * final_weights.unsqueeze(-1)).sum(dim=1)
        return final_output, final_weights

def load_pruned_models(model_paths: list, device: torch.device) -> list:
    """بارگذاری مدل‌های پایه از فایل‌های چک‌پوینت"""
    models = []
    print(f"Loading {len(model_paths)} pruned models...")
    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            print(f" [WARNING] File not found: {path}")
            continue
        print(f" [{i+1}/{len(model_paths)}] Loading: {os.path.basename(path)}")
        try:
            ckpt = torch.load(path, map_location='cpu')
            model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device).eval()
            models.append(model)
        except Exception as e:
            print(f" [ERROR] Failed to load {path}: {e}")
            continue
    if len(models) == 0:
        raise ValueError("No models loaded!")
    print(f"All {len(models)} models loaded!\n")
    return models

def visualize_grad_cam(model, image_tensor, target_layers, class_index):
    """این تابع دقیقا همان تابعی است که قبلا نوشتیم"""
    model.eval()
    input_tensor = image_tensor.unsqueeze(0)
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    targets = [ClassifierOutputTarget(class_index)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    
    rgb_img = np.transpose(image_tensor.cpu().numpy(), (1, 2, 0))
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title(f'Original Image (Predicted: {"Real" if class_index == 1 else "Fake"})')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f'Grad-CAM for Class {"Real" if class_index == 1 else "Fake"}')
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize Grad-CAM for a trained Fuzzy Hesitant Ensemble")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to the trained ensemble checkpoint (.pt file)')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                       help='Paths to the base pruned model checkpoints (same as used in training)')
    parser.add_argument('--model_names', type=str, nargs='+', required=True,
                       help='Names for each model (same as used in training)')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to the input image for visualization')
    parser.add_argument('--model_index', type=int, default=0,
                       help='Index of the base model within the ensemble to visualize (default: 0)')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. بارگذاری مدل‌های پایه (بازسازی انسامبل)
    # این مقادیر باید دقیقا همان مقادیری باشند که در آموزش استفاده شده‌اند
    MEANS = [(0.5207, 0.4258, 0.3806), (0.4868, 0.3972, 0.3624), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2296, 0.2066, 0.2009), (0.2410, 0.2161, 0.2081)]
    
    MEANS = MEANS[:len(args.model_paths)]
    STDS = STDS[:len(args.model_paths)]
    
    base_models = load_pruned_models(args.model_paths, device)

    # 2. بازسازی مدل انسامبل
    ensemble = FuzzyHesitantEnsemble(
        base_models, MEANS, STDS,
        num_memberships=3, # این مقدار باید با مقدار استفاده شده در آموزش یکسان باشد
        freeze_models=True
    ).to(device)

    # 3. بارگذاری وزن‌های آموزش‌دیده بخش فازی
    if os.path.exists(args.checkpoint_path):
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        # در اسکریپت شما، فقط بخش hesitant_fuzzy ذخیره شده است
        ensemble.hesitant_fuzzy.load_state_dict(ckpt['hesitant_state_dict'])
        print("Trained hesitant fuzzy network loaded successfully.")
    else:
        print(f"Error: Checkpoint not found at {args.checkpoint_path}")
        return

    # 4. انتخاب مدل پایه برای مصورسازی
    if args.model_index >= len(base_models):
        print(f"Error: model_index {args.model_index} is out of range. Total models: {len(base_models)}")
        return
        
    model_to_visualize = base_models[args.model_index]
    model_name = args.model_names[args.model_index]
    print(f"\nVisualizing Grad-CAM for model: {model_name} (index {args.model_index})")

    # 5. پیدا کردن لایه هدف برای Grad-CAM
    # برای ResNet، معمولاً آخرین بلاک کانولوشنی مناسب است
    target_layers = [model_to_visualize.layer4[-1]]
    print(f"Target layer for Grad-CAM: {target_layers[0]}")

    # 6. بارگذاری و پیش‌پردازش تصویر
    # از همان ترنسفورمی که برای ارزیابی استفاده کرده‌اید، استفاده کنید
    # نرمال‌سازی با میانگین و انحراف معیار مدل انتخاب شده
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEANS[args.model_index], std=STDS[args.model_index])
    ])
    
    image_pil = Image.open(args.image_path).convert('RGB')
    image_tensor = transform(image_pil).to(device)

    # 7. پیش‌بینی کلاس تصویر توسط کل انسامبل
    ensemble.eval()
    with torch.no_grad():
        # تصویر را برای انسامبل نرمالایز نمی‌کنیم، چون خود انسامبل این کار را انجام می‌دهد
        unnormalized_transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()
        ])
        unnormalized_tensor = unnormalized_transform(image_pil).unsqueeze(0).to(device)
        
        output, weights = ensemble(unnormalized_tensor)
        prediction = torch.sigmoid(output).item()
        predicted_class_index = 1 if prediction > 0.5 else 0
        print(f"Ensemble Prediction: {'Real' if predicted_class_index == 1 else 'Fake'} (with probability {prediction:.2f})")

    # 8. فراخوانی تابع Grad-CAM برای مدل پایه انتخاب شده
    visualize_grad_cam(model_to_visualize, image_tensor, target_layers, predicted_class_index)

if __name__ == "__main__":
    main()
