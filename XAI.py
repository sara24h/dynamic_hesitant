import torch
import torch.nn as nn
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# کتابخانه‌های مورد نیاز برای Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- فقط کلاس مدل اصلی خود را وارد کنید ---
# مطمئن شوید این مسیر صحیح است
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

def visualize_grad_cam(model, image_tensor, target_layers, class_index, mean, std):
    """
    نمایش Grad-CAM برای یک مدل و یک تصویر خاص.
    تصویر اصلی برای نمایش بهتر دِنورمالایز می‌شود.
    """
    model.eval()
    input_tensor = image_tensor.unsqueeze(0)
    
    # ساخت نمونه Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    
    # تعریف هدف: کلاسی که می‌خواهیم توضیح برای آن نمایش داده شود
    targets = [ClassifierOutputTarget(class_index)]
    
    # اجرای Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    
    # --- بخش اصلاح شده برای نمایش بهتر تصویر ---
    # 1. تبدیل تنسور نرمال‌شده به numpy
    normalized_img = image_tensor.cpu().numpy()
    
    # 2. دِنورمالایز کردن تصویر برای نمایش
    # فرمول: (img * std) + mean
    denormalized_img = np.transpose(normalized_img, (1, 2, 0)) # به [H, W, C] تبدیل کن
    denormalized_img = denormalized_img * std + mean
    denormalized_img = np.clip(denormalized_img, 0, 1) # مقادیر را بین 0 و 1 محدود کن
    
    # 3. ادغام نقشه حرارتی با تصویر اصلی (از نسخه نرمال‌شده استفاده می‌کنیم چون show_cam_on_image انتظار این را دارد)
    rgb_img_for_cam = np.transpose(normalized_img, (1, 2, 0))
    visualization = show_cam_on_image(rgb_img_for_cam, grayscale_cam, use_rgb=True)
    
    # نمایش نتایج
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(denormalized_img) # نمایش تصویر دنورمالایز شده
    plt.title(f'Original Image (Predicted: {"Real" if class_index == 1 else "Fake"})')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f'Grad-CAM for Class {"Real" if class_index == 1 else "Fake"}')
    plt.axis('off')
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize Grad-CAM for a single pre-trained model")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the pre-trained model checkpoint (.pt file)')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to the input image for visualization')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. بارگذاری مدل تکی
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    print(f"Loading model from: {args.model_path}")
    try:
        ckpt = torch.load(args.model_path, map_location=device)
        # مدل را با ماسک‌های ذخیره شده می‌سازیم
        model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
        # وزن‌های مدل را بارگذاری می‌کنیم
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        model.eval() # قرار دادن مدل در حالت ارزیابی
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. پیدا کردن لایه هدف (Target Layer)
    # برای ResNet، معمولاً آخرین بلاک کانولوشنی بهترین نتایج را می‌دهد.
    # برای ResNet50، این لایه 'layer4' است.
    target_layers = [model.layer4[-1]]
    print(f"Target layer for Grad-CAM: {target_layers[0]}")

    # 3. بارگذاری و پیش‌پردازش تصویر
    # مهم: از همان ترنسفورم و نرمال‌سازی استفاده کنید که برای آموزش این مدل خاص استفاده شده است.
    # شما باید مقادیر mean و std متناسب با مدل خودتان را جایگزین کنید.
    MODEL_MEAN = (0.5207, 0.4258, 0.3806)  # <-- مقادیر خود را اینجا قرار دهید
    MODEL_STD = (0.2490, 0.2239, 0.2212)   # <-- مقادیر خود را اینجا قرار دهید

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=MODEL_MEAN, std=MODEL_STD)
    ])
    
    image_pil = Image.open(args.image_path).convert('RGB')
    image_tensor = transform(image_pil).to(device)

    # 4. پیش‌بینی کلاس تصویر توسط مدل
    # کد صحیح با تورفتگی اصلاح شده
    with torch.no_grad():
        model_output = model(image_tensor.unsqueeze(0))
        
        # چک می‌کنیم که آیا مدل یک تاپل برمی‌گرداند یا نه
        if isinstance(model_output, (tuple, list)):
            # اگر تاپل بود، معمولاً اولین عنصر آن خروجی اصلی (logits) است
            logits = model_output[0]
        else:
            # اگر تاپل نبود، از همان خروجی استفاده می‌کنیم
            logits = model_output
            
        prediction = torch.sigmoid(logits).item()
        predicted_class_index = 1 if prediction > 0.5 else 0
        print(f"Model Prediction: {'Real' if predicted_class_index == 1 else 'Fake'} (with probability {prediction:.2f})")

    # 5. فراخوانی تابع Grad-CAM
    # توضیح را برای کلاسی که مدل پیش‌بینی کرده نمایش می‌دهیم
    visualize_grad_cam(model, image_tensor, target_layers, predicted_class_index, MODEL_MEAN, MODEL_STD)

if __name__ == "__main__":
    main()
