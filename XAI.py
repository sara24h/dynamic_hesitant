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
# *** تغییر ۱: وارد کردن کلاس صحیح برای مسائل دو کلاسه ***
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- فقط کلاس مدل اصلی خود را وارد کنید ---
# مطمئن شوید این مسیر صحیح است
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

def visualize_grad_cam(model, image_tensor, target_layers, class_index, mean, std):
    """
    نمایش Grad-CAM برای یک مدل و یک تصویر خاص.
    """
    model.eval()
    input_tensor = image_tensor.unsqueeze(0)
    
    # ساخت نمونه Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # *** تغییر ۲: استفاده از BinaryClassifierOutputTarget ***
    # این کلاس برای مسائل دو کلاسه (مانند Real/Fake) مناسب است.
    targets = [BinaryClassifierOutputTarget(class_index)]
    
    # اجرای Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    
    # --- بخش اصلاح شده برای نمایش بهتر تصویر ---
    normalized_img = image_tensor.cpu().numpy()
    denormalized_img = np.transpose(normalized_img, (1, 2, 0))
    denormalized_img = denormalized_img * std + mean
    denormalized_img = np.clip(denormalized_img, 0, 1)
    
    rgb_img_for_cam = np.transpose(normalized_img, (1, 2, 0))
    visualization = show_cam_on_image(rgb_img_for_cam, grayscale_cam, use_rgb=True)
    
    # نمایش نتایج
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(denormalized_img)
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

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    print(f"Loading model from: {args.model_path}")
    try:
        ckpt = torch.load(args.model_path, map_location=device)
        model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    target_layers = [model.layer4[-1]]
    print(f"Target layer for Grad-CAM: {target_layers[0]}")

    MODEL_MEAN = (0.5207, 0.4258, 0.3806)
    MODEL_STD = (0.2490, 0.2239, 0.2212)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=MODEL_MEAN, std=MODEL_STD)
    ])
    
    image_pil = Image.open(args.image_path).convert('RGB')
    image_tensor = transform(image_pil).to(device)

    with torch.no_grad():
        model_output = model(image_tensor.unsqueeze(0))
        if isinstance(model_output, (tuple, list)):
            logits = model_output[0]
        else:
            logits = model_output
            
        prediction = torch.sigmoid(logits).item()
        predicted_class_index = 1 if prediction > 0.5 else 0
        print(f"Model Prediction: {'Real' if predicted_class_index == 1 else 'Fake'} (with probability {prediction:.2f})")

    visualize_grad_cam(model, image_tensor, target_layers, predicted_class_index, MODEL_MEAN, MODEL_STD)

if __name__ == "__main__":
    main()
