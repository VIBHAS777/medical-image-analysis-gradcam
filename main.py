import torch
import numpy as np
from src.models import build_medical_model
from src.gradcam import GradCAM

BRAIN_CLASSES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]
SKIN_CLASSES = ["Melanoma", "Basal Cell Carcinoma", "Benign Keratosis", "Melanocytic Nevus"]

def main():
    print("=" * 70)
    print("  Medical Image Analysis & Grad-CAM Explainability Pipeline")
    print("  Publication: 3rd IEEE ICITEICS 2026")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    # 1. Brain Tumor Diagnostic Demonstration
    print("\n[1] Diagnostic Pipeline: Brain Tumor MRI (VGG16 Backbone)")
    brain_model = build_medical_model("vgg16", num_classes=4).to(device)
    
    # Target last conv layer in VGG16 features
    target_layer = brain_model.features[-1]
    cam_engine = GradCAM(brain_model, target_layer)

    # Synthetic sample MRI tensor (1, 3, 224, 224)
    dummy_mri = torch.randn(1, 3, 224, 224, device=device)
    
    output = brain_model(dummy_mri)
    probs = torch.softmax(output, dim=1)[0].cpu().detach().numpy()
    pred_class_idx = int(np.argmax(probs))

    print(f"    - Diagnosed Class  : {BRAIN_CLASSES[pred_class_idx]}")
    print(f"    - Model Confidence : {probs[pred_class_idx]*100:.2f}%")
    
    heatmap = cam_engine.generate_heatmap(dummy_mri, class_idx=pred_class_idx)
    print(f"    - Generated Grad-CAM Heatmap matrix shape : {heatmap.shape}")
    print(f"    - Heatmap Intensity [Min, Max]            : [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    print("\n[2] Diagnostic Pipeline: ISIC Dermoscopic Skin Lesion (ResNet-50)")
    skin_model = build_medical_model("resnet50", num_classes=4).to(device)
    dummy_skin = torch.randn(1, 3, 224, 224, device=device)
    skin_probs = torch.softmax(skin_model(dummy_skin), dim=1)[0].cpu().detach().numpy()
    pred_skin_idx = int(np.argmax(skin_probs))
    
    print(f"    - Diagnosed Class  : {SKIN_CLASSES[pred_skin_idx]}")
    print(f"    - Model Confidence : {skin_probs[pred_skin_idx]*100:.2f}%")

    print("\n" + "=" * 70)
    print("  CLINICAL BENCHMARK METRICS (From Published Paper)")
    print("=" * 70)
    print("  Brain Tumor MRI Dataset  : 94.12% Accuracy | 0.993 Macro-AUC")
    print("  ISIC Skin Lesion Dataset : 99.04% Accuracy | 0.988 F1-Score")
    print("=" * 70)

if __name__ == "__main__":
    main()
