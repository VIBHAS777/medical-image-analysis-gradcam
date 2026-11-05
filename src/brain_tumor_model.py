import os
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

def create_and_save_model():
    print("Generating PyTorch VGG16 Brain Tumor model...")
    # Load pretrained VGG16
    weights = VGG16_Weights.IMAGENET1K_V1
    model = vgg16(weights=weights)
    
    # Freeze feature extraction layers for standard fine-tuning pattern
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Replace classifier head for 4 classes: Glioma, Meningioma, No Tumor, Pituitary
    # VGG16's classifier is a Sequential block. We replace the last linear layer.
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 4)
    
    # Save the state_dict (better practice than saving whole model)
    # Actually, to make it easier for model_service (like we did for skin), 
    # we can just write a wrapper or save the state dict.
    # Let's write the loading function right in this file.
    
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "brain_tumor_vgg16.pth")
    
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved model weights to {save_path} (size: {os.path.getsize(save_path)/(1024*1024):.1f} MB)")

def load_brain_tumor_model(path):
    """Utility to load the model used by model_service.py"""
    model = vgg16(weights=None)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 4)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model

if __name__ == "__main__":
    create_and_save_model()
