import os
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torch.nn import functional as F
import torch
# ==========================================
# 1. Shared Backbone: MobileNetV2 Encoder
# ==========================================
class MobileNetV2Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Instantiate the bare architecture with NO weights initially
        base_model = mobilenet_v2(weights=None)
        
        if pretrained:
            # 1. Dynamically resolve the absolute path to checkpoints/pretrained
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            ckpt_dir = os.path.join(project_root, "checkpoints", "pretrained")
            
            # Ensure the directory exists
            os.makedirs(ckpt_dir, exist_ok=True)
            
            # The exact URL for the official MobileNetV2 weights
            weight_url = "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth"
            
            # 2. Smart Loading Mechanism:
            # PyTorch will automatically check if the .pth file exists in 'model_dir'.
            # If YES: It directly loads from your local checkpoints/pretrained/ folder.
            # If NO: It downloads it, saves it to that folder, and then loads it.
            state_dict = torch.hub.load_state_dict_from_url(
                weight_url, 
                model_dir=ckpt_dir, 
                file_name="mobilenet_v2.pth",
                progress=True
            )
            
            # Load the weights into our bare model
            base_model.load_state_dict(state_dict)
            print(f"✅ MobileNetV2 weights loaded successfully from: {ckpt_dir}")

        # Extract features from the initialized model
        features = base_model.features
        
        # Extract features at different scales
        self.layer1 = features[:4]   # Output channels: 24
        self.layer2 = features[4:7]  # Output channels: 32
        self.layer3 = features[7:14] # Output channels: 96
        
        # Stop at index 18 to exclude the final 1x1 expansion conv layer
        # This ensures the output is exactly 320 channels, not 1280.
        self.layer4 = features[14:18] 

    def forward(self, x):
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return l1, l2, l3, l4