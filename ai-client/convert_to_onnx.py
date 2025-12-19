"""
PyTorch to ONNX Model Converter
================================
Run this script ONCE in 64-bit Python environment to convert your models.

Requirements (64-bit Python):
pip install torch onnx

Usage:
python convert_to_onnx.py
"""

import torch
import torch.nn as nn
import os

# Import your original model definitions
from models import FogPredictor

class TinyQNet(nn.Module):
    """Same as your original DQN model"""
    def __init__(self, H, W, num_actions=5):
        super().__init__()
        self.H = H
        self.W = W
        self.num_actions = num_actions
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, num_actions, 1),
        )

    def forward(self, x):
        return self.conv(x)


def convert_dqn_to_onnx(pytorch_model_path: str, 
                        onnx_output_path: str, 
                        map_size: int = 20):
    """
    Convert DQN PyTorch model to ONNX format
    
    Args:
        pytorch_model_path: Path to .pth file
        onnx_output_path: Path to save .onnx file
        map_size: Map dimensions
    """
    print(f"Converting DQN model: {pytorch_model_path}")
    
    # Load PyTorch model
    model = TinyQNet(map_size, map_size)
    model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu'))
    model.eval()
    
    # Create dummy input (batch_size=1, channels=2, height=map_size, width=map_size)
    dummy_input = torch.randn(1, 2, map_size, map_size)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=11,  # Use opset 11 for better compatibility
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ DQN model saved to: {onnx_output_path}")
    
    # Verify the model
    import onnx
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified successfully")


def convert_fog_predictor_to_onnx(pytorch_model_path: str, 
                                   onnx_output_path: str, 
                                   map_size: int = 20,
                                   seq_len: int = 3):
    """
    Convert Fog Predictor PyTorch model to ONNX format
    
    Args:
        pytorch_model_path: Path to .pth file
        onnx_output_path: Path to save .onnx file
        map_size: Map dimensions
        seq_len: Sequence length
    """
    print(f"\nConverting Fog Predictor model: {pytorch_model_path}")
    
    # Define fog classes (same as in your code)
    fog_classes = [-2, -1, 0, 1, 2, 100, 101, 102, 201, 202]
    num_classes = len(fog_classes)
    
    # Load PyTorch model
    model = FogPredictor(map_size, map_size, fog_classes)
    model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu'))
    model.eval()
    
    # Create dummy input (batch_size=1, seq_len=3, num_classes=10, height, width)
    dummy_input = torch.randn(1, seq_len, num_classes, map_size, map_size)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Fog Predictor saved to: {onnx_output_path}")
    
    # Verify
    import onnx
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified successfully")


def main():
    """Main conversion function"""
    print("=" * 60)
    print("PyTorch to ONNX Model Converter")
    print("=" * 60)
    
    # Configuration
    map_size = 20
    models_dir = "models"
    
    # Ensure output directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Convert DQN model
    dqn_pytorch = f"{models_dir}/dqn_agent_20x20.pth"
    dqn_onnx = f"{models_dir}/dqn_agent_20x20.onnx"
    
    if os.path.exists(dqn_pytorch):
        convert_dqn_to_onnx(dqn_pytorch, dqn_onnx, map_size)
    else:
        print(f"⚠ Warning: DQN model not found at {dqn_pytorch}")
        print("  Skipping DQN conversion")
    
    # Convert Fog Predictor model
    fog_pytorch = f"{models_dir}/fog_predictor_20x20.pth"
    fog_onnx = f"{models_dir}/fog_predictor_20x20.onnx"
    
    if os.path.exists(fog_pytorch):
        convert_fog_predictor_to_onnx(fog_pytorch, fog_onnx, map_size)
    else:
        print(f"\n⚠ Warning: Fog Predictor not found at {fog_pytorch}")
        print("  Skipping Fog Predictor conversion")
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Copy the .onnx files to your 32-bit Python environment")
    print("2. Install: pip install onnxruntime numpy")
    print("3. Use the ONNX version of the agent in LabVIEW")
    print("\nConverted files:")
    if os.path.exists(dqn_onnx):
        print(f"  - {dqn_onnx}")
    if os.path.exists(fog_onnx):
        print(f"  - {fog_onnx}")


if __name__ == "__main__":
    main()