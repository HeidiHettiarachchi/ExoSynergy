"""
U-Net model architecture for CRISM hyperspectral mineral segmentation.
Implements encoder-decoder architecture with skip connections for precise
pixel-level semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from . import config


class DoubleConv(nn.Module):
    """
    Double convolution block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    This is the basic building block used throughout the U-Net architecture.
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling block: MaxPool -> DoubleConv
    Used in the encoder (contracting) path of the U-Net.
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block: Upsample -> Concat -> DoubleConv
    Used in the decoder (expanding) path with skip connections from encoder.
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super(Up, self).__init__()
        # Use transposed convolution for upsampling
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Feature map from decoder (lower resolution)
            x2: Feature map from encoder (higher resolution, skip connection)
        """
        # Upsample x1
        x1 = self.up(x1)
        
        # Handle size mismatches due to padding
        # Calculate difference in height and width
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        # Pad x1 to match x2's dimensions
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final output convolution: 1x1 conv to map features to class predictions.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation of hyperspectral data.
    
    Architecture:
    - Encoder: 4 downsampling blocks (contracting path)
    - Bottleneck: Central processing block
    - Decoder: 4 upsampling blocks with skip connections (expanding path)
    - Output: Final convolution to produce class logits
    
    The network processes input of shape (batch_size, num_bands, height, width)
    and outputs logits of shape (batch_size, num_classes, height, width).
    """
    
    def __init__(self, n_channels: int = None, n_classes: int = None, 
                 initial_features: int = None, dropout_rate: float = None):
        super(UNet, self).__init__()
        
        # Use config defaults if not specified
        self.n_channels = n_channels or config.NUM_BANDS
        self.n_classes = n_classes or config.NUM_CLASSES
        initial_features = initial_features or config.INITIAL_FEATURES
        dropout_rate = dropout_rate or config.DROPOUT_RATE
        
        # Initial convolution with higher capacity for hyperspectral data
        # Inspired by CRISM ML approaches for high-dimensional spectral data
        self.inc = DoubleConv(self.n_channels, initial_features, dropout_rate)
        
        # Encoder (contracting path)
        self.down1 = Down(initial_features, initial_features * 2, dropout_rate)
        self.down2 = Down(initial_features * 2, initial_features * 4, dropout_rate)
        self.down3 = Down(initial_features * 4, initial_features * 8, dropout_rate)
        self.down4 = Down(initial_features * 8, initial_features * 16, dropout_rate)
        
        # Decoder (expanding path)
        self.up1 = Up(initial_features * 16, initial_features * 8, dropout_rate)
        self.up2 = Up(initial_features * 8, initial_features * 4, dropout_rate)
        self.up3 = Up(initial_features * 4, initial_features * 2, dropout_rate)
        self.up4 = Up(initial_features * 2, initial_features, dropout_rate)
        
        # Output convolution
        self.outc = OutConv(initial_features, self.n_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input tensor of shape (batch_size, n_channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, n_classes, height, width)
        """
        # Encoder path with skip connections saved
        x1 = self.inc(x)     # Initial features
        x2 = self.down1(x1)  # 1/2 resolution
        x3 = self.down2(x2)  # 1/4 resolution
        x4 = self.down3(x3)  # 1/8 resolution
        x5 = self.down4(x4)  # 1/16 resolution (bottleneck)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)  # Skip from down3
        x = self.up2(x, x3)   # Skip from down2
        x = self.up3(x, x2)   # Skip from down1
        x = self.up4(x, x1)   # Skip from initial conv
        
        # Final classification layer
        logits = self.outc(x)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate class predictions from input.
        
        Args:
            x: Input tensor of shape (batch_size, n_channels, height, width)
            
        Returns:
            Predicted classes of shape (batch_size, height, width)
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract feature maps at different scales for visualization/analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature maps from encoder stages
        """
        features = []
        
        # Encoder path
        x1 = self.inc(x)
        features.append(x1)
        
        x2 = self.down1(x1)
        features.append(x2)
        
        x3 = self.down2(x2)
        features.append(x3)
        
        x4 = self.down3(x3)
        features.append(x4)
        
        x5 = self.down4(x4)
        features.append(x5)
        
        return features


def create_model(pretrained_path: str = None) -> UNet:
    """
    Factory function to create and optionally load a pre-trained U-Net model.
    
    Args:
        pretrained_path: Path to saved model weights (optional)
        
    Returns:
        UNet model instance
    """
    model = UNet()
    
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pre-trained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=config.DEVICE)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("Pre-trained weights loaded successfully")
    
    # Move model to device
    model = model.to(config.DEVICE)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully:")
    print(f"  - Input channels: {model.n_channels}")
    print(f"  - Output classes: {model.n_classes}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Device: {config.DEVICE}")
    
    return model


if __name__ == "__main__":
    # Test the model with dummy data
    import os
    
    print("Testing U-Net model...")
    
    # Create model
    model = create_model()
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, config.NUM_BANDS, 64, 64).to(config.DEVICE)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
        predictions = model.predict(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Unique predictions: {torch.unique(predictions)}")
    
    # Test feature extraction
    features = model.get_feature_maps(dummy_input)
    print(f"Number of feature map levels: {len(features)}")
    for i, feat in enumerate(features):
        print(f"  Level {i}: {feat.shape}")
    
    print("Model test completed successfully!")
