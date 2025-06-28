# models.py
"""
Contains definitions for machine learning models used in the RedShield AI system.
This modularization improves code organization and maintainability.
"""
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    logger.info("PyTorch found. TCNN model is available.")
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not found. TCNN model will be disabled. For full functionality, install with 'pip install torch'.")


class TCNN(nn.Module if TORCH_AVAILABLE else object):
    """
    Temporal Convolutional Network (TCNN) for multi-horizon time-series forecasting.
    A TCNN uses dilated convolutions to capture long-range temporal dependencies efficiently.
    It processes all zones' features at once to make predictions.
    """
    def __init__(self, input_size: int, output_size: int, num_channels: list, kernel_size: int = 2, dropout: float = 0.2):
        if not TORCH_AVAILABLE:
            # This check prevents instantiation if torch is not available.
            raise ImportError("Cannot initialize TCNN model, PyTorch is not installed.")
        super(TCNN, self).__init__()

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Causal Convolution: padding ensures output length is same as input and it doesn't 'see the future'.
            # We treat the zones as the 'sequence length' in this interpretation.
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=(kernel_size - 1) * dilation_size, 
                             dilation=dilation_size)
            
            layers.append(nn.utils.weight_norm(conv))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)
        # The output from the Conv1D will be (batch_size, num_channels[-1], num_zones).
        # We need to flatten this to predict for all zones at once.
        self.flatten = nn.Flatten()
        # The linear layer size must match the flattened output size. Assume num_zones=3 for default config.
        # This is a simplification; a more robust approach would pass num_zones as a parameter.
        # For now, let's assume the output is pooled or averaged.
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TCNN.
        Args:
            x: Input tensor of shape (batch_size, num_zones, num_features)
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Input to Conv1d should be (batch, channels/features, seq_len/zones)
        y = self.network(x.permute(0, 2, 1))
        
        # Use adaptive pooling to handle variable number of zones and get a fixed-size representation
        y_pooled = self.final_pool(y).squeeze(-1)

        return self.linear(y_pooled)
