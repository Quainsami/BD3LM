# models/controller.py
import torch
import torch.nn as nn

class MetaController(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2_s = nn.Linear(hidden_dim, 1) # For s_b (scale)
        self.fc2_o = nn.Linear(hidden_dim, 1) # For o_b (offset)

    def forward(self, block_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            block_features (torch.Tensor): Shape (batch_size, num_blocks, feature_dim)
        Returns:
            s_params (torch.Tensor): Shape (batch_size, num_blocks, 1)
            o_params (torch.Tensor): Shape (batch_size, num_blocks, 1)
        """
        x = self.fc1(block_features)
        x = self.activation(x)
        x = self.dropout(x)
        
        s_params = self.fc2_s(x)
        o_params = self.fc2_o(x)
        
        return s_params, o_params