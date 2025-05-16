import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

class MetaController(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config.algo.meta_controller # Specific config for the controller
        
        feature_dim = self.config.feature_dim
        hidden_dim = self.config.hidden_dim
        
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1) # Output one scalar: log_s_tilde

    def forward(self, block_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            block_features: Tensor of shape (batch_size, num_blocks, feature_dim)
        Returns:
            log_s_tilde_params: Tensor of shape (batch_size, num_blocks, 1)
        """
        x = self.fc1(block_features)
        x = self.activation(x)
        log_s_tilde_params = self.fc2(x)
        return log_s_tilde_params

    def get_s_b(self, log_s_tilde_params: torch.Tensor) -> torch.Tensor:
        """
        Derives s_b from log_s_tilde_params.
        s_b = softplus(tanh(log_s_tilde) * C_s) + eps_s
        """
        s_b = F.softplus(
            torch.tanh(log_s_tilde_params) * self.config.s_tilde_squash_factor
        ) + self.config.s_min_epsilon
        return s_b