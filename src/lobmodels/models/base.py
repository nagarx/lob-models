"""
Base model class for all LOB models.

Provides a consistent interface for:
- Forward pass
- Model information
- Configuration access

Design Principles (RULE.md):
- Consistent interface across models (§3)
- Configuration-driven (§4)
- Explicit output types (§1)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import torch
import torch.nn as nn

from lobmodels.config import BaseConfig


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all LOB prediction models.
    
    All models should inherit from this class and implement:
    - forward(): The forward pass
    - name property: Human-readable model name
    
    Provides common functionality:
    - Parameter counting
    - Configuration access
    - Model info dictionary
    """
    
    def __init__(self, config: BaseConfig):
        super().__init__()
        self._config = config
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (shape depends on model)
        
        Returns:
            Logits tensor [batch, num_classes]
            
        Note:
            Returns raw logits (pre-softmax). Use CrossEntropyLoss which
            applies log_softmax internally, or apply softmax for probabilities.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name for logging."""
        pass
    
    @property
    def config(self) -> BaseConfig:
        """Model configuration."""
        return self._config
    
    @property
    def num_parameters(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get model information dictionary.
        
        Returns:
            Dict with:
            - name: Model name
            - num_params: Total parameters
            - trainable_params: Trainable parameters
            - config: Configuration dict
        """
        return {
            "name": self.name,
            "num_params": self.num_parameters,
            "trainable_params": self.num_trainable_parameters,
            "config": self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config),
        }
    
    def freeze(self) -> None:
        """Freeze all parameters (for transfer learning)."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

