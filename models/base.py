from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor


def check_type(module, expected_type):
    if hasattr(module, "unwrapped_module"):
        assert isinstance(module.unwrapped_module, expected_type), \
            f"{type(module.unwrapped_module)} != {expected_type}"
    else:
        assert isinstance(module, expected_type), f"{type(module)} != {expected_type}"

class BaseModel(nn.Module):
    """Base class for src models."""

    def __init__(self):
        super().__init__()
    
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
    
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        raise NotImplementedError("Model must implement the build_model method")
    
    def get_targets(self, sample, net_output):
        """get targets from either the sample or the net's output."""
        return sample["target"]
    
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        if torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim = -1)
            else:
                return F.softmax(logits, dim = -1)
        raise NotImplementedError
    
    def extract_features(self, *args, **kwargs):
        """Similar to *forward* but only return features."""
        return self(*args, **kwargs)
    
    def upgrade_state_dict(self, state_dict):
        """Upgrade old state dicts to work with newer code."""
        self.upgrade_state_dict_named(state_dict, "")
    
    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code.

        Args:
            state_dict (dict): state dictionary to upgrade, in place
            name (str): the state dict key corresponding to the current module
        """
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if len(prefix) > 0:
                prefix += "."
            
            for n, c in m.named_children():
                name = prefix + n
                if hasattr(c, "upgrade_state_dict_named"):
                    c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, "upgrade_state_dict"):
                    c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)
        
    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        for m in self.modules():
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)
                
                
class PretrainingModel(BaseModel):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrate a (possibly old) state dict for new versions."""
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        raise NotImplementedError("Model must implement the build_model method")

    @classmethod
    def from_pretrained(cls, **kwargs):
        """
        Load a :class:`~src.models.PretrainingModel` from a pre-trained model
        checkpoint.
        """
        raise NotImplementedError("PretrainingModel must implement the from_pretrained method")

    def extract_features(self, **kwargs):
        raise NotImplementedError()

    def get_logits(self, **kwargs):
        raise NotImplementedError()
    
    def get_targets(self, **kwargs):
        raise NotImplementedError()
    
    def forward(self, **kwargs):
        raise NotImplementedError()
    