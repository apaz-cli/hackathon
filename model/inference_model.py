from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import triton.testing

from .mamba2 import Mamba2
from .retnet import RetNet
from .hawk import Hawk
from .xlstm import XLSTM
from .modules import SimpleMLP


class LayerType(Enum):
    MAMBA2 = "mamba2"
    RETNET = "retnet"
    HAWK = "hawk"
    XLSTM = "xlstm"


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    hidden_size: int
    proj_size: int
    tower_depth: int
    num_heads: int
    num_features: int


def create_layer(layer_type: LayerType, hidden_size: int, num_heads: int = 8):
    """Factory function to create baseline models

    Args:
        layer_type: Type of layer to create
        hidden_size: Hidden dimension
        num_heads: Number of attention heads (for models that use it)
    """
    match layer_type:
        case LayerType.MAMBA2:
            return Mamba2(hidden_size=hidden_size)
        case LayerType.RETNET:
            return RetNet(hidden_size=hidden_size, num_heads=num_heads)
        case LayerType.HAWK:
            return Hawk(hidden_size=hidden_size)
        case LayerType.XLSTM:
            return XLSTM(
                hidden_size=hidden_size,
                mlstm_num_heads=num_heads,
                slstm_num_heads=num_heads // 2,
            )
        case _:
            raise ValueError(f"Unknown layer type: {layer_type}.")


class Block(nn.Module):
    def __init__(self, layer_type: LayerType, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size)
        self.layer = create_layer(layer_type, config.hidden_size, config.num_heads)
        self.norm2 = nn.RMSNorm(config.hidden_size)
        self.mlp = SimpleMLP(config.hidden_size, config.proj_size)

    def forward(self, x: torch.Tensor, state):
        skip = x
        x, new_state = self.layer(self.norm1(x), state)
        x = x + skip
        x = x + self.mlp(self.norm2(x))
        return x, new_state

    def init_state(self, batch_size: int, device: torch.device):
        return self.layer.init_state(batch_size, device)


class Tower(nn.Module):
    def __init__(self, config: ModelConfig, layer_type):
        super().__init__()

        self.input_up = nn.Linear(config.num_features, config.proj_size)
        self.input_down = nn.Linear(config.proj_size, config.hidden_size)

        self.blocks = nn.ModuleList(
            [Block(layer_type, config) for _ in range(config.tower_depth)]
        )

    def forward(self, x: torch.Tensor, state):
        new_state = []

        x = self.input_up(x)
        x = nn.functional.relu(x)
        x = self.input_down(x)

        for block, block_state in zip(self.blocks, state, strict=True):
            x, new_block_state = block(x, block_state)
            new_state.append(new_block_state)
        return x, new_state

    def init_state(self, batch_size: int, device: torch.device):
        return [
            block.init_state(batch_size, device) for block in self.blocks
        ]  # pyright: ignore[reportCallIssue]


class MultiTowerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        layer_types = [
            LayerType.XLSTM,
            LayerType.MAMBA2,
            LayerType.RETNET,
            LayerType.HAWK,
        ]

        self.towers = nn.ModuleList(
            [Tower(config, layer_type) for layer_type in layer_types]
        )
        self.layer_types = layer_types  # Store for profiling
        self.output_proj = nn.Linear(config.hidden_size, 1)
        self.profile_enabled = False
        self.last_tower_timings = None

    def enable_profiling(self, enable: bool = True):
        self.profile_enabled = enable

    def forward(self, x: torch.Tensor, state):
        if self.profile_enabled:
            # Profile each tower separately
            results = []
            tower_timings = []
            
            for i, (tower_state, tower) in enumerate(zip(state, self.towers, strict=True)):
                tower_name = self.layer_types[i].value
                
                def tower_forward():
                    return tower(x, tower_state)
                
                # Benchmark the tower
                timing_ms = triton.testing.do_bench(tower_forward, warmup=10, rep=100)
                tower_timings.append((tower_name, timing_ms))
                
                # Get the actual result
                result = tower_forward()
                results.append(result)
            
            # Store timing results instead of printing
            self.last_tower_timings = dict(tower_timings)
            
            xs, new_state = zip(*results, strict=True)
        else:
            # Normal forward pass without profiling
            results = [
                tower(x, tower_state)
                for tower_state, tower in zip(state, self.towers, strict=True)
            ]
            xs, new_state = zip(*results, strict=True)
            self.last_tower_timings = None

        xs = [self.output_proj(x) for x in xs]
        return torch.concat(xs, dim=1), list(new_state)
    
    def get_last_tower_timings(self):
        """Return the timing results from the last forward pass"""
        return self.last_tower_timings

    def init_state(self, batch_size, device):
        return [tower.init_state(batch_size, device) for tower in self.towers]
