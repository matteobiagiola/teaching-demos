import torch
import torch.nn as nn
from typing import Callable, Dict, Tuple

class NeuronCoverage:
    """
    Tracks neuron coverage across chosen layers.
    By default, for Conv layers we collapse spatial dims via max -> one value per channel.
    """
    def __init__(
        self, 
        model,
        device: str = "cpu",
        threshold: float = 0.0, 
        conv_reduce: str = "max"
    ) -> None:
        """
        Args:
            model: nn.Module
            device: "cuda" or "cpu"
            threshold: absolute threshold after ReLU (or raw outputs if no ReLU). You can tune this.
                       A common variant is to use a relative threshold based on activation stats; see notes below.
            conv_reduce: "max" | "mean" | "none"
                - "max"  -> one value per channel (spatial max)
                - "mean" -> one value per channel (spatial mean)
                - "none" -> treat each spatial location as a separate neuron (can be huge)
        """
        self.model = model.to(device).eval()
        self.device = device
        self.threshold = float(threshold)
        self.conv_reduce = conv_reduce
        
        assert conv_reduce in ("max", "mean", "none"), "Invalid conv_reduce"
            
        self.layers = []
        
        for name, m in self.model.named_modules():
            
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.layers.append((name, m))
                
        # excluding the last layer (usually softmax)
        if self.layers:
            self.layers = self.layers[:-1]

        self._handles = []
        self._last_batch_activations = {}   # temp activations captured by hooks
        self._neuron_counts = {}     # total neurons per layer
        self._covered = {}           # boolean coverage per layer (1D bool tensor)

        # Register forward hooks
        for lname, module in self.layers:
            h = module.register_forward_hook(self._hook_maker(lname))
            self._handles.append(h)

    def _hook_maker(self, lname: str) -> Callable[[nn.Module, torch.Tensor], None]:
        
        def hook(module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
            
            # input is not used in this case (but it is needed for the function signature)
            # as we are measuring post-activation outputs
            
            with torch.no_grad():
                # output shape handling
                x = output
                
                if isinstance(x, (tuple, list)):
                    x = x[0]
                # Move to CPU for bookkeeping
                x = x.detach().to("cpu")

                if isinstance(module, nn.Conv2d):
                    # x: [B, C, H, W]
                    if self.conv_reduce == "max":
                        # one value per (B, C): spatial max (equivalent to x.amax(dim=(2, 3)))
                        x = x.amax(dim=(-1, -2))  # [B, C]
                    elif self.conv_reduce == "mean":
                        x = x.mean(dim=(-1, -2))  # [B, C]
                    elif self.conv_reduce == "none":
                        # Treat each spatial location as a neuron (B,C,H,W).
                        # We'll flatten to [B, C*H*W] below.
                        pass
                # Linear: x is [B, N] ideally (if extra dims, flatten)
                # ReLU: same as its input shape; we treat each unit independently.

                # Flatten all but batch
                if x.dim() > 2:
                    x = x.flatten(start_dim=1)  # [B, N]

                self._last_batch_activations[lname] = x  # [B, N]
                
        return hook

    def _ensure_initialized_for_batch(self) -> None:
        
        # After first hook call(s), initialize coverage shapes
        for lname, _ in self.layers:
            
            if lname in self._neuron_counts:
                continue
            
            if lname not in self._last_batch_activations:
                # layer might not have fired yet (unused path); skip for now
                continue
            
            _, N = self._last_batch_activations[lname].shape
            self._neuron_counts[lname] = N
            self._covered[lname] = torch.zeros(N, dtype=torch.bool)

    @torch.no_grad()
    def update_with_batch(self) -> None:
        
        # Initialize coverage vectors on first batch
        self._ensure_initialized_for_batch()

        for lname, _ in self.layers:
            
            if lname not in self._last_batch_activations:
                continue
            
            activations = self._last_batch_activations[lname]  # [B, N]
            # Threshold check
            fired = (activations > self.threshold)       # [B, N] bool
            fired_any = fired.any(dim=0)          # [N] bool
            
            if lname not in self._covered:
                self._covered[lname] = fired_any.clone()
                self._neuron_counts[lname] = activations.shape[1]
            else:
                self._covered[lname] |= fired_any

        # Clear stash to avoid accidental reuse
        self._last_batch_activations.clear()

    def coverage_by_layer(self) -> Dict[str, Tuple[int, int, float]]:
        
        out = {}
        
        for lname in self._covered:
            covered = int(self._covered[lname].sum().item())
            total = int(self._neuron_counts[lname])
            out[lname] = (covered, total, covered / max(total, 1))
            
        return out

    def overall_coverage(self) -> Tuple[int, int, float]:
        
        covered = 0
        total = 0
        
        for lname in self._covered:
            covered += int(self._covered[lname].sum().item())
            total   += int(self._neuron_counts[lname])
            
        return covered, total, covered / max(total, 1)

    def close(self) -> None:
        
        for h in self._handles:
            h.remove()
            
        self._handles.clear()
