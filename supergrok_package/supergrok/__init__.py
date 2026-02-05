"""
SuperGrok: Ultimate Optimizer for Generalization

Combines the best of:
- NeuralGrok: Learned gradient transformation via neural amplifier
- Grokfast: EMA-based slow gradient amplification  
- GrokAdamW: AdamW base with adaptive alpha and layer-wise β₁ decay

Key innovations:
- Decoupled memory: mu buffer tracks raw gradients, not transformed ones
- Tensor-wise cosine gating: Binary stability clutch based on alignment
- Zero-loss fix: Clears momentum buffer post-convergence
- Cached adaptive alpha: Lazy updates requiring explicit val_loss

Usage:
    from supergrok import SuperGrok
    
    optimizer = SuperGrok(model.parameters(), lr=1e-3)

Author: SuperGrok Development
License: Apache 2.0
"""

__version__ = "1.0.0"
__all__ = ["SuperGrok", "SimpleMetaNet"]

import math
from typing import Optional, Callable, Iterable, Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


class SimpleMetaNet(nn.Module):
    """
    Lightweight gradient transformer for SuperGrok.
    
    Implements NeuralGrok-style mask transformation:
        output = softmax(MLP(grad)) * rescale * grad
    
    Uses learned base scaling + dynamic norm correction (Option C).
    
    The MLP processes each gradient element independently (shared weights),
    making it efficient for arbitrary parameter shapes.
    
    Args:
        hidden_dim: Hidden dimension of the MLP (default: 32)
    
    Example:
        >>> meta_net = SimpleMetaNet(hidden_dim=32)
        >>> grad = torch.randn(100, 50)
        >>> smart_grad = meta_net(grad)
        >>> smart_grad.shape
        torch.Size([100, 50])
    """
    
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 3-layer MLP for per-element transformation
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Learned base rescaling coefficient
        self.rescale = nn.Parameter(torch.tensor(1.0))
        
        # Initialize weights for stable training
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
    
    def forward(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Transform gradient through learned mask.
        
        Args:
            grad: Gradient tensor of any shape
            
        Returns:
            Transformed gradient (same shape as input)
        """
        if grad.numel() == 0:
            return grad
            
        original_shape = grad.shape
        
        # Flatten to (N, 1) for per-element processing
        flat = grad.view(-1, 1)
        
        # Compute mask logits through MLP
        mask_logits = self.net(flat)  # (N, 1)
        mask_logits = mask_logits.view(-1)  # (N,)
        
        # Softmax across all elements to get probability mask
        # Use numerically stable softmax
        mask = F.softmax(mask_logits, dim=0)  # (N,)
        mask = mask.view(original_shape)  # Restore shape
        
        # Apply mask to gradient
        masked_grad = mask * grad
        
        # Dynamic norm correction: preserve original gradient magnitude
        grad_norm = torch.norm(grad)
        masked_norm = torch.norm(masked_grad)
        
        # Avoid division by zero
        eps = 1e-8
        if masked_norm > eps:
            dynamic_scale = grad_norm / (masked_norm + eps)
        else:
            dynamic_scale = 1.0
        
        # Combined rescaling: learned base * dynamic correction (Option C)
        c = self.rescale * dynamic_scale
        
        # Final output
        smart_grad = c * masked_grad
        
        return smart_grad
    
    def __repr__(self) -> str:
        return f"SimpleMetaNet(hidden_dim={self.hidden_dim}, rescale={self.rescale.item():.4f})"


class SuperGrok(Optimizer):
    """
    SuperGrok: Ultimate Optimizer for Generalization
    
    Combines:
    - NeuralGrok: Learned gradient transformation via meta_net
    - Grokfast: EMA-based slow gradient amplification
    - GrokAdamW: AdamW base with adaptive alpha and layer-wise β₁ decay
    - Cosine Gating: Stability clutch based on alignment between smart_grad and momentum
    
    Key Design Principles:
    - Rule 1 (Decoupled Memory): The Grokfast buffer (mu) is updated using RAW 
      gradients only, never the transformed gradients. This keeps the momentum 
      track stable and unpolluted by meta-net fluctuations.
    
    - Rule 2 (Cosine Gating): Compute tensor-wise cosine similarity between 
      smart_grad and mu. If aligned (>0), apply full Grokfast amplification.
      If misaligned (<=0), disengage momentum and use only smart_grad.
    
    - Rule 3 (Zero-Loss Fix): When train_loss approaches zero, use a high 
      grokking signal to decay alpha, clearing the buffer to prevent drift.
    
    - Adaptive Alpha (Cached/Lazy): Alpha only updates when val_loss is 
      explicitly provided. Otherwise, the cached value is reused.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for Adam moving averages (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: Decoupled weight decay coefficient (default: 0.01)
        alpha_init: Initial EMA momentum for Grokfast buffer (default: 0.98)
        lamb: Amplification factor for slow gradients (default: 2.0)
        gamma: Layer-wise β₁ decay rate (default: 0.1)
        kappa: Grokking signal decay rate for adaptive alpha (default: 0.1)
        warmup_steps: Steps before enabling gating/amplification (default: 100)
        gradient_clipping: Max gradient norm, 0 to disable (default: 1.0)
        alpha_update_freq: Steps between alpha recalculations (default: 100)
        meta_net: Optional nn.Module for gradient transformation. If None,
                  a SimpleMetaNet is automatically created (default: None)
        meta_hidden_dim: Hidden dimension for auto-created SimpleMetaNet (default: 32)
    
    Note on Meta-Net Training:
        The meta_net parameters are accessible via `optimizer.meta_net.parameters()`.
        For bilevel optimization, create a separate optimizer for the meta_net:
        
        >>> meta_optimizer = torch.optim.Adam(optimizer.meta_net.parameters(), lr=1e-4)
    
    Example (Simple - like any other optimizer):
        >>> model = nn.Linear(10, 1)
        >>> optimizer = SuperGrok(model.parameters(), lr=1e-3)
        >>> 
        >>> for data, target in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = criterion(model(data), target)
        ...     loss.backward()
        ...     optimizer.step(train_loss=loss.item())
    
    Example (With validation loss for adaptive alpha):
        >>> for step, (train_batch, val_batch) in enumerate(dataloader):
        ...     optimizer.zero_grad()
        ...     loss = criterion(model(train_batch), train_targets)
        ...     loss.backward()
        ...     
        ...     # Update alpha every 100 steps using validation loss
        ...     if step % 100 == 0:
        ...         with torch.no_grad():
        ...             val_loss = criterion(model(val_batch), val_targets)
        ...         optimizer.step(train_loss=loss.item(), val_loss=val_loss.item())
        ...     else:
        ...         optimizer.step(train_loss=loss.item())  # Uses cached alpha
    """
    
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        alpha_init: float = 0.98,
        lamb: float = 2.0,
        gamma: float = 0.1,
        kappa: float = 0.1,
        warmup_steps: int = 100,
        gradient_clipping: float = 1.0,
        alpha_update_freq: int = 100,
        meta_net: Optional[nn.Module] = None,
        meta_hidden_dim: int = 32,
    ):
        # Validation
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha_init < 1.0:
            raise ValueError(f"Invalid alpha_init value: {alpha_init}")
        if lamb < 0.0:
            raise ValueError(f"Invalid lamb value: {lamb}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if kappa < 0.0:
            raise ValueError(f"Invalid kappa value: {kappa}")
        if warmup_steps < 0:
            raise ValueError(f"Invalid warmup_steps value: {warmup_steps}")
        if gradient_clipping < 0.0:
            raise ValueError(f"Invalid gradient_clipping value: {gradient_clipping}")
        if alpha_update_freq < 1:
            raise ValueError(f"Invalid alpha_update_freq value: {alpha_update_freq}")
        if meta_hidden_dim < 1:
            raise ValueError(f"Invalid meta_hidden_dim value: {meta_hidden_dim}")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            alpha_init=alpha_init,
            lamb=lamb,
            gamma=gamma,
            kappa=kappa,
            warmup_steps=warmup_steps,
            gradient_clipping=gradient_clipping,
            alpha_update_freq=alpha_update_freq,
        )
        
        super().__init__(params, defaults)
        
        # Auto-create meta_net if not provided
        if meta_net is None:
            self.meta_net = SimpleMetaNet(hidden_dim=meta_hidden_dim)
        else:
            self.meta_net = meta_net
        
        self._meta_hidden_dim = meta_hidden_dim
        
        # Cached adaptive alpha (initialized to alpha_init)
        self._cached_alpha: float = alpha_init
        self._alpha_init: float = alpha_init
        self._kappa: float = kappa
        
        # Global step counter
        self._global_step: int = 0
        
        # Cache for layer-wise beta1 values (computed lazily)
        self._beta1_cache: Dict[Tuple[int, float, float], float] = {}
        
        # Pre-compute and cache layer indices for each parameter
        self._layer_indices: Dict[int, int] = {}
        layer_idx = 0
        for group in self.param_groups:
            for p in group['params']:
                self._layer_indices[id(p)] = layer_idx
                layer_idx += 1
        
        self._total_params = layer_idx
    
    def _get_layer_beta1(self, param: torch.Tensor, beta1_init: float, gamma: float) -> float:
        """
        Get layer-wise β₁ with caching.
        
        Formula: β1_l = β1_init * (1 - γ)^l
        
        Args:
            param: Parameter tensor
            beta1_init: Initial β₁ value
            gamma: Decay rate
            
        Returns:
            Layer-specific β₁ value
        """
        param_id = id(param)
        layer_idx = self._layer_indices.get(param_id, 0)
        
        # Check cache with composite key
        cache_key = (layer_idx, beta1_init, gamma)
        if cache_key not in self._beta1_cache:
            self._beta1_cache[cache_key] = beta1_init * ((1.0 - gamma) ** layer_idx)
        
        return self._beta1_cache[cache_key]
    
    def _get_adaptive_alpha(
        self, 
        train_loss: Optional[float] = None, 
        val_loss: Optional[float] = None
    ) -> float:
        """
        Get adaptive alpha with strict val_loss requirement and caching.
        
        Behavior:
        - Only updates when val_loss is explicitly provided
        - Returns cached value otherwise (lazy evaluation)
        - Implements zero-loss fix (high signal when train_loss < 1e-6)
        
        Args:
            train_loss: Current training loss (optional)
            val_loss: Current validation loss (required for alpha update)
            
        Returns:
            Adaptive alpha value for EMA computation
        """
        # STRICT: Only update if val_loss is explicitly provided
        if val_loss is not None:
            if train_loss is not None and train_loss < 1e-6:
                # Zero-loss fix: high signal → low alpha → memory clears rapidly
                signal = 10.0
            elif train_loss is not None and train_loss >= 1e-6:
                # Normal grokking signal: (val - train) / train
                signal = (val_loss - train_loss) / train_loss
            else:
                # val_loss provided but train_loss is None — edge case
                # Cannot compute meaningful signal, keep cached value
                return self._cached_alpha
            
            # Compute new alpha and cache it
            # α_t = α_init * exp(-κ * signal)
            self._cached_alpha = self._alpha_init * math.exp(-self._kappa * signal)
            
            # Clamp to valid range
            self._cached_alpha = max(0.0, min(self._cached_alpha, 1.0 - 1e-8))
        
        # LAZY: Always return cached value (whether just updated or not)
        return self._cached_alpha
    
    def _compute_cosine_similarity(
        self, 
        tensor_a: torch.Tensor, 
        tensor_b: torch.Tensor
    ) -> float:
        """
        Compute tensor-wise cosine similarity.
        
        Flattens both tensors to vectors and computes a single alignment score.
        
        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            
        Returns:
            Cosine similarity in range [-1, 1]
        """
        flat_a = tensor_a.view(-1)
        flat_b = tensor_b.view(-1)
        
        # Handle zero vectors
        norm_a = torch.norm(flat_a)
        norm_b = torch.norm(flat_b)
        
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        
        # Cosine similarity
        cos_sim = torch.dot(flat_a, flat_b) / (norm_a * norm_b)
        
        return cos_sim.item()
    
    @torch.no_grad()
    def step(
        self, 
        closure: Optional[Callable[[], float]] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
    ) -> Optional[float]:
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)
            train_loss: Current training loss for adaptive alpha (optional)
            val_loss: Current validation loss for adaptive alpha 
                     (required for alpha update, optional for step execution)
            
        Returns:
            Loss value if closure provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Get adaptive alpha (cached/lazy)
        alpha = self._get_adaptive_alpha(train_loss, val_loss)
        
        # Increment global step
        self._global_step += 1
        
        for group in self.param_groups:
            beta1_init, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            weight_decay = group['weight_decay']
            lamb = group['lamb']
            gamma = group['gamma']
            warmup_steps = group['warmup_steps']
            gradient_clipping = group['gradient_clipping']
            
            # Collect parameters and their processed gradients
            params_with_grads = []
            final_grads = []
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if grad.is_sparse:
                    raise RuntimeError("SuperGrok does not support sparse gradients")
                
                # Initialize state if needed
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # AdamW momentum buffers
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Grokfast EMA buffer - initialize with first gradient (warmup Option B)
                    state['mu'] = grad.clone()
                
                state['step'] += 1
                
                # === RULE 1: Store raw gradient for decoupled Grokfast buffer ===
                raw_grad = grad.clone()
                
                # === NEURALGROK: Transform gradient through meta_net ===
                # Enable grad temporarily for meta_net forward pass
                # (needed if user wants to train meta_net later)
                with torch.enable_grad():
                    grad_input = grad.detach().clone()
                    grad_input.requires_grad_(False)
                    smart_grad = self.meta_net(grad_input)
                    smart_grad = smart_grad.detach()
                
                # === RULE 1 (continued): Update Grokfast buffer with RAW gradients ===
                # mu_t = α * mu_{t-1} + (1-α) * raw_grad
                state['mu'].mul_(alpha).add_(raw_grad, alpha=(1.0 - alpha))
                
                # === RULE 2: Cosine Gating (Stability Clutch) ===
                if self._global_step <= warmup_steps:
                    # During warmup: skip amplification, just use smart_grad
                    final_grad = smart_grad.clone()
                else:
                    # Compute tensor-wise cosine similarity
                    cos_sim = self._compute_cosine_similarity(smart_grad, state['mu'])
                    
                    # Binary gating decision
                    if cos_sim > 0:
                        # Aligned: apply full Grokfast amplification
                        # final_grad = smart_grad + λ * mu
                        final_grad = smart_grad + lamb * state['mu']
                    else:
                        # Misaligned: disengage momentum, just use smart_grad
                        # This allows the meta-net to take sharp turns
                        final_grad = smart_grad.clone()
                
                params_with_grads.append(p)
                final_grads.append(final_grad)
            
            # === GRADIENT CLIPPING (after gating, before AdamW) ===
            if gradient_clipping > 0 and len(final_grads) > 0:
                # Compute total gradient norm across all parameters
                total_norm_sq = sum(g.norm().pow(2) for g in final_grads)
                total_norm = torch.sqrt(total_norm_sq)
                
                # Clip if exceeds threshold
                clip_coef = gradient_clipping / (total_norm + 1e-6)
                if clip_coef < 1.0:
                    for g in final_grads:
                        g.mul_(clip_coef)
            
            # === ADAMW UPDATE ===
            for p, final_grad in zip(params_with_grads, final_grads):
                state = self.state[p]
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                step = state['step']
                
                # Layer-wise β₁ decay (with caching)
                beta1 = self._get_layer_beta1(p, beta1_init, gamma)
                
                # Decoupled weight decay (applied before momentum update)
                # θ = θ * (1 - lr * wd)
                if weight_decay > 0:
                    p.mul_(1.0 - lr * weight_decay)
                
                # Update biased first moment estimate
                # m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                exp_avg.mul_(beta1).add_(final_grad, alpha=(1.0 - beta1))
                
                # Update biased second moment estimate
                # v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
                exp_avg_sq.mul_(beta2).addcmul_(final_grad, final_grad, value=(1.0 - beta2))
                
                # Bias correction
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                
                # Compute step size with bias correction
                step_size = lr / bias_correction1
                
                # AdamW update: θ = θ - step_size * m_hat / (sqrt(v_hat) + ε)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss
    
    def get_cached_alpha(self) -> float:
        """Return the current cached alpha value."""
        return self._cached_alpha
    
    def get_global_step(self) -> int:
        """Return the current global step count."""
        return self._global_step
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Return a summary of optimizer state for debugging."""
        mu_norms = []
        exp_avg_norms = []
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state.get(p, {})
                if 'mu' in state:
                    mu_norms.append(state['mu'].norm().item())
                if 'exp_avg' in state:
                    exp_avg_norms.append(state['exp_avg'].norm().item())
        
        return {
            'global_step': self._global_step,
            'cached_alpha': self._cached_alpha,
            'total_params': self._total_params,
            'avg_mu_norm': sum(mu_norms) / len(mu_norms) if mu_norms else 0,
            'avg_exp_avg_norm': sum(exp_avg_norms) / len(exp_avg_norms) if exp_avg_norms else 0,
        }
    
    def __repr__(self) -> str:
        return (
            f"SuperGrok(\n"
            f"  lr={self.defaults['lr']},\n"
            f"  betas={self.defaults['betas']},\n"
            f"  weight_decay={self.defaults['weight_decay']},\n"
            f"  alpha_init={self.defaults['alpha_init']},\n"
            f"  lamb={self.defaults['lamb']},\n"
            f"  gamma={self.defaults['gamma']},\n"
            f"  warmup_steps={self.defaults['warmup_steps']},\n"
            f"  gradient_clipping={self.defaults['gradient_clipping']},\n"
            f"  meta_net={self.meta_net.__class__.__name__}\n"
            f")"
        )
