# SuperGrok

**Ultimate Optimizer for Generalization**

SuperGrok combines the best of three grokking-acceleration techniques into a single, easy-to-use PyTorch optimizer:

- **NeuralGrok**: Learned gradient transformation via neural amplifier
- **Grokfast**: EMA-based slow gradient amplification  
- **GrokAdamW**: AdamW base with adaptive alpha and layer-wise β₁ decay

## Installation

```bash
pip install supergrok
```

Or install from source:

```bash
git clone https://github.com/peterc04/supergrok.git
cd supergrok
pip install -e .
```

## Quick Start

SuperGrok works like any other PyTorch optimizer:

```python
from supergrok import SuperGrok

model = YourModel()
optimizer = SuperGrok(model.parameters(), lr=1e-3)

for data, target in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    optimizer.step(train_loss=loss.item())
```

## Features

### 🧠 Neural Gradient Transformation
An internal neural network learns to transform gradients for better generalization. No manual tuning required - it's created automatically.

### 🎯 Cosine Gating (Stability Clutch)
Automatically detects when the learned gradient direction conflicts with historical momentum and disengages amplification to prevent oscillation.

### 📉 Adaptive Alpha with Validation Loss
Provide validation loss periodically for smarter momentum adaptation:

```python
for step, (train_batch, val_batch) in enumerate(dataloader):
    optimizer.zero_grad()
    loss = criterion(model(train_batch), train_targets)
    loss.backward()
    
    # Update alpha every 100 steps
    if step % 100 == 0:
        with torch.no_grad():
            val_loss = criterion(model(val_batch), val_targets)
        optimizer.step(train_loss=loss.item(), val_loss=val_loss.item())
    else:
        optimizer.step(train_loss=loss.item())
```

### 🔧 Zero-Loss Fix
Automatically clears momentum buffer when training loss approaches zero, preventing post-convergence drift.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-3 | Learning rate |
| `betas` | (0.9, 0.999) | Adam momentum coefficients |
| `eps` | 1e-8 | Numerical stability term |
| `weight_decay` | 0.01 | Decoupled weight decay (AdamW style) |
| `alpha_init` | 0.98 | Initial EMA momentum for Grokfast buffer |
| `lamb` | 2.0 | Amplification factor for slow gradients |
| `gamma` | 0.1 | Layer-wise β₁ decay rate |
| `kappa` | 0.1 | Grokking signal decay rate |
| `warmup_steps` | 100 | Steps before enabling amplification |
| `gradient_clipping` | 1.0 | Max gradient norm (0 to disable) |
| `meta_net` | None | Custom gradient transformer (auto-created if None) |
| `meta_hidden_dim` | 32 | Hidden dimension for auto-created meta-net |

## Advanced Usage

### Custom Meta-Net

Provide your own gradient transformation network:

```python
from supergrok import SuperGrok, SimpleMetaNet

# Custom meta-net with larger capacity
meta_net = SimpleMetaNet(hidden_dim=64)

optimizer = SuperGrok(
    model.parameters(),
    lr=1e-3,
    meta_net=meta_net,
)
```

### Bilevel Optimization (Training the Meta-Net)

For maximum performance, train the meta-net on validation loss:

```python
from supergrok import SuperGrok

optimizer = SuperGrok(model.parameters(), lr=1e-3)
meta_optimizer = torch.optim.Adam(optimizer.meta_net.parameters(), lr=1e-4)

for step, (train_batch, val_batch) in enumerate(dataloader):
    # Inner loop: train model
    optimizer.zero_grad()
    train_loss = criterion(model(train_batch), train_targets)
    train_loss.backward()
    optimizer.step(train_loss=train_loss.item())
    
    # Outer loop: train meta-net (every N steps)
    if step % 10 == 0:
        meta_optimizer.zero_grad()
        val_loss = criterion(model(val_batch), val_targets)
        val_loss.backward()
        meta_optimizer.step()
```

### Inspecting Optimizer State

```python
# Current adaptive alpha
alpha = optimizer.get_cached_alpha()

# Global step count
step = optimizer.get_global_step()

# Full state summary
summary = optimizer.get_state_summary()
print(summary)
# {'global_step': 1000, 'cached_alpha': 0.95, 'avg_mu_norm': 0.023, ...}
```

## How It Works

### Rule 1: Decoupled Memory
The Grokfast momentum buffer (`mu`) is updated using **raw gradients only**, never the transformed gradients. This keeps the momentum track stable.

```
mu_t = α * mu_{t-1} + (1-α) * raw_grad
```

### Rule 2: Cosine Gating
Before applying momentum amplification, we check if the smart gradient aligns with historical momentum:

```
cos_sim = dot(smart_grad, mu) / (||smart_grad|| * ||mu||)

if cos_sim > 0:
    final_grad = smart_grad + λ * mu   # Amplify
else:
    final_grad = smart_grad            # Disengage
```

### Rule 3: Zero-Loss Fix
When `train_loss < 1e-6`, we use a high grokking signal to rapidly decay alpha, clearing the momentum buffer:

```
signal = 10.0  # When near-zero loss
α_t = α_init * exp(-κ * signal)  # ≈ 0.36 with defaults
```

## Citation

If you use SuperGrok in your research, please cite the underlying papers:

```bibtex
@article{lee2024grokfast,
  title={Grokfast: Accelerated Grokking by Amplifying Slow Gradients},
  author={Lee, Jaerin and Kang, Bong Gyun and Kim, Kihoon and Lee, Kyoung Mu},
  journal={arXiv preprint arXiv:2405.20233},
  year={2024}
}

@article{zhou2025neuralgrok,
  title={NeuralGrok: Accelerate Grokking by Neural Gradient Transformation},
  author={Zhou, Xinyu and Fan, Simin and Jaggi, Martin and Fu, Jie},
  journal={arXiv preprint arXiv:2504.17243},
  year={2025}
}
```

## License

Apache 2.0
