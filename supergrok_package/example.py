"""
SuperGrok Usage Examples

This script demonstrates various ways to use the SuperGrok optimizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Import SuperGrok - that's all you need!
from supergrok import SuperGrok


# =============================================================================
# Example 1: Basic Usage (Like Any Other Optimizer)
# =============================================================================

def example_basic():
    """Simplest possible usage - just like Adam or SGD."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    
    # Create optimizer - just like any other!
    optimizer = SuperGrok(model.parameters(), lr=1e-3)
    
    # Training loop
    criterion = nn.MSELoss()
    
    for step in range(100):
        # Generate random data
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        # Standard training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        # Pass train_loss for adaptive features
        optimizer.step(train_loss=loss.item())
        
        if step % 20 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")
    
    print(f"\n  Final alpha: {optimizer.get_cached_alpha():.4f}")
    print("  ✓ Basic training complete!\n")


# =============================================================================
# Example 2: With Validation Loss (Recommended for Best Results)
# =============================================================================

def example_with_validation():
    """Using validation loss for adaptive alpha updates."""
    print("=" * 60)
    print("Example 2: With Validation Loss")
    print("=" * 60)
    
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    
    optimizer = SuperGrok(
        model.parameters(),
        lr=1e-3,
        warmup_steps=20,
    )
    
    criterion = nn.MSELoss()
    
    # Simulated train/val split
    x_train = torch.randn(100, 10)
    y_train = torch.randn(100, 1)
    x_val = torch.randn(20, 10)
    y_val = torch.randn(20, 1)
    
    alpha_history = []
    
    for step in range(100):
        # Training step
        optimizer.zero_grad()
        train_loss = criterion(model(x_train), y_train)
        train_loss.backward()
        
        # Update alpha every 10 steps with validation loss
        if step % 10 == 0:
            with torch.no_grad():
                val_loss = criterion(model(x_val), y_val)
            optimizer.step(
                train_loss=train_loss.item(),
                val_loss=val_loss.item()
            )
            alpha_history.append(optimizer.get_cached_alpha())
            print(f"  Step {step}: train={train_loss.item():.4f}, "
                  f"val={val_loss.item():.4f}, alpha={alpha_history[-1]:.4f}")
        else:
            optimizer.step(train_loss=train_loss.item())
    
    print(f"\n  Alpha range: [{min(alpha_history):.4f}, {max(alpha_history):.4f}]")
    print("  ✓ Training with validation complete!\n")


# =============================================================================
# Example 3: Custom Configuration
# =============================================================================

def example_custom_config():
    """Customizing all hyperparameters."""
    print("=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)
    
    model = nn.Linear(10, 1)
    
    optimizer = SuperGrok(
        model.parameters(),
        # AdamW parameters
        lr=5e-4,
        betas=(0.9, 0.99),
        weight_decay=0.05,
        # Grokfast parameters
        alpha_init=0.95,
        lamb=3.0,
        # Layer-wise decay
        gamma=0.15,
        # Grokking signal
        kappa=0.2,
        # Other settings
        warmup_steps=50,
        gradient_clipping=0.5,
        meta_hidden_dim=64,  # Larger meta-net
    )
    
    print(f"\n  Optimizer configuration:")
    print(optimizer)
    
    # Quick training
    criterion = nn.MSELoss()
    for step in range(20):
        optimizer.zero_grad()
        loss = criterion(model(torch.randn(16, 10)), torch.randn(16, 1))
        loss.backward()
        optimizer.step(train_loss=loss.item())
    
    print("\n  ✓ Custom config training complete!\n")


# =============================================================================
# Example 4: Bilevel Optimization (Training the Meta-Net)
# =============================================================================

def example_bilevel():
    """Full bilevel optimization for maximum performance."""
    print("=" * 60)
    print("Example 4: Bilevel Optimization")
    print("=" * 60)
    
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    
    # Main optimizer
    optimizer = SuperGrok(model.parameters(), lr=1e-3)
    
    # Separate optimizer for meta-net
    meta_optimizer = torch.optim.Adam(
        optimizer.meta_net.parameters(),
        lr=1e-4
    )
    
    criterion = nn.MSELoss()
    
    # Data
    x_train = torch.randn(100, 10)
    y_train = torch.randn(100, 1)
    x_val = torch.randn(20, 10)
    y_val = torch.randn(20, 1)
    
    print("\n  Training with bilevel optimization...")
    
    for step in range(100):
        # === Inner Loop: Train Model ===
        optimizer.zero_grad()
        train_loss = criterion(model(x_train), y_train)
        train_loss.backward()
        optimizer.step(train_loss=train_loss.item())
        
        # === Outer Loop: Train Meta-Net (every 5 steps) ===
        if step % 5 == 0:
            meta_optimizer.zero_grad()
            
            # Forward through model (meta-net gradients flow)
            with torch.enable_grad():
                val_loss = criterion(model(x_val), y_val)
                val_loss.backward()
            
            meta_optimizer.step()
            
            if step % 20 == 0:
                print(f"  Step {step}: train={train_loss.item():.4f}, "
                      f"val={val_loss.item():.4f}")
    
    print("\n  ✓ Bilevel optimization complete!\n")


# =============================================================================
# Example 5: Inspecting Optimizer State
# =============================================================================

def example_inspection():
    """Debugging and monitoring the optimizer."""
    print("=" * 60)
    print("Example 5: State Inspection")
    print("=" * 60)
    
    model = nn.Linear(10, 1)
    optimizer = SuperGrok(model.parameters(), lr=1e-3)
    
    criterion = nn.MSELoss()
    
    # Train a few steps
    for _ in range(10):
        optimizer.zero_grad()
        loss = criterion(model(torch.randn(16, 10)), torch.randn(16, 1))
        loss.backward()
        optimizer.step(train_loss=loss.item())
    
    # Inspect state
    print(f"\n  Global step: {optimizer.get_global_step()}")
    print(f"  Cached alpha: {optimizer.get_cached_alpha():.4f}")
    
    summary = optimizer.get_state_summary()
    print(f"\n  State summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")
    
    # Access meta-net parameters
    meta_params = sum(p.numel() for p in optimizer.meta_net.parameters())
    print(f"\n  Meta-net parameters: {meta_params}")
    print(f"  Meta-net rescale: {optimizer.meta_net.rescale.item():.4f}")
    
    print("\n  ✓ State inspection complete!\n")


# =============================================================================
# Example 6: Classification Task
# =============================================================================

def example_classification():
    """Using SuperGrok for classification."""
    print("=" * 60)
    print("Example 6: Classification Task")
    print("=" * 60)
    
    # Simple classifier
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )
    
    optimizer = SuperGrok(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
    )
    
    criterion = nn.CrossEntropyLoss()
    
    print("\n  Training classifier...")
    
    for step in range(50):
        # Random classification data
        x = torch.randn(64, 20)
        y = torch.randint(0, 10, (64,))
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step(train_loss=loss.item())
        
        if step % 10 == 0:
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean()
            print(f"  Step {step}: loss={loss.item():.4f}, acc={acc.item():.2%}")
    
    print("\n  ✓ Classification training complete!\n")


# =============================================================================
# Run All Examples
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SuperGrok Usage Examples")
    print("=" * 60 + "\n")
    
    example_basic()
    example_with_validation()
    example_custom_config()
    example_bilevel()
    example_inspection()
    example_classification()
    
    print("=" * 60)
    print("All examples completed successfully! ✓")
    print("=" * 60)
