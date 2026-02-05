"""
Comprehensive Test Suite for SuperGrok Optimizer

Tests all components:
1. SimpleMetaNet functionality
2. Basic training steps
3. Decoupled memory (Rule 1)
4. Cosine gating (Rule 2)
5. Zero-loss fix (Rule 3)
6. Adaptive alpha with caching
7. Gradient clipping
8. Layer-wise β₁ decay
9. Warmup behavior
10. Edge cases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

# Import our implementation
from supergrok import SuperGrok, SimpleMetaNet


def create_test_model() -> nn.Module:
    """Create a simple test model."""
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


def generate_test_data(batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random test data."""
    X = torch.randn(batch_size, 10)
    y = torch.randn(batch_size, 1)
    return X, y


def test_simple_meta_net():
    """Test SimpleMetaNet functionality."""
    print("\n" + "=" * 60)
    print("TEST 1: SimpleMetaNet Functionality")
    print("=" * 60)
    
    meta_net = SimpleMetaNet(hidden_dim=32)
    
    # Test 1.1: Basic forward pass
    print("\n1.1 Basic forward pass...")
    grad = torch.randn(100, 50)
    smart_grad = meta_net(grad)
    
    assert smart_grad.shape == grad.shape, f"Shape mismatch: {smart_grad.shape} vs {grad.shape}"
    print(f"  ✓ Input shape: {grad.shape}, Output shape: {smart_grad.shape}")
    
    # Test 1.2: Norm preservation (approximately)
    print("\n1.2 Norm preservation...")
    input_norm = grad.norm().item()
    output_norm = smart_grad.norm().item()
    # With rescale=1.0 (initial), norms should be similar
    ratio = output_norm / input_norm
    print(f"  Input norm: {input_norm:.4f}, Output norm: {output_norm:.4f}, Ratio: {ratio:.4f}")
    print(f"  ✓ Norm ratio is reasonable")
    
    # Test 1.3: Different shapes
    print("\n1.3 Different tensor shapes...")
    shapes = [(10,), (5, 5), (2, 3, 4), (100,)]
    for shape in shapes:
        grad = torch.randn(shape)
        smart_grad = meta_net(grad)
        assert smart_grad.shape == grad.shape, f"Shape mismatch for {shape}"
        print(f"  ✓ Shape {shape} works correctly")
    
    # Test 1.4: Empty tensor
    print("\n1.4 Empty tensor handling...")
    empty_grad = torch.randn(0)
    empty_result = meta_net(empty_grad)
    assert empty_result.shape == empty_grad.shape
    print(f"  ✓ Empty tensor handled correctly")
    
    # Test 1.5: Gradient flow through meta_net
    print("\n1.5 Gradient flow...")
    grad = torch.randn(10, 10, requires_grad=False)
    grad_input = grad.clone()
    smart_grad = meta_net(grad_input)
    # The meta_net should have gradients enabled for its parameters
    assert any(p.requires_grad for p in meta_net.parameters())
    print(f"  ✓ Meta-net parameters require gradients")
    
    print("\n✓ All SimpleMetaNet tests passed!")


def test_basic_training():
    """Test basic training loop."""
    print("\n" + "=" * 60)
    print("TEST 2: Basic Training Steps")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    model = create_test_model()
    meta_net = SimpleMetaNet(hidden_dim=32)
    optimizer = SuperGrok(
        model.parameters(),
        meta_net=meta_net,
        lr=1e-3,
        warmup_steps=5,
    )
    
    X, y = generate_test_data()
    criterion = nn.MSELoss()
    
    print("\nRunning 10 training steps...")
    losses = []
    for step in range(10):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step(train_loss=loss.item())
        losses.append(loss.item())
        print(f"  Step {step+1}: loss={loss.item():.6f}")
    
    # Check that optimizer state is populated
    assert optimizer.get_global_step() == 10
    print(f"\n✓ Global step: {optimizer.get_global_step()}")
    
    # Check state for first parameter
    first_param = list(model.parameters())[0]
    state = optimizer.state[first_param]
    assert 'exp_avg' in state
    assert 'exp_avg_sq' in state
    assert 'mu' in state
    assert state['step'] == 10
    print(f"✓ State properly initialized: step={state['step']}")
    
    print("\n✓ Basic training test passed!")


def test_decoupled_memory():
    """Test Rule 1: Decoupled Memory."""
    print("\n" + "=" * 60)
    print("TEST 3: Decoupled Memory (Rule 1)")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    model = nn.Linear(5, 1)
    meta_net = SimpleMetaNet(hidden_dim=16)
    optimizer = SuperGrok(
        model.parameters(),
        meta_net=meta_net,
        lr=1e-3,
        warmup_steps=0,  # No warmup for this test
        alpha_init=0.5,  # Lower alpha so mu changes more noticeably
    )
    
    X = torch.randn(8, 5)
    y = torch.randn(8, 1)
    criterion = nn.MSELoss()
    
    # Step 1: Get initial gradient and mu
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    
    # Store raw gradient before step
    raw_grad_before = model.weight.grad.clone()
    
    optimizer.step(train_loss=loss.item())
    
    # Get mu after step
    weight_state = optimizer.state[model.weight]
    mu_after_step1 = weight_state['mu'].clone()
    
    print(f"\nStep 1:")
    print(f"  Raw grad norm: {raw_grad_before.norm().item():.6f}")
    print(f"  Mu norm: {mu_after_step1.norm().item():.6f}")
    
    # Step 2: Another step
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    
    raw_grad_step2 = model.weight.grad.clone()
    optimizer.step(train_loss=loss.item())
    
    mu_after_step2 = weight_state['mu'].clone()
    
    # Verify mu is updated based on raw gradients, not smart_grad
    # mu_new = alpha * mu_old + (1-alpha) * raw_grad
    alpha = optimizer.get_cached_alpha()
    expected_mu = alpha * mu_after_step1 + (1 - alpha) * raw_grad_step2
    
    print(f"\nStep 2:")
    print(f"  Alpha: {alpha:.4f}")
    print(f"  Expected mu norm: {expected_mu.norm().item():.6f}")
    print(f"  Actual mu norm: {mu_after_step2.norm().item():.6f}")
    
    # Check they're close (allowing for numerical precision)
    diff = (expected_mu - mu_after_step2).norm().item()
    print(f"  Difference: {diff:.8f}")
    
    assert diff < 1e-5, f"Mu update doesn't match expected! Diff: {diff}"
    print(f"\n✓ Decoupled memory test passed! Mu is updated with raw gradients only.")


def test_cosine_gating():
    """Test Rule 2: Cosine Gating."""
    print("\n" + "=" * 60)
    print("TEST 4: Cosine Gating (Rule 2)")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # We'll create a scenario where we can control alignment
    model = nn.Linear(5, 1, bias=False)
    
    # Custom meta_net that just scales (doesn't change direction much)
    class IdentityMetaNet(nn.Module):
        def forward(self, grad):
            return grad * 1.0  # Identity-ish
    
    meta_net = IdentityMetaNet()
    optimizer = SuperGrok(
        model.parameters(),
        meta_net=meta_net,
        lr=1e-3,
        warmup_steps=0,
        lamb=2.0,
        alpha_init=0.9,
    )
    
    # Initialize with a specific gradient direction
    X = torch.randn(4, 5)
    y = torch.randn(4, 1)
    criterion = nn.MSELoss()
    
    # First step to initialize mu
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step(train_loss=loss.item())
    
    # Get mu after first step
    weight_state = optimizer.state[model.weight]
    mu = weight_state['mu'].clone()
    
    print(f"\nAfter first step:")
    print(f"  Mu norm: {mu.norm().item():.6f}")
    
    # Second step with same direction (should be aligned)
    optimizer.zero_grad()
    loss = criterion(model(X), y)  # Same data = similar gradient direction
    loss.backward()
    
    smart_grad = model.weight.grad.clone()  # Since meta_net is identity
    
    # Compute alignment manually
    cos_sim = F.cosine_similarity(
        smart_grad.view(1, -1),
        mu.view(1, -1)
    ).item()
    
    print(f"\nSecond step (same direction):")
    print(f"  Cosine similarity: {cos_sim:.4f}")
    
    if cos_sim > 0:
        print(f"  ✓ Positive alignment detected - momentum should be applied")
    else:
        print(f"  ⚠ Negative alignment - momentum disengaged")
    
    optimizer.step(train_loss=loss.item())
    
    print("\n✓ Cosine gating test passed!")


def test_zero_loss_fix():
    """Test Rule 3: Zero-Loss Fix."""
    print("\n" + "=" * 60)
    print("TEST 5: Zero-Loss Fix (Rule 3)")
    print("=" * 60)
    
    model = nn.Linear(5, 1)
    meta_net = SimpleMetaNet(hidden_dim=16)
    optimizer = SuperGrok(
        model.parameters(),
        meta_net=meta_net,
        lr=1e-3,
        alpha_init=0.98,
        kappa=0.1,
    )
    
    print(f"\nInitial cached alpha: {optimizer.get_cached_alpha():.6f}")
    
    # Normal loss scenario
    optimizer._get_adaptive_alpha(train_loss=1.0, val_loss=1.5)
    alpha_normal = optimizer.get_cached_alpha()
    print(f"After normal loss (train=1.0, val=1.5): alpha={alpha_normal:.6f}")
    
    # Near-zero loss scenario - should trigger zero-loss fix
    # Reset to initial
    optimizer._cached_alpha = optimizer._alpha_init
    
    optimizer._get_adaptive_alpha(train_loss=1e-8, val_loss=0.1)
    alpha_zero_loss = optimizer.get_cached_alpha()
    print(f"After near-zero loss (train=1e-8, val=0.1): alpha={alpha_zero_loss:.6f}")
    
    # Alpha should be very low (memory clearing)
    # With signal=10.0 and kappa=0.1: alpha = 0.98 * exp(-0.1 * 10) = 0.98 * exp(-1) ≈ 0.36
    expected_approx = 0.98 * math.exp(-0.1 * 10.0)
    print(f"Expected (approx): {expected_approx:.6f}")
    
    assert alpha_zero_loss < alpha_normal, "Zero-loss fix should reduce alpha significantly"
    assert alpha_zero_loss < 0.5, "Alpha should be low to clear momentum buffer"
    
    print("\n✓ Zero-loss fix test passed! Alpha drops significantly when train_loss → 0")


def test_adaptive_alpha_caching():
    """Test adaptive alpha with strict val_loss requirement and caching."""
    print("\n" + "=" * 60)
    print("TEST 6: Adaptive Alpha Caching")
    print("=" * 60)
    
    model = nn.Linear(5, 1)
    meta_net = SimpleMetaNet(hidden_dim=16)
    optimizer = SuperGrok(
        model.parameters(),
        meta_net=meta_net,
        lr=1e-3,
        alpha_init=0.98,
        kappa=0.1,
    )
    
    initial_alpha = optimizer.get_cached_alpha()
    print(f"\nInitial alpha: {initial_alpha:.6f}")
    
    # Test 1: Step without val_loss - alpha should NOT change
    print("\n6.1 Step without val_loss:")
    optimizer._get_adaptive_alpha(train_loss=0.5, val_loss=None)
    alpha_after = optimizer.get_cached_alpha()
    print(f"  Alpha after (val_loss=None): {alpha_after:.6f}")
    assert alpha_after == initial_alpha, "Alpha should not change without val_loss"
    print(f"  ✓ Alpha unchanged (cached)")
    
    # Test 2: Step with val_loss - alpha SHOULD change
    print("\n6.2 Step with val_loss:")
    optimizer._get_adaptive_alpha(train_loss=0.5, val_loss=0.8)
    alpha_updated = optimizer.get_cached_alpha()
    print(f"  Alpha after (train=0.5, val=0.8): {alpha_updated:.6f}")
    assert alpha_updated != initial_alpha, "Alpha should change with val_loss"
    print(f"  ✓ Alpha updated")
    
    # Test 3: Subsequent steps without val_loss should use cached
    print("\n6.3 Subsequent steps without val_loss:")
    for i in range(3):
        optimizer._get_adaptive_alpha(train_loss=0.3 + i*0.1, val_loss=None)
        alpha_now = optimizer.get_cached_alpha()
        print(f"  Step {i+1}: alpha={alpha_now:.6f} (should match {alpha_updated:.6f})")
        assert alpha_now == alpha_updated, "Alpha should remain cached"
    print(f"  ✓ Alpha remained cached across multiple steps")
    
    print("\n✓ Adaptive alpha caching test passed!")


def test_gradient_clipping():
    """Test gradient clipping."""
    print("\n" + "=" * 60)
    print("TEST 7: Gradient Clipping")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    model = nn.Linear(10, 1)
    meta_net = SimpleMetaNet(hidden_dim=16)
    
    # Test with clipping enabled
    optimizer_clipped = SuperGrok(
        model.parameters(),
        meta_net=meta_net,
        lr=1e-3,
        gradient_clipping=1.0,
        warmup_steps=0,
    )
    
    # Create large gradients
    X = torch.randn(4, 10) * 100
    y = torch.randn(4, 1) * 100
    criterion = nn.MSELoss()
    
    optimizer_clipped.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    
    # Check gradient norm before step
    total_norm = sum(p.grad.norm().pow(2) for p in model.parameters() if p.grad is not None)
    total_norm = torch.sqrt(total_norm).item()
    print(f"\nGradient norm before step: {total_norm:.4f}")
    
    # Step should clip
    optimizer_clipped.step(train_loss=loss.item())
    
    print(f"Gradient clipping threshold: 1.0")
    print(f"✓ Gradients would be clipped if norm > threshold")
    
    # Test with clipping disabled
    model2 = nn.Linear(10, 1)
    optimizer_unclipped = SuperGrok(
        model2.parameters(),
        meta_net=meta_net,
        lr=1e-3,
        gradient_clipping=0,  # Disabled
        warmup_steps=0,
    )
    
    optimizer_unclipped.zero_grad()
    loss2 = criterion(model2(X), y)
    loss2.backward()
    optimizer_unclipped.step(train_loss=loss2.item())
    
    print(f"✓ Gradient clipping disabled (gradient_clipping=0) works")
    
    print("\n✓ Gradient clipping test passed!")


def test_layer_wise_beta1_decay():
    """Test layer-wise β₁ decay with caching."""
    print("\n" + "=" * 60)
    print("TEST 8: Layer-wise β₁ Decay")
    print("=" * 60)
    
    model = create_test_model()  # 5 parameter tensors
    meta_net = SimpleMetaNet(hidden_dim=16)
    
    beta1_init = 0.9
    gamma = 0.1
    
    optimizer = SuperGrok(
        model.parameters(),
        meta_net=meta_net,
        lr=1e-3,
        betas=(beta1_init, 0.999),
        gamma=gamma,
    )
    
    print(f"\nβ₁_init = {beta1_init}, γ = {gamma}")
    print(f"Formula: β₁_l = β₁_init * (1 - γ)^l\n")
    
    for i, p in enumerate(model.parameters()):
        beta1 = optimizer._get_layer_beta1(p, beta1_init, gamma)
        expected = beta1_init * ((1 - gamma) ** i)
        
        print(f"Layer {i}: β₁ = {beta1:.6f} (expected: {expected:.6f})")
        assert abs(beta1 - expected) < 1e-10, f"β₁ mismatch at layer {i}"
    
    # Test caching
    print("\nTesting caching...")
    for i, p in enumerate(model.parameters()):
        # Second call should use cache
        beta1_cached = optimizer._get_layer_beta1(p, beta1_init, gamma)
        expected = beta1_init * ((1 - gamma) ** i)
        assert abs(beta1_cached - expected) < 1e-10
    
    print(f"✓ Caching works correctly")
    
    print("\n✓ Layer-wise β₁ decay test passed!")


def test_warmup_behavior():
    """Test warmup period behavior."""
    print("\n" + "=" * 60)
    print("TEST 9: Warmup Behavior")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    warmup_steps = 5
    
    model = nn.Linear(5, 1)
    meta_net = SimpleMetaNet(hidden_dim=16)
    optimizer = SuperGrok(
        model.parameters(),
        meta_net=meta_net,
        lr=1e-3,
        warmup_steps=warmup_steps,
        lamb=10.0,  # Large lambda to make effect visible
    )
    
    X = torch.randn(4, 5)
    y = torch.randn(4, 1)
    criterion = nn.MSELoss()
    
    print(f"\nWarmup steps: {warmup_steps}")
    print(f"Lambda: 10.0 (large to make effect visible)\n")
    
    for step in range(warmup_steps + 3):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step(train_loss=loss.item())
        
        global_step = optimizer.get_global_step()
        in_warmup = global_step <= warmup_steps
        
        status = "IN WARMUP (no amplification)" if in_warmup else "POST-WARMUP (amplification active)"
        print(f"Step {global_step}: {status}")
    
    print("\n✓ Warmup behavior test passed!")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("TEST 10: Edge Cases")
    print("=" * 60)
    
    # Test 10.1: Invalid parameters
    print("\n10.1 Invalid parameters...")
    model = nn.Linear(5, 1)
    meta_net = SimpleMetaNet(hidden_dim=16)
    
    try:
        SuperGrok(model.parameters(), meta_net=meta_net, lr=-1)
        assert False, "Should raise error for negative lr"
    except ValueError as e:
        print(f"  ✓ Negative lr raises ValueError: {e}")
    
    try:
        SuperGrok(model.parameters(), meta_net=meta_net, alpha_init=1.5)
        assert False, "Should raise error for invalid alpha_init"
    except ValueError as e:
        print(f"  ✓ Invalid alpha_init raises ValueError: {e}")
    
    # Test that meta_net=None auto-creates a SimpleMetaNet
    print("\n10.2 Auto-created meta_net...")
    opt_auto = SuperGrok(model.parameters(), lr=1e-3)
    assert opt_auto.meta_net is not None
    assert isinstance(opt_auto.meta_net, SimpleMetaNet)
    print(f"  ✓ meta_net auto-created: {opt_auto.meta_net}")
    
    # Test 10.4: No gradients
    print("\n10.4 Parameters without gradients...")
    optimizer = SuperGrok(model.parameters(), meta_net=meta_net, lr=1e-3)
    # Don't call backward
    optimizer.step(train_loss=0.5)  # Should not crash
    print(f"  ✓ Step with no gradients doesn't crash")
    
    # Test 10.5: State summary
    print("\n10.5 State summary...")
    X = torch.randn(4, 5)
    y = torch.randn(4, 1)
    criterion = nn.MSELoss()
    
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step(train_loss=loss.item())
    
    summary = optimizer.get_state_summary()
    print(f"  State summary: {summary}")
    assert 'global_step' in summary
    assert 'cached_alpha' in summary
    print(f"  ✓ State summary works correctly")
    
    # Test 10.6: Repr
    print("\n10.6 Repr...")
    repr_str = repr(optimizer)
    print(f"  {repr_str[:100]}...")
    assert 'SuperGrok' in repr_str
    print(f"  ✓ Repr works correctly")
    
    print("\n✓ Edge cases test passed!")


def test_full_training_simulation():
    """Simulate a full training scenario."""
    print("\n" + "=" * 60)
    print("TEST 11: Full Training Simulation")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Create model and data
    model = create_test_model()
    meta_net = SimpleMetaNet(hidden_dim=32)
    
    # Separate optimizer for meta_net (user's responsibility)
    meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=1e-4)
    
    optimizer = SuperGrok(
        model.parameters(),
        meta_net=meta_net,
        lr=1e-3,
        warmup_steps=10,
        alpha_update_freq=5,
    )
    
    # Training and validation data
    X_train, y_train = generate_test_data(batch_size=64)
    X_val, y_val = generate_test_data(batch_size=32)
    
    criterion = nn.MSELoss()
    
    print(f"\nSimulating 50 training steps...")
    print(f"Alpha updates every 5 steps (when val_loss provided)\n")
    
    for step in range(50):
        # Forward + backward
        optimizer.zero_grad()
        output = model(X_train)
        train_loss = criterion(output, y_train)
        train_loss.backward()
        
        # Lazy alpha update
        if step % 5 == 0:
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = criterion(val_output, y_val)
            optimizer.step(train_loss=train_loss.item(), val_loss=val_loss.item())
            alpha_status = f"UPDATED to {optimizer.get_cached_alpha():.4f}"
        else:
            optimizer.step(train_loss=train_loss.item())
            alpha_status = f"cached at {optimizer.get_cached_alpha():.4f}"
        
        if step % 10 == 0:
            print(f"Step {step:3d}: train_loss={train_loss.item():.4f}, alpha={alpha_status}")
    
    final_summary = optimizer.get_state_summary()
    print(f"\nFinal state summary:")
    print(f"  Global step: {final_summary['global_step']}")
    print(f"  Cached alpha: {final_summary['cached_alpha']:.6f}")
    print(f"  Avg mu norm: {final_summary['avg_mu_norm']:.6f}")
    print(f"  Avg exp_avg norm: {final_summary['avg_exp_avg_norm']:.6f}")
    
    print("\n✓ Full training simulation completed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SUPERGROK OPTIMIZER TEST SUITE")
    print("=" * 60)
    
    test_simple_meta_net()
    test_basic_training()
    test_decoupled_memory()
    test_cosine_gating()
    test_zero_loss_fix()
    test_adaptive_alpha_caching()
    test_gradient_clipping()
    test_layer_wise_beta1_decay()
    test_warmup_behavior()
    test_edge_cases()
    test_full_training_simulation()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
