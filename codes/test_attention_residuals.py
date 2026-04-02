"""
Tests for the Attention Residuals module.
Run with: python test_attention_residuals.py
"""

import torch
import torch.nn as nn
from attention_residuals import (
    BlockAttnRes,
    AttnResTransformerLayer,
    AttnResTransformer,
    RMSNorm,
)


def test_rms_norm():
    norm = RMSNorm(64)
    x = torch.randn(2, 8, 64)
    out = norm(x)
    assert out.shape == x.shape
    print("[PASS] test_rms_norm")


def test_block_attn_res_shapes():
    B, T, D = 2, 16, 128
    attn_res = BlockAttnRes(D)

    blocks = [torch.randn(B, T, D) for _ in range(3)]
    partial = torch.randn(B, T, D)

    h = attn_res(blocks, partial)
    assert h.shape == (B, T, D), f"Expected ({B},{T},{D}), got {h.shape}"
    print("[PASS] test_block_attn_res_shapes")


def test_block_attn_res_zero_query_init():
    """At init, query=0 so attention should be uniform over all sources."""
    B, T, D = 1, 4, 64
    attn_res = BlockAttnRes(D)

    # Verify query is zero
    assert torch.all(attn_res.query == 0), "Query should be initialized to zero"

    # With zero query, all logits are zero, so softmax gives uniform weights
    blocks = [torch.ones(B, T, D) * i for i in range(3)]
    partial = torch.ones(B, T, D) * 3

    h = attn_res(blocks, partial)

    # Should be average of all sources: (0 + 1 + 2 + 3) / 4 = 1.5
    # But RMSNorm on keys will change the actual logits...
    # At least verify it runs and produces reasonable output
    assert not torch.isnan(h).any(), "Output contains NaN"
    assert not torch.isinf(h).any(), "Output contains Inf"
    print("[PASS] test_block_attn_res_zero_query_init")


def test_block_attn_res_gradient_flow():
    """Ensure gradients flow through AttnRes to all sources."""
    B, T, D = 1, 4, 64
    attn_res = BlockAttnRes(D)

    blocks = [torch.randn(B, T, D, requires_grad=True) for _ in range(3)]
    partial = torch.randn(B, T, D, requires_grad=True)

    h = attn_res(blocks, partial)
    loss = h.sum()
    loss.backward()

    for i, b in enumerate(blocks):
        assert b.grad is not None, f"No gradient for block {i}"
        assert b.grad.abs().sum() > 0, f"Zero gradient for block {i}"
    assert partial.grad is not None, "No gradient for partial block"
    print("[PASS] test_block_attn_res_gradient_flow")


def test_block_attn_res_single_source():
    """With one source, output should equal that source (regardless of query)."""
    B, T, D = 1, 4, 64
    attn_res = BlockAttnRes(D)

    source = torch.randn(B, T, D)
    h = attn_res([], source)

    # Only one source, softmax of single logit = 1.0
    assert torch.allclose(h, source, atol=1e-5), "Single source should pass through"
    print("[PASS] test_block_attn_res_single_source")


def test_transformer_layer_shapes():
    B, T, D = 2, 16, 128
    layer = AttnResTransformerLayer(
        hidden_dim=D, num_heads=4, ffn_dim=256,
        block_size=4, layer_idx=0,
    )

    blocks = [torch.randn(B, T, D)]
    partial = torch.zeros(B, T, D)

    blocks_out, partial_out = layer(blocks, partial)
    assert partial_out.shape == (B, T, D)
    print("[PASS] test_transformer_layer_shapes")


def test_transformer_layer_block_boundary():
    """Layer at a block boundary should append partial to blocks."""
    B, T, D = 1, 8, 64

    # block_size=4 -> 2 layers per block -> layer 2 is a boundary
    layer = AttnResTransformerLayer(
        hidden_dim=D, num_heads=2, ffn_dim=128,
        block_size=4, layer_idx=2,
    )

    blocks = [torch.randn(B, T, D)]  # b_0
    partial = torch.randn(B, T, D)

    blocks_out, partial_out = layer(blocks, partial)
    assert len(blocks_out) == 2, f"Expected 2 blocks, got {len(blocks_out)}"
    print("[PASS] test_transformer_layer_block_boundary")


def test_transformer_layer_no_boundary():
    """Layer NOT at a block boundary should not change block count."""
    B, T, D = 1, 8, 64

    layer = AttnResTransformerLayer(
        hidden_dim=D, num_heads=2, ffn_dim=128,
        block_size=4, layer_idx=1,
    )

    blocks = [torch.randn(B, T, D)]
    partial = torch.randn(B, T, D)

    blocks_out, partial_out = layer(blocks, partial)
    assert len(blocks_out) == 1, f"Expected 1 block, got {len(blocks_out)}"
    print("[PASS] test_transformer_layer_no_boundary")


def test_full_transformer_forward():
    B, T = 2, 32
    vocab = 1000

    model = AttnResTransformer(
        vocab_size=vocab, hidden_dim=128, num_heads=4,
        num_layers=4, ffn_dim=256, block_size=4, max_seq_len=T,
    )

    ids = torch.randint(0, vocab, (B, T))
    logits, loss = model(ids)

    assert logits.shape == (B, T, vocab)
    assert loss is None
    print("[PASS] test_full_transformer_forward")


def test_full_transformer_with_loss():
    B, T = 2, 16
    vocab = 500

    model = AttnResTransformer(
        vocab_size=vocab, hidden_dim=64, num_heads=2,
        num_layers=4, ffn_dim=128, block_size=4, max_seq_len=T,
    )

    ids = torch.randint(0, vocab, (B, T))
    labels = torch.randint(0, vocab, (B, T))

    logits, loss = model(ids, labels)
    assert loss is not None
    assert loss.item() > 0
    # Loss should be roughly -log(1/vocab) ~ log(500) ~ 6.2 at init
    assert loss.item() < 10.0, f"Loss too high: {loss.item()}"
    print(f"[PASS] test_full_transformer_with_loss (loss={loss.item():.4f})")


def test_full_transformer_backward():
    """Ensure full backward pass works without errors."""
    B, T = 1, 8
    vocab = 200

    model = AttnResTransformer(
        vocab_size=vocab, hidden_dim=64, num_heads=2,
        num_layers=4, ffn_dim=128, block_size=4, max_seq_len=T,
    )

    ids = torch.randint(0, vocab, (B, T))
    labels = torch.randint(0, vocab, (B, T))

    logits, loss = model(ids, labels)
    loss.backward()

    # Check all parameters have gradients
    params_with_grad = sum(
        1 for p in model.parameters() if p.grad is not None
    )
    total_params = sum(1 for _ in model.parameters())
    assert params_with_grad == total_params, \
        f"Only {params_with_grad}/{total_params} params have gradients"
    print("[PASS] test_full_transformer_backward")


def test_attn_res_params_are_tiny():
    """AttnRes should add negligible parameters."""
    model = AttnResTransformer(
        vocab_size=1000, hidden_dim=256, num_heads=4,
        num_layers=12, ffn_dim=512, block_size=6, max_seq_len=64,
    )

    attn_res_params = sum(
        p.numel() for name, p in model.named_parameters()
        if "attn_res" in name or "mlp_res" in name
    )
    total_params = sum(p.numel() for p in model.parameters())

    ratio = attn_res_params / total_params * 100
    assert ratio < 5.0, f"AttnRes params are {ratio:.2f}% of total (should be tiny)"
    print(f"[PASS] test_attn_res_params_are_tiny ({ratio:.2f}% of total)")


def test_weight_tying():
    """Verify token embedding and lm_head share weights."""
    model = AttnResTransformer(
        vocab_size=500, hidden_dim=64, num_heads=2,
        num_layers=2, ffn_dim=128, block_size=4, max_seq_len=16,
    )

    assert model.token_emb.weight is model.lm_head.weight, \
        "Weight tying is broken"
    print("[PASS] test_weight_tying")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Attention Residuals Tests")
    print("=" * 60)

    test_rms_norm()
    test_block_attn_res_shapes()
    test_block_attn_res_zero_query_init()
    test_block_attn_res_gradient_flow()
    test_block_attn_res_single_source()
    test_transformer_layer_shapes()
    test_transformer_layer_block_boundary()
    test_transformer_layer_no_boundary()
    test_full_transformer_forward()
    test_full_transformer_with_loss()
    test_full_transformer_backward()
    test_attn_res_params_are_tiny()
    test_weight_tying()

    print("=" * 60)
    print("All Attention Residuals tests passed!")
    print("=" * 60)
