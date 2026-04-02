"""
Tests for the Combined Engram + AttnRes model.
Run with: python test_combined.py
"""

import torch
from combined_model import CombinedEngramAttnResTransformer, CombinedTransformerLayer


def test_combined_forward():
    B, T = 2, 16
    vocab = 1000

    model = CombinedEngramAttnResTransformer(
        vocab_size=vocab, hidden_dim=128, num_heads=4,
        num_layers=6, ffn_dim=256, block_size=4,
        engram_layers={1, 3},
        engram_config={
            "compressed_vocab_size": 800,
            "engram_dim": 16, "num_heads": 4, "table_size_hint": 1009,
        },
        max_seq_len=T,
    )

    ids = torch.randint(0, vocab, (B, T))
    logits, loss = model(ids)

    assert logits.shape == (B, T, vocab)
    assert loss is None
    print("[PASS] test_combined_forward")


def test_combined_with_loss():
    B, T = 2, 16
    vocab = 500

    model = CombinedEngramAttnResTransformer(
        vocab_size=vocab, hidden_dim=64, num_heads=2,
        num_layers=6, ffn_dim=128, block_size=4,
        engram_layers={1, 3},
        engram_config={
            "compressed_vocab_size": 400,
            "engram_dim": 8, "num_heads": 2, "table_size_hint": 503,
        },
        max_seq_len=T,
    )

    ids = torch.randint(0, vocab, (B, T))
    labels = torch.randint(0, vocab, (B, T))

    logits, loss = model(ids, labels)
    assert loss is not None
    assert loss.item() > 0
    print(f"[PASS] test_combined_with_loss (loss={loss.item():.4f})")


def test_combined_backward():
    B, T = 1, 8
    vocab = 200

    model = CombinedEngramAttnResTransformer(
        vocab_size=vocab, hidden_dim=64, num_heads=2,
        num_layers=4, ffn_dim=128, block_size=4,
        engram_layers={1},
        engram_config={
            "compressed_vocab_size": 150,
            "engram_dim": 8, "num_heads": 2, "table_size_hint": 503,
        },
        max_seq_len=T,
    )

    ids = torch.randint(0, vocab, (B, T))
    labels = torch.randint(0, vocab, (B, T))

    logits, loss = model(ids, labels)
    loss.backward()

    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    assert params_with_grad == total_params, \
        f"Only {params_with_grad}/{total_params} params have gradients"
    print("[PASS] test_combined_backward")


def test_param_summary():
    model = CombinedEngramAttnResTransformer(
        vocab_size=1000, hidden_dim=128, num_heads=4,
        num_layers=6, ffn_dim=256, block_size=4,
        engram_layers={1, 3},
        engram_config={
            "compressed_vocab_size": 800,
            "engram_dim": 16, "num_heads": 4, "table_size_hint": 1009,
        },
        max_seq_len=32,
    )

    summary = model.param_summary()
    assert summary["total"] > 0
    assert summary["engram"] > 0
    assert summary["attn_res"] > 0
    assert summary["backbone"] > 0
    assert abs(summary["total"] - summary["backbone"] - summary["engram"] - summary["attn_res"]) < 10
    print(f"[PASS] test_param_summary")
    print(f"  Total: {summary['total']:,}")
    print(f"  Backbone: {summary['backbone']:,}")
    print(f"  Engram: {summary['engram']:,} ({summary['engram_pct']:.1f}%)")
    print(f"  AttnRes: {summary['attn_res']:,} ({summary['attn_res_pct']:.1f}%)")


def test_engram_only_at_designated_layers():
    """Verify Engram modules exist only at specified layers."""
    model = CombinedEngramAttnResTransformer(
        vocab_size=500, hidden_dim=64, num_heads=2,
        num_layers=8, ffn_dim=128, block_size=4,
        engram_layers={2, 5},
        engram_config={
            "compressed_vocab_size": 400,
            "engram_dim": 8, "num_heads": 2, "table_size_hint": 503,
        },
        max_seq_len=16,
    )

    for i, layer in enumerate(model.layers):
        if i in {2, 5}:
            assert layer.has_engram, f"Layer {i} should have Engram"
            assert hasattr(layer, "engram"), f"Layer {i} missing engram module"
        else:
            assert not layer.has_engram, f"Layer {i} should NOT have Engram"
    print("[PASS] test_engram_only_at_designated_layers")


def test_training_step():
    """Simulate a full training step."""
    vocab = 200
    model = CombinedEngramAttnResTransformer(
        vocab_size=vocab, hidden_dim=64, num_heads=2,
        num_layers=4, ffn_dim=128, block_size=4,
        engram_layers={1},
        engram_config={
            "compressed_vocab_size": 150,
            "engram_dim": 8, "num_heads": 2, "table_size_hint": 503,
        },
        max_seq_len=32,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ids = torch.randint(0, vocab, (4, 32))
    labels = ids[:, 1:].contiguous()
    input_ids = ids[:, :-1].contiguous()

    # Forward
    logits, loss = model(input_ids, labels)
    initial_loss = loss.item()

    # Backward + step
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Second forward -- loss should change
    logits2, loss2 = model(input_ids, labels)
    assert loss2.item() != initial_loss, "Loss did not change after optimizer step"
    print(f"[PASS] test_training_step (loss: {initial_loss:.4f} -> {loss2.item():.4f})")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Combined Model Tests")
    print("=" * 60)

    test_combined_forward()
    test_combined_with_loss()
    test_combined_backward()
    test_param_summary()
    test_engram_only_at_designated_layers()
    test_training_step()

    print("=" * 60)
    print("All Combined Model tests passed!")
    print("=" * 60)
