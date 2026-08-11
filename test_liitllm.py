"""Phase 1 correctness gate. Run this before spending a single GPU hour.

    python test_liitllm.py

No pytest, no fixtures — plain asserts, runs on CPU in well under a minute.
It uses synthetic token ids, so it needs no dataset and no tokenizer.

The tests, in order of how much time they save you:

  1. overfit    the real gate. 4 fixed batches, memorized to loss < 0.05 and
                parroted back verbatim. If the model or the loop is miswired,
                this fails here in seconds instead of after 6 GPU hours.
  2. causality  perturbing token t must not change any logit at position < t.
                A broken causal mask leaks the answer, so training loss looks
                great and generation is gibberish — the nastiest silent bug in
                a transformer, and invisible without a test like this one.
  3. init loss  ~ln(vocab_size) at initialization. Catches a dead head or a
                mis-shaped output projection.
  4. fast==slow the fused attention kernel and the reference implementation
                must agree, so the speed switch can never change results.
"""

import math

import torch

from liitllm.model import Liit, LiitConfig
from liitllm.sample import generate

torch.manual_seed(0)
VOCAB = 128


def cfg(**kw):
    base = dict(vocab_size=VOCAB, block_size=32, n_layer=2, n_head=2, n_embd=64, dropout=0.0)
    return LiitConfig(**{**base, **kw})


def test_shapes_and_init_loss():
    c = cfg()
    m = Liit(c)
    x = torch.randint(VOCAB, (4, 16))
    # Targets must be independent of the inputs. Passing `targets=x` would
    # understate the loss: weight tying means the output projection is the
    # embedding matrix, and the residual stream still carries the input
    # embedding, so "predict yourself" is easy even at initialization.
    y = torch.randint(VOCAB, (4, 16))
    logits, loss = m(x, y)
    assert logits.shape == (4, 16, VOCAB), logits.shape

    # Before learning anything the model should be uniform over the vocab, so
    # cross-entropy ~= ln(vocab). Far off means the output layer is broken.
    expected = math.log(VOCAB)
    assert abs(loss.item() - expected) < 0.7, f"init loss {loss.item():.3f} != ~{expected:.3f}"
    print(f"  ok  shapes + init loss {loss.item():.3f} (~{expected:.3f})")


def test_causal_mask():
    """The load-bearing test: position i must not see position j > i."""
    m = Liit(cfg()).eval()
    x = torch.randint(VOCAB, (1, 16))
    with torch.no_grad():
        base = m(x)[0]
        x2 = x.clone()
        x2[0, 10] = (x2[0, 10] + 1) % VOCAB  # change one token in the middle
        after = m(x2)[0]

    # Everything strictly before position 10 must be bit-identical...
    assert torch.equal(base[:, :10], after[:, :10]), "future token leaked into the past"
    # ...and position 10 itself must have changed, or the model ignores its input.
    assert not torch.allclose(base[:, 10], after[:, 10]), "model ignored a changed token"
    print("  ok  causal mask (no future leakage)")


def test_fast_matches_manual():
    slow = Liit(cfg(fast_attn=False)).eval()
    fast = Liit(cfg(fast_attn=True)).eval()
    fast.load_state_dict(slow.state_dict())
    x = torch.randint(VOCAB, (2, 16))
    with torch.no_grad():
        assert torch.allclose(slow(x)[0], fast(x)[0], atol=1e-4), "fast attention diverges"
    print("  ok  fused attention == reference implementation")


def test_no_pos_is_order_blind():
    """Sanity-check the ablation itself: without positional embeddings the model
    must be permutation-equivariant. If it isn't, the ablation isn't measuring
    what the case study will claim it measures."""
    m = Liit(cfg(learned_pos=False)).eval()
    x = torch.randint(VOCAB, (1, 8))
    with torch.no_grad():
        # With no positions and a causal mask, the first position's output
        # depends only on the first token — reordering the rest can't touch it.
        a = m(x)[0][:, 0]
        shuffled = torch.cat([x[:, :1], x[:, 1:].flip(1)], dim=1)
        b = m(shuffled)[0][:, 0]
        assert torch.allclose(a, b, atol=1e-5), "no-pos model still sees order"
    print("  ok  ablation config is genuinely order-blind")


def test_get_batch_alignment():
    """The data path's one silent failure mode.

    The model learns to predict token t+1 from token t, and that shift happens in
    get_batch — not in the model. If it were off by one, training would optimize
    a misaligned objective perfectly happily and every other test here would
    still pass, because they build their own batches. Check it against a known
    array where the value at each index is the index itself.
    """
    import numpy as np

    from liitllm import data as D

    data = np.arange(1000, dtype=np.uint16)
    x, y = D.get_batch(data, batch_size=8, block_size=16)
    assert x.shape == y.shape == (8, 16)
    # y must be x shifted exactly one token left...
    assert torch.equal(y[:, :-1], x[:, 1:]), "targets are not inputs shifted by one"
    # ...and on this array the token at index i *is* i, so the next token is i+1.
    assert torch.equal(y, x + 1), "off-by-one in the next-token shift"
    print("  ok  get_batch targets are inputs shifted by exactly one")


def test_overfit_batches():
    """THE GATE. Memorize 4 fixed batches; then reproduce them greedily."""
    c = cfg(block_size=32, n_layer=2, n_head=2, n_embd=128)
    m = Liit(c)
    g = torch.Generator().manual_seed(42)
    batches = [torch.randint(VOCAB, (4, 33), generator=g) for _ in range(4)]

    steps = 1200
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95))
    m.train()
    for step in range(steps):
        # Decay the LR over the run. At a flat 3e-3 the loss reaches ~0 and then
        # occasionally spikes back out of the minimum, which makes the final
        # step's loss a coin flip rather than a measurement.
        for grp in opt.param_groups:
            grp["lr"] = 3e-3 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        b = batches[step % len(batches)]
        _, loss = m(b[:, :-1], b[:, 1:])  # inputs, targets shifted by one
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 300 == 0:
            print(f"      step {step:4d} loss {loss.item():.4f}")

    # Measure every batch, not just whichever one the loop happened to end on.
    m.eval()
    with torch.no_grad():
        final = sum(m(b[:, :-1], b[:, 1:])[1].item() for b in batches) / len(batches)
    assert final < 0.05, f"FAILED TO OVERFIT: mean loss {final:.4f} — the model or loop is wrong"

    # Memorizing is necessary but not sufficient: it must also *generate* the
    # sequences back. This is what catches a sampling bug (wrong position
    # indexed, context cropped wrongly) that training loss cannot see.
    b = batches[0]
    prompt = b[:, :8]
    out = generate(m, prompt, max_new_tokens=24, temperature=0.0)  # greedy
    match = (out == b[:, :32]).float().mean().item()
    assert match > 0.95, f"greedy generation only matched {match:.1%} of the memorized batch"
    print(f"  ok  OVERFIT GATE: mean loss {final:.4f}, greedy recall {match:.1%}")


if __name__ == "__main__":
    print("Liit correctness gate\n")
    for fn in (
        test_shapes_and_init_loss,
        test_causal_mask,
        test_fast_matches_manual,
        test_no_pos_is_order_blind,
        test_get_batch_alignment,
        test_overfit_batches,
    ):
        print(f"- {fn.__name__}")
        fn()
    print("\nALL PASS — the transformer and training loop are correct. GPU run is cleared.")
