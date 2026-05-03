# neuromamba_trainer_lab

[![CI](https://github.com/YOUR_ORG/neuromamba_trainer_lab/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_ORG/neuromamba_trainer_lab/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/neuromamba_trainer_lab.svg)](https://crates.io/crates/neuromamba_trainer_lab)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A CPU-first, pure-Rust supervised trainer for **Selective State Space Models (Mamba/SSM)** with Hyperspherical Prototype Networks (HPN) as the output head.

Designed as a standalone crate that can be used as a library dependency or run directly as a binary trainer. Fully independent of any Python / JAX training path.

---

## Features

| Capability | Status |
|---|---|
| SIMD-accelerated SSM forward scan | ✅ |
| SIMD-accelerated SSM backward scan | ✅ |
| SIMD Conv1d + SiLU forward | ✅ |
| LayerNorm forward + backward | ✅ |
| HPN cosine-prototype loss + grads (hidden + prototypes) | ✅ |
| AdamW optimizer with serializable state | ✅ |
| Cached single-layer Mamba forward + backward | ✅ |
| Multi-layer residual stack training step | ✅ |
| Full think-trainer with persistent AdamW and HPN prototypes | ✅ |
| Configurable layer expansion (append / prepend / insert / interleave) | ✅ |
| Configurable layer freezing (first-N or explicit indices) | ✅ |
| Resume-safe checkpointing (bincode) | ✅ |
| JSONL metrics logging | ✅ |
| Token-file data pipeline | ✅ |
| Deterministic parity harness | ✅ |

---

## Architecture overview

```
TokenIDs → Embedding → [Mamba Layer 0] → [Mamba Layer 1] → … → LayerNorm → HPN → Cosine Loss
                          (frozen)          (frozen)         (trainable)
```

- Each Mamba layer runs a depthwise causal conv1d + SiLU, then an SSM scan with discretized state transitions.
- Output head uses fixed or learned hyperspherical prototypes (one per vocab token); loss is squared cosine distance.
- Frozen layers are excluded from gradient updates but remain in the forward pass residual stack.
- Newly inserted think-layers start as zero-residual (identity pass-through) and are learned during think-training.

---

## Crate structure

```
src/
  lib.rs            — crate root, public API
  simd_ops.rs       — SIMD SSM scan kernels (forward + backward), Conv1d+SiLU
  nn.rs             — LayerNorm, HPN loss, prototype gradient
  optim.rs          — AdamW 1D/2D with serializable moment buffers
  layer.rs          — Cached Mamba layer forward_with_cache / backward
  stack.rs          — Freeze-aware multi-layer residual step (SGD, for probes)
  trainer.rs        — Layer params, expansion/freeze orchestration, ExperimentalTrainer
  think_trainer.rs  — Full ThinkTrainer: AdamW, prototype updates, checkpoint/resume
bin/
  train_think.rs    — CLI training binary with token-file support and metrics logging
  think_parity.rs   — Deterministic resume equivalence probe
  parity_lab.rs     — Configurable expansion/freeze harness
  layer_probe.rs    — Single-layer forward/backward probe
  stack_probe.rs    — Multi-layer residual step probe
  bp_probe.rs       — Scalar vs SIMD backward scan parity
  e2e_supervised_probe.rs — LayerNorm + HPN + AdamW end-to-end probe
```

---

## Quick start

```bash
git clone https://github.com/YOUR_ORG/neuromamba_trainer_lab
cd neuromamba_trainer_lab
cargo test
```

Run a short smoke training pass (synthetic tokens, no data file needed):

```bash
cargo run --release --bin train_think -- \
  --steps 200 \
  --batch-size 4 \
  --seq-len 32 \
  --out-dir runs/smoke
```

Run the deterministic parity check (validates checkpoint resume equivalence):

```bash
cargo run --release --bin think_parity
```

---

## Training on real data

Export integer token IDs from your tokenizer (one integer per whitespace-separated token), then:

```bash
cargo run --release --bin train_think -- \
  --token-file /path/to/tokens.txt \
  --out-dir runs/think_v1 \
  --steps 50000 \
  --batch-size 8 \
  --seq-len 64 \
  --d-model 512 \
  --d-state 16 \
  --base-layers 2 \
  --target-layers 6 \
  --placement specific:1,3,4,5 \
  --freeze first:2 \
  --lr 1e-4
```

Resume from a checkpoint:

```bash
cargo run --release --bin train_think -- \
  --resume runs/think_v1/latest.bincode \
  --out-dir runs/think_v1 \
  --steps 20000
```

---

## CLI flags (`train_think`)

| Flag | Default | Description |
|---|---|---|
| `--out-dir PATH` | `runs/RUST_THINK` | Output directory for checkpoint and metrics |
| `--steps N` | `5000` | Training steps to run |
| `--save-every N` | `200` | Checkpoint every N steps |
| `--log-every N` | `20` | Log metrics every N steps |
| `--batch-size N` | `8` | Batch size |
| `--seq-len N` | `64` | Sequence length |
| `--seed N` | `42` | Random seed |
| `--base-layers N` | `2` | Number of base (frozen) layers |
| `--target-layers N` | `6` | Total layers after expansion |
| `--d-model N` | `512` | Model dimension |
| `--d-state N` | `16` | SSM state dimension |
| `--d-conv N` | `4` | Conv1d kernel size |
| `--placement STR` | `specific:1,3,4,5` | Layer insertion placement |
| `--freeze STR` | `first:2` | Freeze selection |
| `--lr F` | `1e-4` | AdamW learning rate |
| `--freeze-embedding 1` | `false` | Freeze the embedding table |
| `--token-file PATH` | — | Whitespace-separated integer token file |
| `--resume PATH` | — | Resume from checkpoint path |
| `--vocab-size N` | auto | Override vocab size (default: max token + 1) |

### Placement formats

| Value | Meaning |
|---|---|
| `append` | Add new layers after existing ones |
| `prepend` | Add new layers before existing ones |
| `insert:N` | Insert all new layers at index N |
| `specific:1,3,4,5` | Place new layers at these exact final indices |

### Freeze formats

| Value | Meaning |
|---|---|
| `first:N` | Freeze the first N layers |
| `indices:0,2,5` | Freeze layers at these exact indices |

---

## Using as a library

Add to your `Cargo.toml`:

```toml
[dependencies]
neuromamba_trainer_lab = "0.1"
```

Minimal example:

```rust
use neuromamba_trainer_lab::think_trainer::{
    ThinkTrainer, default_think_config, make_batch_from_tokens,
};
use neuromamba_trainer_lab::{ExpansionPlacement, FreezeSelection, LayerSpec};

let spec = LayerSpec { d_model: 512, d_state: 16, d_conv: 4 };
let cfg = default_think_config(
    8192,                                          // vocab_size
    spec,
    6,                                             // target_layers
    ExpansionPlacement::SpecificPositions(vec![1, 3, 4, 5]),
    FreezeSelection::FirstN(2),
    false,                                         // freeze_embedding
    1e-4,                                          // lr
);

let mut trainer = ThinkTrainer::new_random(cfg, 2 /* base_layers */, 42 /* seed */);
let tokens: Vec<i64> = (0..8192).collect();
let (ids, targets) = make_batch_from_tokens(&tokens, 0, 8, 64);
let stats = trainer.train_step(&ids, &targets);
println!("loss: {}", stats.loss);
trainer.save_checkpoint("checkpoint.bincode").unwrap();
```

---

## Release process

Releases are automated via GitHub Actions on version tags.

```bash
# bump version in Cargo.toml, commit, then:
git tag v0.2.0
git push origin v0.2.0
```

The `release.yml` workflow will:
1. Run `cargo test --release`
2. Build release binaries for `x86_64-unknown-linux-gnu`
3. Create a GitHub Release with binaries attached
4. Publish to crates.io if `CARGO_REGISTRY_TOKEN` is set

---

## Roadmap

- [ ] Cross-framework one-step parity check against Python/JAX trainer on shared deterministic batch
- [ ] Streaming data pipeline (shard files, shuffle buffer, packed sequences)
- [ ] LR schedule (cosine decay with linear warmup)
- [ ] Gradient clipping
- [ ] Generation / sampling loop (greedy, top-k, temperature)
- [ ] PyO3 bindings for hybrid Python-orchestrated + Rust-computed training
- [ ] Integration gate into `rust_core` after parity and stability checks pass

---

## License

Apache-2.0. See [LICENSE](LICENSE).