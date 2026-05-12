# RUST Trainer

[![CI](https://img.shields.io/badge/CI-ready-brightgreen)](.github/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/rust_trainer.svg)](https://crates.io/crates/rust_trainer)
[![PyPI](https://img.shields.io/pypi/v/rust-trainer.svg)](https://pypi.org/project/rust-trainer/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A CPU-first Rust training package implementing a **Mamba SSM + Hyperspherical Prototype Network (HPN)** architecture.

This is a concrete, working reference implementation — not a blank framework. The model is a stack of Mamba selective state-space layers with an HPN cosine-distance output head and learnable prototype matrix. Teams can use it as-is or fork and replace the layer/loss internals for their own architecture.

It works as both:

- a library dependency in your Rust application
- a ready-to-run CLI training binary

No Python runtime is required.

---

## What this package gives you

| Capability | Status |
|---|---|
| End-to-end train loop binary | ✅ |
| Library API for embedding custom model/training logic | ✅ |
| Serializable optimizer state (AdamW) | ✅ |
| Resume-safe checkpoints (model + optimizer + step) | ✅ |
| JSONL metrics logging | ✅ |
| Configurable layer expansion and freezing | ✅ |
| Deterministic parity probe for save/load correctness | ✅ |
| SIMD math kernels for high-throughput CPU training | ✅ |
| Validation cadence + best-checkpoint tracking | ✅ |
| Early stopping support | ✅ |
| Gradient clipping controls | ✅ |
| LR warmup + cosine decay controls | ✅ |
| Non-finite update guardrails | ✅ |
| Sharded streaming dataset support | ✅ |
| Packed sequence batching on shard streams | ✅ |
| Multi-worker sharded prefetch ingestion | ✅ |
| Run-state resume for stream cursors | ✅ |
| Atomic versioned checkpoints | ✅ |
| Cross-framework parity harness (Rust vs Python/JAX) | ✅ |

---

## Production readiness status

Current state: production-candidate for single-node CPU training with production-critical ingestion and parity validation implemented.

What is already robust:

- deterministic resume behavior (checkpoint + optimizer state + step)
- deterministic resume behavior for streaming shard cursors (`run_state.json`)
- configurable expansion/freeze controls for staged training
- validated SIMD and backward kernels with scalar parity probes
- CI, release, and crate packaging automation

Operational note:

- cross-framework parity runner requires `jax` to be installed in the active Python environment

Detailed roadmap and release milestones are tracked in [roadmap.md](roadmap.md).

---

## Design philosophy

- Keep trainer internals explicit and hackable.
- Favor reproducible runs and resumability.
- Make the package easy to fork and specialize for custom architectures.
- Keep data ingestion simple at first (integer token files), then scale to streaming pipelines.

---

## Repository layout

```
src/
  lib.rs            - crate root and public exports
  generic_trainer.rs  - full trainer state, train step, checkpoint/resume
  trainer.rs        - parameter and expansion/freezing config types
  optim.rs          - AdamW optimizer primitives
  nn.rs             - layer norm and output-loss helpers
  simd_ops.rs       - SIMD kernels used by the model path
  layer.rs          - cached layer forward/backward helpers
  stack.rs          - stack-level supervised step helpers
src/bin/
  train_generic.rs    - main CLI trainer
  trainer_parity.rs   - deterministic parity/resume checker
  parity_lab.rs     - expansion/freeze behavior harness
  *_probe.rs        - low-level probes used for validation
```

---

## Quick start

```bash
git clone https://github.com/npradeep357/rust_trainer
cd rust_trainer
cargo test
```

Run a short smoke training job:

```bash
cargo run --release --bin train_generic -- \
  --steps 200 \
  --batch-size 4 \
  --seq-len 32 \
  --out-dir runs/smoke
```

Run deterministic resume parity check:

```bash
cargo run --release --bin trainer_parity
```

Run Rust vs Python/JAX parity check:

```bash
cargo run --release --bin cross_framework_parity
```

---

## Train your own model data

The default trainer accepts a whitespace-separated integer token file.

```bash
cargo run --release --bin train_generic -- \
  --token-file /path/to/your_tokens.txt \
  --out-dir runs/experiment_v1 \
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

Resume training:

```bash
cargo run --release --bin train_generic -- \
  --resume runs/experiment_v1/latest.bincode \
  --out-dir runs/experiment_v1 \
  --steps 20000
```

---

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--out-dir PATH` | `runs/` | Output directory for checkpoints and metrics |
| `--steps N` | `5000` | Number of train steps |
| `--save-every N` | `200` | Checkpoint interval |
| `--log-every N` | `20` | Metric logging interval |
| `--batch-size N` | `8` | Batch size |
| `--seq-len N` | `64` | Sequence length |
| `--seed N` | `42` | RNG seed |
| `--base-layers N` | `2` | Initial layer count before expansion |
| `--target-layers N` | `6` | Final layer count after expansion |
| `--d-model N` | `512` | Hidden width |
| `--d-state N` | `16` | State width |
| `--d-conv N` | `4` | Convolution kernel width |
| `--placement STR` | `specific:1,3,4,5` | Expansion placement |
| `--freeze STR` | `first:2` | Freeze policy |
| `--lr F` | `1e-4` | AdamW learning rate |
| `--ff-lr F` | `1e-4` | Forward-Forward local learning rate for d_skip updates |
| `--bp-cadence-steps N` | `32` | Apply global BP every N train steps (FF runs each step) |
| `--gradient-surgery-method STR` | `pcgrad` | Conflict handling method: `pcgrad`, `gradnorm`, `cagradstep` |
| `--gradient-surgery-epsilon F` | `1e-8` | Numerical stability epsilon for surgery operations |
| `--gradnorm-alpha F` | `0.2` | GradNorm disagreement scaling factor |
| `--cagrad-lambda F` | `1.0` | CAGradStep conflict-aversion strength |
| `--freeze-embedding 1` | `false` | Freeze embedding table |
| `--token-file PATH` | none | Integer token dataset |
| `--token-dir PATH` | none | Directory of shard files for streaming training |
| `--val-token-file PATH` | none | Optional dedicated validation token dataset |
| `--val-token-dir PATH` | none | Optional validation shard directory |
| `--shard-ext EXT` | `txt` | Extension filter used with `--token-dir` / `--val-token-dir` |
| `--shuffle-shards 1` | `true` | Shuffle shard order each epoch in streaming mode |
| `--packed-sequences 1` | `true` | Use packed contiguous token windows in streaming mode |
| `--prefetch-workers N` | `0` | Number of worker threads for sharded prefetch (`>1` enables multi-worker mode) |
| `--prefetch-buffer N` | `16` | Bounded channel capacity for prefetched worker batches |
| `--resume PATH` | none | Resume checkpoint |
| `--vocab-size N` | auto | Override vocab size |
| `--val-ratio F` | `0.05` | Validation split ratio when `--val-token-file` is not provided |
| `--val-every N` | `200` | Validation cadence in train steps |
| `--eval-batches N` | `8` | Number of validation batches per eval pass |
| `--early-stopping-patience N` | `0` | Stop when validation does not improve for N eval windows (0 disables) |
| `--grad-clip-norm F` | `0.0` | Global gradient clipping threshold (0 disables clipping) |
| `--fail-on-non-finite 1` | `false` | Panic on NaN/Inf detection instead of skipping the update |
| `--lr-warmup-steps N` | `0` | Linear warmup length before decay |
| `--lr-min-scale F` | `0.1` | Minimum LR floor as fraction of base LR for cosine decay |

### Debug + recovery artifacts

- `latest.bincode`: latest atomic, versioned checkpoint (model + optimizer + step)
- `best.bincode`: best validation checkpoint
- `run_state.json`: resumable data-pipeline state (in-memory cursor or shard stream cursor)
- `metrics.jsonl`: train/validation metrics stream for dashboards and debugging

### Placement values

| Value | Meaning |
|---|---|
| `append` | Add new layers at the end |
| `prepend` | Add new layers at the beginning |
| `insert:N` | Insert all new layers starting at index N |
| `specific:1,3,4,5` | Place each new layer at specific final indices |

### Freeze values

| Value | Meaning |
|---|---|
| `first:N` | Freeze first N layers |
| `indices:0,2,5` | Freeze explicit layer indices |

---

## Use as a library

Add dependency:

```toml
[dependencies]
rust_trainer = "0.1"
```

Use the package name from your own `Cargo.toml`.

Minimal integration example:

```rust
use rust_trainer::generic_trainer::{
    GenericTrainer, default_trainer_config, make_batch_from_tokens,
};
use rust_trainer::{ExpansionPlacement, FreezeSelection, LayerSpec};

let spec = LayerSpec { d_model: 512, d_state: 16, d_conv: 4 };
let cfg = default_trainer_config(
    8192,
    spec,
    6,
    ExpansionPlacement::SpecificPositions(vec![1, 3, 4, 5]),
    FreezeSelection::FirstN(2),
    false,
    1e-4,
);

let mut trainer = GenericTrainer::new_random(cfg, 2, 42);
let tokens: Vec<i64> = (0..8192).collect();
let (ids, targets) = make_batch_from_tokens(&tokens, 0, 8, 64);
let stats = trainer.train_step(&ids, &targets);
println!("loss: {}", stats.loss);
trainer.save_checkpoint("checkpoint.bincode").unwrap();
```

---

## Architecture

The default model uses:

| Component | Implementation |
|---|---|
| Sequence layers | Mamba SSM (causal conv1d + SiLU + discretized state scan) |
| Output head | Hyperspherical Prototype Network (HPN) |
| Loss | Squared cosine distance to nearest prototype |
| Optimizer | AdamW with serializable moment buffers |
| Inference path | CPU-only; no GPU required |

## Customize for your own architecture

The package is designed to be forked for other architectures. Replace or extend:

1. Layer forward/backward path in `src/layer.rs` — swap Mamba for Transformer, LSTM, etc.
2. Output loss/head logic in `src/nn.rs` — swap HPN for cross-entropy, contrastive loss, etc.
3. Trainer state wiring in `src/generic_trainer.rs` — add or remove parameter groups
4. Data loading logic in `src/bin/train_generic.rs`

The checkpointing, optimizer state, logging, expansion, and freeze infrastructure are all architecture-independent and can be kept as-is.

---

## Release flow

Releases are tag-driven via GitHub Actions.

```bash
# bump version in Cargo.toml, commit, then:
git tag v0.2.0
git push origin v0.2.0
```

The release workflow runs tests, builds binaries, creates a GitHub Release, and can publish to crates.io when credentials are configured.

---

## License

Apache-2.0. See [LICENSE](LICENSE).