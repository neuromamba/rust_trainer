# Changelog

All notable changes to `rust_trainer_lab` are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Upgraded core dependencies to current versions:
	- `ndarray` `0.17.2`
	- `rand` `0.10.1`
	- `rand_distr` `0.6.0`
	- `rayon` `1.12.0`
	- `wide` `1.3.0`
	- `serde` `1.0.228`
	- `serde_json` `1.0.149`
	- `bincode` `2.0.1` with serde support
- Migrated checkpoint serialization calls to bincode v2 serde API.
- Moved roadmap details from `README.md` to dedicated `roadmap.md`.

### Added
- Streaming shard data pipeline via `src/data_stream.rs` with resumable shard cursor state.
- `train_generic` support for `--token-dir` / `--val-token-dir` with extension filtering and shard shuffling.
- `run_state.json` persistence for deterministic resume of data pipeline cursors.
- Atomic, versioned checkpoint envelope for `GenericTrainer` state.
- Validation path hardening in `train_generic` with best-checkpoint tracking and early stopping.

### Planned
- Cross-framework parity check against Python/JAX trainer on shared deterministic batches
- Streaming data pipeline: shard files, shuffle buffer, packed sequences
- LR schedule: cosine decay with linear warmup
- Gradient clipping
- Generation / sampling loop (greedy, top-k, temperature)
- PyO3 bindings for hybrid Python + Rust training orchestration

---

## [0.1.0] — 2026-05-03

### Added
- **SIMD SSM kernels**: forward and backward scans over state lanes using `wide::f32x8` (`simd_ops.rs`)
- **SIMD Conv1d + SiLU**: depthwise causal convolution with SiLU activation, scalar and SIMD paths
- **LayerNorm**: forward and backward with per-token mean/variance cache (`nn.rs`)
- **HPN loss**: squared cosine-distance loss with gradients for both hidden state and prototype matrix (`nn.rs`)
- **AdamW**: serializable 1D and 2D moment buffers with bias-corrected update (`optim.rs`)
- **Cached Mamba layer**: `forward_with_cache` and `backward` for a single SSM layer (`layer.rs`)
- **Residual stack step**: freeze-aware multi-layer supervised step (`stack.rs`)
- **ExperimentalTrainer**: expansion/freeze/checkpoint orchestration with FF-cycle support (`trainer.rs`)
- **GenericTrainer**: full training with persistent AdamW state, prototype updates, and resume-safe bincode checkpoints (`generic_trainer.rs`)
- **train_generic binary**: CLI trainer with token-file input, JSONL metrics, and periodic checkpointing
- **trainer_parity binary**: deterministic resume equivalence probe
- **parity_lab binary**: configurable expansion/freeze harness
- **CI workflow**: lint, test, build on push/PR
- **Release workflow**: multi-arch binary build, GitHub Release, optional crates.io publish

[Unreleased]: https://github.com/npradeep357/rust_trainer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/npradeep357/rust_trainer/releases/tag/v0.1.0
