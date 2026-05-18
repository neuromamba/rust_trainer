# Rust Trainer Roadmap

This roadmap tracks the path from current production-candidate status to full production-grade training.

## Scope boundary

This file tracks trainer-engine roadmap items only.

Out of scope for this file:

- NeuroMamba pre-release capacity benchmarking phases,
- release-phase orchestration (`R0`-`R5`),
- high-level project claim-validation reporting.

For those items, use NeuroMamba docs:

- `docs/poc-roadmap.md`
- `docs/training.md`
- `CAPACITY_BENCHMARK.md`

## Current baseline

Completed:

- end-to-end CPU training loop binary
- serializable AdamW optimizer state
- resume-safe checkpoints
- JSONL metrics logging
- expansion and freeze policies
- deterministic parity probe for save/load equivalence
- SIMD kernel parity checks
- validation loop with configurable eval cadence
- best-checkpoint tracking and early stopping
- gradient clipping + LR warmup/cosine scheduling controls
- non-finite guardrails for train-step updates
- streaming shard data loader with resumable cursor state
- atomic versioned generic trainer checkpoints

## Phase 1: Training correctness hardening

Goals:

- add cross-framework parity checks against Python/JAX on deterministic shared batches
- verify one-step parity for loss and gradient norms with strict tolerances
- run deterministic replay tests across resume boundaries

Acceptance criteria:

- parity report generated in CI and for local release checks
- parity drift bounds documented and enforced

## Phase 2: Data pipeline and scalability

Goals:

- add streaming token reader with shard support
- add shuffled packed-sequence batching
- avoid full-file memory loading by default

Acceptance criteria:

- training can run on datasets larger than RAM
- throughput and memory use are stable across long runs

## Phase 3: Stability and safety controls

Goals:

- add gradient clipping
- add LR schedules (warmup + cosine/linear decay)
- add non-finite gradient/loss guards and fail-safe checkpointing
- add atomic checkpoint writes and checkpoint schema versioning

Acceptance criteria:

- long-run smoke tests complete without NaN divergence
- restart from checkpoints is robust after forced interruptions

## Phase 4: Production observability

Goals:

- add validation loop and eval cadence
- add best-checkpoint tracking and early stopping hooks
- extend metrics with validation loss, update norms, and stability indicators

Acceptance criteria:

- training quality can be monitored and compared across runs
- operational dashboards can consume metrics.jsonl without custom parsing

## Phase 5: Packaging and ecosystem integration

Goals:

- provide stable public APIs for embedders
- add optional bindings/integration layers where needed
- add docs for multi-environment deployment and reproducibility

Acceptance criteria:

- crate is consumable as a reusable generic trainer package
- release process includes compatibility and migration notes

## Near-term implementation order

1. Cross-framework parity harness
2. Packed sequence batching for shard streams
3. Multi-worker/parallel data ingest for throughput scaling
4. Cross-platform CI matrix expansion for runtime validation
5. Optional PyO3 integration layer for hybrid orchestration
