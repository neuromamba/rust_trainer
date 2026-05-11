# Publishing Guide for rust_trainer

This document describes how to build, test, and publish `rust_trainer` as a standalone crate and Python package.

## Prerequisites

- **Rust**: 1.70+ (see `rust-toolchain.toml`)
- **Python 3.11+** (for PyPI wheels)
- **maturin 1.7+** (for Python wheel building)
- **GitHub repository**: Forked or created at `github.com/neuromamba/rust_trainer`

## Local Development

### Build the Rust library

```bash
cd rust_trainer_lab
cargo build --release
```

### Run tests

```bash
cargo test --all-features
```

### Build Python extension (local)

Requires Python development headers:

```bash
cargo build --release --features python --lib
# or via maturin:
maturin develop --release
```

### Format and lint

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
```

## Publishing to crates.io

### Prerequisites

1. **crates.io account**: Create at https://crates.io
2. **API token**: Get from https://crates.io/me → "API tokens"
3. **Repository secret**: Add `CARGO_REGISTRY_TOKEN` to GitHub repo secrets

### Manual publish

```bash
cargo publish --no-verify
```

### Automated (CI/CD)

- Tag a release: `git tag v0.2.0 && git push origin v0.2.0`
- GitHub Actions will:
  1. Run tests
  2. Build cross-platform binaries (Linux x86_64/aarch64, Windows, etc.)
  3. Create GitHub Release with binaries
  4. Publish to crates.io automatically

## Publishing to PyPI

### Prerequisites

1. **PyPI account**: Create at https://pypi.org
2. **API token**: Get from PyPI account settings
3. **GitHub secret**: Add `PYPI_API_TOKEN` to repo secrets

### Wheels are built automatically

When you push a version tag (e.g., `v0.2.0`), the release workflow:

1. **Builds wheels** for multiple platforms:
   - Linux x86_64 (GNU)
   - Linux aarch64 (GNU)
   - macOS x86_64
   - macOS aarch64 (Apple Silicon)
   - Windows x86_64

2. **Uploads to PyPI** automatically via the `publish-pypi` job

3. **Skips pre-releases** (tags like `v0.2.0-rc1`)

### Manual wheel build

```bash
# Build wheels locally
maturin build --release

# Publish to PyPI
twine upload target/wheels/*
```

## Version Management

### Update version

1. **Cargo.toml**: Update `version = "X.Y.Z"`
2. **pyproject.toml**: Versioned dynamically from Cargo.toml (via maturin)
3. **CHANGELOG.md**: Document release notes under `## [X.Y.Z]`

### Example: Release v0.2.0

```bash
# Update version in Cargo.toml
sed -i 's/version = "0.1.0"/version = "0.2.0"/' Cargo.toml

# Update CHANGELOG.md
cat > CHANGELOG.md << 'EOF'
# Changelog

## [0.2.0] — 2026-05-15

### Added
- Support for distributed gradient surgery across Rust and Python

### Fixed
- Memory alignment bug in SIMD convolution

---

## [0.1.0] — 2026-05-03
...
EOF

# Commit and tag
git add Cargo.toml CHANGELOG.md
git commit -m "Release v0.2.0"
git tag v0.2.0
git push origin main v0.2.0
```

## GitHub Actions Workflows

### ci.yml

Runs on every push and PR:

- Format checking (`cargo fmt`)
- Lint (`cargo clippy`)
- Unit tests (`cargo test`)
- Release binary builds (`cargo build --release`)
- Optional: PyO3 bindings build test

### release.yml

Triggered by version tags (`v*.*.*`):

1. **test**: Run all tests
2. **build**: Cross-platform binary compilation
   - Linux x86_64 / aarch64
   - Windows x86_64
3. **github-release**: Create GitHub Release with:
   - Extracted changelog notes
   - Pre-built binaries
4. **publish-crate**: Publish to crates.io (if not pre-release)
5. **publish-pypi**: Build and publish Python wheels (if not pre-release)

## Troubleshooting

### PyPI publish fails with "exists" error

This means the version already exists on PyPI. Either:
- Use `--skip-existing` in `twine upload` (CI does this)
- Bump version and re-tag

### Build fails on aarch64 Linux

Ensure cross-compilation tools are installed:

```bash
sudo apt-get install -y gcc-aarch64-linux-gnu
```

The CI workflow handles this automatically.

### Wheel not found in artifacts

Maturin may skip wheels for certain Python versions if they're unavailable on the runner. The CI workflow uses PyO3/maturin-action which auto-detects available Python versions.

## Integration with NeuroMamba

If this crate is used as a dependency in the main NeuroMamba repo:

```toml
# NeuroMamba/Cargo.toml
[dependencies]
rust_trainer = "0.2"

# or:
rust_trainer = { git = "https://github.com/neuromamba/rust_trainer", branch = "main" }
```

```toml
# NeuroMamba/pyproject.toml (maturin)
# Point to the external rust_trainer crate
```

## References

- [crates.io Publishing Guide](https://doc.rust-lang.org/cargo/publishing/)
- [PyPI Publishing](https://packaging.python.org/en/latest/tutorials/packaging-python-projects/)
- [maturin Documentation](https://maturin.rs/)
- [PyO3 Guide](https://pyo3.rs/)
