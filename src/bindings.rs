/*!
 * Python bindings for the Rust trainer.
 *
 * Exposes the core training loop as a Python-callable API via PyO3.
 * Entry point: neuromamba_trainer.train_from_config(config_path, output_dir)
 */
// PyO3 macro expansions generate PyErr::from(PyErr) patterns; suppress false positive.
#![allow(clippy::useless_conversion)]

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use std::path::Path;
#[cfg(feature = "python")]
use std::process::Command;

/// Python-facing configuration struct for training.
/// Serializable to/from JSON for easy YAML → JSON conversion in Python.
#[cfg(feature = "python")]
#[pyclass]
pub struct TrainerConfig {
    #[pyo3(get, set)]
    pub config_path: String,
    #[pyo3(get, set)]
    pub output_dir: String,
    #[pyo3(get, set)]
    pub steps: usize,
    #[pyo3(get, set)]
    pub seed: u32,
    #[pyo3(get, set)]
    pub base_ckpt: Option<String>,
}

#[cfg(feature = "python")]
#[pymethods]
impl TrainerConfig {
    #[new]
    #[pyo3(signature = (config_path, output_dir, steps, seed, base_ckpt=None))]
    fn new(
        config_path: String,
        output_dir: String,
        steps: usize,
        seed: u32,
        base_ckpt: Option<String>,
    ) -> Self {
        TrainerConfig {
            config_path,
            output_dir,
            steps,
            seed,
            base_ckpt,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TrainerConfig(config_path='{}', output_dir='{}', steps={}, seed={}, base_ckpt={:?})",
            self.config_path, self.output_dir, self.steps, self.seed, self.base_ckpt
        )
    }
}

/// Main Python entry point: train_from_config
/// Loads a config file (JSON or YAML), initializes trainer, runs training loop.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (config_path, output_dir, steps, seed, base_ckpt=None))]
pub fn train_from_config(
    config_path: String,
    output_dir: String,
    steps: usize,
    seed: u32,
    base_ckpt: Option<String>,
) -> PyResult<()> {
    use std::fs;

    let path = Path::new(&config_path);
    if !path.exists() {
        return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "Config file not found: {}",
            config_path
        )));
    }

    let config_text = fs::read_to_string(&config_path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to read config: {}", e))
    })?;

    // Validate config syntax before launching trainer.
    if config_path.ends_with(".json") {
        let _parsed: serde_json::Value = serde_json::from_str(&config_text).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON config: {}", e))
        })?;
    } else if config_path.ends_with(".yaml") || config_path.ends_with(".yml") {
        let _parsed: serde_yaml::Value = serde_yaml::from_str(&config_text).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid YAML config: {}", e))
        })?;
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Config must be .json or .yaml",
        ));
    }

    let manifest = format!("{}/Cargo.toml", env!("CARGO_MANIFEST_DIR"));
    let mut cmd = Command::new("cargo");
    cmd.arg("run")
        .arg("--manifest-path")
        .arg(&manifest)
        .arg("--release")
        .arg("--bin")
        .arg("train_generic")
        .arg("--")
        .arg("--config")
        .arg(&config_path)
        .arg("--output-dir")
        .arg(&output_dir)
        .arg("--steps")
        .arg(steps.to_string())
        .arg("--seed")
        .arg(seed.to_string());
    if let Some(path) = base_ckpt {
        cmd.arg("--base-ckpt").arg(path);
    }

    let status = cmd.status().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Failed to launch train_generic command: {}",
            e
        ))
    })?;
    if !status.success() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "train_generic exited with status: {}",
            status
        )));
    }

    Ok(())
}

/// Optional: Python module initialization
#[cfg(feature = "python")]
#[pymodule]
pub fn neuromamba_trainer(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_from_config, m)?)?;
    m.add_class::<TrainerConfig>()?;

    m.add("__version__", "0.1.4")?;
    m.add(
        "__doc__",
        "NeuroMamba Trainer — Rust-native trainer with FF+BP cadencing and orthogonal gradient projections.",
    )?;

    Ok(())
}
