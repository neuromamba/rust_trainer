use rust_trainer::nn::hpn_loss_and_grads;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Serialize, Deserialize)]
struct ParityInput {
    z: Vec<Vec<f32>>,
    target: Vec<i64>,
    prototypes: Vec<Vec<f32>>,
    margin: f32,
    scale: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct ParityOutput {
    loss: f32,
    dz: Vec<Vec<f32>>,
    d_prototypes: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DiffSummary {
    loss_abs: f32,
    dz_l1_mean: f32,
    dz_linf: f32,
    dprot_l1_mean: f32,
    dprot_linf: f32,
}

fn flatten(m: &[Vec<f32>]) -> Vec<f32> {
    let mut out = Vec::new();
    for row in m {
        out.extend_from_slice(row);
    }
    out
}

fn l1_mean(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += (a[i] - b[i]).abs();
    }
    sum / a.len() as f32
}

fn linf(a: &[f32], b: &[f32]) -> f32 {
    let mut mx = 0.0f32;
    for i in 0..a.len() {
        mx = mx.max((a[i] - b[i]).abs());
    }
    mx
}

fn main() {
    let rows = 6usize;
    let dim = 8usize;
    let classes = 5usize;

    let mut z = vec![vec![0.0f32; dim]; rows];
    for r in 0..rows {
        for c in 0..dim {
            z[r][c] = ((r * dim + c) as f32 * 0.03).sin() * 0.7 + 0.1 * (c as f32);
        }
    }

    let mut prototypes = vec![vec![0.0f32; dim]; classes];
    for k in 0..classes {
        for c in 0..dim {
            prototypes[k][c] = ((k * dim + c + 11) as f32 * 0.02).cos() * 0.6 + 0.05 * (k as f32);
        }
    }

    let target = vec![0i64, 2, 1, 4, 3, 2];

    let z_flat = flatten(&z);
    let p_flat = flatten(&prototypes);
    let z_arr = Array2::from_shape_vec((rows, dim), z_flat).expect("shape z");
    let p_arr = Array2::from_shape_vec((classes, dim), p_flat).expect("shape prototypes");

    let (loss_rust, dz_rust, dp_rust) = hpn_loss_and_grads(z_arr.view(), &target, &p_arr);

    let input = ParityInput {
        z,
        target,
        prototypes,
        margin: 0.0,
        scale: 0.0,
    };

    let out_dir = PathBuf::from("target/parity");
    fs::create_dir_all(&out_dir).expect("create parity output dir");
    let in_path = out_dir.join("input.json");
    let py_out_path = out_dir.join("jax_output.json");
    let diff_out_path = out_dir.join("diff.json");

    fs::write(&in_path, serde_json::to_vec_pretty(&input).unwrap()).expect("write parity input");

    let output = Command::new("python3")
        .arg("scripts/jax_parity_reference.py")
        .arg(&in_path)
        .arg(&py_out_path)
        .output()
        .expect("spawn python3 for JAX parity");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "python parity reference failed (status: {}). Ensure JAX is installed in the Python environment. stderr: {}",
            output.status,
            stderr.trim()
        );
    }

    let py_raw = fs::read_to_string(&py_out_path).expect("read python parity output");
    let py: ParityOutput = serde_json::from_str(&py_raw).expect("parse python parity json");

    let dz_py = flatten(&py.dz);
    let dp_py = flatten(&py.d_prototypes);
    let dz_rust_flat: Vec<f32> = dz_rust.iter().copied().collect();
    let dp_rust_flat: Vec<f32> = dp_rust.iter().copied().collect();

    let diff = DiffSummary {
        loss_abs: (loss_rust - py.loss).abs(),
        dz_l1_mean: l1_mean(&dz_rust_flat, &dz_py),
        dz_linf: linf(&dz_rust_flat, &dz_py),
        dprot_l1_mean: l1_mean(&dp_rust_flat, &dp_py),
        dprot_linf: linf(&dp_rust_flat, &dp_py),
    };

    fs::write(&diff_out_path, serde_json::to_vec_pretty(&diff).unwrap()).expect("write parity diff");

    println!("Cross-framework parity complete");
    println!("  loss_abs      = {:.8}", diff.loss_abs);
    println!("  dz_l1_mean    = {:.8}", diff.dz_l1_mean);
    println!("  dz_linf       = {:.8}", diff.dz_linf);
    println!("  dprot_l1_mean = {:.8}", diff.dprot_l1_mean);
    println!("  dprot_linf    = {:.8}", diff.dprot_linf);
    println!("  report        = {}", diff_out_path.display());

    let atol = 5e-5f32;
    if diff.loss_abs > atol || diff.dz_linf > atol || diff.dprot_linf > atol {
        panic!("parity failed: diff above atol={atol}");
    }
}
