use ndarray::Array2;
use rust_trainer_lab::stack::supervised_residual_step;
use rust_trainer_lab::trainer::{LayerSpec, MambaLayerParams, TrainerParams};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::json;

fn main() {
    let spec = LayerSpec {
        d_model: 16,
        d_state: 16,
        d_conv: 4,
    };
    let mut rng = StdRng::seed_from_u64(23);
    let mut params = TrainerParams {
        embedding: Array2::from_shape_fn((64, 16), |(v, d)| 0.01 * (1 + v + d) as f32),
        layers: vec![
            MambaLayerParams::random(spec, &mut rng),
            MambaLayerParams::random(spec, &mut rng),
            MambaLayerParams::random(spec, &mut rng),
        ],
    };
    let prototypes = Array2::from_shape_fn((64, 16), |(k, d)| 0.02 * (1 + k + d) as f32);
    let ids = Array2::from_shape_fn((2, 6), |(b, t)| ((b * 6 + t) % 32) as i64);
    let targets = Array2::from_shape_fn((2, 6), |(b, t)| ((b * 6 + t + 1) % 32) as i64);
    let frozen_before = params.layers[0].out_proj_w.clone();
    let stats = supervised_residual_step(&mut params, &prototypes, &ids, &targets, 1e-3, &[0], false);
    let frozen_unchanged = params.layers[0].out_proj_w == frozen_before;

    let out = json!({
        "loss": stats.loss,
        "embedding_grad_norm": stats.embedding_grad_norm,
        "top_grad_norm": stats.top_grad_norm,
        "frozen_unchanged": frozen_unchanged,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}