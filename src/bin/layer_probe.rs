use ndarray::Array3;
use neuromamba_trainer_lab::layer::{backward, forward_with_cache};
use neuromamba_trainer_lab::trainer::{LayerSpec, MambaLayerParams};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::json;

fn main() {
    let spec = LayerSpec {
        d_model: 16,
        d_state: 16,
        d_conv: 4,
    };
    let mut rng = StdRng::seed_from_u64(13);
    let layer = MambaLayerParams::random(spec, &mut rng);
    let x = Array3::from_shape_fn((2, 6, 16), |(b, t, d)| 0.01 * (1 + b + t + d) as f32);
    let dy = Array3::from_shape_fn((2, 6, 16), |(b, t, d)| 0.02 * (1 + b + t + d) as f32);
    let (y, cache) = forward_with_cache(&layer, x.view());
    let (dx, grads) = backward(&layer, dy.view(), &cache);

    let out = json!({
        "y_shape": y.dim(),
        "dx_shape": dx.dim(),
        "dx_norm": dx.iter().map(|v| v * v).sum::<f32>().sqrt(),
        "grad_out_proj_norm": grads.out_proj_w.iter().map(|v| v * v).sum::<f32>().sqrt(),
        "grad_x_proj_norm": grads.x_proj_w.iter().map(|v| v * v).sum::<f32>().sqrt(),
        "grad_conv_norm": grads.conv1d_w.iter().map(|v| v * v).sum::<f32>().sqrt(),
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}