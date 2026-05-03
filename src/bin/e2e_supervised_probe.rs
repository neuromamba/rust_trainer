use ndarray::{Array2, Array3};
use neuromamba_trainer_lab::nn::{hpn_loss_and_grad_z, layer_norm_backward, layer_norm_forward};
use neuromamba_trainer_lab::optim::{adamw_update_2d, Adam2};
use serde_json::json;

fn main() {
    let mut hidden = Array3::from_shape_fn((2, 3, 8), |(b, t, d)| 0.05 * (1 + b + t + d) as f32);
    let prototypes = Array2::from_shape_fn((16, 8), |(k, d)| 0.02 * (1 + k + d) as f32);
    let targets = vec![0, 1, 2, 3, 4, 5];

    let (x_ln, ln_cache) = layer_norm_forward(hidden.view());
    let z_flat = x_ln
        .clone()
        .into_shape_with_order((6, 8))
        .expect("reshape hidden to flat tokens");
    let (loss_before, dz_flat) = hpn_loss_and_grad_z(z_flat.view(), &targets, &prototypes);
    let dx_ln = dz_flat
        .into_shape_with_order((2, 3, 8))
        .expect("reshape grad to sequence");
    let dx = layer_norm_backward(dx_ln.view(), &ln_cache);

    let mut hidden2d = hidden
        .clone()
        .into_shape_with_order((6, 8))
        .expect("flatten hidden param");
    let grad2d = dx
        .clone()
        .into_shape_with_order((6, 8))
        .expect("flatten hidden grad");
    let grad_norm: f32 = grad2d.iter().map(|v| v * v).sum::<f32>().sqrt();

    let mut opt = Adam2::zeros(6, 8);
    adamw_update_2d(&mut hidden2d, &grad2d, &mut opt, 1e-3, 0.9, 0.999, 1e-8, 0.01, 0);

    hidden = hidden2d
        .into_shape_with_order((2, 3, 8))
        .expect("reshape updated hidden");
    let (x_ln_after, _) = layer_norm_forward(hidden.view());
    let z_flat_after = x_ln_after
        .into_shape_with_order((6, 8))
        .expect("reshape hidden after update");
    let (loss_after, _) = hpn_loss_and_grad_z(z_flat_after.view(), &targets, &prototypes);

    let out = json!({
        "loss_before": loss_before,
        "loss_after": loss_after,
        "grad_norm": grad_norm,
        "loss_delta": loss_after - loss_before,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}