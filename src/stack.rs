use crate::layer::{backward as layer_backward, forward_with_cache, LayerForwardCache, LayerGrads};
use crate::nn::{hpn_loss_and_grad_z, layer_norm_backward, layer_norm_forward};
use crate::trainer::TrainerParams;
use ndarray::{Array2, Array3};

fn sgd_update_2d(param: &mut Array2<f32>, grad: &Array2<f32>, lr: f32) {
    for (p, g) in param.iter_mut().zip(grad.iter()) {
        *p -= lr * *g;
    }
}

fn sgd_update_1d(param: &mut ndarray::Array1<f32>, grad: &ndarray::Array1<f32>, lr: f32) {
    for (p, g) in param.iter_mut().zip(grad.iter()) {
        *p -= lr * *g;
    }
}

fn apply_layer_grads(layer: &mut crate::trainer::MambaLayerParams, grads: &LayerGrads, lr: f32) {
    sgd_update_2d(&mut layer.a_log, &grads.a_log, lr);
    sgd_update_1d(&mut layer.d_skip, &grads.d_skip, lr);
    sgd_update_2d(&mut layer.x_proj_w, &grads.x_proj_w, lr);
    sgd_update_2d(&mut layer.dt_proj_w, &grads.dt_proj_w, lr);
    sgd_update_1d(&mut layer.dt_proj_b, &grads.dt_proj_b, lr);
    sgd_update_2d(&mut layer.conv1d_w, &grads.conv1d_w, lr);
    sgd_update_1d(&mut layer.conv1d_b, &grads.conv1d_b, lr);
    sgd_update_2d(&mut layer.out_proj_w, &grads.out_proj_w, lr);
}

#[derive(Debug)]
pub struct SupervisedStepStats {
    pub loss: f32,
    pub embedding_grad_norm: f32,
    pub top_grad_norm: f32,
}

pub fn supervised_residual_step(
    params: &mut TrainerParams,
    prototypes: &Array2<f32>,
    ids: &Array2<i64>,
    targets: &Array2<i64>,
    lr: f32,
    frozen_layer_indices: &[usize],
    freeze_embedding: bool,
) -> SupervisedStepStats {
    let (batch, seq_len) = (ids.shape()[0], ids.shape()[1]);
    let d_model = params.embedding.shape()[1];

    let mut x = Array3::<f32>::zeros((batch, seq_len, d_model));
    for b in 0..batch {
        for t in 0..seq_len {
            let tok = ids[(b, t)].rem_euclid(params.embedding.shape()[0] as i64) as usize;
            for d in 0..d_model {
                x[(b, t, d)] = params.embedding[(tok, d)];
            }
        }
    }

    let mut residual = x.clone();
    let mut caches: Vec<LayerForwardCache> = Vec::with_capacity(params.layers.len());
    for layer in &params.layers {
        let (h, cache) = forward_with_cache(layer, residual.view());
        residual = &residual + &h;
        caches.push(cache);
    }

    let (x_ln, ln_cache) = layer_norm_forward(residual.view());
    let z_flat = x_ln
        .clone()
        .into_shape_with_order((batch * seq_len, d_model))
        .expect("flatten ln output");
    let tgt_flat = targets.iter().copied().collect::<Vec<_>>();
    let (loss, dz_flat) = hpn_loss_and_grad_z(z_flat.view(), &tgt_flat, prototypes);
    let dx_ln = dz_flat
        .into_shape_with_order((batch, seq_len, d_model))
        .expect("reshape dz");
    let mut dx = layer_norm_backward(dx_ln.view(), &ln_cache);
    let top_grad_norm = dx.iter().map(|v| v * v).sum::<f32>().sqrt();

    for li in (0..params.layers.len()).rev() {
        let (dx_input, grads) = layer_backward(&params.layers[li], dx.view(), &caches[li]);
        if frozen_layer_indices.binary_search(&li).is_err() {
            apply_layer_grads(&mut params.layers[li], &grads, lr);
        }
        dx = &dx + &dx_input;
    }

    let mut embedding_grads = Array2::<f32>::zeros(params.embedding.dim());
    for b in 0..batch {
        for t in 0..seq_len {
            let tok = ids[(b, t)].rem_euclid(params.embedding.shape()[0] as i64) as usize;
            for d in 0..d_model {
                embedding_grads[(tok, d)] += dx[(b, t, d)];
            }
        }
    }
    let embedding_grad_norm = embedding_grads.iter().map(|v| v * v).sum::<f32>().sqrt();
    if !freeze_embedding {
        sgd_update_2d(&mut params.embedding, &embedding_grads, lr);
    }

    SupervisedStepStats {
        loss,
        embedding_grad_norm,
        top_grad_norm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::{LayerSpec, MambaLayerParams};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn supervised_step_runs_and_preserves_frozen_layer() {
        let spec = LayerSpec { d_model: 8, d_state: 8, d_conv: 4 };
        let mut rng = StdRng::seed_from_u64(19);
        let mut params = TrainerParams {
            embedding: Array2::from_shape_fn((32, 8), |(v, d)| 0.01 * (1 + v + d) as f32),
            layers: vec![
                MambaLayerParams::random(spec, &mut rng),
                MambaLayerParams::random(spec, &mut rng),
            ],
        };
        let frozen_before = params.layers[0].out_proj_w.clone();
        let prototypes = Array2::from_shape_fn((32, 8), |(k, d)| 0.02 * (1 + k + d) as f32);
        let ids = Array2::from_shape_fn((2, 4), |(b, t)| ((b * 4 + t) % 16) as i64);
        let targets = Array2::from_shape_fn((2, 4), |(b, t)| ((b * 4 + t + 1) % 16) as i64);
        let stats = supervised_residual_step(&mut params, &prototypes, &ids, &targets, 1e-3, &[0], false);
        assert!(stats.loss.is_finite());
        assert!(stats.embedding_grad_norm.is_finite());
        assert_eq!(params.layers[0].out_proj_w, frozen_before);
    }
}