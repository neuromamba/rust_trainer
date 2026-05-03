use crate::layer::{backward as layer_backward, forward_with_cache, LayerForwardCache, LayerGrads};
use crate::nn::{hpn_loss_and_grads, layer_norm_backward, layer_norm_forward};
use crate::optim::{adamw_update_1d, adamw_update_2d, Adam1, Adam2};
use crate::trainer::{
    expand_layers_in_place, resolve_freeze_indices, AdamWConfig, ExpansionConfig, ExpansionPlacement,
    FreezeSelection, LayerSpec, MambaLayerParams, TrainerParams,
};
use ndarray::{Array1, Array2, Array3};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericTrainerConfig {
    pub vocab_size: usize,
    pub layer_spec: LayerSpec,
    pub expansion: ExpansionConfig,
    pub freeze_selection: FreezeSelection,
    pub freeze_embedding: bool,
    pub adamw: AdamWConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAdamState {
    pub a_log: Adam2,
    pub d_skip: Adam1,
    pub x_proj_w: Adam2,
    pub dt_proj_w: Adam2,
    pub dt_proj_b: Adam1,
    pub conv1d_w: Adam2,
    pub conv1d_b: Adam1,
    pub out_proj_w: Adam2,
}

impl LayerAdamState {
    pub fn zeros_like(layer: &MambaLayerParams) -> Self {
        let a_log_dim = layer.a_log.dim();
        let d_skip_len = layer.d_skip.len();
        let x_proj_dim = layer.x_proj_w.dim();
        let dt_proj_w_dim = layer.dt_proj_w.dim();
        let dt_proj_b_len = layer.dt_proj_b.len();
        let conv1d_w_dim = layer.conv1d_w.dim();
        let conv1d_b_len = layer.conv1d_b.len();
        let out_proj_dim = layer.out_proj_w.dim();
        Self {
            a_log: Adam2::zeros(a_log_dim.0, a_log_dim.1),
            d_skip: Adam1::zeros(d_skip_len),
            x_proj_w: Adam2::zeros(x_proj_dim.0, x_proj_dim.1),
            dt_proj_w: Adam2::zeros(dt_proj_w_dim.0, dt_proj_w_dim.1),
            dt_proj_b: Adam1::zeros(dt_proj_b_len),
            conv1d_w: Adam2::zeros(conv1d_w_dim.0, conv1d_w_dim.1),
            conv1d_b: Adam1::zeros(conv1d_b_len),
            out_proj_w: Adam2::zeros(out_proj_dim.0, out_proj_dim.1),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MambaHpnOptimizerState {
    pub embedding: Adam2,
    pub prototypes: Adam2,
    pub layers: Vec<LayerAdamState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericTrainer {
    pub cfg: GenericTrainerConfig,
    pub params: TrainerParams,
    pub prototypes: Array2<f32>,
    pub optimizer: MambaHpnOptimizerState,
    pub frozen_layer_indices: Vec<usize>,
    pub step: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepStats {
    pub step: usize,
    pub loss: f32,
    pub embedding_grad_norm: f32,
    pub prototype_grad_norm: f32,
    pub top_grad_norm: f32,
}

impl GenericTrainer {
    pub fn new_random(cfg: GenericTrainerConfig, base_layers: usize, seed: u64) -> Self {
        let mut base = TrainerParams::random(cfg.vocab_size, cfg.layer_spec, base_layers, seed);
        expand_layers_in_place(
            &mut base.layers,
            cfg.layer_spec,
            cfg.expansion.target_num_layers,
            &cfg.expansion.placement,
        );
        let frozen_layer_indices = resolve_freeze_indices(&cfg.freeze_selection, base.layers.len());

        let mut rng = StdRng::seed_from_u64(seed ^ 0x5a5a_1234_8765_4321);
        let prototypes = Array2::from_shape_fn((cfg.vocab_size, cfg.layer_spec.d_model), |_| {
            rng.sample::<f32, _>(StandardNormal) * 0.02
        });

        let embedding_dim = base.embedding.dim();
        let proto_dim = prototypes.dim();
        let layer_states = base
            .layers
            .iter()
            .map(LayerAdamState::zeros_like)
            .collect::<Vec<_>>();
        let optimizer = MambaHpnOptimizerState {
            embedding: Adam2::zeros(embedding_dim.0, embedding_dim.1),
            prototypes: Adam2::zeros(proto_dim.0, proto_dim.1),
            layers: layer_states,
        };

        Self {
            cfg,
            params: base,
            prototypes,
            optimizer,
            frozen_layer_indices,
            step: 0,
        }
    }

    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let bytes = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|err| format!("serialize failed: {err}"))?;
        fs::write(path, bytes).map_err(|err| format!("checkpoint write failed: {err}"))
    }

    pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|err| format!("checkpoint read failed: {err}"))?;
        let (decoded, _bytes_read) =
            bincode::serde::decode_from_slice::<Self, _>(&bytes, bincode::config::standard())
                .map_err(|err| format!("deserialize failed: {err}"))?;
        Ok(decoded)
    }

    pub fn train_step(&mut self, ids: &Array2<i64>, targets: &Array2<i64>) -> StepStats {
        let (batch, seq_len) = (ids.shape()[0], ids.shape()[1]);
        let d_model = self.params.embedding.shape()[1];

        let mut x = Array3::<f32>::zeros((batch, seq_len, d_model));
        for b in 0..batch {
            for t in 0..seq_len {
                let tok = ids[(b, t)].rem_euclid(self.params.embedding.shape()[0] as i64) as usize;
                for d in 0..d_model {
                    x[(b, t, d)] = self.params.embedding[(tok, d)];
                }
            }
        }

        let mut residual = x.clone();
        let mut caches: Vec<LayerForwardCache> = Vec::with_capacity(self.params.layers.len());
        for layer in &self.params.layers {
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
        let (loss, dz_flat, d_prototypes) = hpn_loss_and_grads(z_flat.view(), &tgt_flat, &self.prototypes);
        let dx_ln = dz_flat
            .into_shape_with_order((batch, seq_len, d_model))
            .expect("reshape dz");
        let mut dx = layer_norm_backward(dx_ln.view(), &ln_cache);
        let top_grad_norm = dx.iter().map(|v| v * v).sum::<f32>().sqrt();

        let mut layer_grads = self
            .params
            .layers
            .iter()
            .map(LayerGrads::zeros_like)
            .collect::<Vec<_>>();
        for li in (0..self.params.layers.len()).rev() {
            let (dx_input, grads) = layer_backward(&self.params.layers[li], dx.view(), &caches[li]);
            layer_grads[li] = grads;
            dx = &dx + &dx_input;
        }

        let mut embedding_grads = Array2::<f32>::zeros(self.params.embedding.dim());
        for b in 0..batch {
            for t in 0..seq_len {
                let tok = ids[(b, t)].rem_euclid(self.params.embedding.shape()[0] as i64) as usize;
                for d in 0..d_model {
                    embedding_grads[(tok, d)] += dx[(b, t, d)];
                }
            }
        }

        let embedding_grad_norm = embedding_grads.iter().map(|v| v * v).sum::<f32>().sqrt();
        let prototype_grad_norm = d_prototypes.iter().map(|v| v * v).sum::<f32>().sqrt();
        self.apply_updates(&embedding_grads, &layer_grads, &d_prototypes);

        self.step += 1;
        StepStats {
            step: self.step,
            loss,
            embedding_grad_norm,
            prototype_grad_norm,
            top_grad_norm,
        }
    }

    fn apply_updates(
        &mut self,
        embedding_grads: &Array2<f32>,
        layer_grads: &[LayerGrads],
        prototype_grads: &Array2<f32>,
    ) {
        let opt = &self.cfg.adamw;
        let step = self.step;

        if !self.cfg.freeze_embedding {
            adamw_update_2d(
                &mut self.params.embedding,
                embedding_grads,
                &mut self.optimizer.embedding,
                opt.lr,
                opt.beta1,
                opt.beta2,
                opt.eps,
                opt.weight_decay,
                step,
            );
        }

        adamw_update_2d(
            &mut self.prototypes,
            prototype_grads,
            &mut self.optimizer.prototypes,
            opt.lr,
            opt.beta1,
            opt.beta2,
            opt.eps,
            opt.weight_decay,
            step,
        );

        for li in 0..self.params.layers.len() {
            if self.frozen_layer_indices.binary_search(&li).is_ok() {
                continue;
            }
            let layer = &mut self.params.layers[li];
            let grads = &layer_grads[li];
            let st = &mut self.optimizer.layers[li];
            adamw_update_2d(
                &mut layer.a_log,
                &grads.a_log,
                &mut st.a_log,
                opt.lr,
                opt.beta1,
                opt.beta2,
                opt.eps,
                opt.weight_decay,
                step,
            );
            adamw_update_1d(
                &mut layer.d_skip,
                &grads.d_skip,
                &mut st.d_skip,
                opt.lr,
                opt.beta1,
                opt.beta2,
                opt.eps,
                opt.weight_decay,
                step,
            );
            adamw_update_2d(
                &mut layer.x_proj_w,
                &grads.x_proj_w,
                &mut st.x_proj_w,
                opt.lr,
                opt.beta1,
                opt.beta2,
                opt.eps,
                opt.weight_decay,
                step,
            );
            adamw_update_2d(
                &mut layer.dt_proj_w,
                &grads.dt_proj_w,
                &mut st.dt_proj_w,
                opt.lr,
                opt.beta1,
                opt.beta2,
                opt.eps,
                opt.weight_decay,
                step,
            );
            adamw_update_1d(
                &mut layer.dt_proj_b,
                &grads.dt_proj_b,
                &mut st.dt_proj_b,
                opt.lr,
                opt.beta1,
                opt.beta2,
                opt.eps,
                opt.weight_decay,
                step,
            );
            adamw_update_2d(
                &mut layer.conv1d_w,
                &grads.conv1d_w,
                &mut st.conv1d_w,
                opt.lr,
                opt.beta1,
                opt.beta2,
                opt.eps,
                opt.weight_decay,
                step,
            );
            adamw_update_1d(
                &mut layer.conv1d_b,
                &grads.conv1d_b,
                &mut st.conv1d_b,
                opt.lr,
                opt.beta1,
                opt.beta2,
                opt.eps,
                opt.weight_decay,
                step,
            );
            adamw_update_2d(
                &mut layer.out_proj_w,
                &grads.out_proj_w,
                &mut st.out_proj_w,
                opt.lr,
                opt.beta1,
                opt.beta2,
                opt.eps,
                opt.weight_decay,
                step,
            );
        }
    }

    pub fn layer_l2_norms(&self) -> Vec<f32> {
        self.params.layers.iter().map(MambaLayerParams::l2_norm).collect()
    }
}

pub fn default_trainer_config(
    vocab_size: usize,
    layer_spec: LayerSpec,
    target_layers: usize,
    placement: ExpansionPlacement,
    freeze: FreezeSelection,
    freeze_embedding: bool,
    lr: f32,
) -> GenericTrainerConfig {
    GenericTrainerConfig {
        vocab_size,
        layer_spec,
        expansion: ExpansionConfig {
            target_num_layers: target_layers,
            placement,
        },
        freeze_selection: freeze,
        freeze_embedding,
        adamw: AdamWConfig {
            lr,
            ..AdamWConfig::default()
        },
    }
}

pub fn make_batch_from_tokens(tokens: &[i64], cursor: usize, batch: usize, seq_len: usize) -> (Array2<i64>, Array2<i64>) {
    assert!(tokens.len() > seq_len + 1, "token stream too short for seq_len");
    let mut ids = Array2::<i64>::zeros((batch, seq_len));
    let mut targets = Array2::<i64>::zeros((batch, seq_len));
    let max_start = tokens.len() - seq_len - 1;
    for b in 0..batch {
        let start = (cursor + b * seq_len) % max_start;
        for t in 0..seq_len {
            ids[(b, t)] = tokens[start + t];
            targets[(b, t)] = tokens[start + t + 1];
        }
    }
    (ids, targets)
}

pub fn tokenize_int_file(input: &str) -> Result<Vec<i64>, String> {
    let raw = fs::read_to_string(input).map_err(|err| format!("failed to read token file: {err}"))?;
    let mut out = Vec::new();
    for part in raw.split_whitespace() {
        let parsed = part
            .parse::<i64>()
            .map_err(|err| format!("bad token '{part}': {err}"))?;
        out.push(parsed);
    }
    if out.is_empty() {
        return Err("token file contained zero integer tokens".to_string());
    }
    Ok(out)
}

pub fn parse_placement(raw: &str) -> ExpansionPlacement {
    if raw == "append" {
        return ExpansionPlacement::Append;
    }
    if raw == "prepend" {
        return ExpansionPlacement::Prepend;
    }
    if let Some(value) = raw.strip_prefix("insert:") {
        return ExpansionPlacement::InsertAt {
            index: value.parse().unwrap_or(0),
        };
    }
    if let Some(value) = raw.strip_prefix("specific:") {
        let positions = value
            .split(',')
            .filter_map(|item| item.parse::<usize>().ok())
            .collect::<Vec<_>>();
        return ExpansionPlacement::SpecificPositions(positions);
    }
    ExpansionPlacement::Append
}

pub fn parse_freeze(raw: &str) -> FreezeSelection {
    if let Some(value) = raw.strip_prefix("first:") {
        return FreezeSelection::FirstN(value.parse().unwrap_or(2));
    }
    if let Some(value) = raw.strip_prefix("indices:") {
        let indices = value
            .split(',')
            .filter_map(|item| item.parse::<usize>().ok())
            .collect::<Vec<_>>();
        return FreezeSelection::Indices(indices);
    }
    FreezeSelection::FirstN(2)
}

pub fn max_token_plus_one(tokens: &[i64]) -> usize {
    tokens
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .saturating_add(1)
        .max(1) as usize
}

pub fn mean_layer_norm(norms: &[f32]) -> f32 {
    if norms.is_empty() {
        return 0.0;
    }
    norms.iter().copied().sum::<f32>() / norms.len() as f32
}

pub fn is_frozen_unchanged(before: &[f32], after: &[f32], frozen: &[usize], tol: f32) -> bool {
    frozen
        .iter()
        .all(|idx| (*idx < before.len()) && (*idx < after.len()) && (before[*idx] - after[*idx]).abs() <= tol)
}

pub fn grad_l2_1d(v: &Array1<f32>) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_resume_is_deterministic_next_step() {
        let spec = LayerSpec {
            d_model: 8,
            d_state: 8,
            d_conv: 4,
        };
        let cfg = default_trainer_config(
            32,
            spec,
            6,
            ExpansionPlacement::Append,
            FreezeSelection::FirstN(2),
            false,
            1e-3,
        );
        let mut trainer_a = GenericTrainer::new_random(cfg, 2, 123);
        let tokens = (0..256).map(|v| (v % 32) as i64).collect::<Vec<_>>();
        let (ids1, tgt1) = make_batch_from_tokens(&tokens, 0, 2, 6);
        let _ = trainer_a.train_step(&ids1, &tgt1);

        let ckpt = std::env::temp_dir().join("generic_trainer_resume_det.bincode");
        trainer_a.save_checkpoint(&ckpt).unwrap();
        let mut trainer_b = GenericTrainer::load_checkpoint(&ckpt).unwrap();

        let (ids2, tgt2) = make_batch_from_tokens(&tokens, 12, 2, 6);
        let a = trainer_a.train_step(&ids2, &tgt2);
        let b = trainer_b.train_step(&ids2, &tgt2);

        assert!((a.loss - b.loss).abs() <= 1e-8);
        assert!((a.embedding_grad_norm - b.embedding_grad_norm).abs() <= 1e-8);
        assert!((a.prototype_grad_norm - b.prototype_grad_norm).abs() <= 1e-8);
        assert_eq!(trainer_a.step, trainer_b.step);
        let emb_err = (&trainer_a.params.embedding - &trainer_b.params.embedding)
            .mapv(f32::abs)
            .sum();
        assert!(emb_err <= 1e-8);
        let _ = std::fs::remove_file(&ckpt);
    }
}