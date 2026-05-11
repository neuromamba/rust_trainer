use crate::layer::{backward as layer_backward, forward_with_cache, LayerForwardCache, LayerGrads};
use crate::loss::{
    cagradstep, gradnorm_ff_scale, pcgrad, GradientSurgeryConfig, GradientSurgeryMethod,
};
use crate::nn::{hpn_loss_and_grad_z, hpn_loss_and_grads, layer_norm_backward, layer_norm_forward};
use crate::optim::{adamw_update_1d, adamw_update_2d, Adam1, Adam2};
use crate::trainer::{
    expand_layers_in_place, resolve_freeze_indices, AdamWConfig, ExpansionConfig,
    ExpansionPlacement, FreezeSelection, LayerSpec, MambaLayerParams, TrainerParams,
};
use ndarray::{Array1, Array2, Array3};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

const GENERIC_TRAINER_CKPT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GenericTrainerCheckpoint {
    version: u32,
    trainer: GenericTrainer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericTrainerConfig {
    pub vocab_size: usize,
    pub layer_spec: LayerSpec,
    pub expansion: ExpansionConfig,
    pub freeze_selection: FreezeSelection,
    pub freeze_embedding: bool,
    pub adamw: AdamWConfig,
    #[serde(default = "default_ff_lr")]
    pub ff_lr: f32,
    #[serde(default = "default_bp_cadence_steps")]
    pub bp_cadence_steps: usize,
    #[serde(default)]
    pub gradient_surgery: GradientSurgeryConfig,
    pub grad_clip_norm: Option<f32>,
    pub fail_on_non_finite: bool,
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
    pub grad_global_norm: f32,
    pub lr: f32,
    pub ff_loss_mean: f32,
    pub bp_applied: bool,
    pub ff_updates_applied: usize,
    pub bp_updates_applied: usize,
    pub conflict_layers: usize,
    pub surgery_method: String,
    pub clipped: bool,
    pub skipped_update: bool,
    pub non_finite_detected: bool,
}

fn default_ff_lr() -> f32 {
    1e-4
}

fn default_bp_cadence_steps() -> usize {
    1
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
        let payload = GenericTrainerCheckpoint {
            version: GENERIC_TRAINER_CKPT_VERSION,
            trainer: self.clone(),
        };
        let bytes = bincode::serde::encode_to_vec(payload, bincode::config::standard())
            .map_err(|err| format!("serialize failed: {err}"))?;
        atomic_write_bytes(path.as_ref(), &bytes)
    }

    pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|err| format!("checkpoint read failed: {err}"))?;
        if let Ok((decoded, _bytes_read)) = bincode::serde::decode_from_slice::<
            GenericTrainerCheckpoint,
            _,
        >(&bytes, bincode::config::standard())
        {
            if decoded.version != GENERIC_TRAINER_CKPT_VERSION {
                return Err(format!(
                    "unsupported checkpoint version: {}",
                    decoded.version
                ));
            }
            return Ok(decoded.trainer);
        }

        // Backward compatibility for pre-versioned checkpoints.
        let (legacy, _bytes_read) =
            bincode::serde::decode_from_slice::<Self, _>(&bytes, bincode::config::standard())
                .map_err(|err| format!("deserialize failed: {err}"))?;
        Ok(legacy)
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

        // Build FF activations from layer inputs (x_in) using a lightweight
        // corruption scheme for negatives (time shift + alternating sign mask).
        let mut ff_grads = Vec::with_capacity(self.params.layers.len());
        let mut ff_losses = Vec::with_capacity(self.params.layers.len());
        let mut ff_updates_applied = 0usize;
        for (li, layer) in self.params.layers.iter().enumerate() {
            if self.frozen_layer_indices.binary_search(&li).is_ok() {
                ff_losses.push(0.0);
                ff_grads.push(Array1::<f32>::zeros(layer.d_skip.len()));
                continue;
            }
            let cache = &caches[li];
            let (b, t, d) = (
                cache.x_in.shape()[0],
                cache.x_in.shape()[1],
                cache.x_in.shape()[2],
            );
            let denom = (b * t) as f32;
            let mut h_pos = Array1::<f32>::zeros(d);
            let mut h_neg = Array1::<f32>::zeros(d);
            for bi in 0..b {
                for ti in 0..t {
                    let src_t = (ti + 1) % t;
                    for di in 0..d {
                        let pv = cache.x_in[(bi, ti, di)];
                        let mask = if (ti + di) % 2 == 0 { 1.0 } else { -1.0 };
                        let nv = cache.x_in[(bi, src_t, di)] * mask;
                        h_pos[di] += pv;
                        h_neg[di] += nv;
                    }
                }
            }
            h_pos.mapv_inplace(|v| v / denom);
            h_neg.mapv_inplace(|v| v / denom);
            let ff_loss = MambaLayerParams::ff_loss_pair(&h_pos, &h_neg, &layer.d_skip);
            let ff_grad = MambaLayerParams::ff_grad_theta(&h_pos, &h_neg, &layer.d_skip);
            if ff_loss > 0.0 {
                ff_updates_applied += 1;
            }
            ff_losses.push(ff_loss);
            ff_grads.push(ff_grad);
        }
        let ff_loss_mean = if ff_losses.is_empty() {
            0.0
        } else {
            ff_losses.iter().copied().sum::<f32>() / ff_losses.len() as f32
        };

        let cadence = self.cfg.bp_cadence_steps.max(1);
        let bp_due = (self.step + 1).is_multiple_of(cadence);

        // FF-only step: update theta (d_skip) locally and skip global BP.
        if !bp_due {
            #[allow(clippy::needless_range_loop)]
            for li in 0..self.params.layers.len() {
                if self.frozen_layer_indices.binary_search(&li).is_ok() {
                    continue;
                }
                let g = &ff_grads[li];
                for i in 0..self.params.layers[li].d_skip.len() {
                    self.params.layers[li].d_skip[i] -= self.cfg.ff_lr * g[i];
                }
            }
            self.step += 1;
            let ff_grad_norm_sq = ff_grads
                .iter()
                .map(|g| g.iter().map(|v| v * v).sum::<f32>())
                .sum::<f32>();
            return StepStats {
                step: self.step,
                loss: ff_loss_mean,
                embedding_grad_norm: 0.0,
                prototype_grad_norm: 0.0,
                top_grad_norm: 0.0,
                grad_global_norm: ff_grad_norm_sq.sqrt(),
                lr: self.cfg.adamw.lr,
                ff_loss_mean,
                bp_applied: false,
                ff_updates_applied,
                bp_updates_applied: 0,
                conflict_layers: 0,
                surgery_method: format!("{:?}", self.cfg.gradient_surgery.method).to_lowercase(),
                clipped: false,
                skipped_update: false,
                non_finite_detected: !ff_loss_mean.is_finite(),
            };
        }

        let (x_ln, ln_cache) = layer_norm_forward(residual.view());
        let z_flat = x_ln
            .clone()
            .into_shape_with_order((batch * seq_len, d_model))
            .expect("flatten ln output");
        let tgt_flat = targets.iter().copied().collect::<Vec<_>>();
        let (loss, dz_flat, mut d_prototypes) =
            hpn_loss_and_grads(z_flat.view(), &tgt_flat, &self.prototypes);
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

        // Blend FF gradients into BP gradients for d_skip using selected surgery.
        let mut conflict_layers = 0usize;
        let ff_to_bp_grad_scale = if self.cfg.adamw.lr.abs() > 1e-12 {
            self.cfg.ff_lr / self.cfg.adamw.lr
        } else {
            1.0
        };
        for li in 0..self.params.layers.len() {
            if self.frozen_layer_indices.binary_search(&li).is_ok() {
                continue;
            }
            let ff_grad = &ff_grads[li];
            let bp_grad = layer_grads[li].d_skip.clone();
            if ff_grad.dot(&bp_grad) < 0.0 {
                conflict_layers += 1;
            }
            let ff_after_surgery = match self.cfg.gradient_surgery.method {
                GradientSurgeryMethod::PcGrad => {
                    pcgrad(ff_grad, &bp_grad, self.cfg.gradient_surgery.epsilon)
                }
                GradientSurgeryMethod::GradNorm => {
                    let scale = gradnorm_ff_scale(
                        ff_grad,
                        &bp_grad,
                        self.cfg.gradient_surgery.gradnorm_alpha,
                        self.cfg.gradient_surgery.epsilon,
                    );
                    ff_grad * scale
                }
                GradientSurgeryMethod::CAGradStep => cagradstep(
                    ff_grad,
                    &bp_grad,
                    self.cfg.gradient_surgery.cagrad_lambda,
                    self.cfg.gradient_surgery.epsilon,
                ),
            };
            layer_grads[li].d_skip += &(ff_after_surgery * ff_to_bp_grad_scale);
        }

        let embedding_grad_norm = embedding_grads.iter().map(|v| v * v).sum::<f32>().sqrt();
        let prototype_grad_norm = d_prototypes.iter().map(|v| v * v).sum::<f32>().sqrt();

        let mut grad_global_norm_sq = embedding_grads.iter().map(|v| v * v).sum::<f32>();
        grad_global_norm_sq += d_prototypes.iter().map(|v| v * v).sum::<f32>();
        for grads in &layer_grads {
            grad_global_norm_sq += layer_grads_l2_sq(grads);
        }
        let grad_global_norm = grad_global_norm_sq.sqrt();

        let non_finite_detected = !loss.is_finite()
            || !embedding_grad_norm.is_finite()
            || !prototype_grad_norm.is_finite()
            || !top_grad_norm.is_finite()
            || !grad_global_norm.is_finite();

        if non_finite_detected {
            if self.cfg.fail_on_non_finite {
                panic!("non-finite detected during train_step");
            }
            return StepStats {
                step: self.step,
                loss,
                embedding_grad_norm,
                prototype_grad_norm,
                top_grad_norm,
                grad_global_norm,
                lr: self.cfg.adamw.lr,
                ff_loss_mean,
                bp_applied: true,
                ff_updates_applied,
                bp_updates_applied: self.params.layers.len() - self.frozen_layer_indices.len(),
                conflict_layers,
                surgery_method: format!("{:?}", self.cfg.gradient_surgery.method).to_lowercase(),
                clipped: false,
                skipped_update: true,
                non_finite_detected: true,
            };
        }

        let mut clipped = false;
        if let Some(clip_norm) = self.cfg.grad_clip_norm {
            if clip_norm > 0.0 && grad_global_norm > clip_norm {
                let scale = clip_norm / grad_global_norm;
                embedding_grads.mapv_inplace(|v| v * scale);
                d_prototypes.mapv_inplace(|v| v * scale);
                for grads in &mut layer_grads {
                    scale_layer_grads(grads, scale);
                }
                clipped = true;
            }
        }

        self.apply_updates(&embedding_grads, &layer_grads, &d_prototypes);

        self.step += 1;
        StepStats {
            step: self.step,
            loss,
            embedding_grad_norm,
            prototype_grad_norm,
            top_grad_norm,
            grad_global_norm,
            lr: self.cfg.adamw.lr,
            ff_loss_mean,
            bp_applied: true,
            ff_updates_applied,
            bp_updates_applied: self.params.layers.len() - self.frozen_layer_indices.len(),
            conflict_layers,
            surgery_method: format!("{:?}", self.cfg.gradient_surgery.method).to_lowercase(),
            clipped,
            skipped_update: false,
            non_finite_detected: false,
        }
    }

    pub fn eval_step(&self, ids: &Array2<i64>, targets: &Array2<i64>) -> f32 {
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

        let mut residual = x;
        for layer in &self.params.layers {
            let (h, _cache) = forward_with_cache(layer, residual.view());
            residual = &residual + &h;
        }

        let (x_ln, _ln_cache) = layer_norm_forward(residual.view());
        let z_flat = x_ln
            .into_shape_with_order((batch * seq_len, d_model))
            .expect("flatten ln output");
        let tgt_flat = targets.iter().copied().collect::<Vec<_>>();
        let (loss, _dz) = hpn_loss_and_grad_z(z_flat.view(), &tgt_flat, &self.prototypes);
        loss
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

        #[allow(clippy::needless_range_loop)]
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
        self.params
            .layers
            .iter()
            .map(MambaLayerParams::l2_norm)
            .collect()
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
        ff_lr: lr,
        bp_cadence_steps: 1,
        gradient_surgery: GradientSurgeryConfig::default(),
        grad_clip_norm: None,
        fail_on_non_finite: false,
    }
}

fn layer_grads_l2_sq(grads: &LayerGrads) -> f32 {
    grads.a_log.iter().map(|v| v * v).sum::<f32>()
        + grads.d_skip.iter().map(|v| v * v).sum::<f32>()
        + grads.x_proj_w.iter().map(|v| v * v).sum::<f32>()
        + grads.dt_proj_w.iter().map(|v| v * v).sum::<f32>()
        + grads.dt_proj_b.iter().map(|v| v * v).sum::<f32>()
        + grads.conv1d_w.iter().map(|v| v * v).sum::<f32>()
        + grads.conv1d_b.iter().map(|v| v * v).sum::<f32>()
        + grads.out_proj_w.iter().map(|v| v * v).sum::<f32>()
}

fn scale_layer_grads(grads: &mut LayerGrads, scale: f32) {
    grads.a_log.mapv_inplace(|v| v * scale);
    grads.d_skip.mapv_inplace(|v| v * scale);
    grads.x_proj_w.mapv_inplace(|v| v * scale);
    grads.dt_proj_w.mapv_inplace(|v| v * scale);
    grads.dt_proj_b.mapv_inplace(|v| v * scale);
    grads.conv1d_w.mapv_inplace(|v| v * scale);
    grads.conv1d_b.mapv_inplace(|v| v * scale);
    grads.out_proj_w.mapv_inplace(|v| v * scale);
}

fn atomic_write_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "checkpoint path has no parent directory".to_string())?;
    fs::create_dir_all(parent).map_err(|err| format!("checkpoint dir create failed: {err}"))?;

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| format!("time error: {err}"))?
        .as_nanos();
    let pid = std::process::id();
    let tmp_name = format!(".tmp_ckpt_{pid}_{stamp}");
    let tmp_path = parent.join(tmp_name);

    fs::write(&tmp_path, bytes).map_err(|err| format!("tmp checkpoint write failed: {err}"))?;
    fs::rename(&tmp_path, path).map_err(|err| format!("atomic checkpoint rename failed: {err}"))
}

pub fn make_batch_from_tokens(
    tokens: &[i64],
    cursor: usize,
    batch: usize,
    seq_len: usize,
) -> (Array2<i64>, Array2<i64>) {
    assert!(
        tokens.len() > seq_len + 1,
        "token stream too short for seq_len"
    );
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
    let raw =
        fs::read_to_string(input).map_err(|err| format!("failed to read token file: {err}"))?;
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
    frozen.iter().all(|idx| {
        (*idx < before.len()) && (*idx < after.len()) && (before[*idx] - after[*idx]).abs() <= tol
    })
}

pub fn grad_l2_1d(v: &Array1<f32>) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::GradientSurgeryMethod;

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

    #[test]
    fn eval_step_returns_finite_loss() {
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
        let trainer = GenericTrainer::new_random(cfg, 2, 321);
        let tokens = (0..256).map(|v| (v % 32) as i64).collect::<Vec<_>>();
        let (ids, tgt) = make_batch_from_tokens(&tokens, 0, 2, 6);
        let loss = trainer.eval_step(&ids, &tgt);
        assert!(loss.is_finite());
    }

    #[test]
    fn grad_clip_activates_with_tiny_threshold() {
        let spec = LayerSpec {
            d_model: 8,
            d_state: 8,
            d_conv: 4,
        };
        let mut cfg = default_trainer_config(
            32,
            spec,
            6,
            ExpansionPlacement::Append,
            FreezeSelection::FirstN(2),
            false,
            1e-3,
        );
        cfg.grad_clip_norm = Some(1e-6);
        let mut trainer = GenericTrainer::new_random(cfg, 2, 777);
        let tokens = (0..256).map(|v| (v % 32) as i64).collect::<Vec<_>>();
        let (ids, tgt) = make_batch_from_tokens(&tokens, 0, 2, 6);
        let stats = trainer.train_step(&ids, &tgt);
        assert!(stats.clipped);
    }

    #[test]
    fn cadence_skips_bp_until_due() {
        let spec = LayerSpec {
            d_model: 8,
            d_state: 8,
            d_conv: 4,
        };
        let mut cfg = default_trainer_config(
            32,
            spec,
            6,
            ExpansionPlacement::Append,
            FreezeSelection::FirstN(2),
            false,
            1e-3,
        );
        cfg.bp_cadence_steps = 3;
        let mut trainer = GenericTrainer::new_random(cfg, 2, 808);
        let tokens = (0..256).map(|v| (v % 32) as i64).collect::<Vec<_>>();
        let (ids, tgt) = make_batch_from_tokens(&tokens, 0, 2, 6);

        let s1 = trainer.train_step(&ids, &tgt);
        let s2 = trainer.train_step(&ids, &tgt);
        let s3 = trainer.train_step(&ids, &tgt);

        assert!(!s1.bp_applied);
        assert!(!s2.bp_applied);
        assert!(s3.bp_applied);
    }

    #[test]
    fn surgery_method_switches_are_active() {
        let spec = LayerSpec {
            d_model: 8,
            d_state: 8,
            d_conv: 4,
        };
        let mut cfg = default_trainer_config(
            32,
            spec,
            6,
            ExpansionPlacement::Append,
            FreezeSelection::FirstN(2),
            false,
            1e-3,
        );
        cfg.bp_cadence_steps = 1;
        cfg.gradient_surgery.method = GradientSurgeryMethod::GradNorm;
        let mut trainer = GenericTrainer::new_random(cfg, 2, 909);
        let tokens = (0..256).map(|v| (v % 32) as i64).collect::<Vec<_>>();
        let (ids, tgt) = make_batch_from_tokens(&tokens, 0, 2, 6);
        let s = trainer.train_step(&ids, &tgt);
        assert!(s.bp_applied);
        assert_eq!(s.surgery_method, "gradnorm");
    }
}
