use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerSpec {
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FreezeSelection {
    FirstN(usize),
    Indices(Vec<usize>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpansionPlacement {
    Append,
    Prepend,
    InsertAt { index: usize },
    SpecificPositions(Vec<usize>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpansionConfig {
    pub target_num_layers: usize,
    pub placement: ExpansionPlacement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentalTrainerConfig {
    pub vocab_size: usize,
    pub layer_spec: LayerSpec,
    pub expansion: ExpansionConfig,
    pub freeze_selection: FreezeSelection,
    pub freeze_embedding: bool,
    pub ff_lr: f32,
    pub ff_threshold: f32,
    pub adamw: AdamWConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MambaLayerParams {
    pub a_log: Array2<f32>,
    pub d_skip: Array1<f32>,
    pub x_proj_w: Array2<f32>,
    pub dt_proj_w: Array2<f32>,
    pub dt_proj_b: Array1<f32>,
    pub conv1d_w: Array2<f32>,
    pub conv1d_b: Array1<f32>,
    pub out_proj_w: Array2<f32>,
}

impl MambaLayerParams {
    pub fn random(spec: LayerSpec, rng: &mut StdRng) -> Self {
        Self {
            a_log: random_matrix(spec.d_model, spec.d_state, rng, 0.02),
            d_skip: random_vector(spec.d_model, rng, 0.02),
            x_proj_w: random_matrix(spec.d_model, spec.d_state * 2 + 1, rng, 0.02),
            dt_proj_w: random_matrix(1, spec.d_model, rng, 0.02),
            dt_proj_b: random_vector(spec.d_model, rng, 0.02),
            conv1d_w: random_matrix(spec.d_model, spec.d_conv, rng, 0.02),
            conv1d_b: random_vector(spec.d_model, rng, 0.02),
            out_proj_w: random_matrix(spec.d_model, spec.d_model, rng, 0.02),
        }
    }

    pub fn zero_residual(spec: LayerSpec) -> Self {
        Self {
            a_log: Array2::zeros((spec.d_model, spec.d_state)),
            d_skip: Array1::zeros(spec.d_model),
            x_proj_w: Array2::zeros((spec.d_model, spec.d_state * 2 + 1)),
            dt_proj_w: Array2::zeros((1, spec.d_model)),
            dt_proj_b: Array1::zeros(spec.d_model),
            conv1d_w: Array2::zeros((spec.d_model, spec.d_conv)),
            conv1d_b: Array1::zeros(spec.d_model),
            out_proj_w: Array2::zeros((spec.d_model, spec.d_model)),
        }
    }

    pub fn l2_norm(&self) -> f32 {
        let mut total = 0.0;
        total += self.a_log.iter().map(|v| v * v).sum::<f32>();
        total += self.d_skip.iter().map(|v| v * v).sum::<f32>();
        total += self.x_proj_w.iter().map(|v| v * v).sum::<f32>();
        total += self.dt_proj_w.iter().map(|v| v * v).sum::<f32>();
        total += self.dt_proj_b.iter().map(|v| v * v).sum::<f32>();
        total += self.conv1d_w.iter().map(|v| v * v).sum::<f32>();
        total += self.conv1d_b.iter().map(|v| v * v).sum::<f32>();
        total += self.out_proj_w.iter().map(|v| v * v).sum::<f32>();
        total.sqrt()
    }

    pub fn ff_goodness(&self) -> f32 {
        let denom = (self.d_skip.len()
            + self.dt_proj_b.len()
            + self.conv1d_b.len()
            + self.a_log.len()
            + self.x_proj_w.len()
            + self.dt_proj_w.len()
            + self.conv1d_w.len()
            + self.out_proj_w.len()) as f32;
        if denom == 0.0 {
            0.0
        } else {
            self.l2_norm() / denom.sqrt()
        }
    }

    pub fn ff_local_update(&mut self, maximize: bool, lr: f32, threshold: f32) -> f32 {
        let goodness = self.ff_goodness();
        let err = if maximize {
            threshold - goodness
        } else {
            goodness - threshold
        };
        if err > 0.0 {
            if maximize && goodness <= 1e-12 {
                let diag = self.out_proj_w.nrows().min(self.out_proj_w.ncols());
                for idx in 0..diag {
                    self.out_proj_w[(idx, idx)] += lr * threshold.max(1e-6);
                }
            } else {
                let scale = if maximize { 1.0 + lr } else { 1.0 - lr };
                self.scale_all(scale.max(0.0));
            }
        }
        self.ff_goodness()
    }

    fn scale_all(&mut self, scale: f32) {
        self.a_log.mapv_inplace(|v| v * scale);
        self.d_skip.mapv_inplace(|v| v * scale);
        self.x_proj_w.mapv_inplace(|v| v * scale);
        self.dt_proj_w.mapv_inplace(|v| v * scale);
        self.dt_proj_b.mapv_inplace(|v| v * scale);
        self.conv1d_w.mapv_inplace(|v| v * scale);
        self.conv1d_b.mapv_inplace(|v| v * scale);
        self.out_proj_w.mapv_inplace(|v| v * scale);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerParams {
    pub embedding: Array2<f32>,
    pub layers: Vec<MambaLayerParams>,
}

impl TrainerParams {
    pub fn random(vocab_size: usize, spec: LayerSpec, num_layers: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let embedding = random_matrix(vocab_size, spec.d_model, &mut rng, 0.02);
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(MambaLayerParams::random(spec, &mut rng));
        }
        Self { embedding, layers }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentalTrainer {
    pub cfg: ExperimentalTrainerConfig,
    pub params: TrainerParams,
    frozen_embedding: Option<Array2<f32>>,
    frozen_layer_indices: Vec<usize>,
    frozen_layers: Vec<MambaLayerParams>,
    pub step: usize,
}

impl ExperimentalTrainer {
    pub fn from_base(base: TrainerParams, cfg: ExperimentalTrainerConfig) -> Self {
        let mut params = base;
        expand_layers_in_place(
            &mut params.layers,
            cfg.layer_spec,
            cfg.expansion.target_num_layers,
            &cfg.expansion.placement,
        );

        let frozen_layer_indices = resolve_freeze_indices(&cfg.freeze_selection, params.layers.len());
        let frozen_layers = frozen_layer_indices
            .iter()
            .map(|&idx| params.layers[idx].clone())
            .collect::<Vec<_>>();
        let frozen_embedding = if cfg.freeze_embedding {
            Some(params.embedding.clone())
        } else {
            None
        };

        Self {
            cfg,
            params,
            frozen_embedding,
            frozen_layer_indices,
            frozen_layers,
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

    pub fn expanded_layer_count(&self) -> usize {
        self.params.layers.len()
    }

    pub fn frozen_layer_indices(&self) -> &[usize] {
        &self.frozen_layer_indices
    }

    pub fn enforce_freeze(&mut self) {
        if let Some(frozen_embedding) = &self.frozen_embedding {
            self.params.embedding.assign(frozen_embedding);
        }
        for (pos, &layer_idx) in self.frozen_layer_indices.iter().enumerate() {
            self.params.layers[layer_idx] = self.frozen_layers[pos].clone();
        }
    }

    pub fn train_ff_cycle(&mut self) -> Vec<f32> {
        let mut goodness = Vec::with_capacity(self.params.layers.len());
        for idx in 0..self.params.layers.len() {
            if self.frozen_layer_indices.binary_search(&idx).is_ok() {
                goodness.push(self.params.layers[idx].ff_goodness());
                continue;
            }
            let g = self.params.layers[idx].ff_local_update(true, self.cfg.ff_lr, self.cfg.ff_threshold);
            goodness.push(g);
        }
        self.enforce_freeze();
        self.step += 1;
        goodness
    }

    pub fn layer_norms(&self) -> Vec<f32> {
        self.params.layers.iter().map(MambaLayerParams::l2_norm).collect()
    }
}

pub fn resolve_freeze_indices(selection: &FreezeSelection, total_layers: usize) -> Vec<usize> {
    let mut indices = match selection {
        FreezeSelection::FirstN(count) => (0..(*count).min(total_layers)).collect::<Vec<_>>(),
        FreezeSelection::Indices(raw) => raw
            .iter()
            .copied()
            .filter(|idx| *idx < total_layers)
            .collect::<Vec<_>>(),
    };
    indices.sort_unstable();
    indices.dedup();
    indices
}

pub fn expand_layers_in_place(
    base_layers: &mut Vec<MambaLayerParams>,
    spec: LayerSpec,
    target_num_layers: usize,
    placement: &ExpansionPlacement,
) {
    assert!(target_num_layers >= base_layers.len(), "target layers must be >= base layers");
    let base_count = base_layers.len();
    let new_count = target_num_layers - base_count;
    if new_count == 0 {
        return;
    }

    let new_layers = (0..new_count)
        .map(|_| MambaLayerParams::zero_residual(spec))
        .collect::<Vec<_>>();

    match placement {
        ExpansionPlacement::Append => {
            base_layers.extend(new_layers);
        }
        ExpansionPlacement::Prepend => {
            let mut merged = new_layers;
            merged.extend(base_layers.clone());
            *base_layers = merged;
        }
        ExpansionPlacement::InsertAt { index } => {
            let insert_at = (*index).min(base_layers.len());
            let tail = base_layers.split_off(insert_at);
            base_layers.extend(new_layers);
            base_layers.extend(tail);
        }
        ExpansionPlacement::SpecificPositions(positions) => {
            assert!(
                positions.len() == new_count,
                "SpecificPositions length must equal number of new layers"
            );
            let mut final_positions = positions.clone();
            final_positions.sort_unstable();
            final_positions.dedup();
            assert!(
                final_positions.len() == new_count,
                "SpecificPositions must contain unique indices"
            );
            assert!(
                final_positions.iter().all(|idx| *idx < target_num_layers),
                "SpecificPositions indices must be < target_num_layers"
            );

            let old_layers = base_layers.clone();
            let mut old_cursor = 0usize;
            let mut new_cursor = 0usize;
            let mut merged = Vec::with_capacity(target_num_layers);
            for final_idx in 0..target_num_layers {
                if final_positions.binary_search(&final_idx).is_ok() {
                    merged.push(new_layers[new_cursor].clone());
                    new_cursor += 1;
                } else {
                    merged.push(old_layers[old_cursor].clone());
                    old_cursor += 1;
                }
            }
            *base_layers = merged;
        }
    }
}

fn random_matrix(rows: usize, cols: usize, rng: &mut StdRng, std: f32) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |_| rng.sample::<f32, _>(StandardNormal) * std)
}

fn random_vector(len: usize, rng: &mut StdRng, std: f32) -> Array1<f32> {
    Array1::from_shape_fn(len, |_| rng.sample::<f32, _>(StandardNormal) * std)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> LayerSpec {
        LayerSpec {
            d_model: 16,
            d_state: 8,
            d_conv: 4,
        }
    }

    fn cfg_with(placement: ExpansionPlacement, freeze: FreezeSelection) -> ExperimentalTrainerConfig {
        ExperimentalTrainerConfig {
            vocab_size: 64,
            layer_spec: spec(),
            expansion: ExpansionConfig {
                target_num_layers: 6,
                placement,
            },
            freeze_selection: freeze,
            freeze_embedding: true,
            ff_lr: 1e-2,
            ff_threshold: 1e-3,
            adamw: AdamWConfig::default(),
        }
    }

    #[test]
    fn append_expansion_adds_new_layers_at_end() {
        let base = TrainerParams::random(64, spec(), 2, 7);
        let trainer = ExperimentalTrainer::from_base(
            base,
            cfg_with(ExpansionPlacement::Append, FreezeSelection::FirstN(2)),
        );
        assert_eq!(trainer.expanded_layer_count(), 6);
        for layer in &trainer.params.layers[2..] {
            assert!(layer.l2_norm() <= 1e-9);
        }
    }

    #[test]
    fn specific_positions_support_interleave_style_insertions() {
        let base = TrainerParams::random(64, spec(), 2, 9);
        let trainer = ExperimentalTrainer::from_base(
            base,
            cfg_with(
                ExpansionPlacement::SpecificPositions(vec![1, 3, 4, 5]),
                FreezeSelection::Indices(vec![0, 2]),
            ),
        );
        assert_eq!(trainer.expanded_layer_count(), 6);
        assert_eq!(trainer.frozen_layer_indices(), &[0, 2]);
        assert!(trainer.params.layers[1].l2_norm() <= 1e-9);
        assert!(trainer.params.layers[3].l2_norm() <= 1e-9);
        assert!(trainer.params.layers[4].l2_norm() <= 1e-9);
        assert!(trainer.params.layers[5].l2_norm() <= 1e-9);
    }

    #[test]
    fn freeze_by_indices_is_enforced_after_training_cycle() {
        let base = TrainerParams::random(64, spec(), 2, 11);
        let mut trainer = ExperimentalTrainer::from_base(
            base,
            cfg_with(
                ExpansionPlacement::Append,
                FreezeSelection::Indices(vec![0, 1]),
            ),
        );
        let before = trainer.layer_norms();
        let _ = trainer.train_ff_cycle();
        let after = trainer.layer_norms();

        assert!((before[0] - after[0]).abs() <= 1e-9);
        assert!((before[1] - after[1]).abs() <= 1e-9);
        assert!(after[2..]
            .iter()
            .zip(before[2..].iter())
            .any(|(a, b)| (a - b).abs() > 1e-9));
    }

    #[test]
    fn checkpoint_roundtrip_preserves_state() {
        let base = TrainerParams::random(64, spec(), 2, 5);
        let mut trainer = ExperimentalTrainer::from_base(
            base,
            cfg_with(ExpansionPlacement::Append, FreezeSelection::FirstN(2)),
        );
        let _ = trainer.train_ff_cycle();
        let ckpt = std::env::temp_dir().join("trainer_lab_roundtrip.bincode");
        trainer.save_checkpoint(&ckpt).unwrap();
        let loaded = ExperimentalTrainer::load_checkpoint(&ckpt).unwrap();
        assert_eq!(trainer.step, loaded.step);
        assert_eq!(trainer.expanded_layer_count(), loaded.expanded_layer_count());
        assert_eq!(trainer.frozen_layer_indices(), loaded.frozen_layer_indices());
        let _ = std::fs::remove_file(&ckpt);
    }
}