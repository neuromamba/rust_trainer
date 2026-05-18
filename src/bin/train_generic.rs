use rust_trainer::data_stream::{
    max_token_plus_one_from_dir, MultiWorkerCursorState, MultiWorkerShardedBatcher,
    ShardedCursorState, ShardedTokenStream,
};
use rust_trainer::generic_trainer::{
    default_trainer_config, make_batch_from_tokens, max_token_plus_one, parse_freeze,
    parse_placement, tokenize_int_file, GenericTrainer,
};
use rust_trainer::loss::GradientSurgeryMethod;
use rust_trainer::LayerSpec;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const RUN_STATE_VERSION: u32 = 1;

#[derive(Debug, Clone)]
struct Args {
    out_dir: String,
    config: Option<String>,
    steps: usize,
    save_every: usize,
    log_every: usize,
    batch_size: usize,
    seq_len: usize,
    seed: u64,
    base_layers: usize,
    target_layers: usize,
    d_model: usize,
    d_state: usize,
    d_conv: usize,
    placement: String,
    freeze: String,
    lr: f32,
    freeze_embedding: bool,
    token_file: Option<String>,
    token_dir: Option<String>,
    val_token_file: Option<String>,
    val_token_dir: Option<String>,
    shard_ext: String,
    shuffle_shards: bool,
    packed_sequences: bool,
    prefetch_workers: usize,
    prefetch_buffer: usize,
    resume: Option<String>,
    vocab_size_override: Option<usize>,
    val_ratio: f32,
    val_every: usize,
    eval_batches: usize,
    early_stopping_patience: usize,
    grad_clip_norm: f32,
    fail_on_non_finite: bool,
    lr_warmup_steps: usize,
    lr_min_scale: f32,
    ff_lr: f32,
    bp_cadence_steps: usize,
    gradient_surgery_method: String,
    gradient_surgery_epsilon: f32,
    gradnorm_alpha: f32,
    cagrad_lambda: f32,
    human_log: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct FileConfig {
    total_steps: Option<usize>,
    seed: Option<u64>,
    dataset: Option<FileDatasetConfig>,
    model: Option<FileModelConfig>,
    ff: Option<FileFfConfig>,
    bp: Option<FileBpConfig>,
    gradient_surgery: Option<FileGradientSurgeryConfig>,
    logging: Option<FileLoggingConfig>,
}

#[derive(Debug, Clone, Deserialize)]
struct FileDatasetConfig {
    max_seq_len: Option<usize>,
    batch_size: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct FileModelConfig {
    num_layers: Option<usize>,
    d_model: Option<usize>,
    d_state: Option<usize>,
    d_conv: Option<usize>,
    num_classes: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct FileFfConfig {
    lr: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
struct FileBpConfig {
    cadence_steps: Option<usize>,
    lr: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
struct FileGradientSurgeryConfig {
    method: Option<String>,
    epsilon: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
struct FileLoggingConfig {
    log_dir: Option<String>,
    run_id: Option<String>,
    log_every: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunState {
    version: u32,
    train_cursor: usize,
    val_cursor: usize,
    train_sharded: Option<ShardedCursorState>,
    train_multiworker: Option<MultiWorkerCursorState>,
    val_sharded: Option<ShardedCursorState>,
    val_multiworker: Option<MultiWorkerCursorState>,
    best_val_loss: Option<f32>,
    no_improve: usize,
}

impl Default for RunState {
    fn default() -> Self {
        Self {
            version: RUN_STATE_VERSION,
            train_cursor: 0,
            val_cursor: 0,
            train_sharded: None,
            train_multiworker: None,
            val_sharded: None,
            val_multiworker: None,
            best_val_loss: None,
            no_improve: 0,
        }
    }
}

#[allow(clippy::large_enum_variant)]
enum BatchSource {
    InMemory {
        tokens: Vec<i64>,
        cursor: usize,
    },
    Sharded {
        stream: ShardedTokenStream,
        packed: bool,
    },
    ShardedMultiWorker {
        batcher: MultiWorkerShardedBatcher,
    },
}

impl BatchSource {
    fn next_batch(
        &mut self,
        batch: usize,
        seq_len: usize,
    ) -> Result<(ndarray::Array2<i64>, ndarray::Array2<i64>), String> {
        match self {
            BatchSource::InMemory { tokens, cursor } => {
                let out = make_batch_from_tokens(tokens, *cursor, batch, seq_len);
                *cursor = cursor.saturating_add(batch * seq_len);
                Ok(out)
            }
            BatchSource::Sharded { stream, packed } => {
                if *packed {
                    stream.next_packed_batch(batch, seq_len)
                } else {
                    stream.next_batch(batch, seq_len)
                }
            }
            BatchSource::ShardedMultiWorker { batcher } => batcher.next_batch(),
        }
    }

    fn set_cursor(&mut self, cursor: usize) {
        if let BatchSource::InMemory { cursor: c, .. } = self {
            *c = cursor;
        }
    }

    fn state_snapshot(&self) -> (Option<usize>, Option<ShardedCursorState>) {
        match self {
            BatchSource::InMemory { cursor, .. } => (Some(*cursor), None),
            BatchSource::Sharded { stream, .. } => (None, Some(stream.state())),
            BatchSource::ShardedMultiWorker { .. } => (Some(0), None),
        }
    }

    fn multiworker_state_snapshot(&self) -> Option<MultiWorkerCursorState> {
        match self {
            BatchSource::ShardedMultiWorker { batcher } => batcher.state(),
            _ => None,
        }
    }

    fn restore_sharded_state(&mut self, state: Option<ShardedCursorState>) -> Result<(), String> {
        if let BatchSource::Sharded { stream, .. } = self {
            if let Some(st) = state {
                stream.set_state(st)?;
            }
        }
        Ok(())
    }
}

fn parse_bool(raw: &str) -> bool {
    matches!(raw, "1" | "true" | "yes" | "y" | "on")
}

fn scheduled_lr(
    base_lr: f32,
    step: usize,
    total_steps: usize,
    warmup_steps: usize,
    min_scale: f32,
) -> f32 {
    let floor = min_scale.clamp(0.0, 1.0) * base_lr;
    if total_steps == 0 {
        return base_lr;
    }
    if warmup_steps > 0 && step < warmup_steps {
        let alpha = (step + 1) as f32 / warmup_steps as f32;
        return (base_lr * alpha).max(floor);
    }
    let denom = total_steps.saturating_sub(warmup_steps).max(1);
    let progressed = step.saturating_sub(warmup_steps).min(denom);
    let frac = progressed as f32 / denom as f32;
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * frac).cos());
    floor + (base_lr - floor) * cosine
}

fn load_file_config(path: &str) -> Result<FileConfig, String> {
    let raw = fs::read_to_string(path).map_err(|err| format!("config read failed: {err}"))?;
    if path.ends_with(".yaml") || path.ends_with(".yml") {
        serde_yaml::from_str::<FileConfig>(&raw).map_err(|err| format!("yaml parse failed: {err}"))
    } else if path.ends_with(".json") {
        serde_json::from_str::<FileConfig>(&raw).map_err(|err| format!("json parse failed: {err}"))
    } else {
        Err("config must end with .yaml/.yml or .json".to_string())
    }
}

fn apply_file_config(args: &mut Args, cfg: &FileConfig) {
    if let Some(v) = cfg.total_steps {
        args.steps = v;
    }
    if let Some(v) = cfg.seed {
        args.seed = v;
    }
    if let Some(dataset) = &cfg.dataset {
        if let Some(v) = dataset.max_seq_len {
            args.seq_len = v;
        }
        if let Some(v) = dataset.batch_size {
            args.batch_size = v;
        }
    }
    if let Some(model) = &cfg.model {
        if let Some(v) = model.num_layers {
            args.target_layers = v;
        }
        if let Some(v) = model.d_model {
            args.d_model = v;
        }
        if let Some(v) = model.d_state {
            args.d_state = v;
        }
        if let Some(v) = model.d_conv {
            args.d_conv = v;
        }
        if let Some(v) = model.num_classes {
            args.vocab_size_override = Some(v);
        }
    }
    if let Some(ff) = &cfg.ff {
        if let Some(v) = ff.lr {
            args.ff_lr = v;
        }
    }
    if let Some(bp) = &cfg.bp {
        if let Some(v) = bp.cadence_steps {
            args.bp_cadence_steps = v;
        }
        if let Some(v) = bp.lr {
            args.lr = v;
        }
    }
    if let Some(gs) = &cfg.gradient_surgery {
        if let Some(v) = &gs.method {
            args.gradient_surgery_method = v.clone();
        }
        if let Some(v) = gs.epsilon {
            args.gradient_surgery_epsilon = v;
        }
    }
    if let Some(logging) = &cfg.logging {
        if let Some(v) = logging.log_every {
            args.log_every = v;
        }
        if let (Some(log_dir), Some(run_id)) = (&logging.log_dir, &logging.run_id) {
            args.out_dir = format!("{}/{}", log_dir.trim_end_matches('/'), run_id);
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args {
        out_dir: "runs/RUST_TRAINER".to_string(),
        config: None,
        steps: 5000,
        save_every: 200,
        log_every: 20,
        batch_size: 8,
        seq_len: 64,
        seed: 42,
        base_layers: 2,
        target_layers: 6,
        d_model: 512,
        d_state: 16,
        d_conv: 4,
        placement: "specific:1,3,4,5".to_string(),
        freeze: "indices:".to_string(),
        lr: 1e-4,
        freeze_embedding: false,
        token_file: None,
        token_dir: None,
        val_token_file: None,
        val_token_dir: None,
        shard_ext: "txt".to_string(),
        shuffle_shards: true,
        packed_sequences: true,
        prefetch_workers: 0,
        prefetch_buffer: 16,
        resume: None,
        vocab_size_override: None,
        val_ratio: 0.05,
        val_every: 200,
        eval_batches: 8,
        early_stopping_patience: 0,
        grad_clip_norm: 0.0,
        fail_on_non_finite: false,
        lr_warmup_steps: 0,
        lr_min_scale: 0.1,
        ff_lr: 1e-4,
        bp_cadence_steps: 32,
        gradient_surgery_method: "pcgrad".to_string(),
        gradient_surgery_epsilon: 1e-8,
        gradnorm_alpha: 0.2,
        cagrad_lambda: 1.0,
        human_log: false,
    };

    let raw = env::args().skip(1).collect::<Vec<_>>();
    let mut i = 0usize;
    while i < raw.len() {
        match raw[i].as_str() {
            "--config" if i + 1 < raw.len() => {
                let cfg_path = raw[i + 1].clone();
                let cfg = load_file_config(&cfg_path)
                    .unwrap_or_else(|err| panic!("failed to load config {}: {}", cfg_path, err));
                args.config = Some(cfg_path);
                apply_file_config(&mut args, &cfg);
                i += 2;
            }
            "--out-dir" if i + 1 < raw.len() => {
                args.out_dir = raw[i + 1].clone();
                i += 2;
            }
            "--output-dir" if i + 1 < raw.len() => {
                args.out_dir = raw[i + 1].clone();
                i += 2;
            }
            "--steps" if i + 1 < raw.len() => {
                args.steps = raw[i + 1].parse().unwrap_or(args.steps);
                i += 2;
            }
            "--save-every" if i + 1 < raw.len() => {
                args.save_every = raw[i + 1].parse().unwrap_or(args.save_every);
                i += 2;
            }
            "--log-every" if i + 1 < raw.len() => {
                args.log_every = raw[i + 1].parse().unwrap_or(args.log_every);
                i += 2;
            }
            "--batch-size" if i + 1 < raw.len() => {
                args.batch_size = raw[i + 1].parse().unwrap_or(args.batch_size);
                i += 2;
            }
            "--seq-len" if i + 1 < raw.len() => {
                args.seq_len = raw[i + 1].parse().unwrap_or(args.seq_len);
                i += 2;
            }
            "--seed" if i + 1 < raw.len() => {
                args.seed = raw[i + 1].parse().unwrap_or(args.seed);
                i += 2;
            }
            "--base-layers" if i + 1 < raw.len() => {
                args.base_layers = raw[i + 1].parse().unwrap_or(args.base_layers);
                i += 2;
            }
            "--target-layers" if i + 1 < raw.len() => {
                args.target_layers = raw[i + 1].parse().unwrap_or(args.target_layers);
                i += 2;
            }
            "--d-model" if i + 1 < raw.len() => {
                args.d_model = raw[i + 1].parse().unwrap_or(args.d_model);
                i += 2;
            }
            "--d-state" if i + 1 < raw.len() => {
                args.d_state = raw[i + 1].parse().unwrap_or(args.d_state);
                i += 2;
            }
            "--d-conv" if i + 1 < raw.len() => {
                args.d_conv = raw[i + 1].parse().unwrap_or(args.d_conv);
                i += 2;
            }
            "--placement" if i + 1 < raw.len() => {
                args.placement = raw[i + 1].clone();
                i += 2;
            }
            "--freeze" if i + 1 < raw.len() => {
                args.freeze = raw[i + 1].clone();
                i += 2;
            }
            "--lr" if i + 1 < raw.len() => {
                args.lr = raw[i + 1].parse().unwrap_or(args.lr);
                i += 2;
            }
            "--freeze-embedding" if i + 1 < raw.len() => {
                args.freeze_embedding = parse_bool(&raw[i + 1]);
                i += 2;
            }
            "--token-file" if i + 1 < raw.len() => {
                args.token_file = Some(raw[i + 1].clone());
                i += 2;
            }
            "--token-dir" if i + 1 < raw.len() => {
                args.token_dir = Some(raw[i + 1].clone());
                i += 2;
            }
            "--val-token-file" if i + 1 < raw.len() => {
                args.val_token_file = Some(raw[i + 1].clone());
                i += 2;
            }
            "--val-token-dir" if i + 1 < raw.len() => {
                args.val_token_dir = Some(raw[i + 1].clone());
                i += 2;
            }
            "--shard-ext" if i + 1 < raw.len() => {
                args.shard_ext = raw[i + 1].clone();
                i += 2;
            }
            "--shuffle-shards" if i + 1 < raw.len() => {
                args.shuffle_shards = parse_bool(&raw[i + 1]);
                i += 2;
            }
            "--packed-sequences" if i + 1 < raw.len() => {
                args.packed_sequences = parse_bool(&raw[i + 1]);
                i += 2;
            }
            "--prefetch-workers" if i + 1 < raw.len() => {
                args.prefetch_workers = raw[i + 1].parse().unwrap_or(args.prefetch_workers);
                i += 2;
            }
            "--prefetch-buffer" if i + 1 < raw.len() => {
                args.prefetch_buffer = raw[i + 1].parse().unwrap_or(args.prefetch_buffer);
                i += 2;
            }
            "--resume" if i + 1 < raw.len() => {
                args.resume = Some(raw[i + 1].clone());
                i += 2;
            }
            "--base-ckpt" if i + 1 < raw.len() => {
                args.resume = Some(raw[i + 1].clone());
                i += 2;
            }
            "--vocab-size" if i + 1 < raw.len() => {
                args.vocab_size_override = raw[i + 1].parse::<usize>().ok();
                i += 2;
            }
            "--val-ratio" if i + 1 < raw.len() => {
                args.val_ratio = raw[i + 1].parse().unwrap_or(args.val_ratio);
                i += 2;
            }
            "--val-every" if i + 1 < raw.len() => {
                args.val_every = raw[i + 1].parse().unwrap_or(args.val_every);
                i += 2;
            }
            "--eval-batches" if i + 1 < raw.len() => {
                args.eval_batches = raw[i + 1].parse().unwrap_or(args.eval_batches);
                i += 2;
            }
            "--early-stopping-patience" if i + 1 < raw.len() => {
                args.early_stopping_patience =
                    raw[i + 1].parse().unwrap_or(args.early_stopping_patience);
                i += 2;
            }
            "--grad-clip-norm" if i + 1 < raw.len() => {
                args.grad_clip_norm = raw[i + 1].parse().unwrap_or(args.grad_clip_norm);
                i += 2;
            }
            "--fail-on-non-finite" if i + 1 < raw.len() => {
                args.fail_on_non_finite = parse_bool(&raw[i + 1]);
                i += 2;
            }
            "--lr-warmup-steps" if i + 1 < raw.len() => {
                args.lr_warmup_steps = raw[i + 1].parse().unwrap_or(args.lr_warmup_steps);
                i += 2;
            }
            "--lr-min-scale" if i + 1 < raw.len() => {
                args.lr_min_scale = raw[i + 1].parse().unwrap_or(args.lr_min_scale);
                i += 2;
            }
            "--ff-lr" if i + 1 < raw.len() => {
                args.ff_lr = raw[i + 1].parse().unwrap_or(args.ff_lr);
                i += 2;
            }
            "--bp-cadence-steps" if i + 1 < raw.len() => {
                args.bp_cadence_steps = raw[i + 1].parse().unwrap_or(args.bp_cadence_steps);
                i += 2;
            }
            "--gradient-surgery-method" if i + 1 < raw.len() => {
                args.gradient_surgery_method = raw[i + 1].clone();
                i += 2;
            }
            "--gradient-surgery-epsilon" if i + 1 < raw.len() => {
                args.gradient_surgery_epsilon =
                    raw[i + 1].parse().unwrap_or(args.gradient_surgery_epsilon);
                i += 2;
            }
            "--gradnorm-alpha" if i + 1 < raw.len() => {
                args.gradnorm_alpha = raw[i + 1].parse().unwrap_or(args.gradnorm_alpha);
                i += 2;
            }
            "--cagrad-lambda" if i + 1 < raw.len() => {
                args.cagrad_lambda = raw[i + 1].parse().unwrap_or(args.cagrad_lambda);
                i += 2;
            }
            "--human-log" => {
                args.human_log = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    args
}

fn parse_surgery_method(raw: &str) -> GradientSurgeryMethod {
    match raw.to_ascii_lowercase().as_str() {
        "pcgrad" => GradientSurgeryMethod::PcGrad,
        "gradnorm" => GradientSurgeryMethod::GradNorm,
        "cagradstep" | "cagrad" => GradientSurgeryMethod::CAGradStep,
        _ => GradientSurgeryMethod::PcGrad,
    }
}

fn load_run_state(path: &Path) -> Result<RunState, String> {
    let raw = fs::read_to_string(path).map_err(|err| format!("run_state read failed: {err}"))?;
    let st: RunState =
        serde_json::from_str(&raw).map_err(|err| format!("run_state parse failed: {err}"))?;
    if st.version != RUN_STATE_VERSION {
        return Err(format!("unsupported run_state version: {}", st.version));
    }
    Ok(st)
}

fn atomic_write_json(path: &Path, value: &serde_json::Value) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "path has no parent directory".to_string())?;
    fs::create_dir_all(parent).map_err(|err| format!("failed creating parent dir: {err}"))?;
    let tmp = parent.join(".tmp_run_state.json");
    fs::write(&tmp, serde_json::to_string_pretty(value).unwrap())
        .map_err(|err| format!("tmp state write failed: {err}"))?;
    fs::rename(&tmp, path).map_err(|err| format!("state rename failed: {err}"))
}

fn main() {
    let args = parse_args();
    fs::create_dir_all(&args.out_dir).expect("create output dir");

    let run_state_path = PathBuf::from(format!("{}/run_state.json", args.out_dir));
    let restored_state = if args.resume.is_some() && run_state_path.exists() {
        load_run_state(&run_state_path).ok()
    } else {
        None
    };

    let mut train_source: BatchSource;
    let mut val_source: Option<BatchSource> = None;

    let inferred_vocab_train = if let Some(dir) = &args.token_dir {
        if args.prefetch_workers > 1 {
            let batcher = MultiWorkerShardedBatcher::from_dir(
                dir,
                &args.shard_ext,
                args.shuffle_shards,
                args.seed,
                args.prefetch_workers,
                args.prefetch_buffer,
                args.batch_size,
                args.seq_len,
                args.packed_sequences,
                restored_state
                    .as_ref()
                    .and_then(|st| st.train_multiworker.clone()),
            )
            .expect("build train multiworker shard batcher");
            train_source = BatchSource::ShardedMultiWorker { batcher };
        } else {
            let stream =
                ShardedTokenStream::from_dir(dir, &args.shard_ext, args.shuffle_shards, args.seed)
                    .expect("build train shard stream");
            train_source = BatchSource::Sharded {
                stream,
                packed: args.packed_sequences,
            };
        }
        max_token_plus_one_from_dir(dir, &args.shard_ext).expect("infer vocab from train shards")
    } else {
        let tokens = if let Some(path) = &args.token_file {
            tokenize_int_file(path).expect("read token file")
        } else {
            (0..65536).map(|v| (v % 8192) as i64).collect::<Vec<_>>()
        };

        if let Some(path) = &args.val_token_file {
            let val_tokens = tokenize_int_file(path).expect("read validation token file");
            val_source = Some(BatchSource::InMemory {
                tokens: val_tokens,
                cursor: 0,
            });
            train_source = BatchSource::InMemory { tokens, cursor: 0 };
        } else {
            let ratio = args.val_ratio.clamp(0.0, 0.5);
            let raw_split = ((tokens.len() as f32) * (1.0 - ratio)) as usize;
            let split = raw_split.clamp(
                args.seq_len + 2,
                tokens.len().saturating_sub(args.seq_len + 2),
            );
            let train_tokens = tokens[..split].to_vec();
            let val_tokens = tokens[split..].to_vec();
            train_source = BatchSource::InMemory {
                tokens: train_tokens,
                cursor: 0,
            };
            if val_tokens.len() > args.seq_len + 1 {
                val_source = Some(BatchSource::InMemory {
                    tokens: val_tokens,
                    cursor: 0,
                });
            }
        }

        match &train_source {
            BatchSource::InMemory { tokens, .. } => max_token_plus_one(tokens),
            BatchSource::Sharded { .. } => unreachable!(),
            BatchSource::ShardedMultiWorker { .. } => unreachable!(),
        }
    };

    if let Some(dir) = &args.val_token_dir {
        if args.prefetch_workers > 1 {
            let batcher = MultiWorkerShardedBatcher::from_dir(
                dir,
                &args.shard_ext,
                args.shuffle_shards,
                args.seed ^ 0x11,
                args.prefetch_workers,
                args.prefetch_buffer,
                args.batch_size,
                args.seq_len,
                args.packed_sequences,
                restored_state
                    .as_ref()
                    .and_then(|st| st.val_multiworker.clone()),
            )
            .expect("build val multiworker shard batcher");
            val_source = Some(BatchSource::ShardedMultiWorker { batcher });
        } else {
            let stream = ShardedTokenStream::from_dir(
                dir,
                &args.shard_ext,
                args.shuffle_shards,
                args.seed ^ 0x11,
            )
            .expect("build val shard stream");
            val_source = Some(BatchSource::Sharded {
                stream,
                packed: args.packed_sequences,
            });
        }
    }

    let inferred_vocab_val = match &val_source {
        Some(BatchSource::InMemory { tokens, .. }) => max_token_plus_one(tokens),
        Some(BatchSource::Sharded { .. }) => {
            let dir = args
                .val_token_dir
                .as_ref()
                .expect("val shard dir exists when val sharded source exists");
            max_token_plus_one_from_dir(dir, &args.shard_ext).expect("infer vocab from val shards")
        }
        Some(BatchSource::ShardedMultiWorker { .. }) => {
            let dir = args
                .val_token_dir
                .as_ref()
                .expect("val shard dir exists when val sharded source exists");
            max_token_plus_one_from_dir(dir, &args.shard_ext).expect("infer vocab from val shards")
        }
        None => 1,
    };

    let vocab_size = args
        .vocab_size_override
        .unwrap_or_else(|| inferred_vocab_train.max(inferred_vocab_val));

    let mut trainer = if let Some(path) = &args.resume {
        if Path::new(path).exists() {
            GenericTrainer::load_checkpoint(path).expect("load checkpoint")
        } else {
            panic!("resume checkpoint does not exist: {path}");
        }
    } else {
        let spec = LayerSpec {
            d_model: args.d_model,
            d_state: args.d_state,
            d_conv: args.d_conv,
        };
        let cfg = default_trainer_config(
            vocab_size,
            spec,
            args.target_layers,
            parse_placement(&args.placement),
            parse_freeze(&args.freeze),
            args.freeze_embedding,
            args.lr,
        );
        GenericTrainer::new_random(cfg, args.base_layers, args.seed)
    };

    trainer.cfg.grad_clip_norm = if args.grad_clip_norm > 0.0 {
        Some(args.grad_clip_norm)
    } else {
        None
    };
    trainer.cfg.fail_on_non_finite = args.fail_on_non_finite;
    trainer.cfg.ff_lr = args.ff_lr;
    trainer.cfg.bp_cadence_steps = args.bp_cadence_steps.max(1);
    trainer.cfg.gradient_surgery.method = parse_surgery_method(&args.gradient_surgery_method);
    trainer.cfg.gradient_surgery.epsilon = args.gradient_surgery_epsilon;
    trainer.cfg.gradient_surgery.gradnorm_alpha = args.gradnorm_alpha;
    trainer.cfg.gradient_surgery.cagrad_lambda = args.cagrad_lambda;

    let metrics_path = format!("{}/metrics.jsonl", args.out_dir);
    let ckpt_path = format!("{}/latest.bincode", args.out_dir);
    let best_ckpt_path = format!("{}/best.bincode", args.out_dir);
    let mut best_val = f32::INFINITY;
    let mut no_improve = 0usize;
    let mut stopped_early = false;
    let mut last_val_loss: Option<f32> = None;

    if let Some(st) = restored_state {
        train_source.set_cursor(st.train_cursor);
        let _ = train_source.restore_sharded_state(st.train_sharded);
        if let Some(ref mut vsrc) = val_source {
            vsrc.set_cursor(st.val_cursor);
            let _ = vsrc.restore_sharded_state(st.val_sharded);
        }
        if let Some(v) = st.best_val_loss {
            best_val = v;
        }
        no_improve = st.no_improve;
    }

    let train_start = Instant::now();

    for local_step in 0..args.steps {
        trainer.cfg.adamw.lr = scheduled_lr(
            args.lr,
            local_step,
            args.steps,
            args.lr_warmup_steps,
            args.lr_min_scale,
        );

        let (ids, targets) = train_source
            .next_batch(args.batch_size, args.seq_len)
            .expect("get train batch");
        let stats = trainer.train_step(&ids, &targets);
        let is_last = local_step + 1 == args.steps;

        if local_step % args.log_every == 0 || is_last {
            let rec = json!({
                "step": stats.step,
                "loss": stats.loss,
                "embedding_grad_norm": stats.embedding_grad_norm,
                "prototype_grad_norm": stats.prototype_grad_norm,
                "top_grad_norm": stats.top_grad_norm,
                "grad_global_norm": stats.grad_global_norm,
                "lr": stats.lr,
                "ff_loss_mean": stats.ff_loss_mean,
                "bp_applied": stats.bp_applied,
                "ff_updates_applied": stats.ff_updates_applied,
                "bp_updates_applied": stats.bp_updates_applied,
                "conflict_layers": stats.conflict_layers,
                "surgery_method": stats.surgery_method,
                "clipped": stats.clipped,
                "skipped_update": stats.skipped_update,
                "non_finite_detected": stats.non_finite_detected,
                "layers": trainer.params.layers.len(),
                "frozen": trainer.frozen_layer_indices,
            });
            if args.human_log {
                let elapsed_s = train_start.elapsed().as_secs_f64().max(1e-9);
                let sps = stats.step as f64 / elapsed_s;
                let tok_s = sps * (args.batch_size * args.seq_len) as f64;
                let learn = if stats.ff_updates_applied > 0 || stats.bp_updates_applied > 0 {
                    "yes"
                } else {
                    "no"
                };
                let val_field = last_val_loss
                    .map(|v| format!(" val={v:.4}"))
                    .unwrap_or_default();
                println!(
                    "step {}/{} train={:.4}{} learn={} sps={:.2} tok/s={:.0}",
                    stats.step,
                    args.steps,
                    stats.loss,
                    val_field,
                    learn,
                    sps,
                    tok_s
                );
            } else {
                println!("{}", serde_json::to_string_pretty(&rec).unwrap());
            }
            let mut f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&metrics_path)
                .expect("open metrics jsonl");
            writeln!(f, "{}", serde_json::to_string(&rec).unwrap()).expect("append metrics line");
        }

        if local_step % args.save_every == 0 || is_last {
            trainer
                .save_checkpoint(&ckpt_path)
                .expect("save checkpoint");
        }

        let do_val = val_source.is_some() && (local_step % args.val_every == 0 || is_last);
        if do_val {
            let eval_steps = args.eval_batches.max(1);
            let mut val_sum = 0.0f32;
            for _ in 0..eval_steps {
                let (vids, vtgt) = val_source
                    .as_mut()
                    .expect("val source exists")
                    .next_batch(args.batch_size, args.seq_len)
                    .expect("get val batch");
                val_sum += trainer.eval_step(&vids, &vtgt);
            }
            let val_loss = val_sum / eval_steps as f32;
            let val_rec = json!({
                "step": trainer.step,
                "val_loss": val_loss,
                "best_val_loss": best_val,
            });
            last_val_loss = Some(val_loss);
            if args.human_log {
                // Keep console output minimal; detailed validation history stays in metrics.jsonl.
            } else {
                println!("{}", serde_json::to_string_pretty(&val_rec).unwrap());
            }
            let mut f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&metrics_path)
                .expect("open metrics jsonl");
            writeln!(f, "{}", serde_json::to_string(&val_rec).unwrap())
                .expect("append val metrics line");

            if val_loss < best_val {
                best_val = val_loss;
                no_improve = 0;
                trainer
                    .save_checkpoint(&best_ckpt_path)
                    .expect("save best checkpoint");
            } else {
                no_improve = no_improve.saturating_add(1);
            }

            if args.early_stopping_patience > 0 && no_improve >= args.early_stopping_patience {
                stopped_early = true;
                break;
            }
        }

        if local_step % args.save_every == 0 || is_last {
            let (train_cursor_opt, train_sharded) = train_source.state_snapshot();
            let train_multiworker = train_source.multiworker_state_snapshot();
            let (val_cursor_opt, val_sharded) = if let Some(vsrc) = &val_source {
                vsrc.state_snapshot()
            } else {
                (Some(0), None)
            };
            let val_multiworker = if let Some(vsrc) = &val_source {
                vsrc.multiworker_state_snapshot()
            } else {
                None
            };
            let run_state = RunState {
                version: RUN_STATE_VERSION,
                train_cursor: train_cursor_opt.unwrap_or(0),
                val_cursor: val_cursor_opt.unwrap_or(0),
                train_sharded,
                train_multiworker,
                val_sharded,
                val_multiworker,
                best_val_loss: if best_val.is_finite() {
                    Some(best_val)
                } else {
                    None
                },
                no_improve,
            };
            let state_json = serde_json::to_value(run_state).expect("serialize run_state");
            atomic_write_json(&run_state_path, &state_json).expect("save run_state");
        }
    }

    let summary = json!({
        "final_step": trainer.step,
        "stopped_early": stopped_early,
        "best_val_loss": if best_val.is_finite() { Some(best_val) } else { None::<f32> },
        "layers": trainer.params.layers.len(),
        "frozen": trainer.frozen_layer_indices,
        "ff_lr": trainer.cfg.ff_lr,
        "bp_cadence_steps": trainer.cfg.bp_cadence_steps,
        "gradient_surgery_method": format!("{:?}", trainer.cfg.gradient_surgery.method).to_lowercase(),
        "checkpoint": ckpt_path,
        "best_checkpoint": best_ckpt_path,
        "metrics": metrics_path,
        "run_state": run_state_path,
    });
    let summary_path = format!("{}/summary.json", args.out_dir);
    fs::write(
        &summary_path,
        serde_json::to_string_pretty(&summary).expect("serialize summary"),
    )
    .expect("write summary");
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
}
