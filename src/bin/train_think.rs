use rust_trainer_lab::think_trainer::{
    default_think_config, make_batch_from_tokens, max_token_plus_one, parse_freeze, parse_placement,
    tokenize_int_file, ThinkTrainer,
};
use rust_trainer_lab::LayerSpec;
use serde_json::json;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone)]
struct Args {
    out_dir: String,
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
    resume: Option<String>,
    vocab_size_override: Option<usize>,
}

fn parse_bool(raw: &str) -> bool {
    matches!(raw, "1" | "true" | "yes" | "y" | "on")
}

fn parse_args() -> Args {
    let mut args = Args {
        out_dir: "runs/RUST_THINK".to_string(),
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
        freeze: "first:2".to_string(),
        lr: 1e-4,
        freeze_embedding: false,
        token_file: None,
        resume: None,
        vocab_size_override: None,
    };

    let raw = env::args().skip(1).collect::<Vec<_>>();
    let mut i = 0usize;
    while i < raw.len() {
        match raw[i].as_str() {
            "--out-dir" if i + 1 < raw.len() => {
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
            "--resume" if i + 1 < raw.len() => {
                args.resume = Some(raw[i + 1].clone());
                i += 2;
            }
            "--vocab-size" if i + 1 < raw.len() => {
                args.vocab_size_override = raw[i + 1].parse::<usize>().ok();
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    args
}

fn main() {
    let args = parse_args();
    fs::create_dir_all(&args.out_dir).expect("create output dir");

    let tokens = if let Some(path) = &args.token_file {
        tokenize_int_file(path).expect("read token file")
    } else {
        (0..65536).map(|v| (v % 8192) as i64).collect::<Vec<_>>()
    };
    let vocab_size = args
        .vocab_size_override
        .unwrap_or_else(|| max_token_plus_one(&tokens));

    let mut trainer = if let Some(path) = &args.resume {
        if Path::new(path).exists() {
            ThinkTrainer::load_checkpoint(path).expect("load checkpoint")
        } else {
            panic!("resume checkpoint does not exist: {path}");
        }
    } else {
        let spec = LayerSpec {
            d_model: args.d_model,
            d_state: args.d_state,
            d_conv: args.d_conv,
        };
        let cfg = default_think_config(
            vocab_size,
            spec,
            args.target_layers,
            parse_placement(&args.placement),
            parse_freeze(&args.freeze),
            args.freeze_embedding,
            args.lr,
        );
        ThinkTrainer::new_random(cfg, args.base_layers, args.seed)
    };

    let metrics_path = format!("{}/metrics.jsonl", args.out_dir);
    let ckpt_path = format!("{}/latest.bincode", args.out_dir);
    let mut cursor = 0usize;

    for local_step in 0..args.steps {
        let (ids, targets) = make_batch_from_tokens(&tokens, cursor, args.batch_size, args.seq_len);
        cursor = cursor.saturating_add(args.batch_size * args.seq_len);
        let stats = trainer.train_step(&ids, &targets);
        let is_last = local_step + 1 == args.steps;

        if local_step % args.log_every == 0 || is_last {
            let rec = json!({
                "step": stats.step,
                "loss": stats.loss,
                "embedding_grad_norm": stats.embedding_grad_norm,
                "prototype_grad_norm": stats.prototype_grad_norm,
                "top_grad_norm": stats.top_grad_norm,
                "layers": trainer.params.layers.len(),
                "frozen": trainer.frozen_layer_indices,
            });
            println!("{}", serde_json::to_string_pretty(&rec).unwrap());
            let mut f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&metrics_path)
                .expect("open metrics jsonl");
            writeln!(f, "{}", serde_json::to_string(&rec).unwrap()).expect("append metrics line");
        }

        if local_step % args.save_every == 0 || is_last {
            trainer.save_checkpoint(&ckpt_path).expect("save checkpoint");
        }
    }

    let summary = json!({
        "final_step": trainer.step,
        "layers": trainer.params.layers.len(),
        "frozen": trainer.frozen_layer_indices,
        "checkpoint": ckpt_path,
        "metrics": metrics_path,
    });
    let summary_path = format!("{}/summary.json", args.out_dir);
    fs::write(
        &summary_path,
        serde_json::to_string_pretty(&summary).expect("serialize summary"),
    )
    .expect("write summary");
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
}