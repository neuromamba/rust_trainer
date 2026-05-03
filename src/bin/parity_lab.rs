use neuromamba_trainer_lab::{
    AdamWConfig, ExpansionConfig, ExpansionPlacement, ExperimentalTrainer, ExperimentalTrainerConfig,
    FreezeSelection, LayerSpec, TrainerParams,
};
use serde_json::json;
use std::env;

fn parse_args() -> (ExpansionPlacement, FreezeSelection, usize, usize, u64, usize) {
    let mut placement = ExpansionPlacement::Append;
    let mut freeze = FreezeSelection::FirstN(2);
    let mut target = 6usize;
    let mut base_layers = 2usize;
    let mut seed = 42u64;
    let mut cycles = 1usize;

    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--placement" if idx + 1 < args.len() => {
                placement = parse_placement(&args[idx + 1]);
                idx += 2;
            }
            "--freeze" if idx + 1 < args.len() => {
                freeze = parse_freeze(&args[idx + 1]);
                idx += 2;
            }
            "--target" if idx + 1 < args.len() => {
                target = args[idx + 1].parse().unwrap_or(6);
                idx += 2;
            }
            "--base-layers" if idx + 1 < args.len() => {
                base_layers = args[idx + 1].parse().unwrap_or(2);
                idx += 2;
            }
            "--seed" if idx + 1 < args.len() => {
                seed = args[idx + 1].parse().unwrap_or(42);
                idx += 2;
            }
            "--cycles" if idx + 1 < args.len() => {
                cycles = args[idx + 1].parse().unwrap_or(1);
                idx += 2;
            }
            _ => {
                idx += 1;
            }
        }
    }

    (placement, freeze, target, base_layers, seed, cycles)
}

fn parse_placement(raw: &str) -> ExpansionPlacement {
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

fn parse_freeze(raw: &str) -> FreezeSelection {
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

fn main() {
    let (placement, freeze, target, base_layers, seed, cycles) = parse_args();
    let spec = LayerSpec {
        d_model: 64,
        d_state: 16,
        d_conv: 4,
    };

    let base = TrainerParams::random(512, spec, base_layers, seed);
    let cfg = ExperimentalTrainerConfig {
        vocab_size: 512,
        layer_spec: spec,
        expansion: ExpansionConfig {
            target_num_layers: target,
            placement,
        },
        freeze_selection: freeze,
        freeze_embedding: true,
        ff_lr: 1e-2,
        ff_threshold: 1e-3,
        adamw: AdamWConfig::default(),
    };
    let mut trainer = ExperimentalTrainer::from_base(base, cfg);
    let before = trainer.layer_norms();
    for _ in 0..cycles {
        let _ = trainer.train_ff_cycle();
    }
    let after = trainer.layer_norms();

    let frozen_unchanged = trainer
        .frozen_layer_indices()
        .iter()
        .all(|idx| (before[*idx] - after[*idx]).abs() <= 1e-8);

    let out = json!({
        "expanded_layers": trainer.expanded_layer_count(),
        "frozen_indices": trainer.frozen_layer_indices(),
        "steps": trainer.step,
        "frozen_unchanged": frozen_unchanged,
        "before_norms": before,
        "after_norms": after,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}