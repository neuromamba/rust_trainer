use neuromamba_trainer_lab::think_trainer::{
    default_think_config, make_batch_from_tokens, is_frozen_unchanged, mean_layer_norm, ThinkTrainer,
};
use neuromamba_trainer_lab::{ExpansionPlacement, FreezeSelection, LayerSpec};
use serde_json::json;

fn main() {
    let spec = LayerSpec {
        d_model: 16,
        d_state: 16,
        d_conv: 4,
    };
    let cfg = default_think_config(
        64,
        spec,
        6,
        ExpansionPlacement::SpecificPositions(vec![1, 3, 4, 5]),
        FreezeSelection::FirstN(2),
        false,
        1e-4,
    );

    let mut a = ThinkTrainer::new_random(cfg, 2, 2026);
    let tokens = (0..1024).map(|v| (v % 64) as i64).collect::<Vec<_>>();
    let before = a.layer_l2_norms();
    let (ids1, tgt1) = make_batch_from_tokens(&tokens, 0, 2, 8);
    let s1 = a.train_step(&ids1, &tgt1);
    let mid = a.layer_l2_norms();
    let frozen_ok = is_frozen_unchanged(&before, &mid, &a.frozen_layer_indices, 1e-8);

    let ckpt = std::env::temp_dir().join("think_parity_roundtrip.bincode");
    a.save_checkpoint(&ckpt).unwrap();
    let mut b = ThinkTrainer::load_checkpoint(&ckpt).unwrap();

    let (ids2, tgt2) = make_batch_from_tokens(&tokens, 16, 2, 8);
    let sa = a.train_step(&ids2, &tgt2);
    let sb = b.train_step(&ids2, &tgt2);

    let emb_delta = (&a.params.embedding - &b.params.embedding)
        .mapv(f32::abs)
        .sum();
    let proto_delta = (&a.prototypes - &b.prototypes).mapv(f32::abs).sum();
    let out = json!({
        "first_step_loss": s1.loss,
        "resume_step_loss_a": sa.loss,
        "resume_step_loss_b": sb.loss,
        "resume_loss_abs_diff": (sa.loss - sb.loss).abs(),
        "embedding_abs_diff_after_resume_step": emb_delta,
        "prototypes_abs_diff_after_resume_step": proto_delta,
        "frozen_layers_unchanged": frozen_ok,
        "mean_layer_norm_before": mean_layer_norm(&before),
        "mean_layer_norm_after_one_step": mean_layer_norm(&mid),
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
    let _ = std::fs::remove_file(&ckpt);
}