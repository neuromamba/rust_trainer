pub mod simd_ops;
pub mod trainer;
pub mod nn;
pub mod optim;
pub mod layer;
pub mod stack;
pub mod generic_trainer;

pub use trainer::{
    AdamWConfig, ExpansionConfig, ExpansionPlacement, ExperimentalTrainer,
    ExperimentalTrainerConfig, FreezeSelection, LayerSpec, MambaLayerParams, TrainerParams,
};