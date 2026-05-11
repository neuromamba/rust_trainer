pub mod simd_ops;
pub mod trainer;
pub mod loss;
pub mod nn;
pub mod optim;
pub mod layer;
pub mod stack;
pub mod generic_trainer;
pub mod data_stream;

#[cfg(feature = "python")]
pub mod bindings;

pub use trainer::{
    AdamWConfig, CadencedStepStats, ExpansionConfig, ExpansionPlacement, ExperimentalTrainer,
    ExperimentalTrainerConfig, FreezeSelection, LayerSpec, MambaLayerParams, TrainerParams,
};
pub use loss::{cagradstep, gradnorm_ff_scale, pcgrad, GradientSurgeryConfig, GradientSurgeryMethod};

#[cfg(feature = "python")]
pub use bindings::{train_from_config, TrainerConfig};