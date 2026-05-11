pub mod data_stream;
pub mod generic_trainer;
pub mod layer;
pub mod loss;
pub mod nn;
pub mod optim;
pub mod simd_ops;
pub mod stack;
pub mod trainer;

#[cfg(feature = "python")]
pub mod bindings;

pub use loss::{
    cagradstep, gradnorm_ff_scale, pcgrad, GradientSurgeryConfig, GradientSurgeryMethod,
};
pub use trainer::{
    AdamWConfig, CadencedStepStats, ExpansionConfig, ExpansionPlacement, ExperimentalTrainer,
    ExperimentalTrainerConfig, FreezeSelection, LayerSpec, MambaLayerParams, TrainerParams,
};

#[cfg(feature = "python")]
pub use bindings::{train_from_config, TrainerConfig};
