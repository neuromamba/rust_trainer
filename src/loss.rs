use ndarray::Array1;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GradientSurgeryMethod {
    #[serde(rename = "pcgrad")]
    PcGrad,
    #[serde(rename = "gradnorm")]
    GradNorm,
    #[serde(rename = "cagradstep")]
    CAGradStep,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GradientSurgeryConfig {
    pub method: GradientSurgeryMethod,
    pub epsilon: f32,
    pub gradnorm_alpha: f32,
    pub cagrad_lambda: f32,
}

impl Default for GradientSurgeryConfig {
    fn default() -> Self {
        Self {
            method: GradientSurgeryMethod::PcGrad,
            epsilon: 1e-8,
            gradnorm_alpha: 0.2,
            cagrad_lambda: 1.0,
        }
    }
}

/// Project FF gradients away from BP conflicts (PCGrad).
///
/// If dot(grad_ff, grad_bp) < 0, remove the FF component aligned
/// against BP: grad_ff' = grad_ff - (dot / (||bp||^2 + eps)) * grad_bp.
pub fn pcgrad(grad_ff: &Array1<f32>, grad_bp: &Array1<f32>, epsilon: f32) -> Array1<f32> {
    assert_eq!(grad_ff.len(), grad_bp.len(), "pcgrad shape mismatch");

    let dot = grad_ff.dot(grad_bp);
    if dot >= 0.0 {
        return grad_ff.clone();
    }

    let bp_norm_sq = grad_bp.dot(grad_bp) + epsilon.max(0.0);
    grad_ff - &(grad_bp * (dot / bp_norm_sq))
}

/// GradNorm-style disagreement scaling for FF branch.
///
/// disagreement = ||ff/||ff|| - bp/||bp||||
/// lambda_ff = exp(alpha * disagreement)
pub fn gradnorm_ff_scale(
    grad_ff: &Array1<f32>,
    grad_bp: &Array1<f32>,
    alpha: f32,
    epsilon: f32,
) -> f32 {
    assert_eq!(grad_ff.len(), grad_bp.len(), "gradnorm shape mismatch");
    let ff_norm = grad_ff.dot(grad_ff).sqrt() + epsilon.max(0.0);
    let bp_norm = grad_bp.dot(grad_bp).sqrt() + epsilon.max(0.0);
    let disagreement = grad_ff
        .iter()
        .zip(grad_bp.iter())
        .map(|(f, b)| {
            let d = (*f / ff_norm) - (*b / bp_norm);
            d * d
        })
        .sum::<f32>()
        .sqrt();
    (alpha * disagreement).exp()
}

/// Conflict-averse step: stronger-than-PCGrad conflict removal.
///
/// If dot < 0, remove lambda times the conflicting projection of FF onto BP.
pub fn cagradstep(
    grad_ff: &Array1<f32>,
    grad_bp: &Array1<f32>,
    lambda: f32,
    epsilon: f32,
) -> Array1<f32> {
    assert_eq!(grad_ff.len(), grad_bp.len(), "cagrad shape mismatch");
    let dot = grad_ff.dot(grad_bp);
    if dot >= 0.0 {
        return grad_ff.clone();
    }
    let bp_norm_sq = grad_bp.dot(grad_bp) + epsilon.max(0.0);
    grad_ff - &(grad_bp * ((lambda.max(0.0) * dot) / bp_norm_sq))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn pcgrad_keeps_aligned_gradients() {
        let ff = array![1.0_f32, 2.0, 3.0];
        let bp = array![0.1_f32, 0.2, 0.3];
        let out = pcgrad(&ff, &bp, 1e-8);
        for (a, b) in out.iter().zip(ff.iter()) {
            assert!((a - b).abs() <= 1e-8);
        }
    }

    #[test]
    fn pcgrad_removes_conflicting_component() {
        let ff = array![-1.0_f32, 0.0];
        let bp = array![1.0_f32, 0.0];
        let out = pcgrad(&ff, &bp, 1e-8);
        let dot_after = out.dot(&bp);
        assert!(
            dot_after >= -1e-6,
            "dot after projection should be non-negative"
        );
    }

    #[test]
    fn gradnorm_scale_increases_with_disagreement() {
        let ff = array![1.0_f32, 0.0];
        let bp_aligned = array![1.0_f32, 0.0];
        let bp_opposed = array![-1.0_f32, 0.0];
        let s1 = gradnorm_ff_scale(&ff, &bp_aligned, 0.2, 1e-8);
        let s2 = gradnorm_ff_scale(&ff, &bp_opposed, 0.2, 1e-8);
        assert!(s2 >= s1);
    }

    #[test]
    fn cagradstep_reduces_conflict_when_dot_negative() {
        let ff = array![-1.0_f32, 0.0];
        let bp = array![1.0_f32, 0.0];
        let out = cagradstep(&ff, &bp, 1.5, 1e-8);
        assert!(out.dot(&bp) >= ff.dot(&bp));
    }
}
