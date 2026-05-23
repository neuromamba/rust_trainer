use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adam1 {
    pub m: Array1<f32>,
    pub v: Array1<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adam2 {
    pub m: Array2<f32>,
    pub v: Array2<f32>,
}

impl Adam1 {
    pub fn zeros(n: usize) -> Self {
        Self {
            m: Array1::zeros(n),
            v: Array1::zeros(n),
        }
    }
}

impl Adam2 {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            m: Array2::zeros((rows, cols)),
            v: Array2::zeros((rows, cols)),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn adamw_update_1d(
    p: &mut Array1<f32>,
    grad: &Array1<f32>,
    st: &mut Adam1,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    wd: f32,
    step: usize,
) {
    let bc1 = 1.0 - b1.powi((step + 1) as i32);
    let bc2 = 1.0 - b2.powi((step + 1) as i32);
    for ((p_v, &g), (m, v)) in p
        .iter_mut()
        .zip(grad.iter())
        .zip(st.m.iter_mut().zip(st.v.iter_mut()))
    {
        *m = b1 * *m + (1.0 - b1) * g;
        *v = b2 * *v + (1.0 - b2) * g * g;
        let mhat = *m / bc1;
        let vhat = *v / bc2;
        let upd = (mhat / (vhat.sqrt() + eps)).clamp(-1e4, 1e4);
        let old = *p_v;
        *p_v = old - lr * (upd + wd * old);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn adamw_update_2d(
    p: &mut Array2<f32>,
    grad: &Array2<f32>,
    st: &mut Adam2,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    wd: f32,
    step: usize,
) {
    let bc1 = 1.0 - b1.powi((step + 1) as i32);
    let bc2 = 1.0 - b2.powi((step + 1) as i32);
    for ((p_v, &g), (m, v)) in p
        .iter_mut()
        .zip(grad.iter())
        .zip(st.m.iter_mut().zip(st.v.iter_mut()))
    {
        *m = b1 * *m + (1.0 - b1) * g;
        *v = b2 * *v + (1.0 - b2) * g * g;
        let mhat = *m / bc1;
        let vhat = *v / bc2;
        let upd = (mhat / (vhat.sqrt() + eps)).clamp(-1e4, 1e4);
        let old = *p_v;
        *p_v = old - lr * (upd + wd * old);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adamw_updates_parameters() {
        let mut p = Array1::from(vec![1.0, 2.0, 3.0]);
        let g = Array1::from(vec![0.1, -0.2, 0.3]);
        let before = p.clone();
        let mut st = Adam1::zeros(3);
        adamw_update_1d(&mut p, &g, &mut st, 1e-3, 0.9, 0.999, 1e-8, 0.01, 0);
        assert!(p
            .iter()
            .zip(before.iter())
            .any(|(a, b)| (a - b).abs() > 0.0));
    }
}
