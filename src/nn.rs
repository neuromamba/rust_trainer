use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};

const LN_EPS: f32 = 1e-5;
const NORM_EPS: f32 = 1e-8;

#[derive(Debug, Clone)]
pub struct LnCache {
    pub y_hat: Array3<f32>,
    pub inv_std: Array2<f32>,
}

pub fn layer_norm_forward(x: ArrayView3<f32>) -> (Array3<f32>, LnCache) {
    let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let mut y = Array3::<f32>::zeros((b, t, d));
    let mut inv_std = Array2::<f32>::zeros((b, t));
    let dn = d as f32;
    for bi in 0..b {
        for ti in 0..t {
            let mut mean = 0.0;
            for di in 0..d {
                mean += x[(bi, ti, di)];
            }
            mean /= dn;
            let mut var = 0.0;
            for di in 0..d {
                let v = x[(bi, ti, di)] - mean;
                var += v * v;
            }
            var /= dn;
            let inv = 1.0 / (var + LN_EPS).sqrt();
            inv_std[(bi, ti)] = inv;
            for di in 0..d {
                y[(bi, ti, di)] = (x[(bi, ti, di)] - mean) * inv;
            }
        }
    }
    (y.clone(), LnCache { y_hat: y, inv_std })
}

pub fn layer_norm_backward(dy: ArrayView3<f32>, cache: &LnCache) -> Array3<f32> {
    let (b, t, d) = (dy.shape()[0], dy.shape()[1], dy.shape()[2]);
    let mut dx = Array3::<f32>::zeros((b, t, d));
    let dn = d as f32;
    for bi in 0..b {
        for ti in 0..t {
            let inv = cache.inv_std[(bi, ti)];
            let mut mean_dy = 0.0;
            let mut mean_dy_y = 0.0;
            for di in 0..d {
                let dyi = dy[(bi, ti, di)];
                mean_dy += dyi;
                mean_dy_y += dyi * cache.y_hat[(bi, ti, di)];
            }
            mean_dy /= dn;
            mean_dy_y /= dn;
            for di in 0..d {
                let dyi = dy[(bi, ti, di)];
                dx[(bi, ti, di)] = inv * (dyi - mean_dy - cache.y_hat[(bi, ti, di)] * mean_dy_y);
            }
        }
    }
    dx
}

pub fn hpn_loss_and_grad_z(
    z_flat: ArrayView2<f32>,
    targets: &[i64],
    prototypes: &Array2<f32>,
) -> (f32, Array2<f32>) {
    let (loss, dz, _d_proto) = hpn_loss_and_grads(z_flat, targets, prototypes);
    (loss, dz)
}

pub fn hpn_loss_and_grads(
    z_flat: ArrayView2<f32>,
    targets: &[i64],
    prototypes: &Array2<f32>,
) -> (f32, Array2<f32>, Array2<f32>) {
    let (n, d) = (z_flat.shape()[0], z_flat.shape()[1]);
    assert_eq!(targets.len(), n);
    let k = prototypes.shape()[0];

    let mut z_norm = Array2::<f32>::zeros((n, d));
    let mut z_invnorm = Array1::<f32>::zeros(n);
    let mut cos = Array1::<f32>::zeros(n);

    for i in 0..n {
        let mut sq = 0.0;
        for di in 0..d {
            let v = z_flat[(i, di)];
            sq += v * v;
        }
        let nrm = sq.sqrt().max(NORM_EPS);
        let inv = 1.0 / nrm;
        z_invnorm[i] = inv;
        let yi = targets[i].rem_euclid(k as i64) as usize;
        let mut c = 0.0;
        for di in 0..d {
            z_norm[(i, di)] = z_flat[(i, di)] * inv;
            c += z_norm[(i, di)] * prototypes[(yi, di)];
        }
        cos[i] = c;
    }

    let mut loss = 0.0;
    for &c in cos.iter() {
        let r = 1.0 - c;
        loss += r * r;
    }
    loss /= n as f32;

    let mut dz_norm = Array2::<f32>::zeros((n, d));
    let mut d_prototypes = Array2::<f32>::zeros((k, d));
    let coeff = -2.0 / (n as f32);
    for i in 0..n {
        let dc = coeff * (1.0 - cos[i]);
        let yi = targets[i].rem_euclid(k as i64) as usize;
        for di in 0..d {
            dz_norm[(i, di)] = dc * prototypes[(yi, di)];
            d_prototypes[(yi, di)] += dc * z_norm[(i, di)];
        }
    }

    let mut dz = Array2::<f32>::zeros((n, d));
    for i in 0..n {
        let mut dot = 0.0;
        for di in 0..d {
            dot += dz_norm[(i, di)] * z_norm[(i, di)];
        }
        let inv = z_invnorm[i];
        for di in 0..d {
            dz[(i, di)] = inv * (dz_norm[(i, di)] - dot * z_norm[(i, di)]);
        }
    }

    (loss, dz, d_prototypes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_norm_backward_preserves_shape_and_finiteness() {
        let x = Array3::from_shape_fn((2, 3, 4), |(b, t, d)| 0.1 * (1 + b + t + d) as f32);
        let dy = Array3::from_shape_fn((2, 3, 4), |(b, t, d)| 0.05 * (1 + b + t + d) as f32);
        let (_y, cache) = layer_norm_forward(x.view());
        let dx = layer_norm_backward(dy.view(), &cache);
        assert_eq!(dx.dim(), (2, 3, 4));
        assert!(dx.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn hpn_loss_and_grad_returns_finite_outputs() {
        let z = Array2::from_shape_fn((6, 8), |(n, d)| 0.03 * (1 + n + d) as f32);
        let prototypes = Array2::from_shape_fn((10, 8), |(k, d)| 0.02 * (1 + k + d) as f32);
        let targets = vec![0, 1, 2, 3, 4, 5];
        let (loss, dz) = hpn_loss_and_grad_z(z.view(), &targets, &prototypes);
        assert!(loss.is_finite());
        assert_eq!(dz.dim(), (6, 8));
        assert!(dz.iter().all(|v| v.is_finite()));

        let (loss2, dz2, dproto) = hpn_loss_and_grads(z.view(), &targets, &prototypes);
        assert!((loss - loss2).abs() <= 1e-8);
        assert_eq!(dz2.dim(), (6, 8));
        assert_eq!(dproto.dim(), (10, 8));
        assert!(dproto.iter().all(|v| v.is_finite()));
    }
}