use crate::simd_ops::{
    conv1d_silu_forward_scalar, conv1d_silu_forward_simd, ssm_scan_backward_scalar,
    ssm_scan_backward_simd, ssm_scan_forward_scalar, ssm_scan_forward_simd, LANES,
};
use crate::trainer::MambaLayerParams;
use ndarray::{s, Array1, Array2, Array3, ArrayView3};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn silu_grad(z: f32) -> f32 {
    let s = sigmoid(z);
    s * (1.0 + z * (1.0 - s))
}

#[derive(Debug, Clone)]
pub struct LayerForwardCachePerBatch {
    pub pre_silu: Array2<f32>,
    pub x_conv: Array2<f32>,
    pub bs: Array2<f32>,
    pub cs: Array2<f32>,
    pub delta_raw: Array2<f32>,
    pub pre_dt: Array2<f32>,
    pub delta: Array2<f32>,
    pub h_traj: Array3<f32>,
    pub y_pre: Array2<f32>,
}

#[derive(Debug, Clone)]
pub struct LayerForwardCache {
    pub x_in: Array3<f32>,
    pub a: Array2<f32>,
    pub per_batch: Vec<LayerForwardCachePerBatch>,
}

#[derive(Debug, Clone)]
pub struct LayerGrads {
    pub a_log: Array2<f32>,
    pub d_skip: Array1<f32>,
    pub x_proj_w: Array2<f32>,
    pub dt_proj_w: Array2<f32>,
    pub dt_proj_b: Array1<f32>,
    pub conv1d_w: Array2<f32>,
    pub conv1d_b: Array1<f32>,
    pub out_proj_w: Array2<f32>,
}

impl LayerGrads {
    pub fn zeros_like(layer: &MambaLayerParams) -> Self {
        Self {
            a_log: Array2::zeros(layer.a_log.dim()),
            d_skip: Array1::zeros(layer.d_skip.dim()),
            x_proj_w: Array2::zeros(layer.x_proj_w.dim()),
            dt_proj_w: Array2::zeros(layer.dt_proj_w.dim()),
            dt_proj_b: Array1::zeros(layer.dt_proj_b.dim()),
            conv1d_w: Array2::zeros(layer.conv1d_w.dim()),
            conv1d_b: Array1::zeros(layer.conv1d_b.dim()),
            out_proj_w: Array2::zeros(layer.out_proj_w.dim()),
        }
    }

    pub fn add_assign(&mut self, other: &LayerGrads) {
        self.a_log += &other.a_log;
        self.d_skip += &other.d_skip;
        self.x_proj_w += &other.x_proj_w;
        self.dt_proj_w += &other.dt_proj_w;
        self.dt_proj_b += &other.dt_proj_b;
        self.conv1d_w += &other.conv1d_w;
        self.conv1d_b += &other.conv1d_b;
        self.out_proj_w += &other.out_proj_w;
    }
}

pub fn forward_with_cache(
    layer: &MambaLayerParams,
    x: ArrayView3<f32>,
) -> (Array3<f32>, LayerForwardCache) {
    let (batch, seq_len, d_model) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let d_state = (layer.x_proj_w.shape()[1] - 1) / 2;
    let a = layer.a_log.mapv(|v| -v.exp());

    let mut pre_silu_all = Array3::<f32>::zeros((batch, seq_len, d_model));
    let mut x_conv_all = Array3::<f32>::zeros((batch, seq_len, d_model));
    for b in 0..batch {
        let x_b = x.slice(s![b, .., ..]);
        let (pre_b, out_b) = if d_model % LANES == 0 {
            conv1d_silu_forward_simd(x_b, layer.conv1d_w.view(), layer.conv1d_b.view())
        } else {
            conv1d_silu_forward_scalar(x_b, layer.conv1d_w.view(), layer.conv1d_b.view())
        };
        pre_silu_all.slice_mut(s![b, .., ..]).assign(&pre_b);
        x_conv_all.slice_mut(s![b, .., ..]).assign(&out_b);
    }

    let mut per_batch = Vec::with_capacity(batch);
    let mut out = Array3::<f32>::zeros((batch, seq_len, d_model));
    for b in 0..batch {
        let x_conv_b = x_conv_all.slice(s![b, .., ..]).to_owned();
        let xz = x_conv_b.dot(&layer.x_proj_w);
        let bs = xz.slice(s![.., 0..d_state]).to_owned();
        let cs = xz.slice(s![.., d_state..2 * d_state]).to_owned();
        let delta_raw = xz.slice(s![.., 2 * d_state..]).to_owned();

        let mut pre_dt = delta_raw.dot(&layer.dt_proj_w);
        for t in 0..seq_len {
            for i in 0..d_model {
                pre_dt[(t, i)] += layer.dt_proj_b[i];
            }
        }
        let delta = pre_dt.mapv(softplus);
        let mut h = Array2::<f32>::zeros((d_model, d_state));
        let mut h_traj = Array3::<f32>::zeros((seq_len, d_model, d_state));
        let mut y_pre = Array2::<f32>::zeros((seq_len, d_model));
        if d_state % LANES == 0 {
            ssm_scan_forward_simd(
                bs.view(),
                cs.view(),
                delta.view(),
                x_conv_b.view(),
                a.view(),
                layer.d_skip.view(),
                &mut h,
                &mut h_traj,
                &mut y_pre,
            );
        } else {
            ssm_scan_forward_scalar(
                bs.view(),
                cs.view(),
                delta.view(),
                x_conv_b.view(),
                a.view(),
                layer.d_skip.view(),
                &mut h,
                &mut h_traj,
                &mut y_pre,
            );
        }

        let yo = y_pre.dot(&layer.out_proj_w);
        out.slice_mut(s![b, .., ..]).assign(&yo);
        per_batch.push(LayerForwardCachePerBatch {
            pre_silu: pre_silu_all.slice(s![b, .., ..]).to_owned(),
            x_conv: x_conv_b,
            bs,
            cs,
            delta_raw,
            pre_dt,
            delta,
            h_traj,
            y_pre,
        });
    }

    (
        out,
        LayerForwardCache {
            x_in: x.to_owned(),
            a,
            per_batch,
        },
    )
}

pub fn backward(
    layer: &MambaLayerParams,
    dy: ArrayView3<f32>,
    cache: &LayerForwardCache,
) -> (Array3<f32>, LayerGrads) {
    let (batch, seq_len, d_model) = (dy.shape()[0], dy.shape()[1], dy.shape()[2]);
    let d_state = (layer.x_proj_w.shape()[1] - 1) / 2;
    let d_conv = layer.conv1d_w.shape()[1];
    let mut total = LayerGrads::zeros_like(layer);
    let mut dx_in = Array3::<f32>::zeros((batch, seq_len, d_model));

    for b in 0..batch {
        let cb = &cache.per_batch[b];
        let dy_b = dy.slice(s![b, .., ..]);
        let dy_pre = dy_b.dot(&layer.out_proj_w.t());
        total.out_proj_w += &cb.y_pre.t().dot(&dy_b);

        let mut dx_conv = Array2::<f32>::zeros((seq_len, d_model));
        let mut d_cs = Array2::<f32>::zeros((seq_len, d_state));
        let mut d_bs = Array2::<f32>::zeros((seq_len, d_state));
        let mut d_delta = Array2::<f32>::zeros((seq_len, d_model));

        let scan = if d_state % LANES == 0 {
            ssm_scan_backward_simd(
                cb.bs.view(),
                cb.cs.view(),
                cb.delta.view(),
                cb.x_conv.view(),
                cache.a.view(),
                layer.d_skip.view(),
                cb.h_traj.view(),
                dy_pre.view(),
            )
        } else {
            ssm_scan_backward_scalar(
                cb.bs.view(),
                cb.cs.view(),
                cb.delta.view(),
                cb.x_conv.view(),
                cache.a.view(),
                layer.d_skip.view(),
                cb.h_traj.view(),
                dy_pre.view(),
            )
        };
        total.a_log += &scan.grad_a_log;
        total.d_skip += &scan.grad_d_skip;
        d_bs.assign(&scan.d_bs);
        d_cs.assign(&scan.d_cs);
        d_delta.assign(&scan.d_delta);
        dx_conv.assign(&scan.dx_conv);

        let mut d_delta_raw = Array2::<f32>::zeros((seq_len, 1));
        for t in 0..seq_len {
            for i in 0..d_model {
                let dpre = d_delta[(t, i)] * sigmoid(cb.pre_dt[(t, i)]);
                total.dt_proj_b[i] += dpre;
                total.dt_proj_w[(0, i)] += dpre * cb.delta_raw[(t, 0)];
                d_delta_raw[(t, 0)] += dpre * layer.dt_proj_w[(0, i)];
            }
        }

        let mut d_xz = Array2::<f32>::zeros((seq_len, 2 * d_state + 1));
        for t in 0..seq_len {
            for s_idx in 0..d_state {
                d_xz[(t, s_idx)] = d_bs[(t, s_idx)];
                d_xz[(t, d_state + s_idx)] = d_cs[(t, s_idx)];
            }
            d_xz[(t, 2 * d_state)] = d_delta_raw[(t, 0)];
        }
        total.x_proj_w += &cb.x_conv.t().dot(&d_xz);
        dx_conv += &d_xz.dot(&layer.x_proj_w.t());

        let mut d_pre_silu = Array2::<f32>::zeros((seq_len, d_model));
        for t in 0..seq_len {
            for d in 0..d_model {
                d_pre_silu[(t, d)] = dx_conv[(t, d)] * silu_grad(cb.pre_silu[(t, d)]);
            }
        }

        let x_in_b = cache.x_in.slice(s![b, .., ..]);
        let mut dx_in_b = Array2::<f32>::zeros((seq_len, d_model));
        for t in 0..seq_len {
            for d in 0..d_model {
                let dp = d_pre_silu[(t, d)];
                total.conv1d_b[d] += dp;
                for k in 0..d_conv {
                    let xk = (t + k) as isize - (d_conv as isize - 1);
                    if xk >= 0 {
                        let xi = xk as usize;
                        total.conv1d_w[(d, k)] += dp * x_in_b[(xi, d)];
                        dx_in_b[(xi, d)] += dp * layer.conv1d_w[(d, k)];
                    }
                }
            }
        }
        dx_in.slice_mut(s![b, .., ..]).assign(&dx_in_b);
    }

    (dx_in, total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::LayerSpec;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn forward_backward_layer_shapes_are_consistent() {
        let spec = LayerSpec {
            d_model: 8,
            d_state: 8,
            d_conv: 4,
        };
        let mut rng = StdRng::seed_from_u64(7);
        let layer = MambaLayerParams::random(spec, &mut rng);
        let x = Array3::from_shape_fn((2, 5, 8), |(b, t, d)| 0.01 * (1 + b + t + d) as f32);
        let (y, cache) = forward_with_cache(&layer, x.view());
        let dy = Array3::from_shape_fn((2, 5, 8), |(b, t, d)| 0.02 * (1 + b + t + d) as f32);
        let (dx, grads) = backward(&layer, dy.view(), &cache);
        assert_eq!(y.dim(), (2, 5, 8));
        assert_eq!(dx.dim(), (2, 5, 8));
        assert_eq!(grads.a_log.dim(), layer.a_log.dim());
        assert_eq!(grads.out_proj_w.dim(), layer.out_proj_w.dim());
        assert!(dx.iter().all(|v| v.is_finite()));
        assert!(grads.out_proj_w.iter().all(|v| v.is_finite()));
    }
}