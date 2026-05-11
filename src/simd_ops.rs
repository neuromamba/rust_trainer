use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use wide::f32x8;

pub const LANES: usize = 8;
pub const DEFAULT_SCAN_CHUNK_LEN: usize = 16;

#[inline]
pub fn recommended_scan_chunk_len(seq_len: usize) -> usize {
    if seq_len >= 512 {
        32
    } else if seq_len >= 256 {
        16
    } else {
        8
    }
}

#[inline]
pub fn simd_width() -> usize {
    LANES
}

#[inline(always)]
fn load8(buf: &[f32], off: usize) -> f32x8 {
    f32x8::from([
        buf[off],
        buf[off + 1],
        buf[off + 2],
        buf[off + 3],
        buf[off + 4],
        buf[off + 5],
        buf[off + 6],
        buf[off + 7],
    ])
}

#[inline(always)]
fn store8(buf: &mut [f32], off: usize, v: f32x8) {
    let arr = v.to_array();
    buf[off..off + LANES].copy_from_slice(&arr);
}

#[inline(always)]
fn hsum(v: f32x8) -> f32 {
    let arr = v.to_array();
    arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7]
}

/// Scalar reference forward SSM scan used for correctness checks.
#[allow(clippy::too_many_arguments)]
pub fn ssm_scan_forward_scalar(
    bs: ArrayView2<f32>,
    cs: ArrayView2<f32>,
    delta: ArrayView2<f32>,
    x_conv: ArrayView2<f32>,
    a: ArrayView2<f32>,
    d_skip: ArrayView1<f32>,
    h: &mut Array2<f32>,
    h_traj: &mut Array3<f32>,
    y_pre: &mut Array2<f32>,
) {
    let seq_len = bs.shape()[0];
    let d_state = bs.shape()[1];
    let d_model = delta.shape()[1];

    for t in 0..seq_len {
        for i in 0..d_model {
            let dt_i = delta[(t, i)];
            let xc_i = x_conv[(t, i)];
            let mut yi = 0.0;
            for s in 0..d_state {
                let a_bar = (dt_i * a[(i, s)]).exp();
                let v = dt_i * bs[(t, s)] * xc_i;
                let h_new = a_bar * h[(i, s)] + v;
                h[(i, s)] = h_new;
                h_traj[(t, i, s)] = h_new;
                yi += h_new * cs[(t, s)];
            }
            y_pre[(t, i)] = yi + d_skip[i] * xc_i;
        }
    }
}

/// SIMD forward SSM scan over state lanes.
#[allow(clippy::too_many_arguments)]
pub fn ssm_scan_forward_simd(
    bs: ArrayView2<f32>,
    cs: ArrayView2<f32>,
    delta: ArrayView2<f32>,
    x_conv: ArrayView2<f32>,
    a: ArrayView2<f32>,
    d_skip: ArrayView1<f32>,
    h: &mut Array2<f32>,
    h_traj: &mut Array3<f32>,
    y_pre: &mut Array2<f32>,
) {
    let seq_len = bs.shape()[0];
    let d_state = bs.shape()[1];
    let d_model = delta.shape()[1];
    assert!(d_state.is_multiple_of(LANES), "d_state must be multiple of 8");

    let bs_s = bs.as_slice().expect("bs must be contiguous");
    let cs_s = cs.as_slice().expect("cs must be contiguous");
    let delta_s = delta.as_slice().expect("delta must be contiguous");
    let xc_s = x_conv.as_slice().expect("x_conv must be contiguous");
    let a_s = a.as_slice().expect("a must be contiguous");
    let d_skip_s = d_skip.as_slice().expect("d_skip must be contiguous");
    let h_s = h.as_slice_mut().expect("h must be contiguous");
    let htraj_s = h_traj.as_slice_mut().expect("h_traj must be contiguous");
    let y_s = y_pre.as_slice_mut().expect("y_pre must be contiguous");

    let n_blocks = d_state / LANES;

    for t in 0..seq_len {
        let bs_row = &bs_s[t * d_state..(t + 1) * d_state];
        let cs_row = &cs_s[t * d_state..(t + 1) * d_state];
        let delta_row = &delta_s[t * d_model..(t + 1) * d_model];
        let xc_row = &xc_s[t * d_model..(t + 1) * d_model];
        let htraj_row_off = t * d_model * d_state;
        let y_row = &mut y_s[t * d_model..(t + 1) * d_model];

        for i in 0..d_model {
            let dt_i = delta_row[i];
            let xc_i = xc_row[i];
            let dt_v = f32x8::splat(dt_i);
            let dxc_v = f32x8::splat(dt_i * xc_i);

            let a_off = i * d_state;
            let h_off = i * d_state;
            let htraj_off = htraj_row_off + i * d_state;

            let mut yi = f32x8::ZERO;
            for blk in 0..n_blocks {
                let s_off = blk * LANES;
                let a_v = load8(a_s, a_off + s_off);
                let bs_v = load8(bs_row, s_off);
                let cs_v = load8(cs_row, s_off);
                let h_prev = load8(h_s, h_off + s_off);

                let a_bar = (dt_v * a_v).exp();
                let v = bs_v * dxc_v;
                let h_new = a_bar.mul_add(h_prev, v);

                store8(h_s, h_off + s_off, h_new);
                store8(htraj_s, htraj_off + s_off, h_new);
                yi = h_new.mul_add(cs_v, yi);
            }
            y_row[i] = hsum(yi) + d_skip_s[i] * xc_i;
        }
    }
}

pub fn scalar_sum8(values: [f32; LANES]) -> f32 {
    let packed = f32x8::from(values);
    packed.to_array().iter().copied().sum()
}

pub fn conv1d_silu_forward_scalar(
    x_in: ArrayView2<f32>,
    conv_w: ArrayView2<f32>,
    conv_b: ArrayView1<f32>,
) -> (Array2<f32>, Array2<f32>) {
    let (seq_len, d_model) = (x_in.shape()[0], x_in.shape()[1]);
    let d_conv = conv_w.shape()[1];
    let mut pre = Array2::<f32>::zeros((seq_len, d_model));
    let mut out = Array2::<f32>::zeros((seq_len, d_model));
    for t in 0..seq_len {
        for d in 0..d_model {
            let mut acc = conv_b[d];
            for k in 0..d_conv {
                let xk = (t + k) as isize - (d_conv as isize - 1);
                if xk >= 0 {
                    acc += x_in[(xk as usize, d)] * conv_w[(d, k)];
                }
            }
            pre[(t, d)] = acc;
            out[(t, d)] = acc / (1.0 + (-acc).exp());
        }
    }
    (pre, out)
}

pub fn conv1d_silu_forward_simd(
    x_in: ArrayView2<f32>,
    conv_w: ArrayView2<f32>,
    conv_b: ArrayView1<f32>,
) -> (Array2<f32>, Array2<f32>) {
    let (seq_len, d_model) = (x_in.shape()[0], x_in.shape()[1]);
    let d_conv = conv_w.shape()[1];
    let mut pre = Array2::<f32>::zeros((seq_len, d_model));
    let mut out = Array2::<f32>::zeros((seq_len, d_model));

    let x_s = x_in.as_slice().expect("x_in contiguous");
    let w_s = conv_w.as_slice().expect("conv_w contiguous");
    let b_s = conv_b.as_slice().expect("conv_b contiguous");
    let pre_s = pre.as_slice_mut().expect("pre contiguous");
    let out_s = out.as_slice_mut().expect("out contiguous");

    if d_model % LANES == 0 {
        let n_blocks = d_model / LANES;
        for t in 0..seq_len {
            let pre_row = &mut pre_s[t * d_model..(t + 1) * d_model];
            let out_row = &mut out_s[t * d_model..(t + 1) * d_model];

            for blk in 0..n_blocks {
                let d_off = blk * LANES;
                let mut acc = load8(b_s, d_off);
                for k in 0..d_conv {
                    let xk = (t + k) as isize - (d_conv as isize - 1);
                    if xk >= 0 {
                        let xv = load8(x_s, (xk as usize) * d_model + d_off);
                        let w_block = f32x8::from([
                            w_s[d_off * d_conv + k],
                            w_s[(d_off + 1) * d_conv + k],
                            w_s[(d_off + 2) * d_conv + k],
                            w_s[(d_off + 3) * d_conv + k],
                            w_s[(d_off + 4) * d_conv + k],
                            w_s[(d_off + 5) * d_conv + k],
                            w_s[(d_off + 6) * d_conv + k],
                            w_s[(d_off + 7) * d_conv + k],
                        ]);
                        acc = xv.mul_add(w_block, acc);
                    }
                }
                store8(pre_row, d_off, acc);
                let neg = -acc;
                let denom = f32x8::splat(1.0) + neg.exp();
                let silu = acc / denom;
                store8(out_row, d_off, silu);
            }
        }
    } else {
        return conv1d_silu_forward_scalar(x_in, conv_w, conv_b);
    }

    (pre, out)
}

#[derive(Debug, Clone)]
pub struct SsmBackwardOutputs {
    pub grad_a_log: Array2<f32>,
    pub grad_d_skip: Array1<f32>,
    pub d_bs: Array2<f32>,
    pub d_cs: Array2<f32>,
    pub d_delta: Array2<f32>,
    pub dx_conv: Array2<f32>,
}

#[allow(clippy::too_many_arguments)]
pub fn ssm_scan_backward_scalar(
    bs: ArrayView2<f32>,
    cs: ArrayView2<f32>,
    delta: ArrayView2<f32>,
    x_conv: ArrayView2<f32>,
    a: ArrayView2<f32>,
    d_skip: ArrayView1<f32>,
    h_traj: ArrayView3<f32>,
    dy_pre: ArrayView2<f32>,
) -> SsmBackwardOutputs {
    let seq_len = bs.shape()[0];
    let d_state = bs.shape()[1];
    let d_model = delta.shape()[1];

    let mut grad_a_log = Array2::<f32>::zeros((d_model, d_state));
    let mut grad_d_skip = Array1::<f32>::zeros(d_model);
    let mut d_bs = Array2::<f32>::zeros((seq_len, d_state));
    let mut d_cs = Array2::<f32>::zeros((seq_len, d_state));
    let mut d_delta = Array2::<f32>::zeros((seq_len, d_model));
    let mut dx_conv = Array2::<f32>::zeros((seq_len, d_model));

    for t in 0..seq_len {
        for i in 0..d_model {
            let dyi = dy_pre[(t, i)];
            grad_d_skip[i] += dyi * x_conv[(t, i)];
            dx_conv[(t, i)] += dyi * d_skip[i];
            for s in 0..d_state {
                d_cs[(t, s)] += dyi * h_traj[(t, i, s)];
            }
        }
    }

    let mut dh_carry = Array2::<f32>::zeros((d_model, d_state));
    for t in (0..seq_len).rev() {
        for i in 0..d_model {
            let dt_i = delta[(t, i)];
            let xc_i = x_conv[(t, i)];
            let mut acc_d_delta = 0.0;
            for s in 0..d_state {
                let dh_state = dy_pre[(t, i)] * cs[(t, s)];
                let dh_total = dh_state + dh_carry[(i, s)];
                let a_bar = (dt_i * a[(i, s)]).exp();
                let h_prev = if t == 0 { 0.0 } else { h_traj[(t - 1, i, s)] };
                let d_v = dh_total;
                let d_a_bar = dh_total * h_prev;
                dh_carry[(i, s)] = dh_total * a_bar;
                let d_da = d_a_bar * a_bar;
                acc_d_delta += d_da * a[(i, s)];
                grad_a_log[(i, s)] += d_da * dt_i * a[(i, s)];
                acc_d_delta += d_v * bs[(t, s)] * xc_i;
                d_bs[(t, s)] += d_v * dt_i * xc_i;
                dx_conv[(t, i)] += d_v * dt_i * bs[(t, s)];
            }
            d_delta[(t, i)] += acc_d_delta;
        }
    }

    SsmBackwardOutputs {
        grad_a_log,
        grad_d_skip,
        d_bs,
        d_cs,
        d_delta,
        dx_conv,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ssm_scan_backward_simd(
    bs: ArrayView2<f32>,
    cs: ArrayView2<f32>,
    delta: ArrayView2<f32>,
    x_conv: ArrayView2<f32>,
    a: ArrayView2<f32>,
    d_skip: ArrayView1<f32>,
    h_traj: ArrayView3<f32>,
    dy_pre: ArrayView2<f32>,
) -> SsmBackwardOutputs {
    let seq_len = bs.shape()[0];
    let d_state = bs.shape()[1];
    let d_model = delta.shape()[1];
    assert!(d_state.is_multiple_of(LANES), "d_state must be multiple of 8");

    let bs_s = bs.as_slice().expect("bs contiguous");
    let cs_s = cs.as_slice().expect("cs contiguous");
    let delta_s = delta.as_slice().expect("delta contiguous");
    let xc_s = x_conv.as_slice().expect("x_conv contiguous");
    let a_s = a.as_slice().expect("a contiguous");
    let h_traj_s = h_traj.as_slice().expect("h_traj contiguous");
    let dy_s = dy_pre.as_slice().expect("dy_pre contiguous");

    let mut grad_a_log = Array2::<f32>::zeros((d_model, d_state));
    let mut grad_d_skip = Array1::<f32>::zeros(d_model);
    let mut d_bs = Array2::<f32>::zeros((seq_len, d_state));
    let mut d_cs = Array2::<f32>::zeros((seq_len, d_state));
    let mut d_delta = Array2::<f32>::zeros((seq_len, d_model));
    let mut dx_conv = Array2::<f32>::zeros((seq_len, d_model));

    let grad_a_s = grad_a_log.as_slice_mut().expect("grad_a contiguous");
    let d_bs_s = d_bs.as_slice_mut().expect("d_bs contiguous");
    let d_cs_s = d_cs.as_slice_mut().expect("d_cs contiguous");
    let d_delta_s = d_delta.as_slice_mut().expect("d_delta contiguous");
    let dx_conv_s = dx_conv.as_slice_mut().expect("dx_conv contiguous");

    let n_blocks = d_state / LANES;

    for t in 0..seq_len {
        let dy_row = &dy_s[t * d_model..(t + 1) * d_model];
        let xc_row = &xc_s[t * d_model..(t + 1) * d_model];
        let dxc_row = &mut dx_conv_s[t * d_model..(t + 1) * d_model];
        let dcs_row = &mut d_cs_s[t * d_state..(t + 1) * d_state];
        let htraj_row_off = t * d_model * d_state;

        for i in 0..d_model {
            let dyi = dy_row[i];
            grad_d_skip[i] += dyi * xc_row[i];
            dxc_row[i] += dyi * d_skip[i];
            let dyi_v = f32x8::splat(dyi);
            for blk in 0..n_blocks {
                let s_off = blk * LANES;
                let h_v = load8(h_traj_s, htraj_row_off + i * d_state + s_off);
                let dcs_v = load8(dcs_row, s_off);
                store8(dcs_row, s_off, dyi_v.mul_add(h_v, dcs_v));
            }
        }
    }

    let mut dh_carry = vec![0.0f32; d_model * d_state];
    for t in (0..seq_len).rev() {
        let dy_row = &dy_s[t * d_model..(t + 1) * d_model];
        let xc_row = &xc_s[t * d_model..(t + 1) * d_model];
        let bs_row = &bs_s[t * d_state..(t + 1) * d_state];
        let cs_row = &cs_s[t * d_state..(t + 1) * d_state];
        let delta_row = &delta_s[t * d_model..(t + 1) * d_model];
        let prev_htraj_off = if t == 0 {
            0
        } else {
            (t - 1) * d_model * d_state
        };
        let dxc_row = &mut dx_conv_s[t * d_model..(t + 1) * d_model];
        let dbs_row = &mut d_bs_s[t * d_state..(t + 1) * d_state];
        let ddelta_row = &mut d_delta_s[t * d_model..(t + 1) * d_model];

        for i in 0..d_model {
            let dt_i = delta_row[i];
            let dt_v = f32x8::splat(dt_i);
            let xc_i = xc_row[i];
            let xc_v = f32x8::splat(xc_i);
            let dyi_v = f32x8::splat(dy_row[i]);
            let a_off = i * d_state;
            let h_off = i * d_state;
            let mut acc_d_delta = f32x8::ZERO;

            for blk in 0..n_blocks {
                let s_off = blk * LANES;
                let cs_v = load8(cs_row, s_off);
                let dh_state_v = dyi_v * cs_v;
                let dh_carry_v = load8(&dh_carry, h_off + s_off);
                let dh_total = dh_state_v + dh_carry_v;
                let a_v = load8(a_s, a_off + s_off);
                let a_bar = (dt_v * a_v).exp();
                let h_prev = if t == 0 {
                    f32x8::ZERO
                } else {
                    load8(h_traj_s, prev_htraj_off + h_off + s_off)
                };
                let d_v = dh_total;
                let d_a_bar = dh_total * h_prev;
                store8(&mut dh_carry, h_off + s_off, dh_total * a_bar);
                let d_da = d_a_bar * a_bar;
                acc_d_delta = d_da.mul_add(a_v, acc_d_delta);
                let ga_v = load8(grad_a_s, a_off + s_off);
                store8(grad_a_s, a_off + s_off, d_da.mul_add(dt_v * a_v, ga_v));
                let bs_v = load8(bs_row, s_off);
                acc_d_delta = (d_v * bs_v).mul_add(xc_v, acc_d_delta);
                let dbs_v = load8(dbs_row, s_off);
                store8(dbs_row, s_off, (d_v * dt_v).mul_add(xc_v, dbs_v));
                dxc_row[i] += hsum(d_v * dt_v * bs_v);
            }

            ddelta_row[i] += hsum(acc_d_delta);
        }
    }

    SsmBackwardOutputs {
        grad_a_log,
        grad_d_skip,
        d_bs,
        d_cs,
        d_delta,
        dx_conv,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn chunk_len_increases_with_sequence() {
        assert_eq!(recommended_scan_chunk_len(128), 8);
        assert_eq!(recommended_scan_chunk_len(256), 16);
        assert_eq!(recommended_scan_chunk_len(1024), 32);
    }

    #[test]
    fn vector_sum_matches_scalar_sum() {
        let got = scalar_sum8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!((got - 36.0).abs() < 1e-6);
    }

    #[test]
    fn simd_forward_matches_scalar_reference() {
        let t = 3usize;
        let d = 4usize;
        let s = 8usize;
        let bs = Array2::from_shape_fn((t, s), |(ti, si)| 0.01 * (ti + si + 1) as f32);
        let cs = Array2::from_shape_fn((t, s), |(ti, si)| 0.02 * (1 + (ti * 3 + si) as i32) as f32);
        let delta = Array2::from_shape_fn((t, d), |(ti, di)| 0.03 * (1 + (ti + di) as i32) as f32);
        let x_conv =
            Array2::from_shape_fn((t, d), |(ti, di)| 0.04 * (1 + (ti * 2 + di) as i32) as f32);
        let a = Array2::from_shape_fn((d, s), |(di, si)| -0.1 + 0.001 * (di * s + si) as f32);
        let d_skip = Array1::from_shape_fn(d, |di| 0.01 * (di + 1) as f32);

        let mut h_scalar = Array2::zeros((d, s));
        let mut htraj_scalar = Array3::zeros((t, d, s));
        let mut y_scalar = Array2::zeros((t, d));
        ssm_scan_forward_scalar(
            bs.view(),
            cs.view(),
            delta.view(),
            x_conv.view(),
            a.view(),
            d_skip.view(),
            &mut h_scalar,
            &mut htraj_scalar,
            &mut y_scalar,
        );

        let mut h_simd = Array2::zeros((d, s));
        let mut htraj_simd = Array3::zeros((t, d, s));
        let mut y_simd = Array2::zeros((t, d));
        ssm_scan_forward_simd(
            bs.view(),
            cs.view(),
            delta.view(),
            x_conv.view(),
            a.view(),
            d_skip.view(),
            &mut h_simd,
            &mut htraj_simd,
            &mut y_simd,
        );

        let y_err = (&y_scalar - &y_simd).mapv(f32::abs).sum();
        let h_err = (&h_scalar - &h_simd).mapv(f32::abs).sum();
        let traj_err = (&htraj_scalar - &htraj_simd).mapv(f32::abs).sum();
        assert!(y_err < 1e-4, "y error too high: {y_err}");
        assert!(h_err < 1e-4, "h error too high: {h_err}");
        assert!(traj_err < 1e-4, "h_traj error too high: {traj_err}");
    }

    #[test]
    fn simd_backward_matches_scalar_reference() {
        let t = 3usize;
        let d = 4usize;
        let s = 8usize;
        let bs = Array2::from_shape_fn((t, s), |(ti, si)| 0.01 * (ti + si + 1) as f32);
        let cs = Array2::from_shape_fn((t, s), |(ti, si)| 0.02 * (1 + (ti * 3 + si) as i32) as f32);
        let delta = Array2::from_shape_fn((t, d), |(ti, di)| 0.03 * (1 + (ti + di) as i32) as f32);
        let x_conv =
            Array2::from_shape_fn((t, d), |(ti, di)| 0.04 * (1 + (ti * 2 + di) as i32) as f32);
        let a = Array2::from_shape_fn((d, s), |(di, si)| -0.1 + 0.001 * (di * s + si) as f32);
        let d_skip = Array1::from_shape_fn(d, |di| 0.01 * (di + 1) as f32);
        let dy_pre = Array2::from_shape_fn((t, d), |(ti, di)| 0.05 * (1 + (ti + di) as i32) as f32);

        let mut h = Array2::zeros((d, s));
        let mut h_traj = Array3::zeros((t, d, s));
        let mut y = Array2::zeros((t, d));
        ssm_scan_forward_scalar(
            bs.view(),
            cs.view(),
            delta.view(),
            x_conv.view(),
            a.view(),
            d_skip.view(),
            &mut h,
            &mut h_traj,
            &mut y,
        );

        let scalar = ssm_scan_backward_scalar(
            bs.view(),
            cs.view(),
            delta.view(),
            x_conv.view(),
            a.view(),
            d_skip.view(),
            h_traj.view(),
            dy_pre.view(),
        );
        let simd = ssm_scan_backward_simd(
            bs.view(),
            cs.view(),
            delta.view(),
            x_conv.view(),
            a.view(),
            d_skip.view(),
            h_traj.view(),
            dy_pre.view(),
        );

        let a_err = (&scalar.grad_a_log - &simd.grad_a_log).mapv(f32::abs).sum();
        let dskip_err = (&scalar.grad_d_skip - &simd.grad_d_skip)
            .mapv(f32::abs)
            .sum();
        let dbs_err = (&scalar.d_bs - &simd.d_bs).mapv(f32::abs).sum();
        let dcs_err = (&scalar.d_cs - &simd.d_cs).mapv(f32::abs).sum();
        let ddelta_err = (&scalar.d_delta - &simd.d_delta).mapv(f32::abs).sum();
        let dxc_err = (&scalar.dx_conv - &simd.dx_conv).mapv(f32::abs).sum();
        assert!(a_err < 1e-4, "grad_a error too high: {a_err}");
        assert!(dskip_err < 1e-4, "grad_d_skip error too high: {dskip_err}");
        assert!(dbs_err < 1e-4, "d_bs error too high: {dbs_err}");
        assert!(dcs_err < 1e-4, "d_cs error too high: {dcs_err}");
        assert!(ddelta_err < 1e-4, "d_delta error too high: {ddelta_err}");
        assert!(dxc_err < 1e-4, "dx_conv error too high: {dxc_err}");
    }

    #[test]
    fn simd_conv_matches_scalar_reference() {
        let t = 5usize;
        let d = 8usize;
        let k = 4usize;
        let x = Array2::from_shape_fn((t, d), |(ti, di)| 0.01 * (1 + ti + di) as f32);
        let w = Array2::from_shape_fn((d, k), |(di, ki)| 0.02 * (1 + di + ki) as f32);
        let b = Array1::from_shape_fn(d, |di| 0.03 * (1 + di) as f32);
        let (pre_scalar, out_scalar) = conv1d_silu_forward_scalar(x.view(), w.view(), b.view());
        let (pre_simd, out_simd) = conv1d_silu_forward_simd(x.view(), w.view(), b.view());
        let pre_err = (&pre_scalar - &pre_simd).mapv(f32::abs).sum();
        let out_err = (&out_scalar - &out_simd).mapv(f32::abs).sum();
        assert!(pre_err < 1e-5, "pre conv error too high: {pre_err}");
        assert!(out_err < 1e-5, "silu conv error too high: {out_err}");
    }
}
