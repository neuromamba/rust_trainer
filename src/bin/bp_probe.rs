use ndarray::{Array1, Array2, Array3};
use rust_trainer::simd_ops::{
    ssm_scan_backward_scalar, ssm_scan_backward_simd, ssm_scan_forward_scalar,
};
use serde_json::json;

fn main() {
    let t = 4usize;
    let d = 8usize;
    let s = 16usize;
    let bs = Array2::from_shape_fn((t, s), |(ti, si)| 0.01 * (1 + ti + si) as f32);
    let cs = Array2::from_shape_fn((t, s), |(ti, si)| 0.02 * (1 + ti * 2 + si) as f32);
    let delta = Array2::from_shape_fn((t, d), |(ti, di)| 0.03 * (1 + ti + di) as f32);
    let x_conv = Array2::from_shape_fn((t, d), |(ti, di)| 0.04 * (1 + ti * 3 + di) as f32);
    let a = Array2::from_shape_fn((d, s), |(di, si)| -0.1 + 0.0005 * (di * s + si) as f32);
    let d_skip = Array1::from_shape_fn(d, |di| 0.01 * (di + 1) as f32);
    let dy_pre = Array2::from_shape_fn((t, d), |(ti, di)| 0.05 * (1 + ti + di) as f32);

    let mut h = Array2::zeros((d, s));
    let mut h_traj = Array3::zeros((t, d, s));
    let mut y_pre = Array2::zeros((t, d));
    ssm_scan_forward_scalar(
        bs.view(),
        cs.view(),
        delta.view(),
        x_conv.view(),
        a.view(),
        d_skip.view(),
        &mut h,
        &mut h_traj,
        &mut y_pre,
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

    let out = json!({
        "grad_a_l1": (&scalar.grad_a_log - &simd.grad_a_log).mapv(f32::abs).sum(),
        "grad_d_skip_l1": (&scalar.grad_d_skip - &simd.grad_d_skip).mapv(f32::abs).sum(),
        "d_bs_l1": (&scalar.d_bs - &simd.d_bs).mapv(f32::abs).sum(),
        "d_cs_l1": (&scalar.d_cs - &simd.d_cs).mapv(f32::abs).sum(),
        "d_delta_l1": (&scalar.d_delta - &simd.d_delta).mapv(f32::abs).sum(),
        "dx_conv_l1": (&scalar.dx_conv - &simd.dx_conv).mapv(f32::abs).sum(),
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}