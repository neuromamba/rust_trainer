#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp


def _l2_normalize_rows(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    denom = jnp.sqrt(jnp.sum(x * x, axis=1, keepdims=True)).clip(min=eps)
    return x / denom


def hpn_loss_and_grads(z: jnp.ndarray, target: jnp.ndarray, prototypes: jnp.ndarray):
    z_norm = _l2_normalize_rows(z)
    yi = target.astype(jnp.int32)
    cos = jnp.sum(z_norm * prototypes[yi], axis=1)
    loss = jnp.mean((1.0 - cos) ** 2)

    def loss_fn(z_in, p_in):
        z_n = _l2_normalize_rows(z_in)
        c = jnp.sum(z_n * p_in[yi], axis=1)
        return jnp.mean((1.0 - c) ** 2)

    dz, dp = jax.grad(loss_fn, argnums=(0, 1))(z, prototypes)
    return loss, dz, dp


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: jax_parity_reference.py <input.json> <output.json>", file=sys.stderr)
        return 2

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    payload = json.loads(in_path.read_text())
    z = jnp.array(payload["z"], dtype=jnp.float32)
    target = jnp.array(payload["target"], dtype=jnp.int32)
    prototypes = jnp.array(payload["prototypes"], dtype=jnp.float32)
    loss, dz, dp = hpn_loss_and_grads(z, target, prototypes)

    out = {
        "loss": float(loss),
        "dz": jnp.asarray(dz).tolist(),
        "d_prototypes": jnp.asarray(dp).tolist(),
    }
    out_path.write_text(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
