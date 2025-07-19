from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax  # type: ignore[import-untyped]
from optax import GradientTransformationExtraArgs, OptState, Params

from gs.loss_function import gs_loss
from gs.projection import project
from gs.rasterization import rasterize


class DataLogger:
    def __init__(self, save_path: Path) -> None:
        self.count = 0
        self.save_path = save_path

        self.save_path.mkdir(parents=True, exist_ok=True)

    def __call__(self, image: jax.Array) -> None:
        plt.imsave(self.save_path / f"output_{self.count:05d}.png", image.clip(0, 1))
        self.count += 1


def get_corrected_params(params: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """パラメータを補正"""
    return {
        "means3d": params["means3d"],
        "quats": params["quats"] / (jnp.linalg.norm(params["quats"], axis=-1, keepdims=True)),
        "scales": jnp.exp(params["scales"]),
        "sh_coeffs": jnp.dstack((params["sh_dc"], params["sh_rest"])),
        "opacities": jax.nn.sigmoid(params["opacities"]),
    }


def make_updater(
    consts: dict[str, Any],
    optimizer: GradientTransformationExtraArgs,
    callback: Callable = lambda _: None,
    *,
    jit: bool = True,
) -> Callable:
    def loss_fn_for_mean2d(
        means_2d: jax.Array, fixed_params: dict[str, jax.Array], target: jax.Array
    ) -> jax.Array:
        """密度制御のためのView-Space Gradientsを中間勾配として取得するための損失関数"""
        projected_gaussians = {"means_2d": means_2d, **fixed_params}
        output = rasterize(projected_gaussians, consts)
        jax.debug.callback(callback, output)
        loss = gs_loss(output, target)
        return loss

    def loss_fn_for_params(
        params: dict[str, jax.Array],
        view: dict[str, jax.Array],
        target: jax.Array,
        active_sh_degree: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        corrected_params = get_corrected_params(params)
        projected_gaussians = project(
            corrected_params, **view, consts=consts, active_sh_degree=active_sh_degree
        )

        # View-Space Gradientsを中間勾配として取得する
        means_2d = projected_gaussians["means_2d"]
        fixed_params = {k: v for k, v in projected_gaussians.items() if k != "means_2d"}
        loss, viewspace_grads = jax.value_and_grad(loss_fn_for_mean2d)(
            means_2d, fixed_params, target
        )

        return loss, viewspace_grads  # viewspace_gradsは補助データとして返す

    compute_loss_and_grad = jax.value_and_grad(loss_fn_for_params, has_aux=True)

    def update(
        params: Params,
        view: dict[str, jax.Array],
        target: jax.Array,
        opt_state: OptState,
        active_sh_degree: int,
    ) -> tuple[Params, OptState, jax.Array, dict[str, jax.Array]]:
        (loss, viewspace_grads), grads = compute_loss_and_grad(
            params, view, target, active_sh_degree
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, viewspace_grads

    return jax.jit(update, donate_argnames=("params",)) if jit else update


def make_render(
    consts: dict[str, Any], active_sh_degree: jax.Array, *, jit: bool = True
) -> Callable:
    def render(params: dict[str, jax.Array], view: dict[str, jax.Array]) -> jax.Array:
        projected_gaussians = project(
            params, **view, consts=consts, active_sh_degree=active_sh_degree
        )

        return rasterize(projected_gaussians, consts)

    return jax.jit(render) if jit else render
