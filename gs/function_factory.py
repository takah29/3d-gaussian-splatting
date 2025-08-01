from collections.abc import Callable
from typing import Any

import jax
import optax  # type: ignore[import-untyped]
from optax import GradientTransformationExtraArgs, OptState, Params

from gs.core.loss_function import gs_loss
from gs.core.projection import project
from gs.core.rasterization import rasterize
from gs.utils import get_corrected_params


def make_updater(
    consts: dict[str, Any],
    optimizer: GradientTransformationExtraArgs,
    callback: Callable = lambda _: None,
    *,
    jit: bool = True,
) -> Callable[
    [Params, dict[str, jax.Array], jax.Array, OptState, int],
    tuple[Params, OptState, jax.Array, dict[str, jax.Array]],
]:
    def loss_fn_for_mean2d(
        means_2d: jax.Array, fixed_params_2d: dict[str, jax.Array], target: jax.Array
    ) -> jax.Array:
        """密度制御のためのView-Space Gradientsを中間勾配として取得するための損失関数"""
        projected_gaussians = {"means_2d": means_2d, **fixed_params_2d}
        output = rasterize(projected_gaussians, consts)
        jax.debug.callback(callback, output)
        loss = gs_loss(output, target)
        return loss

    def loss_fn_for_params(
        raw_params: dict[str, jax.Array],
        view: dict[str, jax.Array],
        target: jax.Array,
        active_sh_degree: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        corrected_params = get_corrected_params(raw_params)
        projected_gaussians = project(
            corrected_params, **view, consts=consts, active_sh_degree=active_sh_degree
        )

        # View-Space Gradientsを中間勾配として取得する
        means_2d = projected_gaussians["means_2d"]
        fixed_params_2d = {k: v for k, v in projected_gaussians.items() if k != "means_2d"}
        loss, viewspace_grads = jax.value_and_grad(loss_fn_for_mean2d)(
            means_2d, fixed_params_2d, target
        )

        return loss, viewspace_grads  # viewspace_gradsは補助データとして返す

    compute_loss_and_grad = jax.value_and_grad(loss_fn_for_params, has_aux=True)

    def update(
        raw_params: Params,
        view: dict[str, jax.Array],
        target: jax.Array,
        opt_state: OptState,
        active_sh_degree: int,
    ) -> tuple[Params, OptState, jax.Array, dict[str, jax.Array]]:
        (loss, viewspace_grads), grads = compute_loss_and_grad(
            raw_params, view, target, active_sh_degree
        )
        updates, opt_state = optimizer.update(grads, opt_state, raw_params)
        raw_params = optax.apply_updates(raw_params, updates)
        return raw_params, opt_state, loss, viewspace_grads

    return jax.jit(update, donate_argnames=("raw_params",)) if jit else update


def make_render(
    consts: dict[str, Any], active_sh_degree: int, *, jit: bool = True
) -> Callable[[dict[str, jax.Array], dict[str, jax.Array]], jax.Array]:
    def render(params: dict[str, jax.Array], view: dict[str, jax.Array]) -> jax.Array:
        projected_gaussians = project(
            params,
            **view,
            consts=consts,
            active_sh_degree=active_sh_degree,  # type: ignore[arg-type]
        )

        return rasterize(projected_gaussians, consts)

    return jax.jit(render) if jit else render
