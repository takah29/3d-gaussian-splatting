from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax  # type: ignore[import-untyped]
from optax import GradientTransformationExtraArgs, OptState, Params

from gs.core.loss_function import gs_loss
from gs.core.projection import project
from gs.core.rasterization import rasterize
from gs.core.tiling_and_sorting import build_tile_data
from gs.utils import fix_quaternions, get_corrected_params


def make_updater(
    consts: dict[str, Any],
    optimizer: GradientTransformationExtraArgs,
    callback: Callable = lambda _: None,
    *,
    jit: bool = True,
) -> Callable[
    [Params, dict[str, jax.Array], jax.Array, OptState, int, float, jax.Array],
    tuple[Params, OptState, jax.Array, jax.Array, dict[str, jax.Array]],
]:
    def loss_fn_for_mean2d(
        means_2d: jax.Array, fixed_params_2d: dict[str, jax.Array], target: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """密度制御のためのView-Space Gradientsを中間勾配として取得するための損失関数"""
        projected_gaussians = {"means_2d": means_2d, **fixed_params_2d}
        tile_depth_decending_indices_batch, tile_upperleft_coord_batch = build_tile_data(
            projected_gaussians, consts
        )

        image, contribution_scores = rasterize(
            projected_gaussians,
            tile_depth_decending_indices_batch,
            tile_upperleft_coord_batch,
            consts,
        )

        jax.debug.callback(callback, image)
        loss = gs_loss(image, target)
        return loss, contribution_scores

    def loss_fn_for_params(
        raw_params: dict[str, jax.Array],
        view: dict[str, jax.Array],
        target: jax.Array,
        active_sh_degree: jax.Array,
        drop_rate: float,
        key: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, dict[str, jax.Array]]]:
        corrected_params = get_corrected_params(raw_params)
        projected_gaussians = project(
            corrected_params, **view, consts=consts, active_sh_degree=active_sh_degree
        )
        n_gaussians = projected_gaussians["means_2d"].shape[0]
        drop_indices = jax.random.choice(
            key, n_gaussians, shape=(int(n_gaussians * drop_rate),), replace=False
        )
        drop_mask = jnp.isin(jnp.arange(n_gaussians), drop_indices)

        projected_gaussians["opacities"] = jnp.where(
            drop_mask[:, None], 0.0, projected_gaussians["opacities"] / (1.0 - drop_rate)
        )

        # View-Space Gradientsを中間勾配として取得する
        means_2d = projected_gaussians["means_2d"]
        fixed_params_2d = {k: v for k, v in projected_gaussians.items() if k != "means_2d"}
        (loss, contribution_scores), viewspace_grads = jax.value_and_grad(
            loss_fn_for_mean2d, has_aux=True
        )(means_2d, fixed_params_2d, target)

        return loss, (contribution_scores, viewspace_grads)  # viewspace_gradsは補助データとして返す

    compute_loss_and_grad = jax.value_and_grad(loss_fn_for_params, has_aux=True)

    def update(
        raw_params: Params,
        view: dict[str, jax.Array],
        target: jax.Array,
        opt_state: OptState,
        active_sh_degree: int,
        drop_rate: float,
        key: jax.Array,
    ) -> tuple[Params, OptState, jax.Array, jax.Array, dict[str, jax.Array]]:
        (loss, (contribution_scores, viewspace_grads)), grads = compute_loss_and_grad(
            raw_params, view, target, active_sh_degree, drop_rate, key
        )
        updates, opt_state = optimizer.update(grads, opt_state, raw_params)
        raw_params = optax.apply_updates(raw_params, updates)
        raw_params = fix_quaternions(raw_params)  # type: ignore[arg-type]
        return raw_params, opt_state, loss, contribution_scores, viewspace_grads

    return (
        jax.jit(update, donate_argnames=("raw_params",), static_argnames=("drop_rate",))
        if jit
        else update
    )


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

        tile_depth_decending_indices_batch, tile_upperleft_coord_batch = build_tile_data(
            projected_gaussians, consts
        )

        image_buffer, _ = rasterize(
            projected_gaussians,
            tile_depth_decending_indices_batch,
            tile_upperleft_coord_batch,
            consts,
        )

        return image_buffer

    return jax.jit(render) if jit else render
