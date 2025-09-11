"""This file implements a custom VJP version for 3DGS color computation.

It has higher memory usage and slower processing compared to the automatic differentiation version,
so it is not used. It stores intermediate variables computed during forward pass and performs
VJP calculations in parallel without using jax.lax.scan during backward pass.

Reference: https://github.com/scomup/EasyGaussianSplatting
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp


@partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
def compute_gaussian_weight(
    pixel_coord: jax.Array,
    mean_2d: jax.Array,
    cov_2d_inv_flat: jax.Array,
    opacity: jax.Array,
) -> jax.Array:
    a11, a12, a22 = cov_2d_inv_flat

    dx = pixel_coord[0] - mean_2d[0]
    dy = pixel_coord[1] - mean_2d[1]

    mahal_dist = jnp.maximum(
        0.0,
        dx * dx * a11 + 2 * dx * dy * a12 + dy * dy * a22,
    )

    return jnp.minimum(0.99, jnp.exp(-0.5 * mahal_dist) * opacity[0])


def compute_color_fwd_base(
    pixel_coord: jax.Array, depth_sorted_indices: jax.Array, gaussians: dict[str, jax.Array]
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
    means_2d = gaussians["means_2d"][depth_sorted_indices]
    covs_2d_inv_flat = gaussians["covs_2d_inv_flat"][depth_sorted_indices]
    opacities = gaussians["opacities"][depth_sorted_indices]
    colors = gaussians["colors"][depth_sorted_indices]

    gaussian_weights, (dalphaprime_dus, dalphaprime_dcinv2ds, dalphaprime_dalphas) = jax.vmap(
        jax.value_and_grad(compute_gaussian_weight, argnums=(1, 2, 3)), in_axes=(None, 0, 0, 0)
    )(pixel_coord, means_2d, covs_2d_inv_flat, opacities)

    @partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
    def body_fun(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        pixel_color, tau, count = carry
        gaussian_idx, gaussian_weight, color = inputs

        @partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
        def true_fun(
            pixel_color: jax.Array, tau: jax.Array, count: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            updated_pixel_color = pixel_color + color * gaussian_weight * tau
            updated_tau = tau * (1.0 - gaussian_weight)
            updated_count = count + 1
            return updated_pixel_color, updated_tau, updated_count

        updated_pixel_color, updated_tau, updated_count = jax.lax.cond(
            (gaussian_idx >= 0) & (gaussian_weight > 1.0 / 255.0) & (tau > 1e-3),
            true_fun,
            lambda pixel_color, tau, count: (pixel_color, tau, count),
            pixel_color,
            tau,
            count,
        )

        return (updated_pixel_color, updated_tau, updated_count), (pixel_color, tau)

    pixel_color = jnp.zeros(3, dtype=jnp.float32)
    tau = 1.0
    count = 0

    (final_pixel_color, _, _), (sum_arr, tau_arr) = jax.lax.scan(
        body_fun,
        (pixel_color, tau, count),
        (depth_sorted_indices, gaussian_weights, colors),
    )

    return final_pixel_color, (
        gaussian_weights,
        sum_arr,
        tau_arr,
        dalphaprime_dus,
        dalphaprime_dcinv2ds,
        dalphaprime_dalphas,
    )


compute_color_fwd_base_vmap = jax.vmap(
    jax.vmap(compute_color_fwd_base, in_axes=(0, None, None)), in_axes=(0, None, None)
)


@jax.custom_vjp
def compute_color_vmap(
    pixel_coords: jax.Array, depth_sorted_indices: jax.Array, gaussians: dict[str, jax.Array]
) -> jax.Array:
    return compute_color_fwd_base_vmap(pixel_coords, depth_sorted_indices, gaussians)[0]


def compute_color_vmap_fwd(
    pixel_coords: jax.Array, depth_sorted_indices: jax.Array, gaussians: dict[str, jax.Array]
) -> tuple[
    jax.Array,
    tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        dict[str, jax.Array],
        jax.Array,
    ],
]:
    final_pixel_color_batch, residuals = compute_color_fwd_base_vmap(
        pixel_coords, depth_sorted_indices, gaussians
    )

    return final_pixel_color_batch, (
        final_pixel_color_batch,
        *residuals,
        gaussians,
        depth_sorted_indices,
    )


def compute_color_bwd_base(
    final_pixel_color: jax.Array,
    gaussian_weights: jax.Array,
    sum_arr: jax.Array,
    tau_arr: jax.Array,
    dalphaprime_dus: jax.Array,
    dalphaprime_dcinv2ds: jax.Array,
    dalphaprime_dalphas: jax.Array,
    colors: jax.Array,
    g: jax.Array,  # dloss_dgamma: shape = (3, )
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    gs_num = colors.shape[0]

    gamma_arr = (final_pixel_color - sum_arr) / tau_arr[:, None]
    gamma_shift_arr = jnp.vstack((gamma_arr[1:], jnp.zeros((1, 3))))

    dgamma_dalphaprimes = (tau_arr[:, None] * (colors - gamma_shift_arr)).reshape(gs_num, 3, 1)

    # compute vjp
    # (gs_num, 1) = (1, 1, 3) @ (gs_num, 3, 1) @ (gs_num, 1, 1)
    dloss_dalphas = (g[None, None, :] @ dgamma_dalphaprimes * dalphaprime_dalphas[:, None]).squeeze(
        1
    )
    # (gs_num, 3) = (1, 1, 3) @ (gs_num, 3, 1) @ (gs_num, 1, 3)
    dloss_dcinv2ds = (
        g[None, None, :] @ (dgamma_dalphaprimes) @ dalphaprime_dcinv2ds[:, None]
    ).squeeze(1)
    # (gs_num, 3) = (3, ) * (gs_num, 1)
    dloss_dcolors = g * (tau_arr * gaussian_weights)[:, None]
    # (gs_num, 2) = (1, 1, 3) @ (gs_num, 3, 1) @ (gs_num, 1, 2)
    dloss_dus = (g[None, None, :] @ dgamma_dalphaprimes @ dalphaprime_dus[:, None]).squeeze(1)

    return (dloss_dalphas, dloss_dcinv2ds, dloss_dcolors, dloss_dus)


compute_color_bwd_base_vmap = jax.vmap(
    jax.vmap(compute_color_bwd_base, in_axes=(0, 0, 0, 0, 0, 0, 0, None, 0)),
    in_axes=(0, 0, 0, 0, 0, 0, 0, None, 0),
)


def compute_color_vmap_bwd(
    residuals: jax.Array, g: jax.Array
) -> tuple[None, None, dict[str, jax.Array]]:
    (
        final_pixel_color_batch,
        gaussian_weights_batch,
        sum_arr_batch,
        tau_arr_batch,
        dalphaprime_dus_batch,
        dalphaprime_dcinv2ds_batch,
        dalphaprime_dalphas_batch,
        gaussians,
        depth_sorted_indices,
    ) = residuals

    (
        dloss_dalphas_batch,
        dloss_dcinv2ds_batch,
        dloss_dcolors_batch,
        dloss_dus_batch,
    ) = compute_color_bwd_base_vmap(
        final_pixel_color_batch,
        gaussian_weights_batch,
        sum_arr_batch,
        tau_arr_batch,
        dalphaprime_dus_batch,
        dalphaprime_dcinv2ds_batch,
        dalphaprime_dalphas_batch,
        gaussians["colors"][depth_sorted_indices],
        g,
    )
    gaussians_vjp = {
        "means_2d": jnp.zeros(gaussians["means_2d"].shape)
        .at[depth_sorted_indices]
        .add(dloss_dus_batch.sum(axis=(0, 1))),
        "covs_2d_inv_flat": jnp.zeros(gaussians["covs_2d_inv_flat"].shape)
        .at[depth_sorted_indices]
        .add(dloss_dcinv2ds_batch.sum(axis=(0, 1))),
        "opacities": jnp.zeros(gaussians["opacities"].shape)
        .at[depth_sorted_indices]
        .add(dloss_dalphas_batch.sum(axis=(0, 1))),
        "colors": jnp.zeros(gaussians["colors"].shape)
        .at[depth_sorted_indices]
        .add(dloss_dcolors_batch.sum(axis=(0, 1))),
        "covs_2d": jnp.zeros(gaussians["covs_2d"].shape),
        "depths": jnp.zeros(gaussians["depths"].shape),
    }
    return (None, None, gaussians_vjp)


compute_color_vmap.defvjp(compute_color_vmap_fwd, compute_color_vmap_bwd)


def rasterize_tile_data(
    depth_decending_indices: jax.Array,
    upperleft_coord: jax.Array,
    gaussians: dict[str, jax.Array],
    tile_size: int,
) -> jax.Array:
    ii, jj = jnp.mgrid[0:tile_size, 0:tile_size]
    pixel_coords = jnp.stack([upperleft_coord[0] + jj + 0.5, upperleft_coord[1] + ii + 0.5], axis=2)

    image_buffer = compute_color_vmap(pixel_coords, depth_decending_indices, gaussians)

    return image_buffer


rasterize_tile_data_vmap = jax.vmap(rasterize_tile_data, in_axes=(0, 0, None, None))


def rasterize(
    gaussians: dict[str, jax.Array],
    tile_depth_decending_indices_batch: jax.Array,
    tile_upperleft_coord_batch: jax.Array,
    consts: dict[str, Any],
) -> jax.Array:
    """projectで射影した2Dガウシアンをラスタライズする.

    Note:
      * 画像配列の次元は[H(y軸), W(x軸), C]で実装を統一する
      * 数値計算の座標は(x, y)で行っていることに注意
    """
    img_shape = consts["img_shape"]
    tile_size = consts["tile_size"]
    tile_chanks = consts["tile_chanks"]

    image_buffer_batch = jax.lax.map(
        lambda args: rasterize_tile_data_vmap(args[0], args[1], gaussians, tile_size),
        (
            tile_depth_decending_indices_batch,
            tile_upperleft_coord_batch,
        ),
        batch_size=(tile_upperleft_coord_batch.shape[0] + tile_chanks - 1) // tile_chanks,
    )

    # タイルごとのバッファを結合
    transposed = jnp.transpose(image_buffer_batch, (0, 2, 1, 3, 4))
    final_buffer = transposed.reshape(
        transposed.shape[0] * transposed.shape[1],
        transposed.shape[2] * transposed.shape[3],
        transposed.shape[4],
    )[: img_shape[0], : img_shape[1]]

    return final_buffer
