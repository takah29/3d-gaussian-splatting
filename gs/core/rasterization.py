from functools import partial
from typing import Any

import jax
import jax.numpy as jnp


@partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
def _gaussian_weight(
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

    return jnp.minimum(0.99, jnp.exp(-0.5 * mahal_dist) * opacity)


@partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
def _render_pixel(
    pixel_coord: jax.Array,
    depth_decending_indices: jax.Array,
    gaussians: dict[str, jax.Array],
    background: tuple[float, float, float],
) -> jax.Array:
    pixel_color = jnp.zeros((3,), dtype=jnp.float32)
    tau = jnp.ones((1,), dtype=jnp.float32)

    means_2d = gaussians["means_2d"][depth_decending_indices]
    covs_2d_inv_flat = gaussians["covs_2d_inv_flat"][depth_decending_indices]
    opacities = gaussians["opacities"][depth_decending_indices]
    colors = gaussians["colors"][depth_decending_indices]

    gaussian_weight_batch = jax.vmap(_gaussian_weight, in_axes=(None, 0, 0, 0))(
        pixel_coord, means_2d, covs_2d_inv_flat, opacities
    )

    @partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
    def body_fun(
        carry: tuple[jax.Array, jax.Array], inputs: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        pixel_color, tau = carry
        gaussian_idx, gaussian_weight, color = inputs

        @partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
        def true_fun(pixel_color: jax.Array, tau: jax.Array) -> tuple[jax.Array, jax.Array]:
            updated_pixel_color = pixel_color + color * gaussian_weight * tau
            updated_tau = tau * (1.0 - gaussian_weight)
            return updated_pixel_color, updated_tau

        updated_pixel_color, updated_tau = jax.lax.cond(
            (gaussian_idx >= 0) & (gaussian_weight[0] > 1.0 / 255.0) & (tau[0] > 1e-3),
            true_fun,
            lambda pixel_color, tau: (pixel_color, tau),
            pixel_color,
            tau,
        )

        return (updated_pixel_color, updated_tau), None

    (pixel_color, tau), _ = jax.lax.scan(
        body_fun,
        (pixel_color, tau),
        (depth_decending_indices, gaussian_weight_batch, colors),
    )
    return pixel_color + jnp.asarray(background) * tau


def rasterize_tile_data(
    depth_decending_indices: jax.Array,
    upperleft_coord: jax.Array,
    gaussians: dict[str, jax.Array],
    background: tuple[float, float, float],
    tile_size: int,
) -> jax.Array:
    ii, jj = jnp.mgrid[0:tile_size, 0:tile_size]
    pixel_coords = jnp.stack([upperleft_coord[0] + jj + 0.5, upperleft_coord[1] + ii + 0.5], axis=2)

    image_buffer = render_pixel_vmap(pixel_coords, depth_decending_indices, gaussians, background)

    return image_buffer


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
    background = consts["background"]
    tile_size = consts["tile_size"]
    tile_chanks = consts["tile_chanks"]

    image_buffer_batch = jax.lax.map(
        lambda args: rasterize_tile_data_vmap(args[0], args[1], gaussians, background, tile_size),
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


rasterize_tile_data_vmap = jax.vmap(rasterize_tile_data, in_axes=(0, 0, None, None, None))
render_pixel_vmap = jax.vmap(
    jax.vmap(_render_pixel, in_axes=(0, None, None, None)), in_axes=(0, None, None, None)
)
