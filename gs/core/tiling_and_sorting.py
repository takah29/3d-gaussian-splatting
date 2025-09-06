from functools import partial
from typing import Any

import jax
import jax.numpy as jnp


@partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
def analytical_max_eigenvalue(mat2x2: jax.Array) -> jax.Array:
    a = mat2x2[0, 0]
    b = mat2x2[0, 1]
    d = mat2x2[1, 1]

    trace = a + d
    diff = a - d
    sqrt_term = jnp.sqrt(diff * diff + 4 * b * b)

    return (trace + sqrt_term) * 0.5


def _depth_sorted_indices(
    tile_index: jax.Array,
    gaussian_index_intervals: jax.Array,
    gaussian_depth: jax.Array,
    tile_max_gs_num: int,
) -> jax.Array:
    visibility_mask = (
        (gaussian_index_intervals[:, 0, 0] <= tile_index[0])
        & (tile_index[0] <= gaussian_index_intervals[:, 1, 0])
        & (gaussian_index_intervals[:, 0, 1] <= tile_index[1])
        & (tile_index[1] <= gaussian_index_intervals[:, 1, 1])
        & (gaussian_depth > 0.2)  # Culling Gaussians closer to the camera
    )
    depth_arr = jnp.where(visibility_mask, gaussian_depth, jnp.inf)
    tile_inverse_depth_topk, tile_depth_topk_indices = jax.lax.top_k(-depth_arr, k=tile_max_gs_num)
    return jnp.where(tile_inverse_depth_topk == -jnp.inf, -1, tile_depth_topk_indices)


def _create_tile_depth_decending_indices_batch(
    gaussians: dict[str, jax.Array],
    tile_index_batch: jax.Array,
    consts: dict[str, Any],
) -> jax.Array:
    # ガウシアンの所属するタイルのインデックスを計算
    gauss_max_eigvals = jax.vmap(analytical_max_eigenvalue)(gaussians["covs_2d"])
    r_batch = 3.0 * jnp.sqrt(gauss_max_eigvals)[:, None]
    gaussian_intervals = jnp.stack(
        (gaussians["means_2d"] - r_batch, gaussians["means_2d"] + r_batch), axis=1
    )
    gaussian_index_intervals = (gaussian_intervals // consts["tile_size"]).astype(
        jnp.int32
    )  # [[x_low_idx, y_low_idx], [x_high_idx, y_high_idx]]

    return jax.lax.map(
        lambda x: depth_sorted_indices_vmap(
            x, gaussian_index_intervals, gaussians["depths"], consts["tile_max_gs_num"]
        ),
        tile_index_batch,
        batch_size=(tile_index_batch.shape[0] + consts["tile_chanks"] - 1) // consts["tile_chanks"],
    )


def build_tile_data(
    gaussians: dict[str, jax.Array],
    consts: dict[str, Any],
) -> tuple[jax.Array, jax.Array]:
    tile_size = consts["tile_size"]
    img_shape = consts["img_shape"]

    height_split_num = (img_shape[0] + tile_size - 1) // tile_size
    width_split_num = (img_shape[1] + tile_size - 1) // tile_size

    ii, jj = jnp.mgrid[0:height_split_num, 0:width_split_num]  # iiがy軸、jjがx軸
    tile_index_batch = jnp.stack([jj, ii], axis=2)

    tile_depth_decending_indices_batch = _create_tile_depth_decending_indices_batch(
        gaussians,
        tile_index_batch,
        consts,
    )

    # タイルごとの左上のピクセル座標値を作成
    tile_upperleft_coord_batch = tile_index_batch * tile_size

    return (
        tile_depth_decending_indices_batch,
        tile_upperleft_coord_batch,
    )


depth_sorted_indices_vmap = jax.vmap(_depth_sorted_indices, in_axes=(0, None, None, None))
