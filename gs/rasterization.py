from functools import partial

import jax
import jax.numpy as jnp

TILE_SIZE = 16  # タイル分割のサイズ（1次元）
MAX_TILE_INDEX_SIZE = 200  # タイルごとの最大ガウシアン登録数


def _inv_strict(mat2x2: jax.Array) -> jax.Array:
    determinant = mat2x2[0, 0] * mat2x2[1, 1] - mat2x2[0, 1] * mat2x2[1, 0]
    inv_mat2x2 = (
        jnp.array([[mat2x2[1, 1], -mat2x2[0, 1]], [-mat2x2[1, 0], mat2x2[0, 0]]]) / determinant
    )
    return inv_mat2x2


def _gaussian_weight(
    pixel_coord: jax.Array,
    gaussian_idx: jax.Array,
    mean_2d: jax.Array,
    cov_2d: jax.Array,
    opacity: jax.Array,
) -> jax.Array:
    def true_fun() -> jax.Array:
        cov_inv = _inv_strict(cov_2d + jnp.eye(2) * 0.3)
        delta = pixel_coord - mean_2d
        mahal_dist = delta @ cov_inv @ delta
        gaussian_weight = jnp.exp(-0.5 * mahal_dist) * opacity
        return gaussian_weight[0]

    return jax.lax.cond(gaussian_idx >= 0, true_fun, lambda: 0.0)


def _render_pixel(
    pixel_coord: jax.Array,
    depth_decending_indices: jax.Array,
    gaussians: dict[str, jax.Array],
    background: jax.Array,
) -> jax.Array:
    pixel_color = background.copy()
    tau = jnp.ones((1,))

    means_2d = gaussians["means_2d"][depth_decending_indices]
    covs_2d = gaussians["covs_2d"][depth_decending_indices]
    opacities = gaussians["opacities"][depth_decending_indices]
    colors = gaussians["colors"][depth_decending_indices]

    gaussian_weight_batch = jax.vmap(_gaussian_weight, in_axes=(None, 0, 0, 0, 0))(
        pixel_coord, depth_decending_indices, means_2d, covs_2d, opacities
    )

    @partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)  # type: ignore[reportPrivateImportUsage]
    def body_fun(
        carry: tuple[jax.Array, jax.Array], inputs: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        pixel_color, tau = carry
        gaussian_idx, gaussian_weight, color = inputs

        def true_fun(pixel_color: jax.Array, tau: jax.Array) -> tuple[jax.Array, jax.Array]:
            updated_pixel_color = pixel_color + color * gaussian_weight * tau
            updated_tau = tau * (1 - gaussian_weight)

            return updated_pixel_color, updated_tau

        updated_pixel_color, updated_tau = jax.lax.cond(
            gaussian_idx >= 0,
            true_fun,
            lambda pixel_color, tau: (pixel_color, tau),
            pixel_color,
            tau,
        )

        return (updated_pixel_color, updated_tau), None

    (pixel_color, _), _ = jax.lax.scan(
        body_fun,
        (pixel_color, tau),
        (depth_decending_indices, gaussian_weight_batch, colors),
    )
    return pixel_color


def rasterize_tile_data(
    depth_decending_indices: jax.Array,
    upperleft_coord: jax.Array,
    gaussians: dict[str, jax.Array],
    background: jax.Array,
) -> jax.Array:
    ii, jj = jnp.mgrid[0:TILE_SIZE, 0:TILE_SIZE]
    pixel_coords = jnp.stack([upperleft_coord[0] + jj + 0.5, upperleft_coord[1] + ii + 0.5], axis=2)

    image_buffer = jax.vmap(
        jax.vmap(_render_pixel, in_axes=(0, None, None, None)), in_axes=(0, None, None, None)
    )(pixel_coords, depth_decending_indices, gaussians, background)

    return image_buffer


rasterize_tile_data_vmap = jax.vmap(
    jax.vmap(rasterize_tile_data, in_axes=(0, 0, None, None)), in_axes=(0, 0, None, None)
)


def _create_tile_depth_map(
    gaussian_depth: jax.Array,
    gaussian_idx_interval: jax.Array,
    height_split_num: jax.Array | int,
    width_split_num: jax.Array | int,
) -> jax.Array:
    inf_depth_map = jnp.full((height_split_num, width_split_num), jnp.inf)

    start_w, start_h = gaussian_idx_interval[0, 0], gaussian_idx_interval[0, 1]  # order (x, y)
    end_w, end_h = gaussian_idx_interval[1, 0], gaussian_idx_interval[1, 1]

    # インデックスグリッドを作成
    h_indices = jnp.arange(inf_depth_map.shape[0])[:, None]
    w_indices = jnp.arange(inf_depth_map.shape[1])[None, :]

    # 更新範囲内かどうかのマスクを作成
    mask = (
        (h_indices >= start_h)
        & (h_indices <= end_h)
        & (w_indices >= start_w)
        & (w_indices <= end_w)
        & (gaussian_depth > 0.2)
    )

    # ガウシアンのデプスで更新
    return jnp.where(mask, gaussian_depth, inf_depth_map)


def _create_tile_depth_decending_indices_batch(
    gaussians: dict[str, jax.Array],
    height_split_num: jax.Array | int,
    width_split_num: jax.Array | int,
    index_size: int,
) -> jax.Array:
    # ガウシアンの所属するタイルのインデックスを計算
    gauss_max_eigvals = jnp.linalg.eigvalsh(gaussians["covs_2d"])[:, 1]
    r_batch = 3.0 * jnp.sqrt(gauss_max_eigvals)[:, None]
    gaussian_intervals = jnp.stack(
        (gaussians["means_2d"] - r_batch, gaussians["means_2d"] + r_batch), axis=1
    )
    gaussian_index_intervals = (gaussian_intervals // TILE_SIZE).astype(
        jnp.int32
    )  # [[x_low_idx, y_low_idx], [x_high_idx, y_high_idx]]

    tile_depth_maps = jax.vmap(_create_tile_depth_map, in_axes=(0, 0, None, None), out_axes=2)(
        gaussians["depths"], gaussian_index_intervals, height_split_num, width_split_num
    )
    tile_inverse_depth_topk_batch, tile_depth_topk_indices_batch = jax.lax.top_k(
        -tile_depth_maps, k=index_size
    )

    return jnp.where(tile_inverse_depth_topk_batch == -jnp.inf, -1, tile_depth_topk_indices_batch)


def build_tile_data(
    gaussians: dict[str, jax.Array], img_shape: jax.Array
) -> tuple[jax.Array, jax.Array]:
    height_split_num = (img_shape[0] + TILE_SIZE - 1) // TILE_SIZE
    width_split_num = (img_shape[1] + TILE_SIZE - 1) // TILE_SIZE

    tile_depth_decending_indices_batch = _create_tile_depth_decending_indices_batch(
        gaussians,
        height_split_num,
        width_split_num,
        index_size=MAX_TILE_INDEX_SIZE,
    )

    # タイルごとの左上の座標値を計算
    ii, jj = jnp.mgrid[0:height_split_num, 0:width_split_num]  # iiがy軸、jjがx軸
    tile_upperleft_coord_batch = jnp.stack([jj * TILE_SIZE, ii * TILE_SIZE], axis=2)

    return (
        tile_depth_decending_indices_batch,
        tile_upperleft_coord_batch,
    )


def rasterize(
    gaussians: dict[str, jax.Array], consts: dict[str, int | float | jax.Array]
) -> jax.Array:
    """projectで射影した2Dガウシアンをラスタライズする.

    Note:
      * 画像配列の次元は[H(y軸), W(x軸), C]で実装を統一する
      * 数値計算の座標は(x, y)で行っていることに注意
    """
    img_shape = consts["img_shape"]
    background = consts["background"]

    # タイルごとに分割
    tile_depth_decending_indices_batch, tile_upperleft_coord_batch = build_tile_data(
        gaussians, img_shape
    )

    image_buffer_batch = rasterize_tile_data_vmap(
        tile_depth_decending_indices_batch,
        tile_upperleft_coord_batch,
        gaussians,
        background,
    )

    # タイルごとのバッファを結合
    transposed = jnp.transpose(image_buffer_batch, (0, 2, 1, 3, 4))
    final_buffer = transposed.reshape(
        transposed.shape[0] * transposed.shape[1],
        transposed.shape[2] * transposed.shape[3],
        transposed.shape[4],
    )[: img_shape[0], : img_shape[1]]

    return final_buffer
