import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

SH_C0_0 = 0.28209479177387814
SH_C1_0 = -0.4886025119029199
SH_C1_1 = 0.4886025119029199
SH_C1_2 = -0.4886025119029199
SH_C2_0 = 1.0925484305920792
SH_C2_1 = -1.0925484305920792
SH_C2_2 = 0.31539156525252005
SH_C2_3 = -1.0925484305920792
SH_C2_4 = 0.5462742152960396
SH_C3_0 = -0.5900435899266435
SH_C3_1 = 2.890611442640554
SH_C3_2 = -0.4570457994644658
SH_C3_3 = 0.3731763325901154
SH_C3_4 = -0.4570457994644658
SH_C3_5 = 1.445305721320277
SH_C3_6 = -0.5900435899266435


def _calc_sh_basis_function_values(direction: jax.Array) -> jax.Array:
    # Reference: https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu
    x, y, z = direction
    xy = x * y
    yz = y * z
    xz = z * x
    xx = x**2
    yy = y**2
    zz = z**2
    vals = jnp.array(
        [
            # degree 0
            SH_C0_0,
            # degree 1
            SH_C1_0 * y,
            SH_C1_1 * z,
            SH_C1_2 * x,
            # degree 2
            SH_C2_0 * xy,
            SH_C2_1 * yz,
            SH_C2_2 * (2.0 * zz - xx - yy),
            SH_C2_3 * xz,
            SH_C2_4 * (xx - yy),
            # degree 3
            SH_C3_0 * y * (3.0 * xx - yy),
            SH_C3_1 * xy * z,
            SH_C3_2 * y * (4.0 * zz - xx - yy),
            SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            SH_C3_4 * x * (4.0 * zz - xx - yy),
            SH_C3_5 * z * (xx - yy),
            SH_C3_6 * x * (xx - 3.0 * yy),
        ]
    )

    return vals


calc_sh_basis_function_values_vmap = jax.vmap(_calc_sh_basis_function_values)


def _compute_sh_deg0(sh_coeff: jax.Array, basis_val: jax.Array) -> jax.Array:
    # sh_coeffs: (3, 16), basis: (16,)
    return sh_coeff[:, :1] @ basis_val[:1, None]


def _compute_sh_deg1(sh_coeff: jax.Array, basis_val: jax.Array) -> jax.Array:
    return sh_coeff[:, :4] @ basis_val[:4, None]


def _compute_sh_deg2(sh_coeff: jax.Array, basis_val: jax.Array) -> jax.Array:
    return sh_coeff[:, :9] @ basis_val[:9, None]


def _compute_sh_deg3(sh_coeff: jax.Array, basis_val: jax.Array) -> jax.Array:
    return sh_coeff[:, :16] @ basis_val[:16, None]


sh_computations_vmap = tuple(
    jax.vmap(f, in_axes=(0, 0))
    for f in (_compute_sh_deg0, _compute_sh_deg1, _compute_sh_deg2, _compute_sh_deg3)
)


def compute_color_from_sh_switch(
    points_3d: jax.Array,
    t_vec: jax.Array,
    sh_coeffs: jax.Array,  # (N, 3, 16)
    active_sh_degree: jax.Array,
) -> jax.Array:
    directions = points_3d - t_vec
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
    basis_values = calc_sh_basis_function_values_vmap(directions)  # (N, 16)

    colors = (
        jax.lax.switch(
            active_sh_degree,
            sh_computations_vmap,
            sh_coeffs,
            basis_values,
        ).squeeze(axis=-1)
        + 0.5
    )

    return jnp.maximum(0.0, colors)


def project_point(
    point_3d: jax.Array, rot_mat: jax.Array, t_vec: jax.Array, intrinsic_vec: jax.Array
) -> tuple[jax.Array, jax.Array]:
    point_cam = rot_mat @ point_3d + t_vec

    x = point_cam[0] / point_cam[2]
    y = point_cam[1] / point_cam[2]

    u = intrinsic_vec[0] * x + intrinsic_vec[2]  # fx * x + cx
    v = intrinsic_vec[1] * y + intrinsic_vec[3]  # fy * y + cy

    return jnp.array([u, v]), point_cam[2]


def _quat_to_rot(quat: jax.Array) -> jax.Array:
    norm = jnp.linalg.norm(quat)
    quat = jnp.where(norm > 1e-8, quat / norm, quat)  # type: ignore[reportAssignmentType]
    return Rotation.from_quat(quat / norm).as_matrix()


def compute_cov(quat: jax.Array, scale: jax.Array) -> jax.Array:
    rot_mat = _quat_to_rot(quat)
    s_mat = jnp.diag(scale)
    prod_mat = rot_mat @ s_mat
    return prod_mat @ prod_mat.T


def to_2dcov(
    mean_3d: jax.Array,
    cov_3d: jax.Array,
    rot_mat: jax.Array,
    t_vec: jax.Array,
    intrinsic_vec: jax.Array,
    img_shape: jax.Array,
) -> jax.Array:
    fx, fy, _, _ = intrinsic_vec[0], intrinsic_vec[1], intrinsic_vec[2], intrinsic_vec[3]
    height, width = img_shape[0], img_shape[1]

    mean_cam = rot_mat @ mean_3d + t_vec

    tan_fovx = 2.0 * jnp.arctan(width / (2.0 * fx))
    tan_fovy = 2.0 * jnp.arctan(height / (2.0 * fy))

    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    z = mean_cam[2]
    x_clipped = jnp.clip(mean_cam[0] / mean_cam[2], -limx, limx) * z
    y_clipped = jnp.clip(mean_cam[1] / mean_cam[2], -limy, limy) * z

    z2 = z * z
    jacobian = jnp.array(
        [
            [fx / z, 0, -fx * x_clipped / (z2)],
            [0, fy / z, -fy * y_clipped / (z2)],
        ]
    )

    prod_mat = jacobian @ rot_mat
    return prod_mat @ cov_3d @ prod_mat.T


def project(
    params: dict[str, jax.Array],
    rot_mat: jax.Array,
    t_vec: jax.Array,
    intrinsic_vec: jax.Array,
    consts: dict[str, jax.Array],
    active_sh_degree: jax.Array,
) -> dict[str, jax.Array]:
    # 球面調和関数による色計算
    colors = compute_color_from_sh_switch(
        params["means3d"],
        t_vec,
        params["sh_coeffs"],
        active_sh_degree,
    )

    # 3D Gaussianの中心点を2D画面に投影するときの座標値を計算
    projected_points, depths = project_point_vmap(params["means3d"], rot_mat, t_vec, intrinsic_vec)

    # 3D Gaussianの3D共分散を計算
    covs = compute_cov_vmap(params["quats"], params["scales"])

    # 3D Gaussianの3D共分散を2D画面に投影したときの2D共分散を計算
    covs_2d_raw = to_2dcov_vmap(
        params["means3d"], covs, rot_mat, t_vec, intrinsic_vec, consts["img_shape"]
    )

    # 特異行列対策
    covs_2d = covs_2d_raw + jnp.eye(2) * 0.3

    return {
        "means_2d": projected_points,
        "covs_2d": covs_2d,
        "colors": colors,
        "opacities": params["opacities"],
        "depths": depths,
    }


project_point_vmap = jax.vmap(project_point, in_axes=(0, None, None, None))
compute_cov_vmap = jax.vmap(compute_cov, in_axes=(0, 0))
to_2dcov_vmap = jax.vmap(to_2dcov, in_axes=(0, 0, None, None, None, None))
