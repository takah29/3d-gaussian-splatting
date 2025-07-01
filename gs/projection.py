import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


def project_point(
    point_3d: jax.Array, rot_mat: jax.Array, t_vec: jax.Array, intrinsic_vec: jax.Array
) -> tuple[jax.Array, jax.Array]:
    point_cam = rot_mat @ point_3d + t_vec

    x = point_cam[0] / point_cam[2]
    y = point_cam[1] / point_cam[2]

    u = intrinsic_vec[0] * x + intrinsic_vec[2]  # fx * x + cx
    v = intrinsic_vec[1] * y + intrinsic_vec[3]  # fy * y + cy

    return jnp.array([u, v]), point_cam[2]


def quat_to_rot(quat: jax.Array) -> jax.Array:
    norm = jnp.linalg.norm(quat)
    quat = jnp.where(norm > 1e-8, quat / norm, quat)
    return Rotation.from_quat(quat / norm).as_matrix()


def compute_cov(quat: jax.Array, scale: jax.Array) -> jax.Array:
    rot_mat = quat_to_rot(quat)
    s_mat = jnp.diag(jnp.exp(scale))
    prod_mat = rot_mat @ s_mat
    return prod_mat @ prod_mat.T


def to_2dcov(
    mean_3d: jax.Array,
    cov_3d: jax.Array,
    rot_mat: jax.Array,
    t_vec: jax.Array,
    intrinsic_vec: jax.Array,
) -> jax.Array:
    f = intrinsic_vec[:2]
    mean_cam = rot_mat @ mean_3d + t_vec
    jacobian = jnp.array(
        [
            [f[0] / mean_cam[2], 0, -f[0] * mean_cam[0] / (mean_cam[2] ** 2)],
            [0, f[1] / mean_cam[2], -f[1] * mean_cam[1] / (mean_cam[2] ** 2)],
        ]
    )
    prod_mat = jacobian @ rot_mat.T
    return prod_mat @ cov_3d @ prod_mat.T


def project(
    params: dict[str, jax.Array], rot_mat: jax.Array, t_vec: jax.Array, intrinsic_vec: jax.Array
) -> dict[str, jax.Array]:
    means3d = params["means3d"]
    quats = params["quats"] / (jnp.linalg.norm(params["quats"], axis=-1, keepdims=True))
    scales = params["scales"]
    colors = jax.nn.sigmoid(params["colors"])
    opacities = jax.nn.sigmoid(params["opacities"])

    # 3D Gaussianの中心点を2D画面に投影するときの座標値を計算
    projected_points, depths = project_point_vmap(means3d, rot_mat, t_vec, intrinsic_vec)

    # 3D Gaussianの3D共分散を計算
    covs = compute_cov_vmap(quats, scales)

    # 3D Gaussianの3D共分散を2D画面に投影したときの2D共分散を計算
    covs_2d = to_2dcov_vmap(means3d, covs, rot_mat, t_vec, intrinsic_vec)

    return {
        "means_2d": projected_points,
        "covs_2d": covs_2d,
        "colors": colors,
        "opacities": opacities,
        "depths": depths,
    }


project_point_vmap = jax.vmap(project_point, in_axes=(0, None, None, None))
compute_cov_vmap = jax.vmap(compute_cov, in_axes=(0, 0))
to_2dcov_vmap = jax.vmap(to_2dcov, in_axes=(0, 0, None, None, None))
