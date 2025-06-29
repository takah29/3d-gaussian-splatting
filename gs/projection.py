import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


def make_projection_matrix(
    rot_mat: jax.Array, t_vec: jax.Array, intrinsic_vec: jax.Array
) -> jax.Array:
    intrinsic_mat = jnp.array(
        [
            [intrinsic_vec[0], 0.0, intrinsic_vec[2]],
            [0.0, intrinsic_vec[1], intrinsic_vec[3]],
            [0.0, 0.0, 1.0],
        ]
    )

    return intrinsic_mat @ jnp.hstack([rot_mat, t_vec[:, None]])


def project_point(point_3d: jax.Array, projection_mat: jax.Array) -> tuple[jax.Array, jax.Array]:
    x = projection_mat @ jnp.hstack([point_3d, jnp.ones(1)])
    x_proj = x / x[2]
    return x_proj[:2], x[2]


def quat_to_rot(quat: jax.Array) -> jax.Array:
    quat_ = jnp.hstack((quat[1:], quat[0]))
    norm = jnp.linalg.norm(quat_)
    quat_ = jnp.where(norm > 1e-8, quat_ / norm, quat_)
    return Rotation.from_quat(quat_ / norm).as_matrix()


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
    mean_cam = jnp.block([rot_mat, t_vec[:, None]]) @ jnp.hstack([mean_3d, jnp.ones(1)])
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
    projection_matrix = make_projection_matrix(rot_mat, t_vec, intrinsic_vec)
    projected_points, depths = project_point_vmap(means3d, projection_matrix)

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


project_point_vmap = jax.vmap(project_point, in_axes=(0, None))
compute_cov_vmap = jax.vmap(compute_cov, in_axes=(0, 0))
to_2dcov_vmap = jax.vmap(to_2dcov, in_axes=(0, 0, None, None, None))
