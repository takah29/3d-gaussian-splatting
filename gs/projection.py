import jax
import jax.numpy as jnp


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

    return intrinsic_mat @ jnp.hstack([rot_mat.T, -rot_mat.T @ t_vec[:, None]])


def project(point_3d: jax.Array, projection_mat: jax.Array) -> jax.Array:
    x = projection_mat @ jnp.hstack([point_3d, jnp.ones(1)])
    x = x / x[2]
    return x[:2]


project_vmap = jax.vmap(project, in_axes=(0, None))
make_projection_matrix_vmap = jax.vmap(make_projection_matrix, in_axes=(0, 0, 0))
