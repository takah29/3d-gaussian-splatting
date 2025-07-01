import jax
import jax.numpy as jnp

from gs.projection import (
    compute_cov_vmap,
    project_point_vmap,
    to_2dcov_vmap,
)


def control_density(params, pos_grads_batch, consts, view):
    # alpha値が低いガウシアンの消去
    params, pos_grads_batch, pruned_num = prune_gaussians(params, pos_grads_batch, consts)

    avg_magnitude = compute_avg_magnitude(pos_grads_batch, consts, view)
    target_indices = avg_magnitude > consts["tau_pos"]

    target_params = jax.tree.map(lambda x: x[target_indices], params)
    target_pos_grads_batch = pos_grads_batch[:, target_indices]

    # view space covarianceの計算
    covs_3d = compute_cov_vmap(target_params["quats"], target_params["scales"])
    covs_2d = to_2dcov_vmap(
        target_params["means3d"], covs_3d, view["rot_mat"], view["t_vec"], view["intrinsic_vec"]
    )
    max_eigvals = jnp.linalg.eigvalsh(covs_2d)[:, -1]

    clone_params, cloned_num = clone_gaussians(
        target_params, target_pos_grads_batch, max_eigvals, consts
    )
    split_params, splited_num = split_gaussians(
        target_params, target_pos_grads_batch, covs_3d, max_eigvals, consts
    )

    new_params = jax.tree.map(
        lambda original, cloned, splitted: jnp.vstack(
            (original[~target_indices], cloned, splitted)
        ),
        params,
        clone_params,
        split_params,
    )

    return new_params, pruned_num, cloned_num, splited_num


def prune_gaussians(params, pos_grads_batch, consts):
    prune_indices = jax.nn.sigmoid(params["opacities"]) < consts["eps_alpha"]
    pruned_params = jax.tree.map(lambda x: x[~prune_indices[:, 0]], params)
    pos_grads_batch = pos_grads_batch[:, ~prune_indices[:, 0]]
    return pruned_params, pos_grads_batch, prune_indices.sum()


def clone_gaussians(params, pos_grads_batch, max_eigvals, consts):
    clone_indices = max_eigvals < consts["eps_eigval"]
    clone_params = jax.tree.map(lambda x: x[clone_indices], params)

    # 公式実装では同じ位置でクローンしたあとに片方のガウシアンだけ勾配更新を適用してずらしているが、
    # ここではすでに勾配更新適用済みなので、クローンしたあと片方のガウシアンだけ逆勾配でもとの位置に戻す
    clone_params["means3d"] = clone_params["means3d"] - pos_grads_batch[-1][clone_indices]
    merged_params = jax.tree.map(
        lambda original, cloned: jnp.vstack((original[clone_indices], cloned)),
        params,
        clone_params,
    )

    return merged_params, clone_indices.sum()


def split_gaussians(params, pos_grads_batch, covs_3d, max_eigvals, consts):
    split_indices = max_eigvals >= consts["eps_eigval"]
    split_params = jax.tree.map(lambda x: x[split_indices], params)

    key = jax.random.PRNGKey(0)
    split_means_3d_sampled = batch_sample_from_covariance(
        key,
        split_params["means3d"],
        covs_3d[split_indices],
        num_samples_per_batch=consts["split_num"],
    )
    split_params_tuple = tuple(
        {
            "means3d": means_3d,
            "scales": jnp.log(  # 実際のstdに変換して処理して戻す
                jnp.exp(split_params["scales"])
                / (consts["split_gaussian_scale"] * consts["split_num"])
            ),
            "quats": split_params["quats"],
            "colors": split_params["colors"],
            "opacities": split_params["opacities"],
        }
        for means_3d in split_means_3d_sampled
    )

    merged_params = jax.tree.map(lambda *v: jnp.vstack(v), *split_params_tuple)

    return merged_params, split_indices.sum() * (consts["split_num"] - 1)


def compute_avg_magnitude(pos_grads_batch, consts, view):
    view_space_mean_grads, _ = project_point_vmap(
        pos_grads_batch.mean(axis=0), view["rot_mat"], view["t_vec"], view["intrinsic_vec"]
    )
    return jnp.linalg.norm(view_space_mean_grads - view["intrinsic_vec"][2:], axis=1)


def batch_sample_from_covariance(key, means, covariances, num_samples_per_batch=1):
    batch_size = means.shape[0]
    keys = jax.random.split(key, batch_size)

    def sample_single(key, mean, cov):
        return jax.random.multivariate_normal(key, mean, cov, shape=(num_samples_per_batch,))

    return jax.vmap(sample_single)(keys, means, covariances + jnp.eye(3) * 1e-6).transpose(1, 0, 2)
