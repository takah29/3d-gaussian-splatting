import jax
import jax.numpy as jnp

from gs.projection import compute_cov_vmap, to_2dcov_vmap
from gs.rasterization import analytical_max_eigenvalue


def prune_gaussians(params, consts):
    prune_indices = jax.nn.sigmoid(params["opacities"]) < consts["eps_alpha"]
    pruned_params = jax.tree.map(lambda x: x[~prune_indices[:, 0]], params)
    return pruned_params, prune_indices.sum()


def densify_gaussians(params, pos_grads, view_space_grads_mean_norm, consts, view):
    target_indices = view_space_grads_mean_norm > consts["tau_pos"]

    target_params = jax.tree.map(lambda x: x[target_indices], params)
    target_pos_grads = pos_grads[target_indices]

    # view space covarianceの計算
    covs_3d = compute_cov_vmap(target_params["quats"], target_params["scales"])
    covs_2d = to_2dcov_vmap(
        target_params["means3d"], covs_3d, view["rot_mat"], view["t_vec"], view["intrinsic_vec"]
    )
    max_eigvals = jax.vmap(analytical_max_eigenvalue)(covs_2d)

    clone_params, cloned_num = clone_gaussians(target_params, target_pos_grads, max_eigvals, consts)
    split_params, splited_num = split_gaussians(
        target_params, target_pos_grads, covs_3d, max_eigvals, consts
    )

    params = jax.tree.map(
        lambda original, cloned, splitted: jnp.vstack(
            (original[~target_indices], cloned, splitted)
        ),
        params,
        clone_params,
        split_params,
    )

    return params, cloned_num, splited_num


def clone_gaussians(params, pos_grads, max_eigvals, consts):
    clone_indices = max_eigvals < consts["eps_eigval"]
    clone_params = jax.tree.map(lambda x: x[clone_indices], params)

    # 公式実装では同じ位置でクローンしたあとに片方のガウシアンだけ勾配更新を適用してずらしているが、
    # ここではすでに勾配更新適用済みなので、クローンしたあと片方のガウシアンだけ逆勾配でもとの位置に戻す
    clone_params["means3d"] = clone_params["means3d"] - pos_grads[clone_indices]
    merged_params = jax.tree.map(
        lambda original, cloned: jnp.vstack((original[clone_indices], cloned)),
        params,
        clone_params,
    )

    return merged_params, clone_indices.sum()


def split_gaussians(params, pos_grads, covs_3d, max_eigvals, consts):
    split_indices = max_eigvals >= consts["eps_eigval"]
    split_params = jax.tree.map(lambda x: x[split_indices], params)

    key = jax.random.PRNGKey(0)
    split_means_3d_sampled = _batch_sample_from_covariance(
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


def _batch_sample_from_covariance(key, means, covariances, num_samples_per_batch=1):
    batch_size = means.shape[0]
    keys = jax.random.split(key, batch_size)

    def sample_single(key, mean, cov):
        return jax.random.multivariate_normal(key, mean, cov, shape=(num_samples_per_batch,))

    return jax.vmap(sample_single)(keys, means, covariances + jnp.eye(3) * 1e-6).transpose(1, 0, 2)
