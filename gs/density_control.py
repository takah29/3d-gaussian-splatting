import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit

from gs.projection import compute_cov_vmap


def prune_gaussians(params, consts):
    prune_indices = (expit(params["opacities"]) < consts["eps_prune_alpha"]).ravel()

    # 空などを表現する巨大なガウシアンが必要ない場合に有効
    if consts["pruning_big_gaussian"]:
        print("pruning big gaussians...")
        scale_prune_indices = np.exp(params["scales"].max(axis=1)) > consts["extent"] * 0.1
        prune_indices = prune_indices | scale_prune_indices

    pruned_params = {key: val[~prune_indices] for key, val in params.items()}
    return pruned_params, prune_indices.sum()


def densify_gaussians(params, view_space_grads_mean_norm, consts):
    tau_pos = consts["tau_pos"]
    max_densification_num = consts["max_points"] - params["means3d"].shape[0]

    while True:
        target_indices = view_space_grads_mean_norm > tau_pos
        target_params = {key: val[target_indices] for key, val in params.items()}

        max_scales = np.exp(target_params["scales"].max(axis=1))
        clone_indices = max_scales < consts["scale_threshold"] * consts["extent"]
        split_indices = max_scales >= consts["scale_threshold"] * consts["extent"]
        split_num = consts["split_num"]

        densification_num = clone_indices.sum() + split_indices.sum() * (split_num - 1)
        if densification_num <= max_densification_num:
            print(f"Changed tau_pos to {tau_pos} to fit within the maximum number of Gaussians")
            break
        tau_pos *= 2.0

    clone_params, cloned_num = clone_gaussians(target_params, clone_indices)
    covs_3d = compute_cov_vmap(
        target_params["quats"] / np.linalg.norm(target_params["quats"], axis=-1, keepdims=True),
        np.exp(target_params["scales"]),
    )
    split_params, splited_num = split_gaussians(
        target_params, covs_3d, split_indices, consts, split_num
    )

    params = {
        key: np.vstack((params[key][~target_indices], clone_params[key], split_params[key]))
        for key in params
    }

    return params, cloned_num, splited_num


def clone_gaussians(params, clone_indices):
    clone_params = {key: val[clone_indices] for key, val in params.items()}

    # 以降で異なる勾配更新が起こり自然と分離するため同じパラメータのガウシアンを複製
    merged_params = {key: np.vstack((val, val)) for key, val in clone_params.items()}

    return merged_params, clone_indices.sum()


def split_gaussians(params, covs_3d, split_indices, consts, split_num):
    split_params = {key: val[split_indices] for key, val in params.items()}

    key = jax.random.PRNGKey(0)
    split_means_3d_sampled = _batch_sample_from_covariance(
        key,
        split_params["means3d"],
        covs_3d[split_indices],
        num_samples_per_batch=split_num,
    )
    split_params_tuple = tuple(
        {
            "means3d": means_3d,
            "scales": np.log(  # 実際のstdに変換して処理して戻す
                np.exp(split_params["scales"]) / (consts["split_gaussian_scale"] * split_num)
            ),
            "quats": split_params["quats"],
            "colors": split_params["colors"],
            "opacities": split_params["opacities"],
        }
        for means_3d in split_means_3d_sampled
    )

    merged_params = {}
    for key in params:
        values = [param_dict[key] for param_dict in split_params_tuple]
        merged_params[key] = np.vstack(values)

    return merged_params, split_indices.sum() * (split_num - 1)


def _batch_sample_from_covariance(key, means, covariances, num_samples_per_batch=1):
    batch_size = means.shape[0]
    keys = jax.random.split(key, batch_size)

    def sample_single(key, mean, cov):
        return jax.random.multivariate_normal(key, mean, cov, shape=(num_samples_per_batch,))

    return jax.vmap(sample_single)(keys, means, covariances + jnp.eye(3) * 1e-6).transpose(1, 0, 2)
