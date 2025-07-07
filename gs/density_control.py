import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit

from gs.projection import compute_cov_vmap, to_2dcov_vmap


def prune_gaussians(params, consts):
    alpha_prune_indices = (expit(params["opacities"]) < consts["eps_prune_alpha"]).ravel()
    scale_prune_indices = np.exp(params["scales"].max(axis=1)) > consts["extent"] * 0.1
    prune_indices = alpha_prune_indices | scale_prune_indices
    pruned_params = {key: val[~prune_indices] for key, val in params.items()}
    return pruned_params, prune_indices.sum()


def densify_gaussians(params, pos_grads, view_space_grads_mean_norm, consts, view):
    target_indices = view_space_grads_mean_norm > consts["tau_pos"]

    target_params = {key: val[target_indices] for key, val in params.items()}
    target_pos_grads = pos_grads[target_indices]

    # view space covarianceの計算
    covs_3d = compute_cov_vmap(target_params["quats"], target_params["scales"])
    covs_2d = to_2dcov_vmap(
        target_params["means3d"], covs_3d, view["rot_mat"], view["t_vec"], view["intrinsic_vec"]
    )
    max_eigvals = np.linalg.eigvalsh(covs_2d).max(axis=1)

    clone_indices = max_eigvals < consts["eps_clone_eigval"]
    split_indices = max_eigvals >= consts["eps_clone_eigval"]
    split_num = consts["split_num"]

    # 分割後のガウシアン数が最大ガウシアン数を超えている場合、最大ガウシアン数に収まるようにランダムサンプルする
    max_densification_num = consts["max_points"] - params["means3d"].shape[0]
    densification_num = clone_indices.sum() + split_indices.sum() * (split_num - 1)
    if max_densification_num < densification_num:
        print(
            f"The Gaussians to be split are {densification_num}, "
            f"but the maximum allowed is {max_densification_num}, "
            "so we'll randomly sample to fit within the maximum limit."
        )
        rng = np.random.default_rng(455)
        ind_arr = np.array(range(max_eigvals.shape[0]))
        clone_indices = np.isin(
            ind_arr, rng.choice(np.where(clone_indices)[0], max_densification_num // 2)
        )
        split_indices = np.isin(
            ind_arr, rng.choice(np.where(split_indices)[0], max_densification_num // 2)
        )
        exclude_indices = ~clone_indices & ~split_indices
        exclude_numbers = np.where(target_indices)[0][exclude_indices]
        target_indices[exclude_numbers] = False
        # 分割対象のガウシアン数を増やすため、分割数を2に設定
        split_num = 2

    clone_params, cloned_num = clone_gaussians(
        target_params, target_pos_grads, clone_indices, consts
    )
    split_params, splited_num = split_gaussians(
        target_params, target_pos_grads, covs_3d, split_indices, consts, split_num
    )

    params = {
        key: np.vstack((params[key][~target_indices], clone_params[key], split_params[key]))
        for key in params
    }

    return params, cloned_num, splited_num


def clone_gaussians(params, pos_grads, clone_indices, consts):
    clone_params = {key: val[clone_indices] for key, val in params.items()}

    # 公式実装では同じ位置でクローンしたあとに片方のガウシアンだけ勾配更新を適用してずらしているが、
    # ここではすでに勾配更新適用済みなので、クローンしたあと片方のガウシアンだけ逆勾配でもとの位置に戻す
    clone_params["means3d"] = clone_params["means3d"] - pos_grads[clone_indices]
    merged_params = {
        key: np.vstack((params[key][clone_indices], clone_params[key])) for key in params
    }

    return merged_params, clone_indices.sum()


def split_gaussians(params, pos_grads, covs_3d, split_indices, consts, split_num):
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
