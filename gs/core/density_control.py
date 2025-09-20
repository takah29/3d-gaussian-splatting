from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from scipy.special import expit, logit

from gs.core.projection import compute_cov_vmap


def prune_gaussians(
    raw_params: dict[str, npt.NDArray], contribution_scores_acc: npt.NDArray, consts: dict[str, Any]
) -> tuple[dict[str, npt.NDArray], npt.NDArray, int]:
    pruning_indices = (expit(raw_params["opacities"]) < consts["eps_prune_alpha"]).ravel()

    # 空などを表現する巨大なガウシアンが必要ない場合に有効
    if consts["pruning_big_gaussian"]:
        print("pruning big gaussians...")
        scale_prune_indices = np.exp(raw_params["scales"].max(axis=1)) > consts["extent"] * 0.1
        pruning_indices = pruning_indices | scale_prune_indices

    pruned_params = {key: val[~pruning_indices] for key, val in raw_params.items()}
    pruned_contribution_scores = contribution_scores_acc[~pruning_indices]
    return pruned_params, pruned_contribution_scores, pruning_indices.sum()


def densify_gaussians(
    raw_params: dict[str, npt.NDArray],
    contribution_scores_acc: npt.NDArray,
    view_space_grads_mean_norm: npt.NDArray,
    consts: dict[str, Any],
) -> tuple[dict[str, npt.NDArray], npt.NDArray, int, int]:
    tau_pos = consts["tau_pos"]
    max_densification_num = consts["max_gaussians"] - raw_params["means3d"].shape[0]

    while True:
        target_indices = view_space_grads_mean_norm > tau_pos
        target_params = {key: val[target_indices] for key, val in raw_params.items()}
        target_contribution_scores = contribution_scores_acc[target_indices]

        max_scales = np.exp(target_params["scales"].max(axis=1))
        clone_indices = max_scales < consts["scale_threshold"] * consts["extent"]
        split_indices = max_scales >= consts["scale_threshold"] * consts["extent"]

        densification_num = clone_indices.sum() + split_indices.sum() * (consts["split_num"] - 1)
        if densification_num <= max_densification_num:
            print(f"Changed tau_pos to {tau_pos} to fit within the maximum number of Gaussians")
            break
        tau_pos *= 2.0

    clone_params, clone_contribution_scores, cloned_num = _clone_gaussians(
        target_params, target_contribution_scores, clone_indices
    )
    covs_3d = compute_cov_vmap(
        target_params["quats"] / np.linalg.norm(target_params["quats"], axis=-1, keepdims=True),
        np.exp(target_params["scales"]),  # type: ignore[arg-type]
    )
    split_params, split_contribution_scores, splited_num = _split_gaussians(
        target_params,
        target_contribution_scores,
        covs_3d,  # type: ignore[arg-type]
        split_indices,
        consts,
    )

    raw_params = {
        key: np.vstack((raw_params[key][~target_indices], clone_params[key], split_params[key]))
        for key in raw_params
    }
    contribution_scores_acc = np.hstack(
        (
            contribution_scores_acc[~target_indices],
            clone_contribution_scores,
            split_contribution_scores,
        )
    )

    return raw_params, contribution_scores_acc, cloned_num, splited_num


def _clone_gaussians(
    raw_params: dict[str, npt.NDArray],
    contribution_scores_acc: npt.NDArray,
    clone_indices: npt.NDArray,
) -> tuple[dict[str, npt.NDArray], npt.NDArray, int]:
    clone_params = {key: val[clone_indices] for key, val in raw_params.items()}
    clone_contribution_scores = contribution_scores_acc[clone_indices]

    # 以降で異なる勾配更新が起こり自然と分離するため同じパラメータのガウシアンを複製
    merged_params = {key: np.vstack((val, val)) for key, val in clone_params.items()}
    merged_contribution_scores = np.hstack((clone_contribution_scores, clone_contribution_scores))
    return merged_params, merged_contribution_scores, clone_indices.sum()


def _split_gaussians(
    raw_params: dict[str, npt.NDArray],
    contribution_scores_acc: npt.NDArray,
    covs_3d: npt.NDArray,
    split_indices: npt.NDArray,
    consts: dict[str, Any],
) -> tuple[dict[str, npt.NDArray], npt.NDArray, int]:
    split_params = {key: val[split_indices] for key, val in raw_params.items()}
    split_contribution_scores = contribution_scores_acc[split_indices]
    split_num = consts["split_num"]
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
            "sh_dc": split_params["sh_dc"],
            "sh_rest": split_params["sh_rest"],
            "opacities": split_params["opacities"],
        }
        for means_3d in split_means_3d_sampled
    )

    merged_params = {}
    for key in raw_params:
        values = [param_dict[key] for param_dict in split_params_tuple]  # type: ignore[index]
        merged_params[key] = np.vstack(values)
    merged_contribution_scores = np.hstack([split_contribution_scores] * split_num)
    return (
        merged_params,
        merged_contribution_scores,
        int(split_indices.sum() * (split_num - 1)),
    )  # type: ignore[return-value]


def split_gaussians_by_long_axis(
    raw_params: dict[str, npt.NDArray],
    contribution_scores_acc: npt.NDArray,
    view_space_grads_mean_norm: npt.NDArray,
    consts: dict[str, Any],
) -> tuple[dict[str, npt.NDArray], npt.NDArray, int]:
    """Long-Axis=splitによるガウシアンの分割

    ref: https://arxiv.org/pdf/2508.12313
    """
    tau_pos = consts["tau_pos"]
    max_densification_num = consts["max_gaussians"] - raw_params["means3d"].shape[0]

    while True:
        target_indices = view_space_grads_mean_norm > tau_pos
        split_num = target_indices.sum()

        if split_num <= max_densification_num:
            print(f"Changed tau_pos to {tau_pos} to fit within the maximum number of Gaussians")
            break

        tau_pos *= 2.0

    target_params = {key: val[target_indices] for key, val in raw_params.items()}
    target_contribution_scores = contribution_scores_acc[target_indices]

    max_scale_indices = np.argmax(target_params["scales"], axis=1)

    linear_scales = np.exp(target_params["scales"])
    max_scales = linear_scales[np.arange(split_num), max_scale_indices]
    d = 0.45 * max_scales
    new_scales_linear = 0.893 * linear_scales  # Rs = R₀ * sqrt(1 - 0.45²) ≈ 0.893 * R₀
    new_scales_linear[np.arange(split_num), max_scale_indices] = 0.55 * max_scales
    target_params["scales"] = np.log(new_scales_linear)
    target_params["opacities"] = logit(expit(target_params["opacities"]) * 0.6)

    local_directions = np.zeros((split_num, 3))
    local_directions[np.arange(split_num), max_scale_indices] = 1.0
    quat_norms = np.linalg.norm(target_params["quats"], axis=1, keepdims=True)
    rotation = Rotation.from_quat(target_params["quats"] / quat_norms)
    world_directions = rotation.apply(local_directions)
    world_directions /= np.linalg.norm(world_directions, axis=1, keepdims=True)

    offset_vecs = d[:, None] * world_directions
    child_positive_means = target_params["means3d"] + offset_vecs
    child_negative_means = target_params["means3d"] - offset_vecs

    child_positive_params = target_params.copy()
    child_positive_params["means3d"] = child_positive_means
    child_negative_params = target_params.copy()
    child_negative_params["means3d"] = child_negative_means

    raw_params = {
        key: np.vstack(
            (
                raw_params[key][~target_indices],
                child_positive_params[key],
                child_negative_params[key],
            )
        )
        for key in raw_params
    }
    contribution_scores_acc = np.hstack(
        (
            contribution_scores_acc[~target_indices],
            target_contribution_scores,
            target_contribution_scores,
        )
    )
    return raw_params, contribution_scores_acc, split_num


def _batch_sample_from_covariance(
    key: jax.Array, means: npt.NDArray, covariances: npt.NDArray, num_samples_per_batch: int = 1
) -> jax.Array:
    batch_size = means.shape[0]
    keys = jax.random.split(key, batch_size)

    def sample_single(key: jax.Array, mean: npt.NDArray, cov: npt.NDArray) -> jax.Array:
        return jax.random.multivariate_normal(key, mean, cov, shape=(num_samples_per_batch,))

    return jax.vmap(sample_single)(keys, means, covariances + jnp.eye(3) * 1e-6).transpose(1, 0, 2)


def prune_gaussians_by_contribution_scores(
    raw_params: dict[str, npt.NDArray],
    contribution_scores_acc: npt.NDArray,
    consts: dict[str, Any],
) -> tuple[dict[str, npt.NDArray], npt.NDArray]:
    non_zero_score_median = np.median(contribution_scores_acc[contribution_scores_acc > 0.0])
    pruning_mask = (
        contribution_scores_acc < consts["contribution_pruning_coeff"] * non_zero_score_median
    )
    raw_params = {key: val[~pruning_mask] for key, val in raw_params.items()}
    return raw_params, pruning_mask.sum()
