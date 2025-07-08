import argparse
import pickle
from pathlib import Path

import jax
import numpy as np
import optax

from gs.density_control import densify_gaussians, prune_gaussians
from gs.make_update import DataLogger, make_updater
from gs.utils import build_params


def get_optimizer(optimizer_class, lr_scale, extent, total_iter):
    # パラメータを分類するための関数
    def partition_params(params):
        """パラメータを異なるグループに分類"""
        partition = {key: key for key in params}

        return partition

    position_lr_scheduler = optax.exponential_decay(
        init_value=1e-4 * extent * lr_scale,
        transition_steps=total_iter,
        decay_rate=0.01,
        end_value=1e-6 * extent * lr_scale,
    )

    # 各グループに異なるオプティマイザーを定義
    optimizers = {
        "means3d": optimizer_class(learning_rate=position_lr_scheduler),
        "colors": optimizer_class(learning_rate=0.001 * lr_scale),
        "scales": optimizer_class(learning_rate=0.005 * lr_scale),
        "quats": optimizer_class(learning_rate=0.001 * lr_scale),
        "opacities": optimizer_class(learning_rate=0.05 * lr_scale),
    }

    # multi_transformオプティマイザーを作成
    optimizer = optax.multi_transform(optimizers, partition_params)

    return optimizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "colmap_data_path",
        type=Path,
        help="path to the colmap dataset (must contain 'images' and 'sparse' directories)",
    )
    parser.add_argument("--max_points", type=int, default=200000, help="max of gaussians")
    parser.add_argument("--image_scale", type=float, default=1.0, help="image scale")
    parser.add_argument("-e", "--n_epochs", type=int, default=1, help="number of epochs")
    args = parser.parse_args()

    params, consts, image_dataloader = build_params(
        args.colmap_data_path, args.max_points, args.image_scale, args.n_epochs
    )

    optimizer = optax.chain(
        # optax.clip(0.01),
        get_optimizer(optax.adam, 1.0, consts["extent"], len(image_dataloader)),
    )
    opt_state = optimizer.init(params)

    logger = DataLogger(Path(__file__).parent / "progress")

    update = make_updater(consts, optimizer, logger, jit=True)

    view_space_grads_norm_acc = np.zeros(params["means3d"].shape[0], dtype=np.float32)
    update_count_arr = np.zeros(params["means3d"].shape[0], dtype=np.int32)

    for i, (view, target) in enumerate(image_dataloader, start=1):
        params, grads, opt_state, loss, viewspace_grads = update(params, view, target, opt_state)
        print(f"Iter {i}: loss={loss}")

        view_space_grads_norm = np.linalg.norm(viewspace_grads, axis=1)
        view_space_grads_norm_acc += view_space_grads_norm
        update_count_arr += view_space_grads_norm > 0.0

        # ガウシアンの分割と除去
        if i >= consts["densify_from_iter"] and i % consts["densification_interval"] == 0:
            # 配列の動的な処理を行うのでnumpy配列に変換
            params = {key: np.array(val) for key, val in params.items()}
            grads_means_3d = np.array(grads["means3d"])
            view_space_grads_norm_acc = np.array(view_space_grads_norm_acc)
            update_count_arr = np.array(update_count_arr)

            enable_mask = view_space_grads_norm_acc > 0.0
            viewspace_grads_mean_norm = np.zeros(params["means3d"].shape[0], dtype=np.float32)
            viewspace_grads_mean_norm[enable_mask] = (
                view_space_grads_norm_acc[enable_mask] / update_count_arr[enable_mask]
            )

            cloned_num, splitted_num = 0, 0
            print("===== Densification and Pruning ======")
            if i <= consts["densify_until_iter"]:
                params, cloned_num, splitted_num = densify_gaussians(
                    params, grads_means_3d, viewspace_grads_mean_norm, consts, view
                )
            # alpha値が低いガウシアンの消去
            params, pruned_num = prune_gaussians(params, consts)
            print(
                f"cloned_num: {cloned_num}, splitted_num: {splitted_num}, pruned_num: {pruned_num}"
            )
            delta_num = cloned_num + splitted_num - pruned_num
            print(
                f"num of gaussian: {params['means3d'].shape[0] - delta_num} -> {params['means3d'].shape[0]}"
            )
            print("======================================")

            opt_state = optimizer.init(params)
            view_space_grads_norm_acc = np.zeros(params["means3d"].shape[0], dtype=np.float32)
            update_count_arr = np.zeros(params["means3d"].shape[0], dtype=np.int32)

    result = {
        "params": params,
        "consts": consts,
        "target_image_dir_path": args.colmap_data_path / "images",
    }
    result = jax.tree.map(lambda x: np.array(x), result)

    output_path = Path(__file__).parent / "reconstructed.pkl"
    with output_path.open("wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
