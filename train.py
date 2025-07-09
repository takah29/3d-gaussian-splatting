import argparse
import pickle
from pathlib import Path

import jax
import numpy as np
import optax

from gs.density_control import densify_gaussians, prune_gaussians
from gs.make_update import get_corrected_params, make_updater
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


def save_params_pkl(save_pkl_path: Path, params, camera_params, consts):
    result = {
        "params": get_corrected_params(params),
        "consts": consts,
        "camera_params": camera_params,
    }
    result = jax.tree.map(lambda x: np.array(x), result)

    with save_pkl_path.open("wb") as f:
        pickle.dump(result, f)


def to_numpy_dict(arr_dict):
    return {key: np.array(val) for key, val in arr_dict.items()}


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
    parser.add_argument("-s", "--checkpoint_cycle", type=int, default=500, help="checkpoint cycle")
    args = parser.parse_args()

    params, consts, image_dataloader = build_params(
        args.colmap_data_path, args.max_points, args.image_scale, args.n_epochs
    )

    optimizer = get_optimizer(optax.adam, 1.0, consts["extent"], len(image_dataloader))
    opt_state = optimizer.init(params)

    save_dirpath = Path(__file__).parent / "output"
    save_dirpath.mkdir(parents=True, exist_ok=True)
    # logger = DataLogger(save_dirpath / "progress")

    update = make_updater(consts, optimizer, jit=True)

    view_space_grads_norm_acc = np.zeros(params["means3d"].shape[0], dtype=np.float32)
    update_count_arr = np.zeros(params["means3d"].shape[0], dtype=np.int32)
    for i, (view, target) in enumerate(image_dataloader, start=1):
        params, grads, opt_state, loss, viewspace_grads = update(params, view, target, opt_state)
        print(f"Iter {i}: loss={loss}")

        # 途中経過のパラメータを保存
        if i % args.checkpoint_cycle == 0:
            save_params_pkl(
                save_dirpath / f"params_checkpoint_iter{i:05d}.pkl",
                *(to_numpy_dict(x) for x in (params, image_dataloader.camera_params, consts)),
            )

        if i <= consts["densify_until_iter"]:
            # 密度化に使用する勾配ノルムを加算
            view_space_grads_norm = np.linalg.norm(viewspace_grads, axis=1)
            view_space_grads_norm_acc += view_space_grads_norm
            update_count_arr += view_space_grads_norm > 0.0

        # ガウシアンの分割と除去
        if i >= consts["densify_from_iter"] and i % consts["densification_interval"] == 0:
            print("===== Densification and Pruning ======")
            # 配列の動的な処理を行うのでnumpy配列に変換
            params = to_numpy_dict(params)

            cloned_num, splitted_num = 0, 0
            if i <= consts["densify_until_iter"]:
                grads_means_3d = np.array(grads["means3d"])
                view_space_grads_norm_acc = np.array(view_space_grads_norm_acc)
                update_count_arr = np.array(update_count_arr)

                enable_mask = view_space_grads_norm_acc > 0.0
                viewspace_grads_mean_norm = np.zeros(params["means3d"].shape[0], dtype=np.float32)
                viewspace_grads_mean_norm[enable_mask] = (
                    view_space_grads_norm_acc[enable_mask] / update_count_arr[enable_mask]
                )

                params, cloned_num, splitted_num = densify_gaussians(
                    params, grads_means_3d, viewspace_grads_mean_norm, consts
                )

            # alpha値が低いガウシアンの消去
            params, pruned_num = prune_gaussians(params, consts)

            print(
                f"cloned_num: {cloned_num}, splitted_num: {splitted_num}, pruned_num: {pruned_num}"
            )
            delta_num = cloned_num + splitted_num - pruned_num
            print(
                f"num of gaussian: {params['means3d'].shape[0] - delta_num} "
                f"-> {params['means3d'].shape[0]}"
            )
            print("======================================")

            # 勾配ノルムの蓄積をリセット
            view_space_grads_norm_acc = np.zeros(params["means3d"].shape[0], dtype=np.float32)
            update_count_arr = np.zeros(params["means3d"].shape[0], dtype=np.int32)

            # paramsのデータ数が変わるのでoptimizerを初期化する
            opt_state = optimizer.init(params)

    save_params_pkl(
        save_dirpath / "params_final.pkl",
        *(to_numpy_dict(x) for x in (params, image_dataloader.camera_params, consts)),
    )


if __name__ == "__main__":
    main()
