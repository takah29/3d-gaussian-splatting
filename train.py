import argparse
import pickle
from pathlib import Path

import jax
import numpy as np
import optax

from gs.density_control import control_density
from gs.make_update import DataLogger, make_updater
from gs.utils import build_params


def get_optimizer(optimizer_class, lr_scale):
    # パラメータを分類するための関数
    def partition_params(params):
        """パラメータを異なるグループに分類"""
        partition = {key: key for key in params}

        return partition

    # 各グループに異なるオプティマイザーを定義
    optimizers = {
        "means3d": optimizer_class(learning_rate=0.001 * lr_scale),
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
        optax.clip(0.01),
        get_optimizer(optax.adam, 1.0),
    )
    opt_state = optimizer.init(params)

    logger = DataLogger(Path(__file__).parent / "progress")

    update = make_updater(consts, optimizer, logger, jit=True)

    pos_grads_list = []
    for i, (view, target) in enumerate(image_dataloader, start=1):
        params, grads, opt_state, loss = update(params, view, target, opt_state)
        print(f"Iter {i}: loss={loss}")

        pos_grads_list.append(np.array(grads["means3d"]))

        if i % consts["densification_interval"] == 0:
            params, pruned_num, cloned_num, splitted_num = control_density(
                params, np.stack(pos_grads_list), consts, view
            )

            print("===== Densification ======")
            print(
                f"pruned_num: {pruned_num}, cloned_num: {cloned_num}, splitted_num: {splitted_num}"
            )
            delta_num = cloned_num + splitted_num - pruned_num
            print(
                f"num of gaussian: {params['means3d'].shape[0] - delta_num} -> {params['means3d'].shape[0]}"
            )
            print("========================")

            opt_state = optimizer.init(params)
            pos_grads_list.clear()

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
