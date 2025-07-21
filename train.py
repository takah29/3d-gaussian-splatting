import argparse
import pickle
from pathlib import Path
from typing import Any

import jax
import numpy as np
import numpy.typing as npt
import optax  # type: ignore[import-untyped]

from gs.config import GsConfig
from gs.core.density_control import densify_gaussians, prune_gaussians
from gs.function_factory import get_corrected_params, make_updater
from gs.utils import build_params


def get_optimizer(optimizer_class, lr_scale: float, extent: float, total_iter: int):  # noqa: ANN001, ANN201
    param_labels = {
        "means3d": "means3d",
        "quats": "quats",
        "scales": "scales",
        "sh_dc": "sh_dc",
        "sh_rest": "sh_rest",
        "opacities": "opacities",
    }

    position_lr_scheduler = optax.exponential_decay(
        init_value=1e-4 * extent * lr_scale,
        transition_steps=total_iter,
        decay_rate=0.01,
        end_value=1e-6 * extent * lr_scale,
    )

    optimizers = {
        "means3d": optimizer_class(learning_rate=position_lr_scheduler),
        "sh_dc": optimizer_class(learning_rate=0.001 * lr_scale),
        "sh_rest": optimizer_class(learning_rate=0.001 / 20.0 * lr_scale),
        "scales": optimizer_class(learning_rate=0.005 * lr_scale),
        "quats": optimizer_class(learning_rate=0.001 * lr_scale),
        "opacities": optimizer_class(learning_rate=0.05 * lr_scale),
    }

    optimizer = optax.multi_transform(optimizers, param_labels)  # type: ignore[reportArgumentType]

    return optimizer


def to_numpy_dict(arr_dict: dict[str, jax.Array]) -> dict[str, np.ndarray]:
    return {key: np.array(val) for key, val in arr_dict.items()}


def save_params_pkl(
    save_pkl_path: Path,
    params: dict[str, npt.NDArray],
    camera_params: dict[str, npt.NDArray],
    consts: dict[str, Any],
) -> None:
    result = {
        "params": get_corrected_params(params),  # type: ignore[arg-type]
        "consts": consts,
        "camera_params": camera_params,
    }
    result = {key: to_numpy_dict(val) for key, val in result.items()}

    with save_pkl_path.open("wb") as f:
        pickle.dump(result, f)


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "colmap_data_path",
        type=Path,
        help="path to the colmap dataset (must contain 'images' and 'sparse' directories)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=(Path(__file__).parent / "output").resolve(),
        help="output directory",
    )
    parser.add_argument("-e", "--n_epochs", type=int, default=1, help="number of epochs")
    parser.add_argument(
        "-c",
        "--config_filepath",
        type=Path,
        default=(Path(__file__).parent / "config" / "default.json").resolve(),
        help="path to the config file",
    )
    parser.add_argument("--checkpoint_cycle", type=int, default=500, help="checkpoint cycle")
    parser.add_argument("--image_scale", type=float, default=1.0, help="image scale")
    args = parser.parse_args()

    gs_config = GsConfig.from_json_file(args.config_filepath)
    params, image_dataloader = build_params(
        args.colmap_data_path, gs_config, args.image_scale, args.n_epochs
    )
    gs_config.derive_additional_property(
        image_batch=image_dataloader.image_batch,
        camera_params=image_dataloader.camera_params,
    )
    gs_config.display()
    consts = gs_config.to_dict()

    # パラメータの保存先
    save_dir = args.output.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # 初期パラメータの保存
    save_params_pkl(
        save_dir / "params_checkpoint_initial.pkl",
        params,
        image_dataloader.camera_params,
        consts,
    )

    optimizer = get_optimizer(optax.adam, 1.0, gs_config.extent, len(image_dataloader))
    opt_state = optimizer.init(params)

    # logger = DataLogger(save_dirpath / "progress")

    update = make_updater(consts, optimizer, jit=True)

    view_space_grads_norm_acc = np.zeros(params["means3d"].shape[0], dtype=np.float32)
    update_count_arr = np.zeros(params["means3d"].shape[0], dtype=np.int32)
    active_sh_degree = 0

    for i, (view, target) in enumerate(image_dataloader, start=1):
        if i in (1000, 2000, 3000):
            active_sh_degree += 1
            print(f"Active SH degree increased to {active_sh_degree}")

        params, opt_state, loss, viewspace_grads = update(
            params, view, target, opt_state, active_sh_degree
        )
        print(f"Iter {i}: loss={loss}")

        # 途中経過のパラメータを保存
        if i % args.checkpoint_cycle == 0:
            save_params_pkl(
                save_dir / f"params_checkpoint_iter{i:05d}.pkl",
                params,
                image_dataloader.camera_params,
                consts,
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
                view_space_grads_norm_acc = np.array(view_space_grads_norm_acc)
                update_count_arr = np.array(update_count_arr)

                enable_mask = view_space_grads_norm_acc > 0.0
                viewspace_grads_mean_norm = np.zeros(params["means3d"].shape[0], dtype=np.float32)
                viewspace_grads_mean_norm[enable_mask] = (
                    view_space_grads_norm_acc[enable_mask] / update_count_arr[enable_mask]
                )

                params, cloned_num, splitted_num = densify_gaussians(
                    params, viewspace_grads_mean_norm, consts
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
        save_dir / "params_final.pkl",
        params,
        image_dataloader.camera_params,
        consts,
    )


if __name__ == "__main__":
    main()
