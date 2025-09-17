import argparse
from pathlib import Path

import jax
import numpy as np
import optax  # type: ignore[import-untyped]

from gs.config import GsConfig
from gs.core.density_control import densify_gaussians, prune_gaussians
from gs.function_factory import make_render, make_updater
from gs.helper import build_params, get_optimizer, print_info
from gs.utils import DataLogger, get_corrected_params, save_params, to_numpy_dict


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
    raw_params, image_dataloader = build_params(
        args.colmap_data_path, gs_config, args.image_scale, args.n_epochs
    )
    print_info(raw_params)

    gs_config.derive_additional_property(
        image_batch=image_dataloader.image_batch,
        camera_params=image_dataloader.camera_params,
        n_gaussians=raw_params["means3d"].shape[0],
    )
    gs_config.display()
    consts = gs_config.to_dict()

    # パラメータの保存先
    save_dir = args.output.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # 初期パラメータの保存
    save_params(
        save_dir / "params_checkpoint_initial",
        raw_params,
        image_dataloader.camera_params,
        gs_config,
    )

    optimizer = get_optimizer(optax.adam, 1.0, gs_config.extent, len(image_dataloader))
    opt_state = optimizer.init(raw_params)

    logger = DataLogger(save_dir / "progress")

    update = make_updater(consts, optimizer, jit=True)
    render = make_render(consts, active_sh_degree=3, jit=True)
    view_space_grads_norm_acc = np.zeros(raw_params["means3d"].shape[0], dtype=np.float32)
    update_count_arr = np.zeros(raw_params["means3d"].shape[0], dtype=np.int32)
    active_sh_degree = 0
    drop_rate = 0.2
    key = jax.random.PRNGKey(1234)
    for i, (view, target) in enumerate(image_dataloader, start=1):
        if i in (1000, 2000, 3000):
            active_sh_degree += 1
            print(f"Active SH degree increased to {active_sh_degree}")

        key, subkey = jax.random.split(key)
        raw_params, opt_state, loss, viewspace_grads = update(
            raw_params,
            view,  # type: ignore[arg-type]
            target,  # type: ignore[arg-type]
            opt_state,
            active_sh_degree,
            drop_rate,
            subkey,
        )
        print(f"Iter {i}: loss={loss}")

        # 途中経過のパラメータを保存
        if i % args.checkpoint_cycle == 0:
            save_params(
                save_dir / f"params_checkpoint_iter{i:05d}",
                raw_params,  # type: ignore[arg-type]
                image_dataloader.camera_params,
                gs_config,
            )
            image = render(get_corrected_params(raw_params), view)  # type: ignore[arg-type]
            logger.save(image, f"output_iter{i:05d}.png")

        if i <= consts["densify_until_iter"]:
            # 密度化に使用する勾配ノルムを加算
            view_space_grads_norm = np.linalg.norm(np.asarray(viewspace_grads), axis=1)
            view_space_grads_norm_acc += view_space_grads_norm
            update_count_arr += view_space_grads_norm > 0.0

        # ガウシアンの分割と除去
        if i >= consts["densify_from_iter"] and i % consts["densification_interval"] == 0:
            print("===== Densification and Pruning ======")
            # 配列の動的な処理を行うのでnumpy配列に変換
            raw_params = to_numpy_dict(raw_params)  # type: ignore[arg-type]

            cloned_num, splitted_num = 0, 0
            if i <= consts["densify_until_iter"]:
                view_space_grads_norm_acc = np.array(view_space_grads_norm_acc)
                update_count_arr = np.array(update_count_arr)

                enable_mask = view_space_grads_norm_acc > 0.0
                viewspace_grads_mean_norm = np.zeros(
                    raw_params["means3d"].shape[0], dtype=np.float32
                )
                viewspace_grads_mean_norm[enable_mask] = (
                    view_space_grads_norm_acc[enable_mask] / update_count_arr[enable_mask]
                )

                raw_params, cloned_num, splitted_num = densify_gaussians(
                    raw_params, viewspace_grads_mean_norm, consts
                )

            # alpha値が低いガウシアンの消去
            raw_params, pruned_num = prune_gaussians(raw_params, consts)

            # タイルあたりのガウシアン数を更新
            gs_config.set_tile_max_gs_num(raw_params["means3d"].shape[0])

            print(
                f"cloned_num: {cloned_num}, splitted_num: {splitted_num}, pruned_num: {pruned_num}"
            )
            delta_num = cloned_num + splitted_num - pruned_num
            print(
                f"num of gaussian: {raw_params['means3d'].shape[0] - delta_num} "
                f"-> {raw_params['means3d'].shape[0]}"
            )
            print("======================================")

            # 勾配ノルムの蓄積をリセット
            view_space_grads_norm_acc = np.zeros(raw_params["means3d"].shape[0], dtype=np.float32)
            update_count_arr = np.zeros(raw_params["means3d"].shape[0], dtype=np.int32)

            # paramsのデータ数が変わるのでoptimizerを初期化する
            opt_state = optimizer.init(raw_params)

    save_params(
        save_dir / "params_final",
        raw_params,  # type: ignore[arg-type]
        image_dataloader.camera_params,
        gs_config,
    )


if __name__ == "__main__":
    main()
