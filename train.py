import argparse
from pathlib import Path

import grain
import numpy as np
import numpy.typing as npt

from gs.utils import compute_nearest_mean_distances, create_view_dataloader, load_colmap_data


def build_params(
    colmap_data_path: Path, max_res: int, n_epochs: int
) -> tuple[dict[str, npt.NDArray], grain.DataLoader]:
    reconstruction_data = load_colmap_data(colmap_data_path / "sparse" / "0")

    # 読み込んだデータの情報を整形して表示
    print("===== Data Information =====")
    for k, v in reconstruction_data.items():
        print(f"{k}: {v.shape}")
    print("============================")

    view_dataloader = create_view_dataloader(
        colmap_data_path / "images", reconstruction_data, max_res, n_epochs
    )

    # ガウシアンの初期プロパティを設定
    num_points = reconstruction_data["points_3d"].shape[0]

    # 近傍の3点の平均距離で設定
    nearest_mean_distances = compute_nearest_mean_distances(reconstruction_data["points_3d"])
    scales = np.log(nearest_mean_distances)[:, np.newaxis].repeat(3, axis=1)

    # 回転なし
    quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (num_points, 1))

    # sigmoidを適用して0.1となるように設定
    alpha = 0.1
    val = np.log(alpha / (1 - alpha))
    opacities = np.full((num_points, 1), val)

    params = {
        "means3d": reconstruction_data["points_3d"],
        "colors": reconstruction_data["colors"],
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
    }

    return params, view_dataloader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "colmap_data_path",
        type=Path,
        help="path to the colmap dataset (must contain 'images' and 'sparse' directories)",
    )
    parser.add_argument("-n", "--n_means", type=int, default=200000, help="number of gaussians")
    parser.add_argument("-e", "--n_epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("-r", "--max_res", type=int, default=1000, help="image fit size")
    args = parser.parse_args()

    params, view_dataset = build_params(args.colmap_data_path, args.max_res, args.n_epochs)


if __name__ == "__main__":
    main()
