import argparse
from pathlib import Path

from gs.utils import create_image_dataloader, load_colmap_data


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

    data = load_colmap_data(args.colmap_data_path / "sparse" / "0")
    image_dataset = create_image_dataloader(
        args.colmap_data_path / "images", args.max_res, args.n_epochs
    )

    # 読み込んだデータの情報を整形して表示
    print("===== Data Information =====")
    print(f"{data['points_3d'].shape=}")
    print(f"{data['quat_batch'].shape=}")
    print(f"{data['t_vec_batch'].shape=}")
    print(f"{data['intrinsic_batch'].shape=}")
    print("==============================")


if __name__ == "__main__":
    main()
