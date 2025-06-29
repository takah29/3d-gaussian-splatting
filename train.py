import argparse
from pathlib import Path

from gs.utils import build_params


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
