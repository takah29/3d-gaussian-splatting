import argparse
import pickle
from pathlib import Path

import jax
import numpy as np
import optax

from gs.make_update import DataLogger, make_updater
from gs.utils import build_params


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
        args.colmap_data_path, args.max_points, args.image_scale
    )

    optimizer = optax.chain(
        optax.clip(0.01),
        optax.adamw(0.01),
    )
    opt_state = optimizer.init(params)

    logger = DataLogger(Path(__file__).parent / "progress")

    update = make_updater(consts, optimizer, logger, jit=True)

    for epoch in range(args.n_epochs):
        for i, (view, target) in enumerate(image_dataloader):
            print(view, target)
            params, _, opt_state, loss = update(params, view, target, opt_state)
            print(f"Iter {i}: loss={loss}")

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
