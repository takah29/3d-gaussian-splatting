import argparse
from pathlib import Path

import fastplotlib as fpl
import jax.numpy as jnp

from gs.projection import make_projection_matrix_vmap, project_vmap
from gs.utils import load_colmap_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", type=Path)
    args = parser.parse_args()

    data = load_colmap_data(args.base_path)

    figure = fpl.Figure(cameras="2d", size=(1600, 1200), names=["projection"])
    projection_mat = make_projection_matrix_vmap(
        data["rot_mat_batch"], data["t_vec_batch"], data["intrinsic_batch"]
    )

    projected = project_vmap(jnp.asarray(data["points_3d"]), projection_mat[0])
    figure["projection"].add_scatter(
        data=projected, name="points_3d", cmap="viridis_r", alpha=0.5, sizes=2
    )

    figure["projection"].axes.visible = False

    figure.show()
    fpl.loop.run()


if __name__ == "__main__":
    main()
