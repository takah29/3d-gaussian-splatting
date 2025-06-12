import argparse
from pathlib import Path

import fastplotlib as fpl
import numpy as np

from utils import read_colmap_data


def plot_reconstruction(reconstruction):
    figure = fpl.Figure(cameras="3d", size=(1600, 1200), names=["reconstruction"])

    figure["reconstruction"].add_scatter(
        data=reconstruction["points_3d"], name="points_3d", cmap="viridis_r", alpha=0.5, sizes=2
    )

    def make_line_collection(rot_mat_batch, t_vec_batch):
        line_collection = []
        for rot_mat, t_vec in zip(rot_mat_batch, t_vec_batch, strict=True):
            # print(rot_vec)
            rot_mat_scaled = rot_mat * 0.3
            line_collection.append(np.array([t_vec, t_vec + rot_mat_scaled[:, 0]]))
            line_collection.append(np.array([t_vec, t_vec + rot_mat_scaled[:, 1]]))
            line_collection.append(np.array([t_vec, t_vec + rot_mat_scaled[:, 2]]))
        return line_collection

    line_collection = make_line_collection(
        reconstruction["rot_mat_batch"], reconstruction["t_vec_batch"]
    )
    figure["reconstruction"].add_line_collection(
        line_collection,
        name="camera_base",
        colors=["red", "green", "blue"] * (len(line_collection) // 3),
    )

    camera_state = {
        "position": np.array([0, 0, 0]),
        "scale": np.array([1.0, 1.0, 1.0]),
        "reference_up": np.array([0.0, -1.0, 0.0]),
        "fov": 50.0,
        # "zoom": 0.75,
        "maintain_aspect": True,
        "depth_range": (0.01, 1000),
    }

    figure["reconstruction"].camera.set_state(camera_state)
    figure["reconstruction"].axes.visible = False

    figure.show()
    fpl.loop.run()

    # try:
    #     input("ビジュアライゼーションを終了するには Enter キーを押してください...")
    # except KeyboardInterrupt:
    #     pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", type=Path)
    args = parser.parse_args()

    result = read_colmap_data(args.base_path)
    plot_reconstruction(result)


if __name__ == "__main__":
    main()
