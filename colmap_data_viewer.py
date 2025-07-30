import argparse
from pathlib import Path

import fastplotlib as fpl  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from gs.utils import load_colmap_data


def plot_reconstruction(
    points: dict[str, npt.NDArray], camera_params: dict[str, npt.NDArray]
) -> None:
    figure = fpl.Figure(
        cameras="3d",
        controller_types="orbit",  # type: ignore[reportArgumentType]
        size=(1600, 1200),
        names=["reconstruction"],
    )

    figure["reconstruction"].add_scatter(
        data=points["points_3d"],
        name="points_3d",
        colors=np.hstack((points["colors"], np.ones((len(points["colors"]), 1)))),
        sizes=2,
    )

    def make_line_collection(
        rot_mat_batch: npt.NDArray, t_vec_batch: npt.NDArray
    ) -> list[npt.NDArray]:
        line_collection = []
        for rot_mat_w2c, t_vec_w2c in zip(rot_mat_batch, t_vec_batch, strict=True):
            rot_mat_c2w = rot_mat_w2c.T
            t_vec_c2w = -rot_mat_c2w @ t_vec_w2c
            rot_mat_c2w *= 0.3
            line_collection.append(np.array([t_vec_c2w, t_vec_c2w + rot_mat_c2w[:, 0]]))
            line_collection.append(np.array([t_vec_c2w, t_vec_c2w + rot_mat_c2w[:, 1]]))
            line_collection.append(np.array([t_vec_c2w, t_vec_c2w + rot_mat_c2w[:, 2]]))
        return line_collection

    line_collection = make_line_collection(
        camera_params["rot_mat_batch"], camera_params["t_vec_batch"]
    )
    figure["reconstruction"].add_line_collection(
        line_collection,
        name="camera_base",
        colors=["red", "green", "blue"] * (len(line_collection) // 3),
    )

    figure.show()

    camera_state = {
        "position": np.array([0.0, -5.0, -5.0]),
        "depth_range": (0.01, 1000),
    }
    figure["reconstruction"].camera.set_state(camera_state)
    figure["reconstruction"].camera.show_pos(
        target=camera_params["t_vec_batch"].mean(axis=0),
        up=[0, -1, 0],
        depth=10.0,
    )
    figure["reconstruction"].axes.visible = False
    fpl.loop.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", type=Path)
    args = parser.parse_args()

    points, camera_params, _ = load_colmap_data(args.base_path, 1.0)

    # 読み込んだデータの情報を整形して表示
    print("===== Data Information =====")
    print(f"{points['points_3d'].shape=}")
    print(f"{camera_params['rot_mat_batch'].shape=}")
    print(f"{camera_params['t_vec_batch'].shape=}")
    print(f"{camera_params['intrinsic_batch'].shape=}")
    print("==============================")

    plot_reconstruction(points, camera_params)


if __name__ == "__main__":
    main()
