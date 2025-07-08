import argparse
from pathlib import Path

import fastplotlib as fpl
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from gs.utils import load_colmap_data


def plot_reconstruction(reconstruction) -> None:
    figure = fpl.Figure(
        cameras="3d", controller_types="orbit", size=(1600, 1200), names=["reconstruction"]
    )

    figure["reconstruction"].add_scatter(
        data=reconstruction["points_3d"],
        name="points_3d",
        cmap="viridis_r",
        alpha=0.5,
        sizes=2,
    )

    def make_line_collection(
        quat_batch: npt.NDArray, t_vec_batch: npt.NDArray
    ) -> list[npt.NDArray]:
        line_collection = []
        for quat, t_vec in zip(quat_batch, t_vec_batch, strict=True):
            rot_mat = Rotation.from_quat(quat).as_matrix()
            rot_mat_scaled = rot_mat.T * 0.3
            t_vec = -rot_mat.T @ t_vec
            line_collection.append(np.array([t_vec, t_vec + rot_mat_scaled[:, 0]]))
            line_collection.append(np.array([t_vec, t_vec + rot_mat_scaled[:, 1]]))
            line_collection.append(np.array([t_vec, t_vec + rot_mat_scaled[:, 2]]))
        return line_collection

    line_collection = make_line_collection(
        reconstruction["quat_batch"], reconstruction["t_vec_batch"]
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
        target=reconstruction["t_vec_batch"].mean(axis=0),
        up=[0, -1, 0],
        depth=10.0,
    )
    figure["reconstruction"].axes.visible = False
    fpl.loop.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", type=Path)
    args = parser.parse_args()

    points, camera_params, image_batch = load_colmap_data(args.base_path, 1.0, quatanion=True)

    # 読み込んだデータの情報を整形して表示
    print("===== Data Information =====")
    print(f"{points['points_3d'].shape=}")
    print(f"{camera_params['quat_batch'].shape=}")
    print(f"{camera_params['t_vec_batch'].shape=}")
    print(f"{camera_params['intrinsic_batch'].shape=}")
    print("==============================")

    plot_reconstruction({**points, **camera_params})


if __name__ == "__main__":
    main()
