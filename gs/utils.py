from pathlib import Path

import numpy as np
import numpy.typing as npt
import pycolmap


def load_colmap_data(base_path: Path) -> dict[str, npt.NDArray]:
    reconstruction = pycolmap.Reconstruction(str(base_path))
    points_3d = np.vstack([pt.xyz for pt in reconstruction.points3D.values()], dtype=np.float32)

    rot_mat_batch = np.stack(
        [img.cam_from_world.rotation.matrix().T for img in reconstruction.images.values()],
        dtype=np.float32,
    )
    t_vec_batch = np.stack(
        [
            -img.cam_from_world.rotation.matrix().T @ img.cam_from_world.translation
            for img in reconstruction.images.values()
        ],
        dtype=np.float32,
    )

    intrinsic_batch = np.stack(
        [
            reconstruction.cameras[img.camera.camera_id].params
            for img in reconstruction.images.values()
        ],
        dtype=np.float32,
    )
    return {
        "points_3d": points_3d,  # (x, y, z) in points_3d
        "rot_mat_batch": rot_mat_batch,  # 3x3 rotation_matrix in rot_mat_batch
        "t_vec_batch": t_vec_batch,  # (tx, ty, tz) in t_vec_batch
        "intrinsic_batch": intrinsic_batch,  # (fx, fy, cx, cy) in intrinsic_batch
    }
