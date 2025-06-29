from pathlib import Path

import grain  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pycolmap
from PIL import Image
from scipy.spatial import cKDTree


def load_colmap_data(base_path: Path, quatanion: bool = False) -> dict[str, npt.NDArray]:
    reconstruction = pycolmap.Reconstruction(str(base_path))
    points_3d = np.vstack([pt.xyz for pt in reconstruction.points3D.values()], dtype=np.float32)
    colors = np.vstack([pt.color for pt in reconstruction.points3D.values()], dtype=np.float32)
    t_vec_batch = np.stack(
        [
            -img.cam_from_world.rotation.matrix().T @ img.cam_from_world.translation
            for img in reconstruction.images.values()
        ],
        dtype=np.float32,
    )

    intrinsic_batch = np.stack(
        [
            reconstruction.cameras[img.camera.camera_id].params  # type: ignore[misc]
            for img in reconstruction.images.values()
        ],
        dtype=np.float32,
    )

    if quatanion:
        quat_batch = np.stack(
            [img.cam_from_world.rotation.quat for img in reconstruction.images.values()],
            dtype=np.float32,
        )
        return {
            "points_3d": points_3d,  # (x, y, z) in points_3d
            "quat_batch": quat_batch,  # (qx, qy, qz, qw) in quat_batch
            "t_vec_batch": t_vec_batch,  # (tx, ty, tz) in t_vec_batch
            "intrinsic_batch": intrinsic_batch,  # (fx, fy, cx, cy) in intrinsic_batch
        }

    rot_mat_batch = np.stack(
        [img.cam_from_world.rotation.matrix().T for img in reconstruction.images.values()],
        dtype=np.float32,
    )
    return {
        "points_3d": points_3d,  # (x, y, z) in points_3d
        "colors": colors,  # (r, g, b) in colors
        "rot_mat_batch": rot_mat_batch,  # 3x3 rotation_matrix in rot_mat_batch
        "t_vec_batch": t_vec_batch,  # (tx, ty, tz) in t_vec_batch
        "intrinsic_batch": intrinsic_batch,  # (fx, fy, cx, cy) in intrinsic_batch
    }


def _load_image_and_fit(image_path: Path, max_res: int) -> npt.NDArray:
    img = Image.open(image_path)
    img_size = np.array(img.size)
    max_current = np.max(img_size)

    if max_current <= max_res:
        return np.asarray(img) / 255.0

    scale = max_res / max_current
    print(scale)
    resized = img.resize((img_size * scale).astype(np.uint32), Image.Resampling.LANCZOS)

    return np.asarray(resized) / 255.0


def create_view_dataloader(
    image_dir_parh: Path,
    reconstruction_data: dict[str, npt.NDArray],
    max_res: int,
    num_epochs: int,
    *,
    shuffle: bool = True,
) -> grain.DataLoader:
    target_image_path_list = [
        pth
        for pth in image_dir_parh.glob("*")
        if pth.suffix.lower() in (".jpg", ".jpeg", ".png", "gif", "bmp", "tiff")
    ]
    camera_param_list = [
        {"rot_mat": rot_mat, "t_vec": t_vec, "intrinsic": intrinsic}
        for rot_mat, t_vec, intrinsic in zip(
            reconstruction_data["rot_mat_batch"],
            reconstruction_data["t_vec_batch"],
            reconstruction_data["intrinsic_batch"],
            strict=True,
        )
    ]

    if len(target_image_path_list) != len(camera_param_list):
        msg = (
            f"The number of images ({len(target_image_path_list)}) "
            f"does not match the number of camera parameters ({len(camera_param_list)})."
        )
        raise ValueError(msg)

    data_source = (
        grain.MapDataset.source(list(zip(camera_param_list, target_image_path_list, strict=False)))
        .shuffle(seed=42)
        .map(lambda inputs: (inputs[0], _load_image_and_fit(inputs[1], max_res)))
    )
    index_sampler = grain.samplers.IndexSampler(
        num_records=len(target_image_path_list),
        shuffle=shuffle,
        num_epochs=num_epochs,
        seed=123,
    )
    return grain.DataLoader(data_source=data_source, sampler=index_sampler)  # type: ignore[arg-type]


def compute_nearest_mean_distances(points: npt.NDArray) -> npt.NDArray:
    if points.shape[0] == 0:
        return np.array([])

    tree = cKDTree(points)

    # 自分自信が含まれるためk=4を設定
    distances, _ = tree.query(points, k=4)
    nearest_3points_distances = distances[:, 1:]

    return np.mean(nearest_3points_distances, axis=1)
