from pathlib import Path

import grain  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pycolmap
from PIL import Image


def load_colmap_data(base_path: Path) -> dict[str, npt.NDArray]:
    reconstruction = pycolmap.Reconstruction(str(base_path))
    points_3d = np.vstack([pt.xyz for pt in reconstruction.points3D.values()], dtype=np.float32)

    quat_batch = np.stack(
        [img.cam_from_world.rotation.quat for img in reconstruction.images.values()],
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
            reconstruction.cameras[img.camera.camera_id].params  # type: ignore[misc]
            for img in reconstruction.images.values()
        ],
        dtype=np.float32,
    )
    return {
        "points_3d": points_3d,  # (x, y, z) in points_3d
        "quat_batch": quat_batch,  # 3x3 rotation_matrix in rot_mat_batch
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


def create_image_dataloader(
    image_dir_parh: Path, max_res: int, num_epochs: int, *, shuffle: bool = True
) -> grain.DataLoader:
    image_path_list = [
        pth
        for pth in image_dir_parh.glob("*")
        if pth.suffix.lower() in (".jpg", ".jpeg", ".png", "gif", "bmp", "tiff")
    ]
    image_data_source = (
        grain.MapDataset.source(image_path_list)
        .shuffle(seed=42)
        .map(lambda path: _load_image_and_fit(path, max_res))
    )
    index_sampler = grain.samplers.IndexSampler(
        num_records=len(image_path_list),
        shuffle=shuffle,
        num_epochs=num_epochs,
        seed=123,
    )
    return grain.DataLoader(data_source=image_data_source, sampler=index_sampler)  # type: ignore[arg-type]
