from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pycolmap
from PIL import Image
from scipy.spatial import cKDTree

from gs.config import GsConfig


def load_colmap_data(
    colmap_data_path: Path, image_scale: float, *, quatanion: bool = False
) -> tuple[dict[str, npt.NDArray], dict[str, npt.NDArray], npt.NDArray]:
    reconstruction = pycolmap.Reconstruction(str(colmap_data_path / "sparse" / "0"))
    points_3d = np.vstack([pt.xyz for pt in reconstruction.points3D.values()], dtype=np.float32)
    colors = (
        np.vstack([pt.color for pt in reconstruction.points3D.values()], dtype=np.float32) / 255.0
    )

    intrinsic_list = []
    image_list = []
    image_filenames = []
    for img in reconstruction.images.values():
        image_arr = _load_image_and_fit(colmap_data_path / "images" / img.name, image_scale)
        height, width = image_arr.shape[:2]
        width_scale = width / img.camera.width
        height_scale = height / img.camera.height
        intrinsic_vec = img.camera.params * np.array(
            [width_scale, height_scale, width_scale, height_scale]
        )

        intrinsic_list.append(intrinsic_vec)
        image_list.append(image_arr)
        image_filenames.append(img.name)

    filename_sort_indices = np.argsort(image_filenames)
    image_batch = np.stack(image_list, axis=0)[filename_sort_indices]
    intrinsic_batch = np.stack(intrinsic_list, axis=0)[filename_sort_indices]

    t_vec_batch = np.stack(
        [img.cam_from_world().translation for img in reconstruction.images.values()],
        dtype=np.float32,
    )[filename_sort_indices]

    if quatanion:
        quat_batch = np.stack(
            [img.cam_from_world().rotation.quat for img in reconstruction.images.values()],
            dtype=np.float32,
        )[filename_sort_indices]
        camera_params = {
            "quat_batch": quat_batch,  # (qx, qy, qz, qw) in quat_batch
            "t_vec_batch": t_vec_batch,  # (tx, ty, tz) in t_vec_batch
            "intrinsic_batch": intrinsic_batch,  # (fx, fy, cx, cy) in intrinsic_batch
        }
    else:
        rot_mat_batch = np.stack(
            [img.cam_from_world().rotation.matrix() for img in reconstruction.images.values()],
            dtype=np.float32,
        )[filename_sort_indices]
        camera_params = {
            "rot_mat_batch": rot_mat_batch,  # 3x3 rotation_matrix in rot_mat_batch
            "t_vec_batch": t_vec_batch,  # (tx, ty, tz) in t_vec_batch
            "intrinsic_batch": intrinsic_batch,  # (fx, fy, cx, cy) in intrinsic_batch
        }

    return (
        {
            "points_3d": points_3d,  # (x, y, z) in points_3d
            "colors": colors,  # (r, g, b) in colors
        },
        camera_params,
        image_batch,
    )


def _load_image_and_fit(image_path: Path, image_scale: float) -> npt.NDArray:
    img = Image.open(image_path)
    img_size = np.array(img.size)

    resized_image = img.resize((img_size * image_scale).astype(np.uint32), Image.Resampling.LANCZOS)

    return np.asarray(resized_image).astype(np.float32) / 255.0


class ImageViewDataLoader:
    def __init__(
        self,
        camera_params: dict[str, npt.NDArray],
        image_batch: npt.NDArray,
        n_epochs: int = 1,
        seed: int = 42,
        *,
        shuffle: bool = True,
    ) -> None:
        self.image_batch = image_batch
        self.camera_params = camera_params

        view_list = [
            {"rot_mat": rot_mat, "t_vec": t_vec, "intrinsic_vec": intrinsic}
            for rot_mat, t_vec, intrinsic in zip(
                camera_params["rot_mat_batch"],
                camera_params["t_vec_batch"],
                camera_params["intrinsic_batch"],
                strict=True,
            )
        ]
        self.data = list(zip(view_list, image_batch, strict=True))
        self.shuffle = shuffle
        self.n_epochs = n_epochs
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __iter__(self) -> Iterator[tuple[dict[str, npt.NDArray], npt.NDArray]]:
        for epoch in range(self.n_epochs):
            # エポックごとに新しいシードを設定
            epoch_rng = np.random.RandomState(self.seed + epoch)

            indices = np.arange(len(self.data))
            if self.shuffle:
                epoch_rng.shuffle(indices)

            for idx in indices:
                yield self.data[idx]

    def __len__(self) -> int:
        return len(self.data) * self.n_epochs


def compute_nearest_distances(points: npt.NDArray) -> npt.NDArray:
    if points.shape[0] == 0:
        return np.array([])

    tree = cKDTree(points)

    distances, _ = tree.query(points, k=2)
    nearest_point_distances = distances[:, 1:].ravel()  # type: ignore[index]

    return nearest_point_distances


def save_params(
    save_path: Path,
    raw_params: dict[str, Any],
    camera_params: dict[str, Any],
    gs_config: GsConfig,
) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path / "params", allow_pickle=False, **to_numpy_dict(get_corrected_params(raw_params))
    )
    np.savez_compressed(
        save_path / "camera_params", allow_pickle=False, **to_numpy_dict(camera_params)
    )
    gs_config.to_json_file(save_path / "config.json")


def to_numpy_dict(arr_dict: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.array(val) for key, val in arr_dict.items()}


class DataLogger:
    def __init__(self, save_path: Path) -> None:
        self.count = 0
        self.save_path = save_path

        self.save_path.mkdir(parents=True, exist_ok=True)

    def __call__(self, image: jax.Array) -> None:
        plt.imsave(self.save_path / f"output_{self.count:05d}.png", image.clip(0, 1))
        self.count += 1

    def save(self, image: jax.Array, filename: str) -> None:
        """画像を保存する"""
        plt.imsave(self.save_path / filename, image.clip(0, 1))


def get_corrected_params(raw_params: dict[str, Any]) -> dict[str, jax.Array]:
    """パラメータを補正"""
    return {
        "means3d": raw_params["means3d"],
        "quats": raw_params["quats"]
        / (jnp.linalg.norm(raw_params["quats"], axis=-1, keepdims=True)),
        "scales": jnp.exp(raw_params["scales"]),
        "sh_coeffs": jnp.dstack((raw_params["sh_dc"], raw_params["sh_rest"])),
        "opacities": jax.nn.sigmoid(raw_params["opacities"]),
    }


def fix_quaternions(params: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """クォータニオンのゼロノルムやNaNを修正し、正規化する"""
    quats = params["quats"]

    nan_mask = jnp.isnan(quats).any(axis=1)
    inf_mask = jnp.isinf(quats).any(axis=1)

    quat_norms = jnp.linalg.norm(quats, axis=1, keepdims=True)
    zero_norm_mask = quat_norms.squeeze() < 1e-8
    invalid_mask = nan_mask | inf_mask | zero_norm_mask

    # 無効なクォータニオンを単位クォータニオン [0, 0, 0, 1] で置き換え
    unit_quat = jnp.array([0.0, 0.0, 0.0, 1.0])
    safe_quats = jnp.where(invalid_mask[:, None], unit_quat[None, :], quats)

    safe_norms = jnp.linalg.norm(safe_quats, axis=1, keepdims=True)
    normalized_quats = safe_quats / safe_norms

    params["quats"] = normalized_quats

    return params
