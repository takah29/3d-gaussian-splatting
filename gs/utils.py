from pathlib import Path

import grain  # type: ignore[import-untyped]
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pycolmap
from PIL import Image
from scipy.spatial import cKDTree


def load_colmap_data(
    colmap_data_path: Path, image_scale: float, quatanion: bool = False
) -> tuple[dict[str, npt.NDArray], dict[str, npt.NDArray], npt.NDArray]:
    reconstruction = pycolmap.Reconstruction(str(colmap_data_path / "sparse" / "0"))
    points_3d = np.vstack([pt.xyz for pt in reconstruction.points3D.values()], dtype=np.float32)
    colors = (
        np.vstack([pt.color for pt in reconstruction.points3D.values()], dtype=np.float32) / 255.0
    )
    t_vec_batch = np.stack(
        [img.cam_from_world.translation for img in reconstruction.images.values()],
        dtype=np.float32,
    )

    intrinsic_list = []
    image_list = []
    for img in reconstruction.images.values():
        image_arr = _load_image_and_fit(colmap_data_path / "images" / img.name, image_scale)
        height, width = image_arr.shape[:2]
        width_scale = width / img.camera.width
        height_scale = height / img.camera.height
        intrinsic_vec = img.camera.params * np.array(
            [width_scale, height_scale, width_scale, height_scale]
        )

        image_list.append(image_arr)
        intrinsic_list.append(intrinsic_vec)

    image_batch = np.stack(image_list, axis=0)
    intrinsic_batch = np.stack(intrinsic_list, axis=0)

    if quatanion:
        quat_batch = np.stack(
            [img.cam_from_world.rotation.quat for img in reconstruction.images.values()],
            dtype=np.float32,
        )
        camera_params = {
            "quat_batch": quat_batch,  # (qx, qy, qz, qw) in quat_batch
            "t_vec_batch": t_vec_batch,  # (tx, ty, tz) in t_vec_batch
            "intrinsic_batch": intrinsic_batch,  # (fx, fy, cx, cy) in intrinsic_batch
        }
    else:
        rot_mat_batch = np.stack(
            [img.cam_from_world.rotation.matrix() for img in reconstruction.images.values()],
            dtype=np.float32,
        )
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


def create_view_dataloader(
    image_batch: npt.NDArray,
    camera_params: dict[str, npt.NDArray],
    n_epochs: int,
    *,
    shuffle: bool = True,
) -> tuple[grain.DataLoader, tuple[int, int]]:
    camera_param_list = [
        {"rot_mat": rot_mat, "t_vec": t_vec, "intrinsic_vec": intrinsic}
        for rot_mat, t_vec, intrinsic in zip(
            camera_params["rot_mat_batch"],
            camera_params["t_vec_batch"],
            camera_params["intrinsic_batch"],
            strict=True,
        )
    ]

    data_source = grain.MapDataset.source(
        list(zip(camera_param_list, image_batch, strict=True))
    ).shuffle(seed=42)

    index_sampler = grain.samplers.IndexSampler(
        num_records=len(image_batch),
        shuffle=shuffle,
        num_epochs=n_epochs,
        seed=123,
    )
    return grain.DataLoader(data_source=data_source, sampler=index_sampler)  # type: ignore[arg-type]


def _compute_nearest_distances(points: npt.NDArray) -> npt.NDArray:
    if points.shape[0] == 0:
        return np.array([])

    tree = cKDTree(points)

    distances, _ = tree.query(points, k=2)
    nearest_point_distances = distances[:, 1:].ravel()  # type: ignore[index]

    return nearest_point_distances


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


def _init_gaussian_property(points_3d: npt.NDArray) -> dict[str, npt.NDArray]:
    num_points = points_3d.shape[0]

    # 近傍の3点の平均距離で設定
    nearest_distances = _compute_nearest_distances(points_3d)
    scale_factor = 0.3
    scales = np.log(np.clip(scale_factor * nearest_distances, 0.01, 3.0))[:, np.newaxis].repeat(
        3, axis=1
    )
    # scales = np.log(np.ones((num_points, 3)) * 0.01)

    # 回転なし
    quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (num_points, 1))

    opacities = np.full((num_points, 1), inverse_sigmoid(0.1))

    return {"scales": scales, "quats": quats, "opacities": opacities}


def _init_consts(
    height: int, width: int, max_points: int, extent: float
) -> dict[str, int | float | npt.NDArray]:
    return {
        "background": np.array([0.0, 0.0, 0.0]),
        "img_shape": np.array([height, width]),
        "max_points": max_points,
        "extent": extent,
        "eps_prune_alpha": 0.05,
        "tau_pos": 0.3,
        "eps_clone_eigval": 5.0,
        "split_gaussian_scale": 0.8,
        "split_num": 2,
        "densify_from_iter": 500,
        "densify_until_iter": 15000,
        "densification_interval": 100,
    }


def build_params(
    colmap_data_path: Path, max_points: int, image_scale: float, n_epochs: int
) -> tuple[dict[str, npt.NDArray], dict[str, int | float | npt.NDArray], npt.NDArray]:
    reconstruction_data, camera_params, image_batch = load_colmap_data(
        colmap_data_path, image_scale
    )
    image_dataloader = create_view_dataloader(
        image_batch, camera_params, n_epochs=n_epochs, shuffle=True
    )

    points_3d = reconstruction_data["points_3d"]
    colors = reconstruction_data["colors"]

    total_points = len(points_3d)
    sample_size = min(max_points, total_points)
    sampled_indices = np.random.default_rng(123).choice(
        total_points, size=sample_size, replace=False
    )

    points_3d = points_3d[sampled_indices]
    colors = colors[sampled_indices]

    params = {
        "means3d": points_3d,
        "colors": inverse_sigmoid(np.clip(colors, 1e-4, 1.0 - 1e-4)),
        **_init_gaussian_property(points_3d),
    }

    height, width = image_batch.shape[1:3]
    extent = jnp.linalg.norm(points_3d.max(axis=0) - points_3d.min(axis=0))
    consts = _init_consts(height, width, max_points, extent)

    print("===== Data Information =====")
    for k, v in params.items():
        print(f"{k}: {v.shape}")
    print("============================")

    return params, consts, image_dataloader
