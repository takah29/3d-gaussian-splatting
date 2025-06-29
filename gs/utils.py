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
) -> tuple[grain.DataLoader, tuple[int, int]]:
    target_image_path_list = [
        pth
        for pth in image_dir_parh.glob("*")
        if pth.suffix.lower() in (".jpg", ".jpeg", ".png", "gif", "bmp", "tiff")
    ]
    camera_param_list = [
        {"rot_mat": rot_mat, "t_vec": t_vec, "intrinsic_vec": intrinsic}
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
    return (
        grain.DataLoader(data_source=data_source, sampler=index_sampler),  # type: ignore[arg-type]
        _load_image_and_fit(target_image_path_list[0], max_res).shape[:2],
    )


def _compute_nearest_mean_distances(points: npt.NDArray) -> npt.NDArray:
    if points.shape[0] == 0:
        return np.array([])

    tree = cKDTree(points)

    # 自分自信が含まれるためk=4を設定
    distances, _ = tree.query(points, k=4)
    nearest_3points_distances = distances[:, 1:]  # type: ignore[index]

    return np.mean(nearest_3points_distances, axis=1)


def _init_gaussian_property(points_3d: npt.NDArray) -> dict[str, npt.NDArray]:
    num_points = points_3d.shape[0]

    # 近傍の3点の平均距離で設定
    nearest_mean_distances = _compute_nearest_mean_distances(points_3d)
    scales = np.log(nearest_mean_distances)[:, np.newaxis].repeat(3, axis=1)

    # 回転なし
    quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (num_points, 1))

    # sigmoidを適用して0.1となるように設定
    alpha = 0.1
    val = np.log(alpha / (1 - alpha))
    opacities = np.full((num_points, 1), val)

    return {"scales": scales, "quats": quats, "opacities": opacities}


def _init_consts(height: int, width: int) -> dict[str, int | float | npt.NDArray]:
    return {
        "background": np.array([0.0, 0.0, 0.0]),
        "img_shape": np.array([height, width]),
        "eps_alpha": 0.05,
        "tau_pos": 0.0005,
        "eps_eigval": 5.0,
        "split_gaussian_scale": 0.8,
        "split_num": 2,
        "max_gaussians_num": 50000,
        "densify_from_iter": 500,
        "densify_until_iter": 15000,
        "densification_interval": 100,
    }


def build_params(
    colmap_data_path: Path, max_res: int, n_epochs: int
) -> tuple[dict[str, npt.NDArray], dict[str, int | float | npt.NDArray], grain.DataLoader]:
    reconstruction_data = load_colmap_data(colmap_data_path / "sparse" / "0")

    # 読み込んだデータの情報を整形して表示
    print("===== Colmap Data Information =====")
    for k, v in reconstruction_data.items():
        print(f"{k}: {v.shape}")
    print("============================")

    view_dataloader, img_size = create_view_dataloader(
        colmap_data_path / "images", reconstruction_data, max_res, n_epochs
    )

    params = {
        "means3d": reconstruction_data["points_3d"],
        "colors": reconstruction_data["colors"],
        **_init_gaussian_property(reconstruction_data["points_3d"]),
    }
    consts = _init_consts(*img_size)

    return params, consts, view_dataloader
