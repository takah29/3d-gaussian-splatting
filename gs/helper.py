from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.special import logit

from gs.config import GsConfig
from gs.core.projection import SH_C0_0
from gs.utils import ImageViewDataLoader, compute_nearest_distances, load_colmap_data


def _init_gaussian_property(points_3d: npt.NDArray) -> dict[str, npt.NDArray]:
    num_points = points_3d.shape[0]

    nearest_distances = compute_nearest_distances(points_3d) ** 2
    scales = np.log(np.clip(nearest_distances, 0.001, 3.0))[:, np.newaxis].repeat(3, axis=1)

    # 回転なし、(x, y, z, w)形式
    quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (num_points, 1))

    opacities = np.full((num_points, 1), logit(0.8))

    return {"scales": scales, "quats": quats, "opacities": opacities}


def print_info(params: dict[str, npt.NDArray]) -> None:
    print("===== params =====")
    for k, v in params.items():
        print(f"{k}: {v.shape}")
    print("==================")


def build_params(
    colmap_data_path: Path, gs_config: GsConfig, image_scale: float, n_epochs: int
) -> tuple[dict[str, npt.NDArray], ImageViewDataLoader]:
    reconstruction_data, camera_params, image_batch = load_colmap_data(
        colmap_data_path, image_scale
    )
    image_dataloader = ImageViewDataLoader(
        camera_params, image_batch, n_epochs=n_epochs, shuffle=True
    )

    points_3d = reconstruction_data["points_3d"]
    colors = reconstruction_data["colors"]

    total_points = len(points_3d)
    sample_size = min(gs_config.max_gaussians, total_points)
    sampled_indices = np.random.default_rng(123).choice(
        total_points, size=sample_size, replace=False
    )

    points_3d = points_3d[sampled_indices]
    colors = colors[sampled_indices]

    params = {
        "means3d": points_3d,
        "sh_dc": ((colors - 0.5) / SH_C0_0)[:, :, None],  # (r, g, b) in sh_deg0
        "sh_rest": np.zeros((sample_size, 3, 15), dtype=np.float32),
        **_init_gaussian_property(points_3d),
    }

    return params, image_dataloader
