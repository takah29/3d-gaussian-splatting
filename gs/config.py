import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class GsConfig:
    """Gaussian Splattingの設定を管理するデータクラス。

    Args:
        background: 背景色 [R, G, B]
        max_gaussians: 最大ガウシアン数
        pruning_big_gaussian: 大きなGaussianのプルーニングを有効にするかどうか
        eps_prune_alpha: プルーニングのアルファ閾値
        tau_pos: 位置の収束判定閾値
        scale_threshold: スケールの閾値
        split_gaussian_scale: Gaussian分割時のスケール係数
        split_num: 分割数
        densify_from_iter: 密度化開始イテレーション
        densify_until_iter: 密度化終了イテレーション
        densification_interval: 密度化間隔
        tile_size: タイルサイズ
        tile_max_gs_num_factor: タイルあたりの最大ガウシアン数を決める係数
        tile_chanks: メモリ消費量削減のためタイル分割数
    """

    background: list[float]
    max_gaussians: int
    pruning_big_gaussian: bool
    eps_prune_alpha: float
    tau_pos: float
    scale_threshold: float
    split_gaussian_scale: float
    split_num: int
    densify_from_iter: int
    densify_until_iter: int
    densification_interval: int
    tile_size: int
    tile_max_gs_num_factor: int
    tile_chanks: int
    drop_rate: float

    img_shape: tuple[int, int] = field(init=False)  # 画像の高さと幅
    extent: float = field(init=False)  # シーンの大きさ
    tile_max_gs_num: int = field(init=False)  # タイルあたりの最大ガウシアン数

    def derive_additional_property(
        self, image_batch: npt.NDArray, camera_params: dict[str, npt.NDArray], n_gaussians: int
    ) -> None:
        self.img_shape = image_batch.shape[1:3]
        self.extent = float(
            np.linalg.norm(
                camera_params["t_vec_batch"] - camera_params["t_vec_batch"].mean(axis=0), axis=1
            ).max()
            * 1.1
        )
        self.set_tile_max_gs_num(n_gaussians)

    def display(self) -> None:
        """設定パラメータを表示する。"""
        print(f"{' Configuration ':=^40}")
        max_len = max(len(field.name) for field in self.__dataclass_fields__.values())

        for field_info in self.__dataclass_fields__.values():
            name = field_info.name
            value = getattr(self, name)

            print(f"{name:<{max_len}} : {value!s}")
        print(f"{'=' * 40}")

    def to_dict(self) -> dict[str, Any]:
        """設定を辞書形式で取得する。

        Returns:
            設定パラメータの辞書
        """
        return asdict(self)

    def to_json_file(self, json_path: Path) -> None:
        """設定をJSONファイルに保存する。

        Args:
            json_path: 保存先のJSONファイルパス
        """
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def set_tile_max_gs_num(self, n_gaussians: int) -> None:
        base_tile_size = 16
        base_gaussian_num = 1000
        tile_scale = self.tile_size / base_tile_size
        self.tile_max_gs_num = int(
            tile_scale**2 * self.tile_max_gs_num_factor * n_gaussians / base_gaussian_num
        )

    @classmethod
    def from_json_file(cls, json_path: Path) -> "GsConfig":
        """JSONファイルから設定インスタンスを作成する。

        Args:
            json_path: 読み込むJSONファイルのパス

        Returns:
            設定インスタンス
        """
        with json_path.open(encoding="utf-8") as f:
            json_data = json.load(f)
        return cls(**json_data)
