import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation


class PostProcess:
    def __init__(self, transform_matrix: npt.NDArray) -> None:
        self.transform_matrix = transform_matrix

    def params_to_gldata(self, params: dict) -> dict:
        """学習済みパラメータをOpenGLの座標系とデータ形式に変換する。"""
        params["means3d"] = (self.transform_matrix @ params["means3d"].T).T

        nan_mask = np.isnan(params["quats"]).any(axis=1)
        params["quats"][nan_mask] = np.array([0, 0, 0, 1])
        params["quats"] /= np.linalg.norm(params["quats"], axis=1, keepdims=True)

        rot_matrices = Rotation.from_quat(params["quats"]).as_matrix()
        transformed_matrices = self.transform_matrix @ rot_matrices
        params["quats"] = Rotation.from_matrix(transformed_matrices).as_quat()
        return params

    def camera_params_to_gldata(self, camera_params: dict) -> dict:
        """カメラパラメータをOpenGLの座標系に変換する。"""
        camera_params["rot_mat_batch"] = (
            self.transform_matrix @ camera_params["rot_mat_batch"] @ self.transform_matrix.T
        )
        camera_params["t_vec_batch"] = camera_params["t_vec_batch"] @ self.transform_matrix
        camera_params["intrinsic_batch"] *= np.array([1, 1, 1, -1])
        return camera_params


class DataManager:
    """PKLファイルの探索、読み込み、キャッシングを管理するクラス。"""

    def __init__(
        self,
        pkl_files: list[Path],
        post_process: PostProcess | None = None,
    ) -> None:
        self.pkl_files = pkl_files
        self.current_data_index: int | None = None
        self.params_cache: dict[int, Any] = {}
        self.camera_params: dict | None = None
        self.consts: dict | None = None
        self.post_process = post_process

    def get_current_filename(self) -> str:
        """現在のファイル名を返す。"""
        return self.pkl_files[self.current_data_index].name

    def get_current_data_index(self):
        if self.current_data_index is None:
            msg = "No data loaded."
            raise ValueError(msg)
        return self.current_data_index, len(self.pkl_files)

    def load_data(self, index: int) -> dict | None:
        """指定インデックスのデータをロードし、キャッシュする。"""
        if index in self.params_cache:
            return self.params_cache[index]

        filepath = self.pkl_files[index]
        print(f"Loading from disk: {filepath.name} ...", end="", flush=True)
        try:
            with filepath.open("rb") as f:
                data = pickle.load(f)

            # 必要に応じてロード後の後処理を実行
            if self.post_process:
                data["params"] = self.post_process.params_to_gldata(data["params"])

            if len(self.params_cache) == 0:
                if self.post_process:
                    self.camera_params = (
                        self.post_process.camera_params_to_gldata(data["camera_params"])
                        if self.post_process
                        else data["camera_params"]
                    )
                else:
                    self.camera_params = data["camera_params"]
                self.consts = data["consts"]
            self.params_cache[index] = data["params"]
            self.current_data_index = index

            print(" Done.")
            return data["params"]

        except Exception as e:
            print(f" Error loading {filepath.name}: {e}", file=sys.stderr)
            return None

    def navigate(self, step: int) -> int:
        """現在のインデックスをstep分移動させる。"""
        num_files = len(self.pkl_files)
        self.current_data_index = (self.current_data_index + step + num_files) % num_files
        return self.current_data_index

    @staticmethod
    def create_for_gldata(pkl_files: list[Path]) -> "DataManager":
        """OpenGL座標系への変換処理クラスを生成する。

        学習時の座標系からOpenGLの座標系への変換行列を定義
        (右手系: X右, Y下, Z奥 -> 右手系: X右, Y上, Z手前)
        """
        transform_matrix = np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ]
        )
        post_process = PostProcess(transform_matrix)
        return DataManager(pkl_files, post_process)
