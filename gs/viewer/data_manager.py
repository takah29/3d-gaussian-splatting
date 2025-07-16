import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation


class PostProcess:
    """座標系の変換用クラス"""

    def __init__(self, transform_matrix: npt.NDArray) -> None:
        self.transform_matrix = transform_matrix

    def params_to_gldata(self, params: dict) -> dict:
        """学習済みパラメータをOpenGLの座標系とデータ形式に変換"""
        params["means3d"] = (self.transform_matrix @ params["means3d"].T).T

        nan_mask = np.isnan(params["quats"]).any(axis=1)
        params["quats"][nan_mask] = np.array([0, 0, 0, 1])
        params["quats"] /= np.linalg.norm(params["quats"], axis=1, keepdims=True)

        rot_matrices = Rotation.from_quat(params["quats"]).as_matrix()
        transformed_matrices = self.transform_matrix @ rot_matrices
        params["quats"] = Rotation.from_matrix(transformed_matrices).as_quat()
        return params

    def camera_params_to_gldata(self, camera_params: dict) -> dict:
        """カメラパラメータをOpenGLの座標系に変換"""
        camera_params["rot_mat_batch"] = (
            self.transform_matrix @ camera_params["rot_mat_batch"] @ self.transform_matrix.T
        )
        camera_params["t_vec_batch"] = camera_params["t_vec_batch"] @ self.transform_matrix
        camera_params["intrinsic_batch"] *= np.array([1, 1, 1, -1])
        return camera_params


class DataManager:
    """PKLファイルの探索、読み込み、キャッシングを管理するクラス

    チェックポイントファイル(params_checkpoint_iter<イテレーション回数>.pkl)が異なっていても
    カメラパラメータは共通であることを想定しているのでカメラパラメータは初回のロード時しか読み込まない
    """

    def __init__(
        self,
        pkl_files: list[Path],
        post_process: PostProcess | None = None,
    ) -> None:
        # params(ガウシアンのデータ)用変数
        self._pkl_files = pkl_files
        self._params_cache: dict[int, Any] = {}
        self._current_data_index = 0

        # カメラパラメータ用変数
        self._camera_params: dict[str, npt.NDArray] | None = None
        self._camera_param_index = 0

        # 設定用変数
        self._consts: dict | None = None

        # 座標変換用クラス
        self._post_process = post_process

    def get_current_filename(self) -> str:
        """現在のファイル名を取得"""
        return self._pkl_files[self._current_data_index].name

    def get_current_data_index(self) -> tuple[int, int]:
        """現在のデータインデックスを取得"""
        return self._current_data_index, len(self._pkl_files)

    def get_current_camera_param_index(self) -> tuple[int, int]:
        """現在のカメラパラメータのインデックス"""
        if self._camera_params is None:
            msg = "No data loaded."
            raise ValueError(msg)

        return self._camera_param_index, len(self._camera_params["rot_mat_batch"])

    def get_camera_param(self, index: int) -> dict[str, npt.NDArray]:
        """指定インデックスのカメラパラメータを取得"""
        if self._camera_params is None:
            msg = "No data loaded."
            raise ValueError(msg)
        return {
            "rot_mat": self._camera_params["rot_mat_batch"][index],
            "t_vec": self._camera_params["t_vec_batch"][index],
            "intrinsic_vec": self._camera_params["intrinsic_batch"][index],
        }

    def get_consts(self) -> tuple[dict, dict]:
        """設定値を取得"""
        if self._consts is None:
            msg = "No data loaded."
            raise ValueError(msg)
        return self._consts

    def move_data_index(self, step: int) -> int:
        """現在のインデックスをstep分移動させる"""
        num_files = len(self._pkl_files)
        self._current_data_index = (self._current_data_index + step + num_files) % num_files
        return self._current_data_index

    def move_camera_param_index(self, step: int) -> int:
        """現在のインデックスをstep分移動させる"""
        if self._camera_params is None:
            msg = "No data loaded."
            raise ValueError(msg)

        num_params = len(self._camera_params["rot_mat_batch"])
        self._camera_param_index = (self._camera_param_index + step + num_params) % num_params
        return self._camera_param_index

    def load_data(self, index: int) -> dict | None:
        """指定インデックスのデータをロードし、キャッシュする"""
        if index in self._params_cache:
            return self._params_cache[index]

        filepath = self._pkl_files[index]
        print(f"Loading from disk: {filepath.name} ...", end="", flush=True)
        try:
            with filepath.open("rb") as f:
                data = pickle.load(f)

            # 必要に応じてロード後の後処理を実行
            if self._post_process:
                data["params"] = self._post_process.params_to_gldata(data["params"])

            if len(self._params_cache) == 0:
                if self._post_process:
                    self._camera_params = (
                        self._post_process.camera_params_to_gldata(data["camera_params"])
                        if self._post_process
                        else data["camera_params"]
                    )
                else:
                    self._camera_params = data["camera_params"]
                self._consts = data["consts"]
            self._params_cache[index] = data["params"]
            self._current_data_index = index

            print(" Done.")
            return data["params"]

        except Exception as e:
            print(f" Error loading {filepath.name}: {e}", file=sys.stderr)
            return None

    @staticmethod
    def create_for_gldata(pkl_files: list[Path]) -> "DataManager":
        """OpenGL座標系への変換処理クラスを生成

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
