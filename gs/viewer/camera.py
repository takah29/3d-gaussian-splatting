from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation


class ControlStrategy(ABC):
    """マウスによるカメラ操作の抽象クラス"""

    @abstractmethod
    def rotate(
        self, position: npt.NDArray, rotation: Rotation, dx: float, dy: float, sensitivity: float
    ) -> tuple[npt.NDArray, Rotation]:
        """視線回転操作"""

    @abstractmethod
    def pan(
        self, position: npt.NDArray, rotation: Rotation, dx: float, dy: float, sensitivity: float
    ) -> tuple[npt.NDArray, Rotation]:
        """パン（平行移動）操作"""

    @abstractmethod
    def zoom(
        self, position: npt.NDArray, rotation: Rotation, delta: float, sensitivity: float
    ) -> tuple[npt.NDArray, Rotation]:
        """ズーム操作"""

    @abstractmethod
    def roll(
        self, position: npt.NDArray, rotation: Rotation, dx: float, sensitivity: float
    ) -> tuple[npt.NDArray, Rotation]:
        """ロール（カメラの前方軸周りの回転）操作"""


class GlControlStrategy(ControlStrategy):
    """OpenGLビューア用のマウスによるカメラ操作クラス"""

    def rotate(
        self, position: npt.NDArray, rotation: Rotation, dx: float, dy: float, sensitivity: float
    ) -> tuple[npt.NDArray, Rotation]:
        rot_y = Rotation.from_rotvec(np.radians(-dx * sensitivity) * np.array([0, 1, 0]))
        rot_x = Rotation.from_rotvec(np.radians(-dy * sensitivity) * np.array([1, 0, 0]))
        new_rotation = rotation * rot_y * rot_x
        return position, new_rotation

    def pan(
        self, position: npt.NDArray, rotation: Rotation, dx: float, dy: float, sensitivity: float
    ) -> tuple[npt.NDArray, Rotation]:
        pan_vector = np.array([-dx, dy, 0]) * sensitivity
        new_position = position + rotation.apply(pan_vector)
        return new_position, rotation

    def zoom(
        self, position: npt.NDArray, rotation: Rotation, delta: float, sensitivity: float
    ) -> tuple[npt.NDArray, Rotation]:
        zoom_vector = np.array([0, 0, -delta]) * sensitivity
        new_position = position + rotation.apply(zoom_vector)
        return new_position, rotation

    def roll(
        self, position: npt.NDArray, rotation: Rotation, dx: float, sensitivity: float
    ) -> tuple[npt.NDArray, Rotation]:
        roll_axis = rotation.apply(np.array([0, 0, 1]))
        roll_rotation = Rotation.from_rotvec(np.radians(-dx * sensitivity) * roll_axis)
        new_rotation = roll_rotation * rotation
        return position, new_rotation


class JaxControlStrategy(ControlStrategy):
    """Jaxビューア用のマウスによるカメラ操作クラス"""

    def rotate(
        self, position: npt.NDArray, rotation: Rotation, dx: float, dy: float, sensitivity: float
    ) -> tuple[npt.NDArray, Rotation]:
        rot_y = Rotation.from_rotvec(np.radians(dx * sensitivity) * np.array([0, 1, 0]))
        rot_x = Rotation.from_rotvec(np.radians(-dy * sensitivity) * np.array([1, 0, 0]))
        new_rotation = rotation * rot_y * rot_x
        return position, new_rotation

    def pan(
        self, position: npt.NDArray, rotation: Rotation, dx: float, dy: float, sensitivity: float
    ) -> tuple[npt.NDArray, Rotation]:
        pan_vector = np.array([-dx, -dy, 0]) * sensitivity
        new_position = position + rotation.apply(pan_vector)
        return new_position, rotation

    def zoom(
        self, position: npt.NDArray, rotation: Rotation, delta: float, sensitivity: float
    ) -> tuple[npt.NDArray, Rotation]:
        zoom_vector = np.array([0, 0, delta]) * sensitivity
        new_position = position + rotation.apply(zoom_vector)
        return new_position, rotation

    def roll(
        self, position: npt.NDArray, rotation: Rotation, dx: float, sensitivity: float
    ) -> tuple[npt.NDArray, Rotation]:
        roll_axis = rotation.apply(np.array([0, 0, 1]))
        roll_rotation = Rotation.from_rotvec(np.radians(dx * sensitivity) * roll_axis)
        new_rotation = roll_rotation * rotation
        return position, new_rotation


class Camera:
    """カメラの位置と回転を管理するクラス"""

    def __init__(
        self,
        position: npt.NDArray,
        rotation: Rotation,
        intrinsic: npt.NDArray,
        control_strategy: ControlStrategy,
    ) -> None:
        self.position = position
        self.rotation = rotation
        self.intrinsic = intrinsic
        self.control_strategy = control_strategy

    def rotate(self, dx: float, dy: float, sensitivity: float) -> None:
        """視点回転（オービット）操作。"""
        self.position, self.rotation = self.control_strategy.rotate(
            self.position, self.rotation, dx, dy, sensitivity
        )

    def pan(self, dx: float, dy: float, sensitivity: float) -> None:
        """パン（平行移動）操作。"""
        self.position, self.rotation = self.control_strategy.pan(
            self.position, self.rotation, dx, dy, sensitivity
        )

    def zoom(self, delta: float, sensitivity: float) -> None:
        """ズーム操作。"""
        self.position, self.rotation = self.control_strategy.zoom(
            self.position, self.rotation, delta, sensitivity
        )

    def roll(self, dx: float, sensitivity: float) -> None:
        """ロール（カメラの前方軸周りの回転）操作。"""
        self.position, self.rotation = self.control_strategy.roll(
            self.position, self.rotation, dx, sensitivity
        )

    def get_view(self) -> dict[str, npt.NDArray]:
        """現在の位置と回転からw2cビュー行列の回転と並進を計算する。"""
        rot_mat = self.rotation.as_matrix().T
        t_vec = -rot_mat @ self.position
        return {"rot_mat": rot_mat, "t_vec": t_vec, "intrinsic_vec": self.intrinsic}

    def set_pose_w2c(self, rot_mat_w2c: npt.NDArray, t_vec_w2c: npt.NDArray) -> None:
        """カメラの位置と回転を設定する。"""
        self.rotation = Rotation.from_matrix(rot_mat_w2c.T)
        self.position = -self.rotation.apply(t_vec_w2c)

    @staticmethod
    def _to_c2w(rot_mat: npt.NDArray, t_vec: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """w2c行列からc2w行列を計算する"""
        rot_mat_c2w = rot_mat.T
        t_vec_c2w = -rot_mat_c2w @ t_vec
        return rot_mat_c2w, t_vec_c2w

    @staticmethod
    def create_gl(rot_mat: npt.NDArray, t_vec: npt.NDArray, intrinsic_vec: npt.NDArray) -> "Camera":
        """OpenGLビューア用のカメラを作成"""
        rot_mat_c2w, t_vec_c2w = Camera._to_c2w(rot_mat, t_vec)
        return Camera(
            t_vec_c2w, Rotation.from_matrix(rot_mat_c2w), intrinsic_vec, GlControlStrategy()
        )

    @staticmethod
    def create_jax(
        rot_mat: npt.NDArray, t_vec: npt.NDArray, intrinsic_vec: npt.NDArray
    ) -> "Camera":
        """JAXビューア用のカメラを作成"""
        rot_mat_c2w, t_vec_c2w = Camera._to_c2w(rot_mat, t_vec)
        return Camera(
            t_vec_c2w, Rotation.from_matrix(rot_mat_c2w), intrinsic_vec, JaxControlStrategy()
        )
