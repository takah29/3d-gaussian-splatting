from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.transform import Rotation


class BaseCamera(ABC):
    """カメラの位置と回転を管理する抽象基底クラス。"""

    def __init__(self, position: np.ndarray, rotation: Rotation):
        self.position = position
        self.rotation = rotation

    @abstractmethod
    def rotate(self, dx: float, dy: float, sensitivity: float):
        """視点回転（オービット）操作。"""

    @abstractmethod
    def pan(self, dx: float, dy: float, sensitivity: float):
        """パン（平行移動）操作。"""

    @abstractmethod
    def zoom(self, delta: float, sensitivity: float):
        """ズーム操作。"""


class GlCamera(BaseCamera):
    """OpenGLビューア用のカメラ。 get_view_matrix を提供する。"""

    def get_view_matrix(self) -> np.ndarray:
        """現在の位置と回転からOpenGL互換の4x4ビュー行列(w2c)を計算する。"""
        rot_mat = self.rotation.as_matrix().T
        t_vec = -rot_mat @ self.position
        view_matrix = np.identity(4, dtype="f4")
        view_matrix[:3, :3] = rot_mat
        view_matrix[:3, 3] = t_vec
        return view_matrix

    def rotate(self, dx: float, dy: float, sensitivity: float):
        """OpenGL版のオービット操作。"""
        rot_y = Rotation.from_rotvec(np.radians(dx * sensitivity) * np.array([0, 1, 0]))
        rot_x = Rotation.from_rotvec(
            self.rotation.apply(np.radians(dy * sensitivity) * np.array([1, 0, 0]))
        )
        self.rotation = rot_x * self.rotation * rot_y

    def pan(self, dx: float, dy: float, sensitivity: float):
        """OpenGL版のパン操作。"""
        pan_vector = np.array([-dx, dy, 0]) * sensitivity
        self.position += self.rotation.apply(pan_vector)

    def zoom(self, delta: float, sensitivity: float):
        """OpenGL版のズーム操作。"""
        zoom_vector = np.array([0, 0, -delta]) * sensitivity
        self.position += self.rotation.apply(zoom_vector)


class JaxCamera(BaseCamera):
    """JAXビューア用のカメラ。 get_view_params を提供する。"""

    def get_view_params(self) -> tuple[np.ndarray, np.ndarray]:
        """現在の位置と回転からw2cビュー行列の回転と並進を計算する。"""
        rot_mat = self.rotation.inv().as_matrix()
        t_vec = -rot_mat @ self.position
        return rot_mat, t_vec

    def rotate(self, dx: float, dy: float, sensitivity: float):
        """JAX版のオービット操作。"""
        rot_y = Rotation.from_rotvec(np.radians(-dx * sensitivity) * np.array([0, 1, 0]))
        rot_x = Rotation.from_rotvec(np.radians(dy * sensitivity) * np.array([1, 0, 0]))
        self.rotation *= rot_y * rot_x

    def pan(self, dx: float, dy: float, sensitivity: float):
        """JAX版のパン操作。"""
        pan_vector = np.array([-dx, -dy, 0]) * sensitivity
        self.position += self.rotation.apply(pan_vector)

    def zoom(self, delta: float, sensitivity: float):
        """JAX版のズーム操作。"""
        zoom_vector = np.array([0, 0, delta]) * sensitivity
        self.position += self.rotation.apply(zoom_vector)
