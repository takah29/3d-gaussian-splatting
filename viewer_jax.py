"""3D Gaussian Splatting Viewer (JAX版)

指定された.pklファイルと同じディレクトリにあるすべての.pklファイルを読み込み、
インタラクティブに表示・切り替えを行うビューア。

操作方法:
- マウス左ドラッグ: 視点の回転 (オービット)
- マウス右ドラッグ: 視点の平行移動 (パン)
- マウスホイール: ズームイン / ズームアウト
- ←/→キー: 学習済みカメラ視点間の移動
- ↑/↓キー: .pklファイルの切り替え
"""

import argparse
import sys
from pathlib import Path

import glfw
import numpy as np
from scipy.spatial.transform import Rotation

# ローカルモジュールがプロジェクト内に存在することを前提とします。
from camera import JaxCamera
from data_manager import DataManager
from gs.utils import calc_tile_max_gs_num
from renderer_jax import RendererJax


class ViewerJax:
    """GLFWウィンドウ、ユーザー入力、レンダリングループを管理するメインクラス。"""

    # --- マウス感度設定 ---
    MOUSE_SENSITIVITY_ORBIT = 0.2
    MOUSE_SENSITIVITY_PAN = 0.002
    MOUSE_SENSITIVITY_ZOOM = 0.3
    MOUSE_SENSITIVITY_ROLL = 0.1
    WINDOW_TITLE = "JAX 3DGS Viewer"

    def __init__(self, data_manager: DataManager, initial_index: int):
        self.data_manager = data_manager
        self.params = self.data_manager.load_data(initial_index)
        self.camera_params, self.consts = self.data_manager.get_camera_params_and_consts()

        # JAX版特有のレンダリング設定
        self.tile_max_gs_num_coeff = 25.0
        self.consts["tile_max_gs_num"] = calc_tile_max_gs_num(
            self.consts["tile_size"],
            self.consts["img_shape"][0],
            self.consts["img_shape"][1],
            self.consts["max_points"],
            self.tile_max_gs_num_coeff,
        )

        # --- パラメータの初期化 ---
        self.initial_width, self.initial_height = self.consts["img_shape"][::-1]
        intrinsic_vec = self.camera_params["intrinsic_batch"][0]
        self.initial_fx, self.initial_fy, _, _ = intrinsic_vec
        self.render_width, self.render_height = self.initial_width, self.initial_height
        self.current_fx, self.current_fy = self.initial_fx, self.initial_fy

        # --- 初期化処理の実行 ---
        self._init_glfw()
        self.renderer = RendererJax(self.params, self.consts)
        self.camera = self._create_initial_camera()
        self._setup_callbacks()

        # 初回のビューポート設定
        self.framebuffer_size_callback(self.window, self.initial_width, self.initial_height)

        # --- 状態変数の初期化 ---
        self.left_mouse_dragging = False
        self.right_mouse_dragging = False
        self.middle_mouse_dragging = False
        self.last_mouse_pos = None
        self.camera_dirty = True  # 再描画が必要かどうかのフラグ

    def run(self):
        """メインループを実行する。"""
        while not glfw.window_should_close(self.window):
            glfw.wait_events()
            if self.camera_dirty:
                self._render_frame()
                self.camera_dirty = False

        self.renderer.shutdown()
        glfw.terminate()

    def _render_frame(self):
        rot_mat, t_vec = self.camera.get_view_params()
        intrinsic_vec = self.camera_params["intrinsic_batch"][self.current_cam_index]
        view_params = {
            "rot_mat": rot_mat,
            "t_vec": t_vec,
            "intrinsic_vec": intrinsic_vec,
        }
        self.renderer.render(view_params)
        glfw.swap_buffers(self.window)

    def _init_glfw(self):
        """GLFWとウィンドウを初期化する。"""
        if not glfw.init():
            sys.exit("FATAL ERROR: glfw initialization failed.")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(
            self.initial_width, self.initial_height, self.WINDOW_TITLE, None, None
        )
        if not self.window:
            glfw.terminate()
            sys.exit("FATAL ERROR: Failed to create glfw window.")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.update_window_title()

    def _setup_callbacks(self):
        """GLFWのイベントコールバックを設定する。"""
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)

    def _create_initial_camera(self) -> JaxCamera:
        """最初の学習済みカメラ視点からCameraオブジェクトを生成する。"""
        self.num_cameras = len(self.camera_params["t_vec_batch"])
        self.current_cam_index = 0
        pos, rot = self._get_camera_state(self.current_cam_index)
        return JaxCamera(position=pos, rotation=rot)

    def _get_camera_state(self, index: int) -> tuple[np.ndarray, Rotation]:
        """指定インデックスの学習済みカメラ情報を取得し、位置と回転を返す。"""
        rot_mat_w2c = self.camera_params["rot_mat_batch"][index]
        t_vec_w2c = self.camera_params["t_vec_batch"][index]
        c2w_rotation = Rotation.from_matrix(rot_mat_w2c.T)
        c2w_position = -c2w_rotation.apply(t_vec_w2c)
        return c2w_position, c2w_rotation

    def load_params(self, index: int):
        """指定インデックスのpklファイルをロードし、レンダラを更新する。"""
        self.params = self.data_manager.load_data(index)
        self.renderer.update_gaussian_data(self.params)
        self.camera_dirty = True
        self.update_window_title()

    def update_window_title(self):
        """現在のファイル名と位置情報でウィンドウタイトルを更新する。"""
        filename = self.data_manager.get_current_filename()
        current_index, n_data = self.data_manager.get_current_data_index()
        title = f"{self.WINDOW_TITLE} ({filename} {current_index + 1}/{n_data})"
        glfw.set_window_title(self.window, title)

    def change_camera_pose(self):
        """学習済みカメラ視点を切り替える。"""
        self.camera.position, self.camera.rotation = self._get_camera_state(self.current_cam_index)
        self.camera_dirty = True

    def framebuffer_size_callback(self, window, width, height):
        """ウィンドウリサイズ時に呼び出され、アスペクト比を維持しつつ表示する。"""
        if height == 0 or width == 0:
            return

        content_aspect = self.initial_width / self.initial_height
        window_aspect = width / height

        # 元の画像の縦横比を保ちつつ、ウィンドウを覆うように描画領域を計算（カバー）
        if window_aspect > content_aspect:  # ウィンドウが横長
            self.render_width = width
            self.render_height = int(width / content_aspect)
            view_x, view_y = 0, (height - self.render_height) // 2
        else:  # ウィンドウが縦長または同じ
            self.render_height = height
            self.render_width = int(height * content_aspect)
            view_x, view_y = (width - self.render_width) // 2, 0

        # 1. レンダラにビューポートを設定
        self.renderer.set_viewport(view_x, view_y, self.render_width, self.render_height)

        self.camera_dirty = True

    def key_callback(self, window, key, scancode, action, mods):
        """キーボード入力イベントを処理する。"""
        if action == glfw.PRESS:
            if key == glfw.KEY_UP:
                current_index = self.data_manager.navigate(-1)
                self.load_params(current_index)
            elif key == glfw.KEY_DOWN:
                current_index = self.data_manager.navigate(1)
                self.load_params(current_index)

        if action in (glfw.PRESS, glfw.REPEAT):
            cam_changed = False
            if key == glfw.KEY_RIGHT:
                self.current_cam_index = (self.current_cam_index + 1) % self.num_cameras
                cam_changed = True
            elif key == glfw.KEY_LEFT:
                self.current_cam_index = (
                    self.current_cam_index - 1 + self.num_cameras
                ) % self.num_cameras
                cam_changed = True

            if cam_changed:
                self.change_camera_pose()
                print(f"Jump to Camera: {self.current_cam_index + 1}/{self.num_cameras}", end="\r")

    def mouse_button_callback(self, window, button, action, mods):
        """マウスボタンのプレス/リリースイベントを処理する。"""
        if action == glfw.PRESS:
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.left_mouse_dragging = True
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.right_mouse_dragging = True
            elif button == glfw.MOUSE_BUTTON_MIDDLE:
                self.middle_mouse_dragging = True
            self.last_mouse_pos = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            self.left_mouse_dragging = False
            self.right_mouse_dragging = False
            self.middle_mouse_dragging = False
            self.last_mouse_pos = None

    def cursor_pos_callback(self, window, xpos, ypos):
        """マウスカーソルの移動イベントを処理する。"""
        if (
            not any(
                [self.left_mouse_dragging, self.right_mouse_dragging, self.middle_mouse_dragging]
            )
            or self.last_mouse_pos is None
        ):
            return

        dx, dy = xpos - self.last_mouse_pos[0], ypos - self.last_mouse_pos[1]
        if self.left_mouse_dragging:
            self.camera.rotate(dx, dy, self.MOUSE_SENSITIVITY_ORBIT)
        if self.right_mouse_dragging:
            self.camera.pan(dx, dy, self.MOUSE_SENSITIVITY_PAN)
        if self.middle_mouse_dragging:
            self.camera.roll(dx, self.MOUSE_SENSITIVITY_ROLL)

        self.camera_dirty = True
        self.last_mouse_pos = (xpos, ypos)

    def scroll_callback(self, window, xoffset, yoffset):
        """マウスホイールのスクロールイベントを処理する。"""
        self.camera.zoom(yoffset, self.MOUSE_SENSITIVITY_ZOOM)
        self.camera_dirty = True


def main():
    """アプリケーションのエントリーポイント。"""
    parser = argparse.ArgumentParser(description="JAX 3D Gaussian Splatting Interactive Viewer")
    parser.add_argument(
        "-f",
        "--params_filepath",
        type=Path,
        default=Path(__file__).parent / "output" / "params_final.pkl",
        help="Path to the initial params pickle file.",
    )
    args = parser.parse_args()

    directory = args.params_filepath.parent
    pkl_files = sorted(directory.glob("*.pkl"))
    if not pkl_files:
        sys.exit(f"Error: No .pkl files found in {directory}")

    try:
        initial_filepath = args.params_filepath.resolve()
        initial_index = [p.resolve() for p in pkl_files].index(initial_filepath)
    except ValueError:
        print(
            f"Warning: Specified file {args.params_filepath} not found. Starting with the first one."
        )
        initial_index = 0

    data_manager = DataManager(pkl_files)
    viewer = ViewerJax(data_manager, initial_index)
    viewer.run()


if __name__ == "__main__":
    main()
