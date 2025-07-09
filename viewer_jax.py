"""3D Gaussian Splatting Viewer

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
import pickle
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import glfw
import moderngl
import numpy as np
from scipy.spatial.transform import Rotation

from gs.make_update import make_render


@dataclass
class Camera:
    """カメラの位置と回転を管理し、ビュー行列を計算するクラス。"""

    position: np.ndarray
    rotation: Rotation

    def get_view_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """現在の位置と回転からw2cビュー行列を計算する。"""
        rot_mat = self.rotation.inv().as_matrix()
        t_vec = -rot_mat @ self.position
        return rot_mat, t_vec

    def rotate(self, dx: float, dy: float, sensitivity: float):
        """現在の回転に対して、相対的な視点回転を行う。"""
        rot_y = Rotation.from_rotvec(np.radians(-dx * sensitivity) * np.array([0, 1, 0]))
        rot_x = Rotation.from_rotvec(np.radians(dy * sensitivity) * np.array([1, 0, 0]))
        self.rotation *= rot_y * rot_x

    def pan(self, dx: float, dy: float, sensitivity: float):
        """現在の向きを基準に、相対的なパン操作を行う。"""
        pan_vector = np.array([-dx, -dy, 0]) * sensitivity
        self.position += self.rotation.apply(pan_vector)

    def zoom(self, delta: float, sensitivity: float):
        """現在の向きを基準に、相対的なズーム操作を行う。"""
        zoom_vector = np.array([0, 0, delta]) * sensitivity
        self.position += self.rotation.apply(zoom_vector)


@dataclass
class GaussianRenderer:
    """ガウシアンスプラッティングのレンダリングを担うクラス。"""

    params: dict
    camera_params: dict
    consts: dict
    render_fn: Callable = field(init=False)

    def __post_init__(self):
        """JITコンパイル済みのレンダリング関数を生成する。"""
        self.render_fn = make_render(self.consts, jit=True)

    def render(self, view_params: dict) -> np.ndarray:
        """指定された視点から画像をレンダリングする。"""
        return np.asarray(self.render_fn(self.params, view_params))


class Viewer:
    """GLFWウィンドウ、ユーザー入力、レンダリングループを管理するメインクラス。"""

    VERTEX_SHADER = """
        #version 330
        in vec2 in_position; in vec2 in_texcoord_0; out vec2 v_uv;
        void main() { gl_Position = vec4(in_position, 0.0, 1.0); v_uv = vec2(in_texcoord_0.x, 1.0 - in_texcoord_0.y); }
    """
    FRAGMENT_SHADER = """
        #version 330
        uniform sampler2D u_texture; in vec2 v_uv; out vec4 f_color;
        void main() { f_color = vec4(texture(u_texture, v_uv).rgb, 1.0); }
    """
    MOUSE_SENSITIVITY_ORBIT = 0.2
    MOUSE_SENSITIVITY_PAN = 0.002
    MOUSE_SENSITIVITY_ZOOM = 0.3

    def __init__(self, pkl_files: list[Path], initial_data: dict, initial_index: int):
        self.pkl_files = pkl_files
        self.current_data_index = initial_index
        self.params_cache = {initial_index: initial_data["params"]}

        # 不変のパラメータを初回ロード時に設定
        self.camera_params = initial_data["camera_params"]
        self.consts = initial_data["consts"]
        self.render_width, self.render_height = self.consts["img_shape"][::-1]

        # 初期化処理
        self._init_glfw()
        self._init_moderngl()

        self.renderer = GaussianRenderer(initial_data["params"], self.camera_params, self.consts)
        self.camera = self._create_initial_camera()
        self._setup_callbacks()

        # 状態変数の初期化
        self.left_mouse_dragging = False
        self.right_mouse_dragging = False
        self.last_mouse_pos = None
        self.camera_dirty = True

    def _init_glfw(self):
        """GLFWとウィンドウを初期化する。"""
        if not glfw.init():
            sys.exit("FATAL ERROR: glfw initialization failed.")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(
            self.render_width, self.render_height, "Viewer", None, None
        )
        if not self.window:
            glfw.terminate()
            sys.exit("FATAL ERROR: Failed to create glfw window.")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.update_window_title()

    def _init_moderngl(self):
        """ModernGLコンテキストと描画オブジェクトを初期化する。"""
        self.ctx = moderngl.create_context(require=330)
        self.program = self.ctx.program(
            vertex_shader=self.VERTEX_SHADER, fragment_shader=self.FRAGMENT_SHADER
        )
        self.image_texture = self.ctx.texture(
            (self.render_width, self.render_height), 3, dtype="f4"
        )

        quad_buffer = self.ctx.buffer(
            np.array(
                [
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    1.0,
                    -1.0,
                    1.0,
                    0.0,
                    -1.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                dtype="f4",
            )
        )
        self.quad_vao = self.ctx.vertex_array(
            self.program, [(quad_buffer, "2f 2f", "in_position", "in_texcoord_0")]
        )
        self.framebuffer_size_callback(self.window, self.render_width, self.render_height)

    def _setup_callbacks(self):
        """GLFWのイベントコールバックを設定する。"""
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
        glfw.set_key_callback(self.window, self.key_event)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)

    def _create_initial_camera(self) -> Camera:
        """最初の学習済みカメラ視点からCameraオブジェクトを生成する。"""
        self.num_cameras = len(self.camera_params["t_vec_batch"])
        self.current_cam_index = 0
        pos, rot = self._get_colmap_camera_state(self.current_cam_index)
        return Camera(position=pos, rotation=rot)

    def _get_colmap_camera_state(self, index: int) -> tuple[np.ndarray, Rotation]:
        """指定インデックスの学習済みカメラ情報を取得する。"""
        rot_mat_w2c = self.camera_params["rot_mat_batch"][index]
        t_vec_w2c = self.camera_params["t_vec_batch"][index]
        c2w_rotation = Rotation.from_matrix(rot_mat_w2c.T)
        c2w_position = -c2w_rotation.apply(t_vec_w2c)
        return c2w_position, c2w_rotation

    def load_params(self, index: int):
        """指定インデックスのparamsをロードし、レンダラを更新する。"""
        if index == self.current_data_index:
            return

        if index in self.params_cache:
            params = self.params_cache[index]
        else:
            filepath = self.pkl_files[index]
            print(f"Loading from disk: {filepath.name} ...", end="", flush=True)
            try:
                with filepath.open("rb") as f:
                    reconstruction = pickle.load(f)
                params = reconstruction["params"]
                self.params_cache[index] = params
                print(" Done.")
            except Exception as e:
                print(f" Error loading {filepath.name}: {e}", file=sys.stderr)
                return

        self.renderer.params = params
        self.current_data_index = index
        self.camera_dirty = True
        self.update_window_title()

    def update_window_title(self):
        """現在のファイル名と位置情報でウィンドウタイトルを更新する。"""
        filename = self.pkl_files[self.current_data_index].name
        k, n = self.current_data_index + 1, len(self.pkl_files)
        title = f"3D Gaussian Splatting Viewer ({filename} {k}/{n})"
        glfw.set_window_title(self.window, title)

    def run(self):
        """メインループを実行する。"""
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            if self.camera_dirty:
                rot_mat, t_vec = self.camera.get_view_matrix()
                intrinsic_vec = self.camera_params["intrinsic_batch"][self.current_cam_index]
                view_params = {"rot_mat": rot_mat, "t_vec": t_vec, "intrinsic_vec": intrinsic_vec}

                image_data = self.renderer.render(view_params)
                self.image_texture.write(image_data.astype("f4").tobytes())
                self.camera_dirty = False

            self.ctx.clear(0.0, 0.0, 0.0)
            self.image_texture.use(location=0)
            self.program["u_texture"].value = 0
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)
            glfw.swap_buffers(self.window)
        glfw.terminate()

    # --- Event Callbacks ---
    def framebuffer_size_callback(self, window, width, height):
        if height == 0:
            return
        aspect_ratio = self.render_width / self.render_height
        window_aspect_ratio = width / height
        if window_aspect_ratio > aspect_ratio:
            new_h, new_w = int(width / aspect_ratio), width
            offset_x, offset_y = 0, (height - new_h) // 2
        else:
            new_w, new_h = int(height * aspect_ratio), height
            offset_x, offset_y = (width - new_w) // 2, 0
        self.ctx.viewport = (offset_x, offset_y, new_w, new_h)
        self.camera_dirty = True

    def key_event(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_UP:
                self.load_params(
                    (self.current_data_index - 1 + len(self.pkl_files)) % len(self.pkl_files)
                )
            elif key == glfw.KEY_DOWN:
                self.load_params((self.current_data_index + 1) % len(self.pkl_files))

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
                self.camera.position, self.camera.rotation = self._get_colmap_camera_state(
                    self.current_cam_index
                )
                self.camera_dirty = True
                print(f"Jump to Camera: {self.current_cam_index + 1}/{self.num_cameras}")

    def mouse_button_callback(self, window, button, action, mods):
        if action == glfw.PRESS:
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.left_mouse_dragging = True
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.right_mouse_dragging = True
            self.last_mouse_pos = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            self.left_mouse_dragging = self.right_mouse_dragging = False
            self.last_mouse_pos = None

    def cursor_pos_callback(self, window, xpos, ypos):
        if (
            not (self.left_mouse_dragging or self.right_mouse_dragging)
            or self.last_mouse_pos is None
        ):
            return
        dx, dy = xpos - self.last_mouse_pos[0], ypos - self.last_mouse_pos[1]
        if self.left_mouse_dragging:
            self.camera.rotate(dx, dy, self.MOUSE_SENSITIVITY_ORBIT)
        if self.right_mouse_dragging:
            self.camera.pan(dx, dy, self.MOUSE_SENSITIVITY_PAN)
        self.camera_dirty = True
        self.last_mouse_pos = (xpos, ypos)

    def scroll_callback(self, window, xoffset, yoffset):
        self.camera.zoom(yoffset, self.MOUSE_SENSITIVITY_ZOOM)
        self.camera_dirty = True


def main():
    """アプリケーションのエントリーポイント。"""
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting Interactive Viewer")
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
        initial_index, initial_filepath = 0, pkl_files[0]

    print(f"Loading initial file: {initial_filepath.name}")
    try:
        with initial_filepath.open("rb") as f:
            initial_data = pickle.load(f)
    except Exception as e:
        sys.exit(f"FATAL ERROR: Could not load initial file {initial_filepath.name}: {e}")

    viewer = Viewer(pkl_files, initial_data, initial_index)
    viewer.run()


if __name__ == "__main__":
    main()
