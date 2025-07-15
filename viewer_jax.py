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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import glfw
import moderngl
import numpy as np
from scipy.spatial.transform import Rotation

from camera import JaxCamera
from data_manager import DataManager
from gs.make_update import make_render
from gs.utils import calc_tile_max_gs_num


@dataclass
class GaussianRenderer:
    """JAXによるガウシアンスプラッティングのレンダリングを担うクラス。"""

    params: dict
    consts: dict
    render_fn: Callable = field(init=False)

    def __post_init__(self):
        """JITコンパイル済みのレンダリング関数を生成する。"""
        self.render_fn = make_render(self.consts, jit=True)

    def render(self, view_params: dict) -> np.ndarray:
        """指定された視点パラメータから画像をレンダリングする。"""
        # JAXの関数は純粋関数であることが望ましいため、引数でパラメータを受け取る
        return np.asarray(self.render_fn(self.params, view_params))


class ViewerJax:
    """GLFWウィンドウ、ユーザー入力、レンダリングループを管理するメインクラス。"""

    # --- シェーダ定義 (画像をテクスチャとして描画するだけのシンプルなシェーダ) ---
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
        self.render_width, self.render_height = self.consts["img_shape"][::-1]

        # --- 初期化処理の実行 ---
        self._init_glfw()
        self._init_moderngl()
        self.renderer = GaussianRenderer(self.params, self.consts)
        self.camera = self._create_initial_camera()
        self._setup_callbacks()

        # --- 状態変数の初期化 ---
        self.left_mouse_dragging = False
        self.right_mouse_dragging = False
        self.middle_mouse_dragging = False
        self.last_mouse_pos = None
        self.camera_dirty = True  # 再描画が必要かどうかのフラグ

    def run(self):
        """メインループを実行する。"""
        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            if self.camera_dirty:
                # 1. カメラから視点パラメータを取得
                rot_mat, t_vec = self.camera.get_view_params()
                intrinsic_vec = self.camera_params["intrinsic_batch"][self.current_cam_index]
                view_params = {"rot_mat": rot_mat, "t_vec": t_vec, "intrinsic_vec": intrinsic_vec}

                # 2. JAXで画像をレンダリング
                image_data = self.renderer.render(view_params)

                # 3. 生成された画像をテクスチャに書き込み
                self.image_texture.write(image_data.astype("f4").tobytes())
                self.camera_dirty = False

            # 描画処理
            self.ctx.clear(0.0, 0.0, 0.0)
            self.image_texture.use(location=0)
            self.program["u_texture"].value = 0
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)
            glfw.swap_buffers(self.window)
        glfw.terminate()

    def _init_glfw(self):
        """GLFWとウィンドウを初期化する。"""
        if not glfw.init():
            sys.exit("FATAL ERROR: glfw initialization failed.")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(
            self.render_width, self.render_height, self.WINDOW_TITLE, None, None
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

        # JAXが生成した画像データを書き込むためのテクスチャ
        self.image_texture = self.ctx.texture(
            (self.render_width, self.render_height), 3, dtype="f4"
        )

        # テクスチャを描画するための四角形の頂点バッファ
        quad_buffer = self.ctx.buffer(
            np.array(
                [
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,  # Bottom Left
                    1.0,
                    -1.0,
                    1.0,
                    0.0,  # Bottom Right
                    -1.0,
                    1.0,
                    0.0,
                    1.0,  # Top Left
                    1.0,
                    1.0,
                    1.0,
                    1.0,  # Top Right
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
        # w2cからc2w(カメラからワールド)への変換
        c2w_rotation = Rotation.from_matrix(rot_mat_w2c.T)
        c2w_position = -c2w_rotation.apply(t_vec_w2c)
        return c2w_position, c2w_rotation

    def load_params(self, index: int):
        """指定インデックスのpklファイルをロードし、レンダラを更新する。"""
        self.params = self.data_manager.load_data(index)

        # レンダラーが保持するパラメータを更新
        self.renderer.params = self.params
        self.camera_dirty = True
        self.update_window_title()

    def update_window_title(self):
        """現在のファイル名と位置情報でウィンドウタイトルを更新する。"""
        filename = self.data_manager.get_current_filename()
        current_index, n_data = self.data_manager.get_current_data_index()
        title = f"{self.WINDOW_TITLE} ({filename} {current_index + 1}/{n_data})"
        glfw.set_window_title(self.window, title)

    # --- Event Callbacks ---
    def framebuffer_size_callback(self, window, width, height):
        """ウィンドウリサイズ時に呼び出され、アスペクト比を維持しつつウィンドウを覆うように表示（カバー）する。"""
        if height == 0:
            return

        aspect_ratio = self.render_width / self.render_height
        window_aspect_ratio = width / height

        # 元の画像の縦横比を保ちつつ、ウィンドウを覆うように描画領域を計算（カバー）
        if window_aspect_ratio > aspect_ratio:  # ウィンドウがコンテンツより横長
            # -> 高さを基準にスケールすると、幅がウィンドウより小さくなる
            # -> 幅を基準にスケールし、上下をクロップする
            new_w = width
            new_h = int(width / aspect_ratio)
            offset_x, offset_y = 0, (height - new_h) // 2
        else:  # ウィンドウがコンテンツより縦長
            # -> 幅を基準にスケールすると、高さがウィンドウより小さくなる
            # -> 高さを基準にスケールし、左右をクロップする
            new_h = height
            new_w = int(height * aspect_ratio)
            offset_x, offset_y = (width - new_w) // 2, 0

        self.ctx.viewport = (offset_x, offset_y, new_w, new_h)
        self.camera_dirty = True

    def key_event(self, window, key, scancode, action, mods):
        """キーボード入力イベントを処理する。"""
        # ファイル切り替え (Up/Down)
        if action == glfw.PRESS:
            if key == glfw.KEY_UP:
                current_index = self.data_manager.navigate(-1)
                self.load_params(current_index)
            elif key == glfw.KEY_DOWN:
                current_index = self.data_manager.navigate(1)
                self.load_params(current_index)

        # 学習済みカメラ視点切り替え (Left/Right)
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
                self.camera_dirty = True
                print(f"Jump to Camera: {self.current_cam_index + 1}/{self.num_cameras}", end="\r")

    def change_camera_pose(self):
        self.camera.position, self.camera.rotation = self._get_camera_state(self.current_cam_index)

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
            not (
                self.left_mouse_dragging or self.right_mouse_dragging or self.middle_mouse_dragging
            )
            or self.last_mouse_pos is None
        ):
            return

        dx, dy = xpos - self.last_mouse_pos[0], ypos - self.last_mouse_pos[1]
        if self.left_mouse_dragging:  # 左ドラッグでオービット
            self.camera.rotate(dx, dy, self.MOUSE_SENSITIVITY_ORBIT)
        if self.right_mouse_dragging:  # 右ドラッグでパン
            self.camera.pan(dx, dy, self.MOUSE_SENSITIVITY_PAN)
        if self.middle_mouse_dragging:  # 中ドラッグでズーム
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

    # 指定されたファイルと同じディレクトリにある全ての.pklファイルを探索
    directory = args.params_filepath.parent
    pkl_files = sorted(directory.glob("*.pkl"))
    if not pkl_files:
        sys.exit(f"Error: No .pkl files found in {directory}")

    # 初期ファイルのインデックスを特定
    try:
        initial_filepath = args.params_filepath.resolve()
        initial_index = [p.resolve() for p in pkl_files].index(initial_filepath)
    except ValueError:
        print(
            f"Warning: Specified file {args.params_filepath} not found. Starting with the first one."
        )
        initial_index = 0

    # ビューアを起動
    data_manager = DataManager(pkl_files)
    viewer = ViewerJax(data_manager, initial_index)
    viewer.run()


if __name__ == "__main__":
    main()
