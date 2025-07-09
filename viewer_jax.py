import pickle
import sys

import glfw
import moderngl
import numpy as np
from scipy.spatial.transform import Rotation

from gs.make_update import make_render


class Camera:
    def __init__(self, initial_pos: np.ndarray, initial_rot: Rotation):
        self.position = initial_pos
        self.rotation = initial_rot

    def get_view_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """現在の位置と回転からw2cビュー行列を計算する"""
        # w2c回転 = c2w回転の逆行列
        rot_mat = self.rotation.inv().as_matrix()
        # w2c移動 = -(w2c回転 @ c2w位置)
        t_vec = -rot_mat @ self.position
        return rot_mat, t_vec

    def rotate(self, dx: float, dy: float, sensitivity: float):
        """現在の回転に対して、相対的な視点回転を行う"""
        # カメラのローカルY軸（下方向）とX軸（右方向）を回転軸とする
        rot_y = Rotation.from_rotvec(np.radians(-dx * sensitivity) * np.array([0, 1, 0]))
        rot_x = Rotation.from_rotvec(np.radians(dy * sensitivity) * np.array([1, 0, 0]))
        # 現在の回転に、ローカルな回転を掛け合わせる
        self.rotation = self.rotation * rot_y * rot_x

    def pan(self, dx: float, dy: float, sensitivity: float):
        """現在の向きを基準に、相対的なパン操作を行う"""
        # カメラのローカルなX, Yベクトルに沿って位置を動かす
        pan_vector = np.array([-dx, -dy, 0]) * sensitivity
        self.position += self.rotation.apply(pan_vector)

    def zoom(self, delta: float, sensitivity: float):
        """現在の向きを基準に、相対的なズーム操作を行う"""
        # カメラのローカルなZベクトル（奥行き）に沿って位置を動かす
        zoom_vector = np.array([0, 0, delta]) * sensitivity
        self.position += self.rotation.apply(zoom_vector)


class GaussianRenderer:
    def __init__(self, pkl_path: str):
        try:
            with open(pkl_path, "rb") as f:
                reconstruction = pickle.load(f)
        except FileNotFoundError:
            print(f"エラー: '{pkl_path}' が見つかりません。")
            sys.exit(1)

        self.params = reconstruction["params"]
        self.consts = reconstruction["consts"]
        self.camera_params = reconstruction["camera_params"]
        self.render_fn = make_render(self.consts, jit=True)

    def render(self, view_params: dict) -> np.ndarray:
        """指定された視点から画像をレンダリングする"""
        rendered_image = self.render_fn(self.params, view_params)
        return np.asarray(rendered_image)


class Viewer:
    VERTEX_SHADER = """
        #version 330
        in vec2 in_position;
        in vec2 in_texcoord_0;
        out vec2 v_uv;
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            v_uv = vec2(in_texcoord_0.x, 1.0 - in_texcoord_0.y);
        }
    """
    FRAGMENT_SHADER = """
        #version 330
        uniform sampler2D u_texture;
        in vec2 v_uv;
        out vec4 f_color;
        void main() {
            f_color = vec4(texture(u_texture, v_uv).rgb, 1.0);
        }
    """

    def __init__(self, window_size=(980, 545), pkl_path="reconstructed.pkl"):
        # --- ウィンドウとコンテキストの初期化 ---
        if not glfw.init():
            sys.exit("FATAL ERROR: glfwの初期化に失敗しました。")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(
            window_size[0], window_size[1], "3D Gaussian Splatting Viewer (Refactored)", None, None
        )
        if not self.window:
            glfw.terminate()
            sys.exit("FATAL ERROR: glfwウィンドウの作成に失敗しました。")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.ctx = moderngl.create_context(require=330)

        # --- イベントハンドラと状態変数の設定 ---
        glfw.set_key_callback(self.window, self.key_event)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)

        self.left_mouse_dragging = False
        self.right_mouse_dragging = False
        self.last_mouse_pos = None
        self.camera_dirty = True

        # --- 専門クラスのインスタンス化 ---
        self.renderer = GaussianRenderer(pkl_path)
        self.num_cameras = len(self.renderer.camera_params["t_vec_batch"])
        self.current_cam_index = 0

        # --- カメラの初期化 ---
        initial_pos, initial_rot = self.get_colmap_camera_state(0)
        self.camera = Camera(initial_pos, initial_rot)
        self.mouse_sensitivity_orbit = 0.2
        self.mouse_sensitivity_pan = 0.002
        self.mouse_sensitivity_zoom = 0.1

        # --- OpenGLリソースの準備 ---
        self.image_texture = self.ctx.texture(window_size, 3, dtype="f4")
        self.program = self.ctx.program(
            vertex_shader=self.VERTEX_SHADER, fragment_shader=self.FRAGMENT_SHADER
        )
        quad_buffer = np.array(
            [-1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype="f4",
        )
        vbo = self.ctx.buffer(quad_buffer)
        self.quad_vao = self.ctx.vertex_array(
            self.program, [(vbo, "2f 2f", "in_position", "in_texcoord_0")]
        )

    def get_colmap_camera_state(self, index: int) -> tuple[np.ndarray, Rotation]:
        """指定されたCOLMAPカメラ(w2c)から、c2wの状態を逆算して返す"""
        rot_mat_w2c = self.renderer.camera_params["rot_mat_batch"][index]
        t_vec_w2c = self.renderer.camera_params["t_vec_batch"][index]

        c2w_rotation = Rotation.from_matrix(rot_mat_w2c.T)
        c2w_position = -c2w_rotation.apply(t_vec_w2c)
        return c2w_position, c2w_rotation

    def key_event(self, window, key, scancode, action, mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_RIGHT:
                self.current_cam_index = (self.current_cam_index + 1) % self.num_cameras
            elif key == glfw.KEY_LEFT:
                self.current_cam_index = (
                    self.current_cam_index - 1 + self.num_cameras
                ) % self.num_cameras

            # カメラをリセット
            pos, rot = self.get_colmap_camera_state(self.current_cam_index)
            self.camera.position = pos
            self.camera.rotation = rot
            self.camera_dirty = True
            print(f"Jump to Camera: {self.current_cam_index + 1} / {self.num_cameras}")

    def mouse_button_callback(self, window, button, action, mods):
        if action == glfw.PRESS:
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.left_mouse_dragging = True
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.right_mouse_dragging = True
            self.last_mouse_pos = glfw.get_cursor_pos(self.window)
        elif action == glfw.RELEASE:
            self.left_mouse_dragging = False
            self.right_mouse_dragging = False
            self.last_mouse_pos = None

    def cursor_pos_callback(self, window, xpos, ypos):
        if not (self.left_mouse_dragging or self.right_mouse_dragging):
            return
        if self.last_mouse_pos is None:
            return

        dx = xpos - self.last_mouse_pos[0]
        dy = ypos - self.last_mouse_pos[1]

        if self.left_mouse_dragging:
            self.camera.rotate(dx, dy, self.mouse_sensitivity_orbit)
            self.camera_dirty = True

        if self.right_mouse_dragging:
            self.camera.pan(dx, dy, self.mouse_sensitivity_pan)
            self.camera_dirty = True

        self.last_mouse_pos = (xpos, ypos)

    def scroll_callback(self, window, xoffset, yoffset):
        self.camera.zoom(yoffset, self.mouse_sensitivity_zoom)
        self.camera_dirty = True

    def run(self):
        """メインループ"""
        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            if self.camera_dirty:
                rot_mat, t_vec = self.camera.get_view_matrix()
                intrinsic_vec = self.renderer.camera_params["intrinsic_batch"][
                    self.current_cam_index
                ]
                view = {"rot_mat": rot_mat, "t_vec": t_vec, "intrinsic_vec": intrinsic_vec}

                image_data = self.renderer.render(view)
                self.image_texture.write(image_data.astype("f4").tobytes())
                self.camera_dirty = False

            self.ctx.clear(0.0, 0.0, 0.0)
            self.image_texture.use(location=0)
            self.program["u_texture"].value = 0
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)

            glfw.swap_buffers(self.window)

        glfw.terminate()


if __name__ == "__main__":
    PKL_FILE_PATH = "reconstructed.pkl"
    try:
        with open(PKL_FILE_PATH, "rb") as f:
            reconstruction = pickle.load(f)
        h, w = reconstruction["consts"]["img_shape"]
    except Exception:
        print(
            f"警告: '{PKL_FILE_PATH}' からウィンドウサイズを読み込めませんでした。デフォルト値を使用します。"
        )
        w, h = 980, 545

    viewer = Viewer(window_size=(w, h), pkl_path=PKL_FILE_PATH)
    viewer.run()
