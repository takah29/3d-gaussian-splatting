import argparse
import pickle
import sys
from pathlib import Path

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
        rot_mat = self.rotation.inv().as_matrix()
        t_vec = -rot_mat @ self.position
        return rot_mat, t_vec

    def rotate(self, dx: float, dy: float, sensitivity: float):
        """現在の回転に対して、相対的な視点回転を行う"""
        rot_y = Rotation.from_rotvec(np.radians(-dx * sensitivity) * np.array([0, 1, 0]))
        rot_x = Rotation.from_rotvec(np.radians(dy * sensitivity) * np.array([1, 0, 0]))
        self.rotation = self.rotation * rot_y * rot_x

    def pan(self, dx: float, dy: float, sensitivity: float):
        """現在の向きを基準に、相対的なパン操作を行う"""
        pan_vector = np.array([-dx, -dy, 0]) * sensitivity
        self.position += self.rotation.apply(pan_vector)

    def zoom(self, delta: float, sensitivity: float):
        """現在の向きを基準に、相対的なズーム操作を行う"""
        zoom_vector = np.array([0, 0, delta]) * sensitivity
        self.position += self.rotation.apply(zoom_vector)


class GaussianRenderer:
    def __init__(self, params, camera_params, consts):
        self.params = params
        self.consts = consts
        self.camera_params = camera_params
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

    def __init__(self, pkl_files, initial_data, initial_index):
        if not glfw.init():
            sys.exit("FATAL ERROR: glfwの初期化に失敗しました。")

        self.pkl_files = pkl_files
        self.current_data_index = initial_index
        # キャッシュにはparamsのみを保存
        self.params_cache = {initial_index: initial_data["params"]}

        # --- 初回ロード時に不変のパラメータを設定 ---
        params = initial_data["params"]
        self.camera_params = initial_data["camera_params"]
        self.consts = initial_data["consts"]

        self.render_width, self.render_height = self.consts["img_shape"][::-1]
        self.render_aspect_ratio = self.render_width / self.render_height

        # --- ウィンドウとOpenGLコンテキストの初期化 ---
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(
            self.render_width, self.render_height, "3D Gaussian Splatting Viewer", None, None
        )
        if not self.window:
            glfw.terminate()
            sys.exit("FATAL ERROR: glfwウィンドウの作成に失敗しました。")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.ctx = moderngl.create_context(require=330)

        # --- イベントコールバックの設定 ---
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
        glfw.set_key_callback(self.window, self.key_event)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)

        # --- レンダラとカメラの初期化 ---
        self.renderer = GaussianRenderer(params, self.camera_params, self.consts)
        self.num_cameras = len(self.camera_params["t_vec_batch"])
        self.current_cam_index = 0

        initial_pos, initial_rot = self.get_colmap_camera_state(self.current_cam_index)
        self.camera = Camera(initial_pos, initial_rot)

        # --- マウスと描画状態の初期化 ---
        self.left_mouse_dragging = False
        self.right_mouse_dragging = False
        self.last_mouse_pos = None
        self.mouse_sensitivity_orbit = 0.2
        self.mouse_sensitivity_pan = 0.002
        self.mouse_sensitivity_zoom = 0.1
        self.camera_dirty = True

        # --- OpenGL描画オブジェクトの初期化 ---
        self.image_texture = self.ctx.texture(
            (self.render_width, self.render_height), 3, dtype="f4"
        )
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

        self.framebuffer_size_callback(self.window, self.render_width, self.render_height)

    def load_data(self, index):
        """指定されたインデックスのparamsをロードし、レンダラを更新する"""
        if index == self.current_data_index:
            return

        # --- キャッシュからparamsを取得、なければディスクからロード ---
        if index in self.params_cache:
            params = self.params_cache[index]
            print(f"Loading from cache: {self.pkl_files[index].name}")
        else:
            filepath = self.pkl_files[index]
            print(f"Loading from disk: {filepath.name} ...", end="", flush=True)
            try:
                with filepath.open("rb") as f:
                    reconstruction = pickle.load(f)
                params = reconstruction["params"]
                self.params_cache[index] = params  # paramsのみをキャッシュに保存
                print(" Done.")
            except Exception as e:
                print(f" Error loading {filepath.name}: {e}", file=sys.stderr)
                return

        # --- paramsをレンダラにセット ---
        self.renderer.params = params
        self.current_data_index = index
        self.camera_dirty = True
        print(
            f"Switched to: {self.pkl_files[self.current_data_index].name} ({self.current_data_index + 1}/{len(self.pkl_files)})"
        )

    def framebuffer_size_callback(self, window, width, height):
        if height == 0:
            return
        window_aspect_ratio = width / height
        if window_aspect_ratio > self.render_aspect_ratio:
            new_height = int(width / self.render_aspect_ratio)
            new_width = width
            x_offset = 0
            y_offset = (height - new_height) // 2
        else:
            new_width = int(height * self.render_aspect_ratio)
            new_height = height
            x_offset = (width - new_width) // 2
            y_offset = 0
        self.ctx.viewport = (x_offset, y_offset, new_width, new_height)
        self.camera_dirty = True

    def get_colmap_camera_state(self, index: int) -> tuple[np.ndarray, Rotation]:
        rot_mat_w2c = self.camera_params["rot_mat_batch"][index]
        t_vec_w2c = self.camera_params["t_vec_batch"][index]
        c2w_rotation = Rotation.from_matrix(rot_mat_w2c.T)
        c2w_position = -c2w_rotation.apply(t_vec_w2c)
        return c2w_position, c2w_rotation

    def key_event(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_UP:
                new_index = (self.current_data_index - 1 + len(self.pkl_files)) % len(
                    self.pkl_files
                )
                self.load_data(new_index)
            elif key == glfw.KEY_DOWN:
                new_index = (self.current_data_index + 1) % len(self.pkl_files)
                self.load_data(new_index)

        if action == glfw.PRESS or action == glfw.REPEAT:
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
        dx, dy = xpos - self.last_mouse_pos[0], ypos - self.last_mouse_pos[1]
        if self.left_mouse_dragging:
            self.camera.rotate(dx, dy, self.mouse_sensitivity_orbit)
        if self.right_mouse_dragging:
            self.camera.pan(dx, dy, self.mouse_sensitivity_pan)
        self.camera_dirty = True
        self.last_mouse_pos = (xpos, ypos)

    def scroll_callback(self, window, xoffset, yoffset):
        self.camera.zoom(yoffset, self.mouse_sensitivity_zoom)
        self.camera_dirty = True

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            if self.camera_dirty:
                rot_mat, t_vec = self.camera.get_view_matrix()
                intrinsic_vec = self.camera_params["intrinsic_batch"][self.current_cam_index]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--params_filepath",
        type=Path,
        default=Path(__file__).parent / "output" / "params_final.pkl",
        help="Path to the params pickle file",
    )
    args = parser.parse_args()

    directory = args.params_filepath.parent
    pkl_files = sorted(list(directory.glob("*.pkl")))

    if not pkl_files:
        print(f"Error: No .pkl files found in {directory}", file=sys.stderr)
        sys.exit(1)

    try:
        initial_filepath = args.params_filepath.resolve()
        initial_index = [p.resolve() for p in pkl_files].index(initial_filepath)
    except ValueError:
        print(
            f"Warning: Specified file {args.params_filepath} not in the list. Starting with the first one."
        )
        initial_index = 0
        initial_filepath = pkl_files[0]

    print(f"Loading initial file: {initial_filepath.name}")
    try:
        with initial_filepath.open("rb") as f:
            initial_data = pickle.load(f)
    except Exception as e:
        print(
            f"FATAL ERROR: Could not load initial file {initial_filepath.name}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    viewer = Viewer(pkl_files, initial_data, initial_index)
    viewer.run()


if __name__ == "__main__":
    main()
