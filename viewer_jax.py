import pickle
import sys

import glfw
import moderngl
import numpy as np
from scipy.spatial.transform import Rotation

from gs.make_update import make_render


class GaussianSplattingViewer:
    """指定された全仕様を実装した最終版ビューア。
    - カメラ状態を位置ベクトルと回転行列で直接管理
    - 左ドラッグ: 相対的な視点回転（Y軸反転修正済み）
    - 右ドラッグ: 相対的なパン
    - ホイール: 相対的なズーム
    - 左右キー: 学習済みカメラ位置へジャンプ
    """

    def __init__(self, window_size=(980, 545)):
        if not glfw.init():
            sys.exit("FATAL ERROR: glfwの初期化に失敗しました。")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(
            window_size[0], window_size[1], "3D Gaussian Splatting Viewer", None, None
        )
        if not self.window:
            glfw.terminate()
            sys.exit("FATAL ERROR: glfwウィンドウの作成に失敗しました。")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.ctx = moderngl.create_context(require=330)

        glfw.set_key_callback(self.window, self.key_event)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)

        self.left_mouse_dragging = False
        self.right_mouse_dragging = False
        self.last_mouse_pos = None
        self.camera_dirty = True

        try:
            with open("reconstructed.pkl", "rb") as f:
                reconstruction = pickle.load(f)
        except FileNotFoundError:
            print("エラー: 'reconstructed.pkl' が見つかりません。")
            sys.exit(1)

        self.params = reconstruction["params"]
        self.consts = reconstruction["consts"]
        self.camera_params = reconstruction["camera_params"]
        self.num_cameras = len(self.camera_params["t_vec_batch"])
        self.current_cam_index = 0

        self.image_width, self.image_height = window_size
        self.render_fn = make_render(self.consts, jit=True)

        self.mouse_sensitivity_orbit = 0.2
        self.mouse_sensitivity_pan = 0.002
        self.mouse_sensitivity_zoom = 0.1

        self.cam_pos = np.zeros(3)
        self.cam_rot = Rotation.identity()
        self.reset_camera_to_colmap(0)

        self.image_texture = self.ctx.texture((self.image_width, self.image_height), 3, dtype="f4")
        self.program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_position; in vec2 in_texcoord_0;
                out vec2 v_uv;
                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                    v_uv = vec2(in_texcoord_0.x, 1.0 - in_texcoord_0.y);
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D u_texture; in vec2 v_uv;
                out vec4 f_color;
                void main() {
                    f_color = vec4(texture(u_texture, v_uv).rgb, 1.0);
                }
            """,
        )
        quad_buffer = np.array(
            [-1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype="f4",
        )
        vbo = self.ctx.buffer(quad_buffer)
        self.quad_vao = self.ctx.vertex_array(
            self.program, [(vbo, "2f 2f", "in_position", "in_texcoord_0")]
        )

    def reset_camera_to_colmap(self, index):
        rot_mat_w2c = self.camera_params["rot_mat_batch"][index]
        t_vec_w2c = self.camera_params["t_vec_batch"][index]

        self.cam_rot = Rotation.from_matrix(rot_mat_w2c.T)
        self.cam_pos = -self.cam_rot.apply(t_vec_w2c)
        print(f"Jump to Camera: {index + 1} / {self.num_cameras}")

    def key_event(self, window, key, scancode, action, mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_RIGHT:
                self.current_cam_index = (self.current_cam_index + 1) % self.num_cameras
                self.reset_camera_to_colmap(self.current_cam_index)
                self.camera_dirty = True
            elif key == glfw.KEY_LEFT:
                self.current_cam_index = (
                    self.current_cam_index - 1 + self.num_cameras
                ) % self.num_cameras
                self.reset_camera_to_colmap(self.current_cam_index)
                self.camera_dirty = True

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
            rot_y = Rotation.from_rotvec(
                np.radians(-dx * self.mouse_sensitivity_orbit) * np.array([0, 1, 0])
            )
            rot_x = Rotation.from_rotvec(
                np.radians(dy * self.mouse_sensitivity_orbit) * np.array([1, 0, 0])
            )
            self.cam_rot = self.cam_rot * rot_y * rot_x
            self.camera_dirty = True

        if self.right_mouse_dragging:
            self.cam_pos += self.cam_rot.apply(np.array([-dx, -dy, 0]) * self.mouse_sensitivity_pan)
            self.camera_dirty = True

        self.last_mouse_pos = (xpos, ypos)

    def scroll_callback(self, window, xoffset, yoffset):
        self.cam_pos += self.cam_rot.apply(np.array([0, 0, yoffset]) * self.mouse_sensitivity_zoom)
        self.camera_dirty = True

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            if self.camera_dirty:
                rot_mat = self.cam_rot.inv().as_matrix()
                t_vec = -rot_mat @ self.cam_pos

                intrinsic_vec = self.camera_params["intrinsic_batch"][self.current_cam_index]
                view = {"rot_mat": rot_mat, "t_vec": t_vec, "intrinsic_vec": intrinsic_vec}

                rendered_image = self.render_fn(self.params, view)
                image_np = np.asarray(rendered_image)
                self.image_texture.write(image_np.astype("f4").tobytes())

                self.camera_dirty = False

            self.ctx.clear(0.0, 0.0, 0.0)
            self.image_texture.use(location=0)
            self.program["u_texture"].value = 0
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)

            glfw.swap_buffers(self.window)

        glfw.terminate()


if __name__ == "__main__":
    try:
        with open("reconstructed.pkl", "rb") as f:
            reconstruction = pickle.load(f)
        h, w = reconstruction["consts"]["img_shape"]
    except Exception:
        w, h = 980, 545

    help_msg = """Gaussian spattering viewer.
    - Directly manages camera state using position vector and rotation matrix
    - Left drag: Relative viewpoint rotation (Y-axis inversion correction implemented)
    - Right drag: Relative panning
    - Wheel: Relative zooming
    - Left/Right arrow keys: Jump to learned camera positions
    """

    print(help_msg)
    viewer = GaussianSplattingViewer(window_size=(w, h))
    viewer.run()
