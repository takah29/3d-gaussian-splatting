import pickle
import sys

import glfw
import jax
import jax.numpy as jnp
import moderngl
import numpy as np

from gs.make_update import make_render


class GaussianSplattingViewer:
    """学習済みカメラパラメータを直接利用し、描画の上下反転を修正した最終版。
    ホイールでカメラ切り替え、ドラッグでパン操作。
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

        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)

        self.mouse_dragging = False
        self.last_mouse_pos = None

        self.camera_dirty = True
        try:
            with open("reconstructed.pkl", "rb") as f:
                reconstruction = pickle.load(f)
        except FileNotFoundError:
            print("エラー: 'reconstructed.pkl' が見つかりません。")
            sys.exit(1)

        self.params = jax.tree_util.tree_map(jnp.array, reconstruction["params"])
        self.consts = reconstruction["consts"]
        self.camera_params = reconstruction["camera_params"]
        self.num_cameras = len(self.camera_params["t_vec_batch"])
        self.current_cam_index = 0
        self.pan_offset = np.zeros(3)

        self.image_width, self.image_height = window_size
        self.render_fn = make_render(self.consts, jit=True)

        self.image_texture = self.ctx.texture((self.image_width, self.image_height), 3, dtype="f4")
        self.program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_position;
                in vec2 in_texcoord_0;
                out vec2 v_uv;
                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                    // --- [最重要修正] Y座標(v)を反転させて、上下を正しく描画する ---
                    v_uv = vec2(in_texcoord_0.x, 1.0 - in_texcoord_0.y);
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D u_texture;
                in vec2 v_uv;
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

    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.mouse_dragging = True
                self.last_mouse_pos = glfw.get_cursor_pos(self.window)
            elif action == glfw.RELEASE:
                self.mouse_dragging = False

    def cursor_pos_callback(self, window, xpos, ypos):
        if self.mouse_dragging:
            if self.last_mouse_pos is None:
                self.last_mouse_pos = (xpos, ypos)
                return

            dx = xpos - self.last_mouse_pos[0]
            dy = ypos - self.last_mouse_pos[1]

            rot_mat = self.camera_params["rot_mat_batch"][self.current_cam_index]
            c2w_rot = rot_mat.T

            move_speed = 0.005
            pan_vec = np.array([-dx * move_speed, -dy * move_speed, 0])
            self.pan_offset += c2w_rot @ pan_vec

            self.camera_dirty = True
            self.last_mouse_pos = (xpos, ypos)

    def scroll_callback(self, window, xoffset, yoffset):
        if yoffset > 0:
            self.current_cam_index = (self.current_cam_index + 1) % self.num_cameras
        elif yoffset < 0:
            self.current_cam_index = (
                self.current_cam_index - 1 + self.num_cameras
            ) % self.num_cameras

        print(f"Switching to Camera: {self.current_cam_index + 1} / {self.num_cameras}")
        self.pan_offset = np.zeros(3)
        self.camera_dirty = True

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            if self.camera_dirty:
                rot_mat = self.camera_params["rot_mat_batch"][self.current_cam_index]
                t_vec = self.camera_params["t_vec_batch"][self.current_cam_index]
                intrinsic_vec = self.camera_params["intrinsic_batch"][self.current_cam_index]

                t_vec_panned = t_vec - rot_mat @ self.pan_offset

                view = {"rot_mat": rot_mat, "t_vec": t_vec_panned, "intrinsic_vec": intrinsic_vec}

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

    viewer = GaussianSplattingViewer(window_size=(w, h))
    viewer.run()
