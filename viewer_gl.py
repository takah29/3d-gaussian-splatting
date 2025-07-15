import argparse
import sys
from pathlib import Path

import glfw
import jax
import jax.numpy as jnp
import moderngl
import numpy as np
import pyglm.glm as pglm
from scipy.spatial.transform import Rotation

from camera import GlCamera
from data_manager import DataManager


class ViewerGl:
    """GLFWウィンドウ、ユーザー入力、OpenGLレンダリングループを管理するメインクラス。"""

    # --- シェーダ定義 ---
    VERTEX_SHADER = """
        #version 430 core
        layout(location = 0) in vec2 in_vert;
        // SSBO bindings
        layout (std430, binding = 0) buffer gaussian_data { float g_data[]; };
        layout (std430, binding = 1) buffer gaussian_order { int gi[]; };
        // Uniforms
        uniform mat4 view_matrix;
        uniform mat4 projection_matrix;
        uniform float focal_x;
        uniform float focal_y;
        uniform vec2 u_resolution;
        // Outputs to Fragment Shader
        out vec3 pass_color;
        out float pass_alpha;
        out vec3 pass_conic;
        out vec2 pass_coordxy;

        mat3 quat_to_rotmat(vec4 q) {
            float x = q.x, y = q.y, z = q.z, w = q.w;
            return mat3(
                1.f - 2.f * (y * y + z * z), 2.f * (x * y + w * z), 2.f * (x * z - w * y),
                2.f * (x * y - w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + w * x),
                2.f * (x * z + w * y), 2.f * (y * z - w * x), 1.f - 2.f * (x * x + y * y)
            );
        }

        void main() {
            int instance_idx = gi[gl_InstanceID];
            int data_dim = 14; // (pos:3, rot:4, scale:3, opacity:1, color:3)
            int offset = instance_idx * data_dim;

            vec3 pos_world = vec3(g_data[offset], g_data[offset+1], g_data[offset+2]);
            vec4 rot_quat = vec4(g_data[offset+3], g_data[offset+4], g_data[offset+5], g_data[offset+6]);
            vec3 scale = vec3(g_data[offset+7], g_data[offset+8], g_data[offset+9]);
            pass_alpha = g_data[offset+10];
            pass_color = vec3(g_data[offset+11], g_data[offset+12], g_data[offset+13]);

            vec4 pos_view = view_matrix * vec4(pos_world, 1.0);
            if (pos_view.z >= -0.2) { // 近すぎるガウシアンは描画しない
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // 画面外に飛ばす
                return;
            }

            mat3 R = quat_to_rotmat(rot_quat);
            mat3 S = mat3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
            mat3 cov3D_world = R * S * transpose(S) * transpose(R);

            mat3 W = mat3(view_matrix);
            mat3 cov3D_view = W * cov3D_world * transpose(W);

            float tan_fovx = 2.0 * atan(u_resolution.x / (2.0 * focal_x));
            float tan_fovy = 2.0 * atan(u_resolution.y / (2.0 * focal_y));
            float limx = 1.3 * tan_fovx;
            float limy = 1.3 * tan_fovy;

            float z = pos_view.z;
            float x_clipped = clamp(pos_view.x / z, -limx, limx) * z;
            float y_clipped = clamp(pos_view.y / z, -limy, limy) * z;

            mat3 J = mat3(
                focal_x / z, 0.0, 0.0,
                0.0, focal_y / z, 0.0,
                -focal_x * x_clipped / (z*z), -focal_y * y_clipped / (z*z), 0.0
            );

            mat3 cov2D = J * cov3D_view * transpose(J);
            cov2D[0][0] += 0.3;
            cov2D[1][1] += 0.3;

            float det = determinant(mat2(cov2D));
            if (det == 0.0) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                return;
            }

            float det_inv = 1.0 / det;
            pass_conic = vec3(cov2D[1][1] * det_inv, -cov2D[0][1] * det_inv, cov2D[0][0] * det_inv);

            float mid = 0.5 * (cov2D[0][0] + cov2D[1][1]);
            float lambda1 = mid + sqrt(max(0.1, mid*mid - det));
            float radius = 3.0 * sqrt(lambda1);

            vec2 quad_radius_ndc = radius / u_resolution * 2.0;
            vec4 pos_clip = projection_matrix * pos_view;
            gl_Position = pos_clip / pos_clip.w + vec4(in_vert * quad_radius_ndc, 0.0, 0.0);
            pass_coordxy = in_vert * radius;
        }
    """
    FRAGMENT_SHADER = """
        #version 430 core
        in vec3 pass_color;
        in float pass_alpha;
        in vec3 pass_conic;
        in vec2 pass_coordxy;
        out vec4 FragColor;

        void main() {
            float power = -0.5 * (pass_conic.x * pass_coordxy.x * pass_coordxy.x + pass_conic.z * pass_coordxy.y * pass_coordxy.y) - pass_conic.y * pass_coordxy.x * pass_coordxy.y;
            if (power > 0.0) discard;

            float G_alpha = min(0.99, pass_alpha * exp(power));
            if (G_alpha < 1.0/255.0) discard;

            FragColor = vec4(pass_color, G_alpha);
        }
    """

    # --- マウス感度設定 ---
    MOUSE_SENSITIVITY_ORBIT = 0.2
    MOUSE_SENSITIVITY_PAN = 0.002
    MOUSE_SENSITIVITY_ZOOM = 0.3
    MOUSE_SENSITIVITY_ROLL = 0.1
    WINDOW_TITLE = "OpenGL 3DGS Viewer"

    def __init__(self, data_manager: DataManager, initial_index: int):
        self.data_manager = data_manager
        self.params = self.data_manager.load_data(initial_index)
        self.camera_params, self.consts = self.data_manager.get_camera_params_and_consts()

        # --- パラメータの初期化 ---
        self.initial_width, self.initial_height = self.consts["img_shape"][::-1]
        intrinsic_vec = self.camera_params["intrinsic_batch"][0]
        self.initial_fx, self.initial_fy, _, _ = intrinsic_vec
        self.render_width, self.render_height = self.initial_width, self.initial_height
        self.current_fx, self.current_fy = self.initial_fx, self.initial_fy

        # --- 初期化処理の実行 ---
        self._init_glfw()
        self._init_moderngl()
        self.camera = self._create_initial_camera()
        self._setup_callbacks()

        # --- 状態変数の初期化 ---
        self.means3d_jax = jnp.asarray(self.params["means3d"])
        self.sorter = jax.jit(self._get_sorted_indices_jax)
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
                # 1. ビュー行列とプロジェクション行列の更新
                view_matrix = self.camera.get_view_matrix()
                fovy = 2 * np.arctan(self.initial_height / (2 * self.initial_fy))
                aspect = self.initial_width / self.initial_height
                projection_matrix = pglm.perspective(fovy, aspect, 0.1, 1000.0)

                # 2. ガウシアンのソート
                self._sort_gaussians(view_matrix)

                # 3. Uniform変数をシェーダに送信
                self.program["view_matrix"].write(pglm.mat4(view_matrix))
                self.program["projection_matrix"].write(projection_matrix)
                self.program["focal_x"].value = self.current_fx
                self.program["focal_y"].value = self.current_fy
                self.program["u_resolution"].value = (self.render_width, self.render_height)

                self.camera_dirty = False

            # 描画処理
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)
            if self.num_gaussians > 0:
                self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=self.num_gaussians)

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
            self.initial_width, self.initial_height, self.WINDOW_TITLE, None, None
        )
        if not self.window:
            glfw.terminate()
            sys.exit("FATAL ERROR: Failed to create glfw window.")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.update_window_title()

    def _init_moderngl(self):
        """ModernGLコンテキストと描画オブジェクトを初期化する。"""
        self.ctx = moderngl.create_context(require=430)
        self.program = self.ctx.program(
            vertex_shader=self.VERTEX_SHADER, fragment_shader=self.FRAGMENT_SHADER
        )

        # ブレンディング設定
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.disable(moderngl.DEPTH_TEST)

        self._init_buffers(self.params)

        # 描画対象の四角形（スプライト）の頂点バッファ
        quad_verts = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype="f4")
        vbo_quad = self.ctx.buffer(quad_verts)
        self.vao = self.ctx.vertex_array(self.program, [(vbo_quad, "2f", "in_vert")])

        width, height = glfw.get_framebuffer_size(self.window)
        self.framebuffer_size_callback(self.window, width, height)

    def _setup_callbacks(self):
        """GLFWのイベントコールバックを設定する。"""
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
        glfw.set_key_callback(self.window, self.key_event)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)

    def _create_initial_camera(self) -> GlCamera:
        """最初の学習済みカメラ視点からCameraオブジェクトを生成する。"""
        self.num_cameras = len(self.camera_params["t_vec_batch"])
        self.current_cam_index = 0
        pos, rot = self._get_camera_state(self.current_cam_index)
        return GlCamera(position=pos, rotation=rot)

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
        self._update_buffers(self.params)
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
        """ウィンドウリサイズ時に呼び出され、ビューポートと解像度関連の値を更新する。"""
        if width == 0 or height == 0:
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

        # 1. ビューポートを更新し、中央に配置
        self.ctx.viewport = (view_x, view_y, self.render_width, self.render_height)

        # 2. 新しい解像度に合わせて焦点距離をスケーリング
        scale = self.render_width / self.initial_width
        self.current_fx = self.initial_fx * scale
        self.current_fy = self.initial_fy * scale

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
        # 基準となる焦点距離を切り替え後のカメラのものに更新
        new_intrinsics = self.camera_params["intrinsic_batch"][self.current_cam_index]
        self.initial_fx, self.initial_fy, _, _ = new_intrinsics
        # 現在のウィンドウサイズに合わせてビューポートと焦点距離を再計算
        w, h = glfw.get_framebuffer_size(self.window)
        self.framebuffer_size_callback(self.window, w, h)
        # カメラの位置と回転を更新
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

    def _init_buffers(self, params: dict):
        """パラメータからSSBO(Shader Storage Buffer Object)を作成または更新する。"""
        self.num_gaussians = params["means3d"].shape[0]
        if "colors" not in params:
            raise KeyError("The provided .pkl file does not contain the required 'colors' key.")

        # シェーダに渡すデータを一つの配列にまとめる
        flat_data = np.concatenate(
            [
                params["means3d"].astype("f4"),
                params["quats"].astype("f4"),
                params["scales"].astype("f4"),
                params["opacities"].astype("f4"),
                params["colors"].astype("f4"),
            ],
            axis=1,
        ).ravel()

        self.ssbo_gaussians = self.ctx.buffer(flat_data.tobytes())
        initial_indices = np.arange(self.num_gaussians, dtype="i4")
        self.ssbo_indices = self.ctx.buffer(initial_indices.tobytes(), dynamic=True)

        self.ssbo_gaussians.bind_to_storage_buffer(0)
        self.ssbo_indices.bind_to_storage_buffer(1)

    def _update_buffers(self, params: dict):
        """既存のバッファを新しいパラメータで更新する。"""
        new_num_gaussians = params["means3d"].shape[0]
        if new_num_gaussians != self.num_gaussians:
            self._init_buffers(params)  # ガウシアン数が変わる場合はバッファを再生成
        else:
            if "colors" not in params:
                raise KeyError("The provided .pkl file does not contain the required 'colors' key.")

            flat_data = np.concatenate(
                [
                    params["means3d"].astype("f4"),
                    params["quats"].astype("f4"),
                    params["scales"].astype("f4"),
                    params["opacities"].astype("f4"),
                    params["colors"].astype("f4"),
                ],
                axis=1,
            ).ravel()
            self.ssbo_gaussians.write(flat_data.tobytes())

        self.means3d_jax = jnp.asarray(params["means3d"])

    @staticmethod
    def _get_sorted_indices_jax(means3d_jax, view_matrix_jax):
        """JAXを使用して、カメラ視点からの深度に基づいてガウシアンをソートする。"""
        means_homo = jnp.hstack([means3d_jax, jnp.ones((means3d_jax.shape[0], 1))])
        means_view = means_homo @ view_matrix_jax.T
        depths = means_view[:, 2]
        return jnp.argsort(depths)

    def _sort_gaussians(self, view_matrix: np.ndarray):
        """ガウシアンをソートし、インデックスバッファを更新する。"""
        sorted_indices_jax = self.sorter(self.means3d_jax, jnp.array(view_matrix))
        self.ssbo_indices.write(np.asarray(sorted_indices_jax).astype("i4").tobytes())


def main():
    """アプリケーションのエントリーポイント。"""
    parser = argparse.ArgumentParser(description="OpenGL 3D Gaussian Splatting Interactive Viewer")
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
    data_manager = DataManager.create_for_gldata(pkl_files)
    viewer = ViewerGl(data_manager, initial_index)
    viewer.run()


if __name__ == "__main__":
    main()
