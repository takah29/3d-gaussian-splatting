import argparse
import pickle
import sys
from pathlib import Path

import glfw
import jax
import jax.numpy as jnp
import moderngl
import numpy as np
import pyglm.glm as pglm
from scipy.spatial.transform import Rotation


class Camera:
    """カメラの位置と回転を管理し、ビュー行列を計算するクラス。(JAX版から流用・一部修正)"""

    def __init__(self, position: np.ndarray, rotation: Rotation):
        self.position = position
        self.rotation = rotation

    def get_view_matrix(self) -> np.ndarray:
        """現在の位置と回転からOpenGL互換のビュー行列(w2c)を計算する。"""
        rot_mat = self.rotation.as_matrix().T
        t_vec = -rot_mat @ self.position

        view_matrix = np.identity(4, dtype="f4")
        view_matrix[:3, :3] = rot_mat
        view_matrix[:3, 3] = t_vec
        return view_matrix

    def rotate(self, dx: float, dy: float, sensitivity: float):
        """現在の回転に対して、相対的な視点回転を行う (オービット操作)。"""
        # Y軸周りの回転 (水平方向) はワールド座標のY軸を基準にする
        rot_y = Rotation.from_rotvec(np.radians(dx * sensitivity) * np.array([0, 1, 0]))
        # X軸周りの回転 (垂直方向) はカメラのローカルX軸を基準にする
        rot_x = Rotation.from_rotvec(
            self.rotation.apply(np.radians(dy * sensitivity) * np.array([1, 0, 0]))
        )
        # より一般的なオービット操作の回転合成順序
        self.rotation = rot_x * self.rotation * rot_y

    def pan(self, dx: float, dy: float, sensitivity: float):
        """現在の向きを基準に、相対的なパン操作を行う。JAX版の仕様に統一。"""
        pan_vector = np.array([-dx, dy, 0]) * sensitivity
        self.position += self.rotation.apply(pan_vector)

    def zoom(self, delta: float, sensitivity: float):
        """現在の向きを基準に、相対的なズーム操作を行う。"""
        # yoffsetは上下で値の符号が異なるため、-deltaで方向を統一
        zoom_vector = np.array([0, 0, -delta]) * sensitivity
        self.position += self.rotation.apply(zoom_vector)


class Viewer:
    """GLFWウィンドウ、ユーザー入力、OpenGLレンダリングループを管理するメインクラス。"""

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

            // GLSL defines directly in column-major order
            return mat3(
                1.f - 2.f * (y * y + z * z), 2.f * (x * y + w * z), 2.f * (x * z - w * y),
                2.f * (x * y - w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + w * x),
                2.f * (x * z + w * y), 2.f * (y * z - w * x), 1.f - 2.f * (x * x + y * y)
            );
        }

        void main() {
            int instance_idx = gi[gl_InstanceID];
            int data_dim = 14;
            int offset = instance_idx * data_dim;

            vec3 pos_world = vec3(g_data[offset], g_data[offset+1], g_data[offset+2]);
            vec4 rot_quat = vec4(g_data[offset+3], g_data[offset+4], g_data[offset+5], g_data[offset+6]);
            vec3 scale = vec3(g_data[offset+7], g_data[offset+8], g_data[offset+9]);
            pass_alpha = g_data[offset+10];
            pass_color = vec3(g_data[offset+11], g_data[offset+12], g_data[offset+13]);

            vec4 pos_view = view_matrix * vec4(pos_world, 1.0);
            if (pos_view.z >= -0.2) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                return;
            }

            mat3 R = quat_to_rotmat(rot_quat);
            mat3 S = mat3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
            mat3 cov3D_world = R * S * transpose(S) * transpose(R);

            mat3 W = mat3(view_matrix);
            mat3 cov3D_view = W * cov3D_world * transpose(W);

            // クリッピング処理を追加
            float tan_fovx = 2.0 * atan(u_resolution.x / (2.0 * focal_x));
            float tan_fovy = 2.0 * atan(u_resolution.y / (2.0 * focal_y));
            float limx = 1.3 * tan_fovx;
            float limy = 1.3 * tan_fovy;

            float z = pos_view.z;
            float x_clipped = clamp(pos_view.x / z, -limx, limx) * z;
            float y_clipped = clamp(pos_view.y / z, -limy, limy) * z;

            // 更新されたJacobian行列
            mat3 J = mat3(
                    focal_x / z, 0.0, 0.0,                    // 第1列
                    0.0, focal_y / z, 0.0,                    // 第2列
                    -focal_x * x_clipped / (z*z), -focal_y * y_clipped / (z*z), 0.0  // 第3列
                );

            mat3 cov2D = J * cov3D_view * transpose(J);
            cov2D[0][0] += 0.3;
            cov2D[1][1] += 0.3;

            float det = cov2D[0][0] * cov2D[1][1] - cov2D[0][1] * cov2D[1][0];
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

    MOUSE_SENSITIVITY_ORBIT = 0.2
    MOUSE_SENSITIVITY_PAN = 0.002
    MOUSE_SENSITIVITY_ZOOM = 0.3

    def __init__(self, pkl_files: list[Path], initial_data: dict, initial_index: int):
        # Transformation matrix from learning coordinate system to OpenGL coordinate system
        # Right-handed coordinate system: x-axis pointing right, y-axis pointing down, z-axis pointing forward
        # -> Right-handed coordinate system: x-axis pointing right, y-axis pointing up, z-axis pointing backward
        self.to_gl_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        self.pkl_files = pkl_files
        self.current_data_index = initial_index
        self.params_cache = {initial_index: self.params_to_gl_data(initial_data["params"])}
        self.params = initial_data["params"]

        self.camera_params = self.camera_params_to_gl_data(initial_data["camera_params"])
        self.consts = initial_data["consts"]

        # 初期パラメータを保存
        self.initial_width, self.initial_height = self.consts["img_shape"][::-1]
        # 最初のカメラの内部パラメータを基準の初期値として保存
        intrinsic_vec = self.camera_params["intrinsic_batch"][initial_index]
        self.initial_fx, self.initial_fy, _, _ = intrinsic_vec

        # 動的パラメータを初期化
        self.render_width, self.render_height = self.initial_width, self.initial_height
        self.current_fx, self.current_fy = self.initial_fx, self.initial_fy

        self._init_glfw()
        self._init_moderngl()

        self.camera = self._create_initial_camera()
        self._setup_callbacks()

        self.means3d_jax = jnp.asarray(self.params["means3d"])
        self.sorter = jax.jit(self._get_sorted_indices_jax)

        self.left_mouse_dragging = False
        self.right_mouse_dragging = False
        self.last_mouse_pos = None
        self.camera_dirty = True

    def params_to_gl_data(self, params):
        """paramsをGLのデータ形式に変換する。"""
        params["means3d"] = (self.to_gl_matrix @ params["means3d"].T).T

        nan_mask = np.isnan(params["quats"]).any(axis=1)
        params["quats"][nan_mask] = np.array([0, 0, 0, 1])  # nanデータは無回転にする
        params["quats"] = params["quats"] / np.linalg.norm(params["quats"], axis=1, keepdims=True)

        transformed_matrix = self.to_gl_matrix @ Rotation.from_quat(params["quats"]).as_matrix()
        params["quats"] = Rotation.from_matrix(transformed_matrix).as_quat()

        return params

    def camera_params_to_gl_data(self, camera_params):
        camera_params["rot_mat_batch"] = (
            self.to_gl_matrix @ camera_params["rot_mat_batch"] @ self.to_gl_matrix.T
        )
        camera_params["t_vec_batch"] = camera_params["t_vec_batch"] @ self.to_gl_matrix
        camera_params["intrinsic_batch"] = camera_params["intrinsic_batch"] * np.array(
            [1, 1, 1, -1]
        )

        return camera_params

    def _init_glfw(self):
        """GLFWとウィンドウを初期化する。"""
        if not glfw.init():
            sys.exit("FATAL ERROR: glfw initialization failed.")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(
            self.initial_width, self.initial_height, "OpenGL 3DGS Viewer", None, None
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

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.disable(moderngl.DEPTH_TEST)

        self._init_buffers(self.params)

        quad_verts = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype="f4")
        vbo_quad = self.ctx.buffer(quad_verts)
        self.vao = self.ctx.vertex_array(self.program, [(vbo_quad, "2f", "in_vert")])

        width, height = glfw.get_framebuffer_size(self.window)
        self.framebuffer_size_callback(self.window, width, height)

    def _init_buffers(self, params: dict):
        """パラメータからSSBOを作成または更新する。"""
        self.num_gaussians = params["means3d"].shape[0]

        if "colors" not in params:
            raise KeyError("The provided .pkl file does not contain the required 'colors' key.")

        colors = params["colors"].astype("f4")
        quats_xyzw = params["quats"].astype("f4")
        scales = params["scales"].astype("f4")
        opacities = params["opacities"].astype("f4")

        # データを一つの配列にまとめる
        flat_data = np.concatenate(
            [params["means3d"].astype("f4"), quats_xyzw, scales, opacities, colors], axis=1
        ).ravel()

        # 2. SSBOを作成・書き込み
        self.ssbo_gaussians = self.ctx.buffer(flat_data.tobytes())
        initial_indices = np.arange(self.num_gaussians, dtype="i4")
        self.ssbo_indices = self.ctx.buffer(initial_indices.tobytes(), dynamic=True)

        # 3. SSBOをシェーダにバインド
        self.ssbo_gaussians.bind_to_storage_buffer(0)
        self.ssbo_indices.bind_to_storage_buffer(1)

    def _update_buffers(self, params: dict):
        """既存のバッファを新しいパラメータで更新する。ガウシアン数が変わる場合は再生成。"""
        new_num_gaussians = params["means3d"].shape[0]
        if new_num_gaussians != self.num_gaussians:
            # ガウシアン数が異なる場合はバッファを再作成
            self._init_buffers(params)
        else:
            if "colors" not in params:
                raise KeyError("The provided .pkl file does not contain the required 'colors' key.")

            colors = params["colors"].astype("f4")
            quats_xyzw = params["quats"].astype("f4")
            scales = params["scales"].astype("f4")
            opacities = params["opacities"].astype("f4")

            flat_data = np.concatenate(
                [params["means3d"].astype("f4"), quats_xyzw, scales, opacities, colors], axis=1
            ).ravel()
            self.ssbo_gaussians.write(flat_data.tobytes())

        self.means3d_jax = jnp.asarray(params["means3d"])

    @staticmethod
    def _get_sorted_indices_jax(means3d_jax, view_matrix_jax):
        """JAXを使って深度ソートを行う。"""
        means_homo = jnp.hstack([means3d_jax, jnp.ones((means3d_jax.shape[0], 1))])
        means_view = means_homo @ view_matrix_jax.T
        depths = means_view[:, 2]
        return jnp.argsort(depths)

    def _sort_gaussians(self, view_matrix: np.ndarray):
        """カメラのビュー行列に基づいてガウシアンをソートし、インデックスバッファを更新する。"""
        sorted_indices_jax = self.sorter(self.means3d_jax, jnp.array(view_matrix))
        self.ssbo_indices.write(np.asarray(sorted_indices_jax).astype("i4").tobytes())

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

        self.params = params
        self._update_buffers(params)
        self.current_data_index = index
        self.camera_dirty = True
        self.update_window_title()

    def update_window_title(self):
        """現在のファイル名と位置情報でウィンドウタイトルを更新する。"""
        filename = self.pkl_files[self.current_data_index].name
        k, n = self.current_data_index + 1, len(self.pkl_files)
        title = f"OpenGL 3DGS Viewer ({filename} {k}/{n})"
        glfw.set_window_title(self.window, title)

    def run(self):
        """メインループを実行する。"""
        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            if self.camera_dirty:
                # 1. ビュー行列の更新
                view_matrix = self.camera.get_view_matrix()

                # 2. プロジェクション行列の更新 (FOVとアスペクト比を初期値で固定)
                fovy = 2 * np.arctan(self.initial_height / (2 * self.initial_fy))
                aspect = self.initial_width / self.initial_height
                projection_matrix = pglm.perspective(fovy, aspect, 0.1, 1000.0)

                # 3. ガウシアンのソート
                self._sort_gaussians(view_matrix)

                # 4. Uniform変数をシェーダに送信
                self.program["view_matrix"].write(pglm.mat4(view_matrix))
                self.program["projection_matrix"].write(projection_matrix)
                # 解像度に合わせてスケールされたfx, fyを渡す
                self.program["focal_x"].value = self.current_fx
                self.program["focal_y"].value = self.current_fy
                # 新しい描画解像度を渡す
                self.program["u_resolution"].value = (self.render_width, self.render_height)

                self.camera_dirty = False

            # 描画
            self.ctx.clear(0.0, 0.0, 0.0)
            if self.num_gaussians > 0:
                self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=self.num_gaussians)

            glfw.swap_buffers(self.window)
        glfw.terminate()

    # --- Event Callbacks (JAX版からほぼ流用) ---
    def framebuffer_size_callback(self, window, width, height):
        """ウィンドウリサイズ時に呼び出され、解像度、焦点距離、ビューポートを更新する。"""
        if width == 0 or height == 0:
            return

        content_aspect = self.initial_width / self.initial_height
        window_aspect = width / height

        # ウィンドウを覆うように新しい描画解像度を計算（カバー）
        if window_aspect > content_aspect:
            # ウィンドウが横長 -> 幅基準でスケール
            self.render_width = width
            self.render_height = int(width / content_aspect)
            view_x = 0
            view_y = int((height - self.render_height) / 2)
        else:
            # ウィンドウが縦長 -> 高さ基準でスケール
            self.render_height = height
            self.render_width = int(height * content_aspect)
            view_y = 0
            view_x = int((width - self.render_width) / 2)

        # 1. ビューポートを設定（クロップと中央配置）
        self.ctx.viewport = (view_x, view_y, self.render_width, self.render_height)

        # 2. 新しい解像度に合わせて焦点距離をスケーリング
        scale = self.render_width / self.initial_width
        self.current_fx = self.initial_fx * scale
        self.current_fy = self.initial_fy * scale

        # 3. 再描画をトリガー
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
                # カメラを切り替えたら、基準となるfx,fyもそのカメラのものに更新
                new_intrinsics = self.camera_params["intrinsic_batch"][self.current_cam_index]
                self.initial_fx, self.initial_fy, _, _ = new_intrinsics
                # 現在のウィンドウサイズに合わせて各種パラメータを再計算
                self.framebuffer_size_callback(self.window, *glfw.get_framebuffer_size(self.window))
                self.camera.position, self.camera.rotation = self._get_colmap_camera_state(
                    self.current_cam_index
                )
                self.camera_dirty = True
                print(f"Jump to Camera: {self.current_cam_index + 1}/{self.num_cameras}", end="\r")

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
    parser = argparse.ArgumentParser(description="OpenGL 3D Gaussian Splatting Interactive Viewer")
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
