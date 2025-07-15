import jax
import jax.numpy as jnp
import moderngl
import numpy as np
import pyglm.glm as pglm


class RendererGl:
    """OpenGL/ModernGLのレンダリング関連の処理をカプセル化するクラス。"""

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

    def __init__(self, initial_params: dict):
        self.ctx = moderngl.create_context(require=430)

        # シェーダとプログラム
        self.program = self.ctx.program(
            vertex_shader=self.VERTEX_SHADER, fragment_shader=self.FRAGMENT_SHADER
        )

        # レンダリング設定
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.disable(moderngl.DEPTH_TEST)

        # バッファの作成
        self.num_gaussians = 0
        self.ssbo_gaussians = None
        self.ssbo_indices = None
        self.update_gaussian_data(initial_params)

        # 頂点配列オブジェクト (VAO)
        quad_verts = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype="f4")
        vbo_quad = self.ctx.buffer(quad_verts)
        self.vao = self.ctx.vertex_array(self.program, [(vbo_quad, "2f", "in_vert")])

        # ガウシアンの深度ソート
        self.means3d_jax = jnp.asarray(initial_params["means3d"])
        self.sorter = jax.jit(self._get_sorted_indices_jax)

    def render(self, view_matrix, projection_matrix, focal_lengths, resolution):
        """シーンを描画する。"""
        # Uniform変数をシェーダに送信
        self.program["view_matrix"].write(pglm.mat4(view_matrix))
        self.program["projection_matrix"].write(projection_matrix)
        self.program["focal_x"].value = focal_lengths[0]
        self.program["focal_y"].value = focal_lengths[1]
        self.program["u_resolution"].value = resolution

        # 描画処理
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        if self.num_gaussians > 0:
            self._sort_and_update_gaussians(view_matrix)
            self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=self.num_gaussians)

    def update_gaussian_data(self, params: dict):
        """ガウシアンデータ用のSSBOを作成または更新する。"""
        new_num_gaussians = params["means3d"].shape[0]
        flat_data = self._flatten_gaussian_data(params)
        self.means3d_jax = jnp.asarray(params["means3d"])

        if new_num_gaussians != self.num_gaussians:
            # ガウシアン数が変わった場合、バッファを再生成
            if self.ssbo_gaussians:
                self.ssbo_gaussians.release()
            if self.ssbo_indices:
                self.ssbo_indices.release()

            self.num_gaussians = new_num_gaussians
            self.ssbo_gaussians = self.ctx.buffer(flat_data.tobytes())
            initial_indices = np.arange(self.num_gaussians, dtype="i4")
            self.ssbo_indices = self.ctx.buffer(initial_indices.tobytes(), dynamic=True)

            self.ssbo_gaussians.bind_to_storage_buffer(0)
            self.ssbo_indices.bind_to_storage_buffer(1)
        else:
            # ガウシアン数が同じなら、データを書き込むだけ
            self.ssbo_gaussians.write(flat_data.tobytes())

    def update_sorted_indices(self, sorted_indices: np.ndarray):
        """ソート済みインデックスのSSBOを更新する。"""
        if self.ssbo_indices:
            self.ssbo_indices.write(sorted_indices.astype("i4").tobytes())

    @staticmethod
    def _flatten_gaussian_data(params: dict) -> np.ndarray:
        """シェーダに渡すためにガウシアンデータをフラット化する。"""
        if "colors" not in params:
            raise KeyError("The provided .pkl file does not contain the required 'colors' key.")

        return np.concatenate(
            [
                params["means3d"].astype("f4"),
                params["quats"].astype("f4"),
                params["scales"].astype("f4"),
                params["opacities"].astype("f4"),
                params["colors"].astype("f4"),
            ],
            axis=1,
        ).ravel()

    def set_viewport(self, x, y, width, height):
        """ビューポートを設定する。"""
        self.ctx.viewport = (x, y, width, height)

    @staticmethod
    def _get_sorted_indices_jax(means3d_jax, view_matrix_jax):
        """JAXを使用して、カメラ視点からの深度に基づいてガウシアンをソートする。"""
        means_homo = jnp.hstack([means3d_jax, jnp.ones((means3d_jax.shape[0], 1))])
        means_view = means_homo @ view_matrix_jax.T
        depths = means_view[:, 2]
        return jnp.argsort(depths)

    def _sort_and_update_gaussians(self, view_matrix: np.ndarray):
        """ガウシアンをソートし、レンダラのインデックスバッファを更新する。"""
        sorted_indices_jax = self.sorter(self.means3d_jax, jnp.array(view_matrix))
        self.update_sorted_indices(np.asarray(sorted_indices_jax))

    def shutdown(self):
        """リソースを解放する。"""
        if self.ssbo_gaussians:
            self.ssbo_gaussians.release()
        if self.ssbo_indices:
            self.ssbo_indices.release()
        self.program.release()
        self.ctx.release()
