from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
import moderngl
import numpy as np
import numpy.typing as npt
import pyglm.glm as pglm

from gs.function_factory import make_render


class GsRendererBase(ABC):
    """3D Gaussian Splattingのレンダリングを行うクラスの基底クラス。"""

    @abstractmethod
    def render(
        self,
        view: dict[str, npt.NDArray],
        focal_lengths: tuple[float, float],
        resolution_wh: tuple[int, int],
    ) -> None:
        """指定したパラメータでレンダリングする"""

    @abstractmethod
    def update_gaussian_data(self, params: dict[str, npt.NDArray]) -> None:
        """レンダリングに使用するガウシアンパラメータを更新する。"""

    @abstractmethod
    def set_viewport(self, x: int, y: int, width: int, height: int) -> None:
        """ビューポートを設定する。"""

    @abstractmethod
    def shutdown(self) -> None:
        """ModernGLのリソースを解放する。"""


class GsRendererJax(GsRendererBase):
    """JAXによる計算とModernGLによる表示の両方を担当するレンダラ。"""

    # --- シェーダ定義 ---
    VERTEX_SHADER = """
        #version 430
        in vec2 in_position;
        in vec2 in_texcoord_0;
        out vec2 v_uv;
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            v_uv = vec2(in_texcoord_0.x, 1.0 - in_texcoord_0.y);
        }
    """
    FRAGMENT_SHADER = """
        #version 430
        uniform sampler2D u_texture;
        in vec2 v_uv;
        out vec4 f_color;
        void main() {
            f_color = vec4(texture(u_texture, v_uv).rgb, 1.0);
        }
    """

    def __init__(self, initial_params: dict, consts: dict[str, Any]) -> None:
        """JAXのレンダリング関数とModernGLの描画オブジェクトを初期化する。"""
        self.ctx = moderngl.create_context(require=430)
        # --- JAX部分の初期化 ---
        self.params = initial_params
        self.consts = consts
        self.active_sh_degree = 3
        self.render_fn = make_render(self.consts, active_sh_degree=self.active_sh_degree, jit=True)

        # --- ModernGL部分の初期化 ---
        self.program = self.ctx.program(
            vertex_shader=self.VERTEX_SHADER, fragment_shader=self.FRAGMENT_SHADER
        )

        # JAX版特有のレンダリング設定
        self.consts["tile_max_gs_num"] = self._calc_tile_max_gs_num(
            self.consts
        )  # 学習時よりタイルあたりのガウシアン数を増やす

        width, height = self.consts["img_shape"][::-1]
        self.image_texture = self.ctx.texture((width, height), 3, dtype="f4")
        self.program["u_texture"].value = 0  # テクスチャユニット0を使用

        # fmt: off
        vertices = [
            -1.0, -1.0, 0.0, 0.0,  # Bottom Left
            1.0, -1.0, 1.0, 0.0,  # Bottom Right
            -1.0,  1.0, 0.0, 1.0,  # Top Left
            1.0,  1.0, 1.0, 1.0,  # Top Right
        ]
        # fmt: on

        quad_buffer = self.ctx.buffer(np.array(vertices, dtype="f4"))
        self.quad_vao = self.ctx.vertex_array(
            self.program, [(quad_buffer, "2f 2f", "in_position", "in_texcoord_0")]
        )

    @staticmethod
    def _calc_tile_max_gs_num(consts: dict[str, Any]) -> int:
        base_tile_size = 16
        base_gaussian_num = 1000
        tile_scale = consts["tile_size"] / base_tile_size
        return int(
            tile_scale**2
            * consts["tile_max_gs_num_factor"]
            * consts["max_gaussians"]
            / base_gaussian_num
        )

    def render(
        self,
        view: dict[str, npt.NDArray],
        focal_lengths: tuple[float, float],  # noqa: ARG002
        resolution_wh: tuple[int, int],  # noqa: ARG002
    ) -> None:
        """JAXで画像を計算し、その結果をModernGLで画面に描画する。"""
        # 1. JAXで画像をレンダリング
        image_data = np.asarray(self.render_fn(self.params, view))  # type: ignore[arg-type]

        # 2. 生成された画像をテクスチャに書き込み、画面に描画
        self.image_texture.write(image_data.astype("f4").tobytes())
        self.ctx.clear(0.0, 0.0, 0.0)
        self.image_texture.use(location=0)
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

    def update_gaussian_data(self, params: dict[str, npt.NDArray]) -> None:
        """レンダリングに使用するガウシアンパラメータを更新する。"""
        self.params = params

    def set_viewport(self, x: int, y: int, width: int, height: int) -> None:
        """ビューポートを設定する。"""
        self.ctx.viewport = (x, y, width, height)

    def shutdown(self) -> None:
        """ModernGLのリソースを解放する。"""
        self.quad_vao.release()
        self.program.release()
        self.image_texture.release()
        self.ctx.release()


class GsRendererGl(GsRendererBase):
    """OpenGL/ModernGLのレンダリング関連の処理をカプセル化するクラス。"""

    # --- シェーダ定義 ---
    VERTEX_SHADER = """
        #version 430 core

        const float SH_C0_0 = 0.28209479177387814;
        const float SH_C1_0 = -0.4886025119029199;
        const float SH_C1_1 = 0.4886025119029199;
        const float SH_C1_2 = -0.4886025119029199;
        const float SH_C2_0 = 1.0925484305920792;
        const float SH_C2_1 = -1.0925484305920792;
        const float SH_C2_2 = 0.31539156525252005;
        const float SH_C2_3 = -1.0925484305920792;
        const float SH_C2_4 = 0.5462742152960396;
        const float SH_C3_0 = -0.5900435899266435;
        const float SH_C3_1 = 2.890611442640554;
        const float SH_C3_2 = -0.4570457994644658;
        const float SH_C3_3 = 0.3731763325901154;
        const float SH_C3_4 = -0.4570457994644658;
        const float SH_C3_5 = 1.445305721320277;
        const float SH_C3_6 = -0.5900435899266435;

        layout(location = 0) in vec2 in_vert;

        // SSBO bindings
        layout (std430, binding = 0) buffer gaussian_data { float g_data[]; };
        layout (std430, binding = 1) buffer gaussian_order { int gi[]; };

        // Uniforms
        uniform mat3 rot_mat;
        uniform vec3 t_vec;
        uniform mat4 projection_matrix;
        uniform float focal_x;
        uniform float focal_y;
        uniform vec2 u_resolution;

        // Outputs to Fragment Shader
        out vec3 pass_color;
        out float pass_alpha;
        out vec3 pass_cov_2d_inv;
        out vec2 pass_coordxy;

        mat3 quat_to_rotmat(vec4 quat) {
            float x = quat.x, y = quat.y, z = quat.z, w = quat.w;
            return mat3(
                1.f - 2.f * (y * y + z * z), 2.f * (x * y + w * z), 2.f * (x * z - w * y),
                2.f * (x * y - w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + w * x),
                2.f * (x * z + w * y), 2.f * (y * z - w * x), 1.f - 2.f * (x * x + y * y)
            );
        }

        mat3 compute_cov(vec4 quat, vec3 scale){
            mat3 rot_mat = quat_to_rotmat(quat);
            mat3 s_mat = mat3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
            mat3 prod_mat = rot_mat * s_mat;
            return prod_mat * transpose(prod_mat);
        }

        mat2 compute_cov_2d(mat3 cov_3d_world, mat3 rot_mat, vec3 pos_view) {
            float tan_fovx = 2.0 * atan(u_resolution.x / (2.0 * focal_x));
            float tan_fovy = 2.0 * atan(u_resolution.y / (2.0 * focal_y));
            float limx = 1.3 * tan_fovx;
            float limy = 1.3 * tan_fovy;

            float z = pos_view.z;
            float x_clipped = clamp(pos_view.x / z, -limx, limx) * z;
            float y_clipped = clamp(pos_view.y / z, -limy, limy) * z;

            mat3 jacobian = mat3(
                focal_x / z, 0.0, 0.0,
                0.0, focal_y / z, 0.0,
                -focal_x * x_clipped / (z*z), -focal_y * y_clipped / (z*z), 0.0
            );

            mat3 prod_mat = jacobian * rot_mat;

            return mat2(prod_mat * cov_3d_world * transpose(prod_mat));
        }

        float max_eigenvalue(mat2 cov_2d, float det) {
            float mid = 0.5 * (cov_2d[0][0] + cov_2d[1][1]);
            float lambda1 = mid + sqrt(max(0.1, mid*mid - det));
            return sqrt(lambda1);
        }

        vec3 compute_color_from_sh(vec3 direction, float sh_coeff[48]) {
            // Jax版と同様の実装を使うためにカメラ座標系をGL -> JAXへ変換
            float x = direction.x, y = -direction.y, z = -direction.z;

            float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;

            float basis[16];
            basis[0] = SH_C0_0;
            basis[1] = SH_C1_0 * y;
            basis[2] = SH_C1_1 * z;
            basis[3] = SH_C1_2 * x;
            basis[4] = SH_C2_0 * xy;
            basis[5] = SH_C2_1 * yz;
            basis[6] = SH_C2_2 * (2.0 * zz - xx - yy);
            basis[7] = SH_C2_3 * xz;
            basis[8] = SH_C2_4 * (xx - yy);
            basis[9] = SH_C3_0 * y * (3.0 * xx - yy);
            basis[10] = SH_C3_1 * xy * z;
            basis[11] = SH_C3_2 * y * (4.0 * zz - xx - yy);
            basis[12] = SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
            basis[13] = SH_C3_4 * x * (4.0 * zz - xx - yy);
            basis[14] = SH_C3_5 * z * (xx - yy);
            basis[15] = SH_C3_6 * x * (xx - 3.0 * yy);

            float r = 0.0, g = 0.0, b = 0.0;
            for (int i = 0; i < 16; i++) {
                r += sh_coeff[i] * basis[i];
                g += sh_coeff[i + 16] * basis[i];
                b += sh_coeff[i + 32] * basis[i];
            }

            return max(vec3(r, g, b)+0.5, 0.0);
        }

        void main() {
            int instance_idx = gi[gl_InstanceID];
            int data_dim = 59; // (pos:3, rot_quat:4, scale:3, opacity:1, coeffs:48)
            int offset = instance_idx * data_dim;

            vec3 pos_world = vec3(g_data[offset], g_data[offset+1], g_data[offset+2]);
            vec4 rot_quat = vec4(
                g_data[offset+3], g_data[offset+4], g_data[offset+5], g_data[offset+6]
            );
            vec3 scale = vec3(g_data[offset+7], g_data[offset+8], g_data[offset+9]);
            pass_alpha = g_data[offset+10];

            vec3 pos_view = rot_mat * pos_world + t_vec;
            if (pos_view.z >= -0.2) { // 近すぎるガウシアンは描画しない
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // 画面外に飛ばす
                return;
            }

            vec3 direction = normalize(pos_world - t_vec);

            float sh_coeff[48];
            int sh_start = offset + 11; // pos:3 + rot:4 + scale:3 + opacity:1 = 11
            for (int i = 0; i < 48; i++) {
                sh_coeff[i] = g_data[sh_start + i];
            }

            pass_color = compute_color_from_sh(direction, sh_coeff);

            mat3 cov_3d_world = compute_cov(rot_quat, scale);
            mat2 cov_2d = compute_cov_2d(cov_3d_world, rot_mat, pos_view);

            cov_2d[0][0] += 0.3;
            cov_2d[1][1] += 0.3;

            float det = determinant(cov_2d);
            if (det == 0.0) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                return;
            }

            float det_inv = 1.0 / det;
            pass_cov_2d_inv = det_inv * vec3(cov_2d[1][1], -cov_2d[0][1], cov_2d[0][0]);

            float radius = 3.0 * max_eigenvalue(cov_2d, det);

            vec2 quad_radius_ndc = radius / u_resolution * 2.0;
            vec4 pos_clip = projection_matrix * vec4(pos_view, 1.0);
            gl_Position = pos_clip / pos_clip.w + vec4(in_vert * quad_radius_ndc, 0.0, 0.0);
            pass_coordxy = in_vert * radius;
        }
    """
    FRAGMENT_SHADER = """
        #version 430 core
        in vec3 pass_color;
        in float pass_alpha;
        in vec3 pass_cov_2d_inv;
        in vec2 pass_coordxy;
        out vec4 FragColor;

        void main() {
            float power = -0.5 * (
                pass_cov_2d_inv.x * pass_coordxy.x * pass_coordxy.x + pass_cov_2d_inv.z * pass_coordxy.y * pass_coordxy.y
            ) - pass_cov_2d_inv.y * pass_coordxy.x * pass_coordxy.y;
            if (power > 0.0) discard;

            float gaussian_weight = min(0.99, pass_alpha * exp(power));
            if (gaussian_weight <= 1.0/255.0) discard;

            FragColor = vec4(pass_color, gaussian_weight);
        }
    """  # noqa: E501

    def __init__(self, initial_params: dict[str, npt.NDArray]) -> None:
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

    def render(
        self,
        view: dict[str, npt.NDArray],
        focal_lengths: tuple[float, float],
        resolution_wh: tuple[int, int],
    ) -> None:
        """シーンを描画する。"""
        fovy = 2 * np.arctan(resolution_wh[1] / (2 * focal_lengths[1]))
        aspect = resolution_wh[0] / resolution_wh[1]
        projection_matrix = pglm.perspective(fovy, aspect, 0.2, 1000.0)

        # Uniform変数をシェーダに送信
        self.program["rot_mat"].write(pglm.mat3(view["rot_mat"]))
        self.program["t_vec"].write(pglm.vec3(view["t_vec"]))
        self.program["projection_matrix"].write(projection_matrix)
        self.program["focal_x"].value = focal_lengths[0]
        self.program["focal_y"].value = focal_lengths[1]
        self.program["u_resolution"].value = resolution_wh

        # 描画処理
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        if self.num_gaussians > 0:
            self._sort_and_update_gaussians(view["rot_mat"], view["t_vec"])
            self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=self.num_gaussians)

    def update_gaussian_data(self, params: dict[str, npt.NDArray]) -> None:
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
            self.ssbo_gaussians.write(flat_data.tobytes())  # type: ignore[reportOptionalMemberAccess]

    def set_viewport(self, x: int, y: int, width: int, height: int) -> None:
        """ビューポートを設定する。"""
        self.ctx.viewport = (x, y, width, height)

    def shutdown(self) -> None:
        """リソースを解放する。"""
        if self.ssbo_gaussians:
            self.ssbo_gaussians.release()
        if self.ssbo_indices:
            self.ssbo_indices.release()
        self.program.release()
        self.ctx.release()

    def _update_sorted_indices(self, sorted_indices: npt.NDArray) -> None:
        """ソート済みインデックスのSSBOを更新する。"""
        if self.ssbo_indices:
            self.ssbo_indices.write(sorted_indices.astype("i4").tobytes())

    def _sort_and_update_gaussians(self, rot_mat: npt.NDArray, t_vec: npt.NDArray) -> None:
        """ガウシアンをソートし、レンダラのインデックスバッファを更新する。"""
        sorted_indices_jax = self.sorter(self.means3d_jax, jnp.asarray(rot_mat), jnp.asarray(t_vec))
        self._update_sorted_indices(np.asarray(sorted_indices_jax))

    @staticmethod
    def _flatten_gaussian_data(params: dict[str, npt.NDArray]) -> npt.NDArray:
        """シェーダに渡すためにガウシアンデータをフラット化する。"""
        return np.concatenate(
            [
                params["means3d"].astype("f4"),
                params["quats"].astype("f4"),
                params["scales"].astype("f4"),
                params["opacities"].astype("f4"),
                params["sh_coeffs"].reshape(-1, 3 * 16).astype("f4"),
            ],
            axis=1,
        ).ravel()

    @staticmethod
    def _get_sorted_indices_jax(
        means3d_jax: jax.Array, rot_mat: jax.Array, t_vec: jax.Array
    ) -> jax.Array:
        """JAXを使用して、カメラ視点からの深度に基づいてガウシアンをソートする。"""
        means_view = means3d_jax @ rot_mat.T + t_vec
        depths = means_view[:, 2]
        return jnp.argsort(depths)
