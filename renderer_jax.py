"""JAXによる画像生成とModernGLによる画面表示を統合したレンダリンクラス。"""

from collections.abc import Callable
from dataclasses import dataclass, field

import moderngl
import numpy as np

from gs.make_update import make_render


@dataclass
class RendererJax:
    """JAXによる計算とModernGLによる表示の両方を担当するレンダラ。"""

    initial_params: dict
    consts: dict

    # JAX関連
    params: dict = field(init=False)
    render_fn: Callable = field(init=False)

    # ModernGL関連
    program: moderngl.Program = field(init=False)
    image_texture: moderngl.Texture = field(init=False)
    quad_vao: moderngl.VertexArray = field(init=False)

    # --- シェーダ定義 ---
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

    def __post_init__(self):
        """JAXのレンダリング関数とModernGLの描画オブジェクトを初期化する。"""
        self.ctx = moderngl.create_context(require=330)
        # --- JAX部分の初期化 ---
        self.params = self.initial_params
        self.render_fn = make_render(self.consts, jit=True)

        # --- ModernGL部分の初期化 ---
        self.program = self.ctx.program(
            vertex_shader=self.VERTEX_SHADER, fragment_shader=self.FRAGMENT_SHADER
        )

        width, height = self.consts["img_shape"][::-1]
        self.image_texture = self.ctx.texture((width, height), 3, dtype="f4")
        self.program["u_texture"].value = 0  # テクスチャユニット0を使用

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

    def render(self, view_params: dict):
        """JAXで画像を計算し、その結果をModernGLで画面に描画する。"""
        # 1. JAXで画像をレンダリング
        image_data = np.asarray(self.render_fn(self.params, view_params))

        # 2. 生成された画像をテクスチャに書き込み、画面に描画
        self.image_texture.write(image_data.astype("f4").tobytes())
        self.ctx.clear(0.0, 0.0, 0.0)
        self.image_texture.use(location=0)
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

    def update_gaussian_data(self, params: dict):
        """レンダリングに使用するガウシアンパラメータを更新する。"""
        self.params = params

    def set_viewport(self, x: int, y: int, width: int, height: int):
        """ビューポートを設定する。"""
        self.ctx.viewport = (x, y, width, height)

    def shutdown(self):
        """ModernGLのリソースを解放する。"""
        self.quad_vao.release()
        self.program.release()
        self.image_texture.release()
        self.ctx.release()
