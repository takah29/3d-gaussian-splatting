import sys

import glfw  # type: ignore[import-untyped]

from gs.viewer.camera import Camera
from gs.viewer.data_manager import DataManager
from gs.viewer.renderer import GsRendererBase


class Viewer:
    """ユーザー入力、カメラ制御、データ管理を行い、Rendererに描画指示を出すクラス。"""

    # --- マウス感度設定 ---
    MOUSE_SENSITIVITY_ROTATE = 0.05
    MOUSE_SENSITIVITY_PAN = 0.002
    MOUSE_SENSITIVITY_ZOOM = 0.3
    MOUSE_SENSITIVITY_ROLL = 0.1

    def __init__(
        self,
        camera: Camera,
        data_manager: DataManager,
        initial_index: int,
        window_title: str = "3DGS Viewer",
    ) -> None:
        self.window_title = window_title
        self.camera = camera
        self.data_manager = data_manager

        self.params = self.data_manager.load_data(initial_index)

        view = self.camera.get_view()
        consts = self.data_manager.get_consts()
        # --- パラメータの初期化 ---
        self.initial_width, self.initial_height = consts["img_shape"][::-1]
        self.initial_fx, self.initial_fy, _, _ = view["intrinsic_vec"]
        self.render_width, self.render_height = self.initial_width, self.initial_height
        self.current_fx, self.current_fy = self.initial_fx, self.initial_fy

        # --- 状態変数の初期化 ---
        self.left_mouse_dragging = False
        self.right_mouse_dragging = False
        self.middle_mouse_dragging = False
        self.last_mouse_pos = None
        self.camera_dirty = True  # 再描画が必要かどうかのフラグ

        # --- 初期化処理の実行 ---
        self._init_glfw()
        self._setup_callbacks()

    def set_renderer(self, renderer: GsRendererBase) -> None:
        self.renderer = renderer

        # 初回のビューポート設定
        self._framebuffer_size_callback(self.window, self.initial_width, self.initial_height)

    def set_windows_size(self, width: int, height: int) -> None:
        """ウィンドウのサイズを設定し、ビューポートを更新する。"""
        glfw.set_window_size(self.window, width, height)
        self._framebuffer_size_callback(self.window, width, height)

    def run(self) -> None:
        """メインループを実行する。"""
        while not glfw.window_should_close(self.window):
            glfw.wait_events()
            if self.camera_dirty:
                self._render_frame()
                self.camera_dirty = False

        self.renderer.shutdown()
        glfw.terminate()

    def _render_frame(self) -> None:
        view = self.camera.get_view()
        self.renderer.render(
            view,
            focal_lengths=(self.current_fx, self.current_fy),
            resolution_wh=(self.render_width, self.render_height),
        )
        glfw.swap_buffers(self.window)

    def _init_glfw(self) -> None:
        """GLFWとウィンドウを初期化する。"""
        if not glfw.init():
            sys.exit("FATAL ERROR: glfw initialization failed.")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(
            self.initial_width, self.initial_height, self.window_title, None, None
        )
        if not self.window:
            glfw.terminate()
            sys.exit("FATAL ERROR: Failed to create glfw window.")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self._update_window_title()

    def _load_params(self, index: int) -> None:
        """指定インデックスのpklファイルをロードし、レンダラを更新する。"""
        self.params = self.data_manager.load_data(index)
        self.renderer.update_gaussian_data(self.params)  # type: ignore[arg-type]

        self.camera_dirty = True
        self._update_window_title()

    def _update_window_title(self) -> None:
        """現在のファイル名と位置情報でウィンドウタイトルを更新する。"""
        filename = self.data_manager.get_current_filename()
        current_index, n_data = self.data_manager.get_current_data_index()
        title = f"{self.window_title} ({filename} {current_index + 1}/{n_data})"
        glfw.set_window_title(self.window, title)

    def _change_camera_pose(self) -> None:
        """学習済みカメラ視点を切り替える。"""
        view = self.data_manager.get_camera_param()
        self.camera.set_pose_w2c(view["rot_mat"], view["t_vec"])
        self.camera_dirty = True

    def _setup_callbacks(self) -> None:
        """GLFWのイベントコールバックを設定する。"""
        glfw.set_framebuffer_size_callback(self.window, self._framebuffer_size_callback)
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)

    def _framebuffer_size_callback(self, window, width: int, height: int) -> None:  # noqa: ANN001, ARG002
        """ウィンドウリサイズ時に呼び出され、ビューポートと解像度関連の値を更新する。"""
        if height == 0 or width == 0:
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

        # 1. レンダラにビューポートを設定
        self.renderer.set_viewport(view_x, view_y, self.render_width, self.render_height)

        # 2. 新しい解像度に合わせて焦点距離をスケーリング
        # OpenGL版のみ使用
        scale = self.render_width / self.initial_width
        self.current_fx = self.initial_fx * scale
        self.current_fy = self.initial_fy * scale

        self.camera_dirty = True

    def _key_callback(self, window, key, scancode, action, mods) -> None:  # noqa: ANN001, ARG002
        """キーボード入力イベントを処理する。"""
        # ファイル切り替え (Up/Down)
        if action == glfw.PRESS:
            if key == glfw.KEY_UP:
                current_index = self.data_manager.move_data_index(-1)
                self._load_params(current_index)
            elif key == glfw.KEY_DOWN:
                current_index = self.data_manager.move_data_index(1)
                self._load_params(current_index)

        # 学習済みカメラ視点切り替え (Left/Right)
        if action in (glfw.PRESS, glfw.REPEAT):
            cam_changed = False
            if key == glfw.KEY_RIGHT:
                self.current_cam_index = self.data_manager.move_camera_param_index(1)
                cam_changed = True
            elif key == glfw.KEY_LEFT:
                self.current_cam_index = self.data_manager.move_camera_param_index(-1)
                cam_changed = True

            if cam_changed:
                self._change_camera_pose()
                current_camera_param_index, num_cameras = (
                    self.data_manager.get_current_camera_param_index()
                )
                print(f"Jump to Camera: {current_camera_param_index + 1}/{num_cameras}", end="\r")

    def _mouse_button_callback(self, window, button, action, mods) -> None:  # noqa: ANN001, ARG002
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

    def _cursor_pos_callback(self, window, xpos, ypos) -> None:  # noqa: ANN001, ARG002
        """マウスカーソルの移動イベントを処理する。"""
        if (
            not (
                self.left_mouse_dragging or self.right_mouse_dragging or self.middle_mouse_dragging
            )
            or self.last_mouse_pos is None
        ):
            return

        dx, dy = xpos - self.last_mouse_pos[0], ypos - self.last_mouse_pos[1]
        if self.left_mouse_dragging:  # 左ドラッグで回転
            self.camera.rotate(dx, dy, self.MOUSE_SENSITIVITY_ROTATE)
        if self.right_mouse_dragging:  # 右ドラッグでパン
            self.camera.pan(dx, dy, self.MOUSE_SENSITIVITY_PAN)
        if self.middle_mouse_dragging:  # 中ドラッグでロール
            self.camera.roll(dx, self.MOUSE_SENSITIVITY_ROLL)

        self.camera_dirty = True
        self.last_mouse_pos = (xpos, ypos)

    def _scroll_callback(self, window, xoffset, yoffset) -> None:  # noqa: ANN001, ARG002
        """マウスホイールのスクロールイベントを処理する。"""
        self.camera.zoom(yoffset, self.MOUSE_SENSITIVITY_ZOOM)
        self.camera_dirty = True

    @staticmethod
    def help_messege() -> None:
        """ヘルプメッセージを表示する。"""
        print(
            f"{' Help ':=^30}\n"
            "Arrow Key\n"
            "  Up/Down: Change dataset\n"
            "  Left/Right: Change camera pose\n"
            "\n"
            "Mouse Button\n"
            "  Left Drag: Rotate\n"
            "  Right Drag: Pan\n"
            "  Middle Drag: Roll\n"
            "  Scroll: Forward/Backward\n"
            f"{'=' * 30}"
        )
