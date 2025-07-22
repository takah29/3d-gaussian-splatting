import argparse
import sys
from pathlib import Path

from gs.viewer.camera import Camera
from gs.viewer.data_manager import DataManager
from gs.viewer.renderer import GsRendererGl
from gs.viewer.viewer import Viewer


def create_gl_viewer(params_dirs: list[Path], start_index: int) -> Viewer:
    data_manager = DataManager.create_for_gldata(params_dirs)
    params = data_manager.load_data(start_index)

    view = data_manager.get_camera_param()
    camera = Camera.create_gl(**view)
    viewer = Viewer(camera, data_manager, start_index, window_title="OpenGL 3DGS Viewer")

    # OpenGLの初期化後に作成する必要があるので作成後にviewerに設定
    renderer = GsRendererGl(params)  # type: ignore[arg-type]
    viewer.set_renderer(renderer)

    return viewer


def main() -> None:
    """アプリケーションのエントリーポイント。"""
    parser = argparse.ArgumentParser(description="OpenGL 3D Gaussian Splatting Interactive Viewer")
    parser.add_argument(
        "-f",
        "--params_dir",
        type=Path,
        default=Path(__file__).parent / "output" / "params_final",
        help="Checkpoint directory with parameter files",
    )
    args = parser.parse_args()

    # 指定されたファイルと同じディレクトリにある全ての.pklファイルを探索
    output_dir = args.params_dir.parent
    params_dirs = sorted(output_dir.glob("*/"))
    if len(params_dirs) == 0:
        sys.exit(f"Error: No checkpoint directory found in {params_dirs}")

    # 初期ファイルのインデックスを特定
    try:
        start_params_dir = args.params_dir.resolve()
        start_index = [p.resolve() for p in params_dirs].index(start_params_dir)
    except ValueError:
        print(f"Warning: Specified file {args.params_dir} not found. Starting with the first one.")
        start_index = 0

    viewer = create_gl_viewer(params_dirs, start_index)
    viewer.help_messege()
    viewer.run()


if __name__ == "__main__":
    main()
