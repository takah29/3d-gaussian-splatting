import argparse
import sys
from pathlib import Path

from scipy.spatial.transform import Rotation

from gs.viewer.camera import CameraJax
from gs.viewer.data_manager import DataManager
from gs.viewer.renderer import GsRendererJax
from gs.viewer.viewer import Viewer


def create_jax_viewer(pkl_files: list[Path], initial_index: int) -> Viewer:
    data_manager = DataManager(pkl_files)
    params = data_manager.load_data(initial_index)

    camera_index = 0
    view = data_manager.get_camera_param(camera_index)
    rot_mat_w2c = view["rot_mat"]
    t_vec_w2c = view["t_vec"]
    c2w_rotation = Rotation.from_matrix(rot_mat_w2c.T)
    c2w_position = -c2w_rotation.apply(t_vec_w2c)

    camera = CameraJax(c2w_position, c2w_rotation, view["intrinsic_vec"])
    viewer = Viewer(camera, data_manager, initial_index, window_title="JAX 3DGS Viewer")

    # OpenGLの初期化後に作成する必要があるので作成後にviewerに設定
    consts = data_manager.get_consts()
    renderer = GsRendererJax(params, consts)  # type: ignore[arg-type]
    viewer.set_renderer(renderer)

    return viewer


def main() -> None:
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
            f"Warning: Specified file {args.params_filepath} "
            "not found. Starting with the first one."
        )
        initial_index = 0

    viewer = create_jax_viewer(pkl_files, initial_index)
    viewer.run()


if __name__ == "__main__":
    main()
