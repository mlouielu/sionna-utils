import sionna.rt
from pathlib import Path

from ipywidgets.embed import embed_minimal_html


def scene_export_html(
    scene: sionna.rt.Scene,
    fpath: Path | str,
    title: str = "Sionna Scene Preview",
    **kwargs,
):
    """Export a Sionna RT scene as an interactive HTML file.

    Creates a standalone HTML file containing an interactive 3D visualization of the scene
    that can be viewed in a web browser without requiring a Jupyter notebook.

    Args:
        scene: Sionna RT scene to export
        fpath: Output file path for the HTML file (string or Path object)
        title: Title for the HTML page (default: "Sionna Scene Preview")
        **kwargs: Additional keyword arguments passed to scene.preview(),
                 such as show_orientations, show_paths, etc.

    Example:
        >>> scene = sionna.rt.load_scene(sionna.rt.scene.munich)
        >>> scene_export_html(scene, "scene.html", show_orientations=True)
    """
    scene.preview(**kwargs)
    fig_ = scene._preview_widget
    embed_minimal_html(
        fpath,
        views=[fig_._renderer],
        title="Sionna RT Scene Preview",
    )
