import sionna.rt
from pathlib import Path

from ipywidgets.embed import embed_minimal_html


def scene_export_html(
    scene: sionna.rt.Scene,
    fpath: Path | str,
    title: str = "Sionna Scene Preview",
    **kwargs,
):
    scene.preview(**kwargs)
    fig_ = scene._preview_widget
    embed_minimal_html(
        fpath,
        views=[fig_._renderer],
        title="Sionna RT Scene Preview",
    )
