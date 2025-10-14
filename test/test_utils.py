import pytest
import tempfile

import sionna.rt
from sionna.rt import PlanarArray, Transmitter, Receiver

import sionna_utils


def test_scene_export_html():
    scene = sionna.rt.load_scene(sionna.rt.scene.munich)

    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="V",
    )

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross",
    )

    # Create transmitter
    tx = Transmitter(name="tx", position=[8.5, 21, 27], display_radius=2)

    # Add transmitter instance to scene
    scene.add(tx)

    # Create a receiver
    rx = Receiver(name="rx", position=[45, 90, 1.5], display_radius=2)

    # Add receiver instance to scene
    scene.add(rx)
    tx.look_at(rx)  # Transmitter points towards receiver

    # Export it as HTML using a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=True) as tmp:
        sionna_utils.utils.scene_export_html(scene, tmp.name, show_orientations=True)

        # Read and verify the content
        with open(tmp.name, 'r') as f:
            content = f.read()
            assert len(content) > 0, "HTML file is empty"
            assert "<html" in content.lower(), "Missing HTML tags"
