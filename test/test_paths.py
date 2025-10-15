import pytest
import numpy as np

import sionna.rt
import mitsuba as mi
from sionna.rt import PlanarArray, Transmitter, Receiver, PathSolver

import sionna_utils


@pytest.fixture
def scene():
    scene = sionna.rt.load_scene(sionna.rt.scene.simple_reflector)
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=2,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="V",
    )

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=3,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross",
    )

    dist = 5
    d = float(dist / np.sqrt(2))
    scene.add(Transmitter(name="tx", position=[-d, 0, d]))
    scene.add(Receiver(name="rx", position=[d, 0, d]))

    return scene


def test_paths_filter_only_paths_all_valid(scene):
    # Create a mock Paths object with fake valid data
    # Shape: (num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths)
    # Using: (2, 1, 5, 1, 3) - 2 rx, 5 tx, 3 paths

    # Create valid array where some paths are all valid across all rx/tx, some are not
    valid_data = np.array(
        [
            [  # rx 0
                [  # rx_ant 0
                    [[True, False, True]],  # tx 0: paths 0,2 valid; path 1 invalid
                    [[True, False, True]],  # tx 1: paths 0,2 valid; path 1 invalid
                    [[True, False, False]],  # tx 2: paths 0 valid; paths 1,2 invalid
                    [[True, False, True]],  # tx 3: paths 0,2 valid; path 1 invalid
                    [[True, False, True]],  # tx 4: paths 0,2 valid; path 1 invalid
                ]
            ],
            [  # rx 1
                [  # rx_ant 0
                    [[True, False, True]],  # tx 0: paths 0,2 valid; path 1 invalid
                    [[True, False, True]],  # tx 1: paths 0,2 valid; path 1 invalid
                    [[True, False, True]],  # tx 2: paths 0,2 valid; path 1 invalid
                    [[True, False, True]],  # tx 3: paths 0,2 valid; path 1 invalid
                    [[True, False, True]],  # tx 4: paths 0,2 valid; path 1 invalid
                ]
            ],
        ]
    )

    valid_data = mi.TensorXb(valid_data)
    # Expected: path 0 is valid everywhere, path 1 is invalid everywhere,
    # path 2 is invalid for rx0/tx2 (so not all valid)

    # Check that the correct paths are marked as valid
    # Path 0: valid everywhere -> True
    # Path 1: invalid everywhere -> False
    # Path 2: invalid at rx0/tx2 -> False
    expected_mask = np.array([True, False, False])

    # Test passing tensor directly
    mask = sionna_utils.paths.get_all_valid_paths_mask(valid_data)
    assert mask.shape == (3,)
    np.testing.assert_array_equal(mask, expected_mask)

    # Test with paths 1
    p_solver = PathSolver()
    paths = p_solver(
        scene,
        max_depth=3,
        los=False,
        specular_reflection=True,
        diffuse_reflection=True,
        synthetic_array=False,
        seed=42,
    )

    mask = sionna_utils.paths.get_all_valid_paths_mask(paths)
    assert mask.shape == (paths.valid.shape[-1],)
    np.testing.assert_array_equal(mask, [True])

    # Test with paths 2
    for name, obj in scene.objects.items():
        obj.radio_material.scattering_coefficient = 0.1
    p_solver = PathSolver()
    paths = p_solver(
        scene,
        max_depth=3,
        los=False,
        specular_reflection=True,
        diffuse_reflection=True,
        synthetic_array=False,
        seed=42,
    )

    mask = sionna_utils.paths.get_all_valid_paths_mask(paths)
    assert mask.shape == (paths.valid.shape[-1],)
    np.testing.assert_array_equal(
        mask,
        [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
        ],
    )


def test_get_path_depths(scene):
    # Test with shape [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    # Using: [3, 2, 1, 2, 1, 4] - max_depth=3, 2 rx, 2 tx, 4 paths

    NONE = sionna.rt.constants.InteractionType.NONE
    REFRACTION = sionna.rt.constants.InteractionType.REFRACTION
    DIFFRACTION = sionna.rt.constants.InteractionType.DIFFRACTION

    # Create interaction types tensor
    # Path 0: 0 interactions (direct/LOS path - all NONE)
    # Path 1: 1 interaction (one reflection)
    # Path 2: 2 interactions (two reflections)
    # Path 3: 3 interactions (three reflections)
    types_data = np.array(
        [
            [  # depth 0
                [  # rx 0
                    [  # rx_ant 0
                        [[NONE, REFRACTION, REFRACTION, REFRACTION]],  # tx 0
                        [[NONE, REFRACTION, REFRACTION, REFRACTION]],  # tx 1
                    ]
                ],
                [  # rx 1
                    [  # rx_ant 0
                        [[NONE, REFRACTION, REFRACTION, REFRACTION]],  # tx 0
                        [[NONE, REFRACTION, REFRACTION, REFRACTION]],  # tx 1
                    ]
                ],
            ],
            [  # depth 1
                [  # rx 0
                    [  # rx_ant 0
                        [[NONE, NONE, DIFFRACTION, REFRACTION]],  # tx 0
                        [[NONE, NONE, DIFFRACTION, REFRACTION]],  # tx 1
                    ]
                ],
                [  # rx 1
                    [  # rx_ant 0
                        [[NONE, NONE, DIFFRACTION, REFRACTION]],  # tx 0
                        [[NONE, NONE, DIFFRACTION, REFRACTION]],  # tx 1
                    ]
                ],
            ],
            [  # depth 2
                [  # rx 0
                    [  # rx_ant 0
                        [[NONE, NONE, NONE, REFRACTION]],  # tx 0
                        [[NONE, NONE, NONE, REFRACTION]],  # tx 1
                    ]
                ],
                [  # rx 1
                    [  # rx_ant 0
                        [[NONE, NONE, NONE, REFRACTION]],  # tx 0
                        [[NONE, NONE, NONE, REFRACTION]],  # tx 1
                    ]
                ],
            ],
        ],
        dtype=np.uint32,
    )

    types_tensor = mi.TensorXu(types_data)

    # Calculate depths - test passing tensor directly
    depths = sionna_utils.paths.get_path_depths(types_tensor)

    # Check shape - should remove the max_depth dimension
    assert depths.shape == (
        2,
        1,
        2,
        1,
        4,
    ), f"Expected shape (2, 1, 2, 1, 4), got {depths.shape}"

    # Check that all rx/tx combinations have the same depths for each path
    # Path 0: 0 interactions, Path 1: 1 interaction, Path 2: 2 interactions, Path 3: 3 interactions
    expected_depths = np.array([0, 1, 2, 3])

    # Check each rx/tx combination
    for rx in range(2):
        for tx in range(2):
            path_depths = depths[rx, 0, tx, 0, :]
            np.testing.assert_array_equal(
                path_depths,
                expected_depths,
                err_msg=f"Depths mismatch for rx={rx}, tx={tx}",
            )

    # Test with simplified shape [max_depth, num_rx, num_tx, num_paths]
    # Using: [2, 2, 2, 3] - max_depth=2, 2 rx, 2 tx, 3 paths
    types_data_simple = np.array(
        [
            [  # depth 0
                [
                    [NONE, REFRACTION, REFRACTION],  # rx 0, tx 0
                    [NONE, REFRACTION, REFRACTION],
                ],  # rx 0, tx 1
                [
                    [NONE, REFRACTION, REFRACTION],  # rx 1, tx 0
                    [NONE, REFRACTION, REFRACTION],
                ],  # rx 1, tx 1
            ],
            [  # depth 1
                [
                    [NONE, NONE, DIFFRACTION],  # rx 0, tx 0
                    [NONE, NONE, DIFFRACTION],
                ],  # rx 0, tx 1
                [
                    [NONE, NONE, DIFFRACTION],  # rx 1, tx 0
                    [NONE, NONE, DIFFRACTION],
                ],  # rx 1, tx 1
            ],
        ],
        dtype=np.uint32,
    )

    types_tensor_simple = mi.TensorXu(types_data_simple)
    depths_simple = sionna_utils.paths.get_path_depths(types_tensor_simple)

    # Check shape
    assert depths_simple.shape == (
        2,
        2,
        3,
    ), f"Expected shape (2, 2, 3), got {depths_simple.shape}"

    # Check depths: path 0 has 0 interactions, path 1 has 1, path 2 has 2
    expected_depths_simple = np.array([0, 1, 2])
    for rx in range(2):
        for tx in range(2):
            np.testing.assert_array_equal(
                depths_simple[rx, tx, :],
                expected_depths_simple,
                err_msg=f"Depths mismatch for rx={rx}, tx={tx}",
            )

    # Test on path 1
    p_solver = PathSolver()
    paths = p_solver(
        scene,
        max_depth=3,
        los=False,
        specular_reflection=True,
        diffuse_reflection=True,
        synthetic_array=False,
        seed=42,
    )
    depths_paths = sionna_utils.paths.get_path_depths(paths)
    assert depths_paths.shape == (1, 6, 1, 2, 1)
    assert np.all(depths_paths == 1)

    # Test on path 2
    for name, obj in scene.objects.items():
        obj.radio_material.scattering_coefficient = 0.1
    p_solver = PathSolver()
    paths = p_solver(
        scene,
        max_depth=3,
        los=True,
        specular_reflection=True,
        diffuse_reflection=True,
        synthetic_array=False,
        seed=42,
    )

    depths_paths = sionna_utils.paths.get_path_depths(paths)
    assert depths_paths.shape == (1, 6, 1, 2, 22)
    assert depths_paths.min() == 0
    assert depths_paths.max() == 1


def test_get_a(scene):
    for name, obj in scene.objects.items():
        obj.radio_material.scattering_coefficient = 0.1
    p_solver = PathSolver()
    paths = p_solver(
        scene,
        max_depth=3,
        los=True,
        specular_reflection=True,
        diffuse_reflection=True,
        synthetic_array=False,
        seed=42,
    )

    a = sionna_utils.paths.get_a(paths)
    a_mag = sionna_utils.paths.get_a_mag(paths)

    assert a.shape == paths.a[0].shape
    assert a_mag.shape == paths.a[0].shape
