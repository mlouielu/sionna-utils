import pytest
import numpy as np

import sionna.rt
import mitsuba as mi

import sionna_utils


def test_paths_filter_only_paths_all_valid():
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
    # Expected: path 0 is valid everywhere, path 1 is invalid everywhere,
    # path 2 is invalid for rx0/tx2 (so not all valid)

    # Create a minimal mock Paths object
    class MockPaths:
        def __init__(self, valid_array):
            self._valid = mi.TensorXu(valid_array.astype(np.uint32))

        @property
        def valid(self):
            return self._valid

    paths = MockPaths(valid_data)

    # Test without applying mask
    mask = sionna_utils.paths.filter_only_paths_all_valid(paths, apply_mask=False)

    # Check that mask has correct shape (should reduce to just the path dimension)
    assert mask.shape == (3,), f"Expected mask shape (3,), got {mask.shape}"

    # Check that the correct paths are marked as valid
    # Path 0: valid everywhere -> True
    # Path 1: invalid everywhere -> False
    # Path 2: invalid at rx0/tx2 -> False
    expected_mask = np.array([True, False, False])
    np.testing.assert_array_equal(mask, expected_mask)

    # Test with applying mask
    paths2 = MockPaths(valid_data.copy())
    mask2 = sionna_utils.paths.filter_only_paths_all_valid(paths2, apply_mask=True)

    # Verify the mask was applied to paths._valid
    assert paths2._valid is not None
    # The applied mask should have the same shape as original valid
    assert paths2.valid.numpy().shape == valid_data.shape


def test_get_path_depths():
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

    # Calculate depths
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
