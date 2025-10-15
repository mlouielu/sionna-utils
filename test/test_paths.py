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
