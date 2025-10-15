import sionna.rt
import mitsuba as mi
import numpy as np


def get_path_depths(types: mi.TensorXu) -> np.ndarray:
    """Calculate the depth (number of interactions) for each path.

    Counts the number of non-NONE interactions along each path. The depth represents
    how many times the signal interacted with objects (reflections, diffractions, etc.)
    before reaching the receiver.

    Args:
        types: Interaction types tensor from paths.types
               Shape: [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
               or [max_depth, num_rx, num_tx, num_paths]

    Returns:
        Integer numpy array with the depth of each path. The shape depends on input:
        - If input is [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths],
          returns shape [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
        - If input is [max_depth, num_rx, num_tx, num_paths],
          returns shape [num_rx, num_tx, num_paths]

    Example:
        >>> depths = get_path_depths(paths.types)
        >>> # Find paths with exactly 2 interactions (e.g., one reflection)
        >>> two_bounce_mask = (depths == 2)
    """
    types_np = types.numpy()
    # Count non-NONE interactions along the depth axis (axis 0)
    depth_counts = (types_np != sionna.rt.constants.InteractionType.NONE).sum(axis=0)
    return depth_counts


def filter_only_paths_all_valid(paths: sionna.rt.Paths, apply_mask=False):
    """Filter propagation paths to identify those valid across all receivers and transmitters.

    Computes a boolean mask indicating which paths are valid across all batch dimensions,
    receivers, receiver antennas, transmitters, and transmitter antennas.

    Args:
        paths: Sionna RT Paths object containing propagation path data
        apply_mask: If True, applies the computed mask to paths._valid in-place,
                   filtering out invalid paths. If False, only returns the mask
                   without modifying the paths object (default: False)

    Returns:
        Boolean numpy array of shape (num_paths,) where True indicates a path
        is valid across all dimensions

    Example:
        >>> mask = filter_only_paths_all_valid(paths)
        >>> # Use mask to select valid paths
        >>> valid_indices = np.where(mask)[0]
    """
    mask = paths.valid.numpy().all(axis=(0, 1, 2, 3))

    if apply_mask:
        paths._valid = mi.TensorXu(paths.valid.numpy() & mask)

    return mask
