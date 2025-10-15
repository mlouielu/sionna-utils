import sionna.rt
import mitsuba as mi


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
