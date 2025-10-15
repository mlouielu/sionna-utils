from functools import wraps
from typing import Union

import sionna.rt
import mitsuba as mi
import drjit as dr
import numpy as np


def _w(attr_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(data: Union[sionna.rt.Paths, mi.TensorXu], *args, **kwargs):
            if isinstance(data, sionna.rt.Paths):
                tensor = getattr(data, attr_name)
                return func(tensor, *args, **kwargs)
            else:
                return func(data, *args, **kwargs)

        return wrapper

    return decorator


@_w("interactions")
def get_path_depths(interactions: mi.TensorXu) -> np.ndarray:
    """Calculate the depth (number of interactions) for each path.

    Counts the number of non-NONE interactions along each path. The depth represents
    how many times the signal interacted with before reaching the receiver.

    Args:
        interactions: Interaction tensor from paths.interactions
               Shape: [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
               or [max_depth, num_rx, num_tx, num_paths]

    Returns:
        Integer numpy array with the depth of each path. The shape depends on input:
        - If input is [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths],
          returns shape [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
        - If input is [max_depth, num_rx, num_tx, num_paths],
          returns shape [num_rx, num_tx, num_paths]

    Example:
        >>> depths = get_path_depths(paths.interactions)
        >>> # Find paths with exactly 2 interactions (e.g., one reflection)
        >>> two_bounce_mask = (depths == 2)
    """
    if not isinstance(interactions, mi.TensorXu):
        raise TypeError(f"Input must be a mi.TensorXu, got {type(interactions)}")

    # Count non-NONE interactions along the depth axis (axis 0)
    depth_counts = (
        interactions.numpy() != sionna.rt.constants.InteractionType.NONE
    ).sum(axis=0)
    return depth_counts


@_w("valid")
def get_all_valid_paths_mask(valid: mi.TensorXb) -> np.ndarray:
    if not isinstance(valid, mi.TensorXb):
        raise TypeError(f"Input must be a mi.TensorXb, got {type(valid)}")
    return valid.numpy().all(axis=tuple(range(valid.numpy().ndim - 1)))


@_w("a")
def get_a(a: tuple[mi.TensorXf, mi.TensorXf]) -> np.ndarray:
    if not isinstance(a, tuple):
        raise TypeError(f"Input must be a tuple, got {type(a)}")

    if not isinstance(a[0], mi.TensorXf) or not isinstance(a[1], mi.TensorXf):
        raise TypeError(
            f"Tuple elements must be mi.TensorXf, got {type(a[0])} and {type(a[1])}"
        )

    return a[0].numpy() + 1j * a[1].numpy()


def get_a_mag(obj) -> np.ndarray:
    """Get channel coefficient magnitude (full shape maintained).

    Args:
        obj: Paths object or tuple of (real, imag) tensors

    Returns:
        Magnitude array with original shape preserved

    Example:
        >>> a_mag = get_a_mag(paths)
        >>> # Shape: (1, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths)
    """
    return np.abs(get_a(obj))


@_w("a")
def get_a_mag_reduced(
    a: tuple[mi.TensorXf, mi.TensorXf], mode: str = "max"
) -> np.ndarray:
    """Get channel coefficient magnitude reduced across rx/tx dimensions.

    Reduces the magnitude across all dimensions except the path dimension,
    allowing for easy filtering based on aggregate magnitude criteria.

    Args:
        a: Paths object or tuple of (real, imag) tensors from paths.a
        mode: Reduction mode - 'max', 'min', 'mean', 'median' (default: 'max')

    Returns:
        Array of shape (num_paths,) with reduced magnitude values

    Example:
        >>> # Get worst-case (min) magnitude per path
        >>> mag = get_a_mag_reduced(paths, mode='min')
        >>> mask = ((mag > 0.1) & (mag < 0.2)) | ((mag > 0.5) & (mag < 0.7))
        >>>
        >>> # Get best-case (max) magnitude per path
        >>> mag = get_a_mag_reduced(paths, mode='max')
        >>> mask = mag > 0.5
        >>>
        >>> # Get average magnitude per path
        >>> mag = get_a_mag_reduced(paths, mode='mean')
        >>> mask = mag < 0.2
    """
    a_mag = get_a_mag(a)

    # Reduce across all dimensions except the last (paths dimension)
    axes = tuple(range(a_mag.ndim - 1))

    if mode == "max":
        return a_mag.max(axis=axes)
    elif mode == "min":
        return a_mag.min(axis=axes)
    elif mode == "mean":
        return a_mag.mean(axis=axes)
    elif mode == "median":
        return np.median(a_mag, axis=axes)
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Must be one of: 'max', 'min', 'mean', 'median'"
        )
