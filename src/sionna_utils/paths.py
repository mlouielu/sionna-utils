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


@_w("objects")
def get_paths_hit_objects(
    objects: mi.TensorXu, object_ids: int | list[int], mode: str = "any"
) -> np.ndarray:
    """Check if paths hit specific object(s) in any order.

    Args:
        objects: Paths object or objects tensor from paths.objects
                Shape: [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
                or [max_depth, num_rx, num_tx, num_paths]
        object_ids: Single object ID (int) or list of object IDs to check
        mode: Reduction mode across rx/tx dimensions:
              'any' - path hits object for at least one rx/tx (default)
              'all' - path hits object for all rx/tx combinations
              'per_link' - path hits object in each rx/tx pair link

    Returns:
        If object_ids is int: Boolean array of shape (num_paths,)
        If object_ids is list: Boolean array of shape (num_objects, num_paths)

    Example:
        >>> # Paths that hit object 5 at some point
        >>> mask = get_paths_hit_objects(paths, 5)
        >>>
        >>> # Check multiple objects
        >>> masks = get_paths_hit_objects(paths, [1, 5, 10])
        >>> # Paths hitting object 1 OR 5 OR 10
        >>> mask = masks.any(axis=0)
        >>> # Paths hitting object 1 AND 5 AND 10 (any order)
        >>> mask = masks.all(axis=0)
        >>> # Paths hitting object 1 but NOT 5
        >>> mask = masks[0] & ~masks[1]
    """
    if not isinstance(objects, mi.TensorXu):
        raise TypeError(f"Input must be a mi.TensorXu, got {type(objects)}")

    objects_np = objects.numpy()

    # Handle single vs multiple IDs
    if isinstance(object_ids, int):
        object_ids = [object_ids]
        return_single = True
    else:
        return_single = False

    results = []
    for obj_id in object_ids:
        # Check if the object ID appears at any depth for each path
        hit_mask = (objects_np == obj_id).any(axis=0)

        # Reduce across rx/tx dimensions
        axes = tuple(range(hit_mask.ndim - 1))

        if mode == "per_link":
            results.append(hit_mask)
        elif mode == "any":
            results.append(hit_mask.any(axis=axes))
        elif mode == "all":
            results.append(hit_mask.all(axis=axes))
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be one of: 'any', 'all'")

    result = np.array(results)
    return result[0] if return_single else result


@_w("objects")
def get_paths_hit_sequence(
    objects: mi.TensorXu, sequence: list[int], mode: str = "any"
) -> np.ndarray:
    """Check if paths hit objects in a specific sequence.

    Checks if the path's interaction sequence matches the given object ID sequence,
    useful for identifying specific reflection patterns like ghost reflections.

    Args:
        objects: Paths object or objects tensor from paths.objects
                Shape: [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
                or [max_depth, num_rx, num_tx, num_paths]
        sequence: List of object IDs in the order they should be hit
                 e.g., [1, 2, 1] for bounce off obj1, then obj2, then obj1 again
        mode: Reduction mode across rx/tx dimensions:
              'any' - match for at least one rx/tx (default)
              'all' - match for all rx/tx combinations
              'per_link' - don't reduct across rx/tx dimensions

    Returns:
        Boolean array of shape (num_paths,) where True indicates the path
        matches the sequence

    Example:
        >>> # Find ghost reflections: object1 -> object2 -> object1
        >>> mask = get_paths_hit_sequence(paths, [1, 2, 1])
        >>>
        >>> # Double bounce on same object
        >>> mask = get_paths_hit_sequence(paths, [5, 5])
        >>>
        >>> # Combine with depth check
        >>> depths = get_path_depths(paths)
        >>> mask = get_paths_hit_sequence(paths, [1, 2, 1]) & (depths == 3)
    """
    if not isinstance(objects, mi.TensorXu):
        raise TypeError(f"Input must be a mi.TensorXu, got {type(objects)}")

    if not isinstance(sequence, (list, tuple)) or len(sequence) == 0:
        raise ValueError("sequence must be a non-empty list or tuple of object IDs")

    objects_np = objects.numpy()
    seq_len = len(sequence)

    # Check if we have enough depth for this sequence
    if objects_np.shape[0] < seq_len:
        # Not enough depth, no paths can match
        return np.zeros(objects_np.shape[-1], dtype=bool)

    # Extract the first seq_len interactions for each path
    interactions_subset = objects_np[:seq_len]

    # Check if each depth matches the sequence
    matches = np.ones(interactions_subset.shape[1:], dtype=bool)
    for depth_idx, expected_obj_id in enumerate(sequence):
        matches &= interactions_subset[depth_idx] == expected_obj_id

    # Reduce across rx/tx dimensions
    axes = tuple(range(matches.ndim - 1))

    if mode == "per_link":
        return matches
    elif mode == "any":
        return matches.any(axis=axes)
    elif mode == "all":
        return matches.all(axis=axes)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be one of: 'any', 'all'")
