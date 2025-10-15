import sionna.rt
import mitsuba as mi


def filter_only_paths_all_valid(paths: sionna.rt.Paths, apply_mask=False):
    mask = paths.valid.numpy().all(axis=(0, 1, 2, 3))

    if apply_mask:
        paths._valid = mi.TensorXu(paths.valid.numpy() & mask)

    return mask
