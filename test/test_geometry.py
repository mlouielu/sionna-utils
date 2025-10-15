import pytest

import open3d as o3d
import sionna.rt
import mitsuba as mi
import matplotlib.pyplot as plt

import sionna_utils


def test_load_mesh_from_open3d():
    mesh = o3d.geometry.TriangleMesh.create_box()

    mesh_name = "box"
    mi_mesh = sionna_utils.geometry.load_mesh_from_open3d(mesh, mesh_name)

    so = sionna.rt.SceneObject(
        mi_mesh=mi_mesh,
        name=mesh_name,
        radio_material=sionna.rt.ITURadioMaterial("test_box_mat", "metal", 1.0),
    )

    scene = sionna.rt.load_scene(sionna.rt.scene.simple_reflector, merge_shapes=False)
    scene.edit(add=so)


def test_create_coordinate_frame():
    scene = sionna.rt.load_scene(sionna.rt.scene.simple_reflector, merge_shapes=False)

    sionna_utils.geometry.create_coordinate_frame(scene, position=[0, 0, 0])
