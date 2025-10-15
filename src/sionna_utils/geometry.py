import open3d as o3d
import sionna.rt
import mitsuba as mi
import drjit as dr
import numpy as np


def load_mesh_from_open3d(mesh: o3d.geometry.TriangleMesh, name: str) -> mi.Mesh:
    """Convert an Open3D triangle mesh to a Mitsuba mesh.

    Args:
        mesh: Open3D triangle mesh to convert
        name: Name identifier for the Mitsuba mesh

    Returns:
        Mitsuba mesh with vertices, normals, and faces transferred from the Open3D mesh
    """
    props = mi.Properties()
    props.set_id(name)
    props["face_normals"] = mesh.has_triangle_normals()

    mi_mesh = mi.Mesh(
        name=name,
        vertex_count=len(mesh.vertices),
        face_count=len(mesh.triangles),
        props=props,
        has_vertex_normals=mesh.has_vertex_normals(),
        has_vertex_texcoords=False,
    )

    params = mi.traverse(mi_mesh)
    params["vertex_positions"] = mi.TensorXf(np.asarray(mesh.vertices).ravel())
    params["vertex_normals"] = mi.TensorXf(np.asarray(mesh.vertex_normals).ravel())
    params["faces"] = mi.TensorXu(np.asarray(mesh.triangles).ravel())
    params.update()

    return mi_mesh


def create_coordinate_frame(
    scene: sionna.rt.Scene, scale: float = 1.0, position: list[float] = [0.0, 0.0, 0.0]
) -> mi.Mesh:
    """Create and add a 3D coordinate frame to a Sionna scene for visualization.

    Creates a coordinate frame with three colored arrows representing X (red), Y (green),
    and Z (blue) axes, plus a black sphere at the origin. The frame is automatically
    added to the scene.

    Args:
        scene: Sionna RT scene to add the coordinate frame to
        scale: Scale factor for the coordinate frame size (default: 1.0)
        position: 3D position [x, y, z] where the coordinate frame origin should be placed (default: [0, 0, 0])

    Returns:
        Mitsuba mesh of the Z-axis arrow (for compatibility, though all components are added to scene)

    Note:
        This function creates internal materials named '_coord_black', '_coord_red',
        '_coord_green', and '_coord_blue' if they don't already exist in the scene.
    """
    if scene.get("_coord_black") is None:
        # Initialize materials only once
        scene.add(sionna.rt.RadioMaterial("_coord_black", 1.0, color=[0, 0, 0]))
        scene.add(sionna.rt.RadioMaterial("_coord_red", 1.0, color=[1, 0, 0]))
        scene.add(sionna.rt.RadioMaterial("_coord_green", 1.0, color=[0, 1, 0]))
        scene.add(sionna.rt.RadioMaterial("_coord_blue", 1.0, color=[0, 0, 1]))

    # Add coordinate frame for visualization
    origin_name = "_coord_origin"
    origin = load_mesh_from_open3d(
        o3d.geometry.TriangleMesh.create_sphere(0.05 * scale), origin_name
    )

    length = 1.0 * scale

    arrow_x_name = "_coord_arrow_x"
    arrow_x = load_mesh_from_open3d(
        o3d.geometry.TriangleMesh.create_arrow(
            cone_height=0.1 * scale,
            cone_radius=0.03 * scale,
            cylinder_height=length - 0.1 * scale,
            cylinder_radius=0.01 * scale,
            resolution=20,
            cylinder_split=4,
            cone_split=1,
        ),
        arrow_x_name,
    )

    arrow_y_name = "_coord_arrow_y"
    arrow_y = load_mesh_from_open3d(
        o3d.geometry.TriangleMesh.create_arrow(
            cone_height=0.1 * scale,
            cone_radius=0.03 * scale,
            cylinder_height=length - 0.1 * scale,
            cylinder_radius=0.01 * scale,
            resolution=20,
            cylinder_split=4,
            cone_split=1,
        ),
        arrow_y_name,
    )

    arrow_z_name = "_coord_arrow_z"
    arrow_z = load_mesh_from_open3d(
        o3d.geometry.TriangleMesh.create_arrow(
            cone_height=0.1 * scale,
            cone_radius=0.03 * scale,
            cylinder_height=length - 0.1 * scale,
            cylinder_radius=0.01 * scale,
            resolution=20,
            cylinder_split=4,
            cone_split=1,
        ),
        arrow_z_name,
    )

    so_origin = sionna.rt.SceneObject(
        name=origin_name,
        mi_mesh=origin,
        radio_material=scene.get("_coord_black"),
    )
    so_arrow_x = sionna.rt.SceneObject(
        name=arrow_x_name,
        mi_mesh=arrow_x,
        radio_material=scene.get("_coord_red"),
    )
    so_arrow_y = sionna.rt.SceneObject(
        name=arrow_y_name,
        mi_mesh=arrow_y,
        radio_material=scene.get("_coord_green"),
    )
    so_arrow_z = sionna.rt.SceneObject(
        name=arrow_z_name,
        mi_mesh=arrow_z,
        radio_material=scene.get("_coord_blue"),
    )

    scene.edit(add=[so_origin, so_arrow_x, so_arrow_y, so_arrow_z])

    pos = np.array(position)

    so_origin.position = mi.Point3f(pos)

    so_arrow_x.position = mi.Point3f(np.array([length / 2, 0, 0]) + pos)
    so_arrow_x.orientation = [0, np.pi / 2, 0]

    so_arrow_y.position = mi.Point3f(np.array([0, length / 2, 0]) + pos)
    so_arrow_y.orientation = [0, 0, -np.pi / 2]

    so_arrow_z.position = mi.Point3f(np.array([0, 0, length / 2]) + pos)
