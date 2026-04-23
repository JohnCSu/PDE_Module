import numpy as np
from typing import Literal
import gmsh
from pde_module.mesh.mesh import Mesh


CellType = Literal["hex", "tet", "wedge"]


def generate_cube_mesh(
    size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    divisions: tuple[int, int, int] = (2, 2, 2),
    cell_type: CellType = "hex",
    mesh_size_factor: float = 0.1,
    show_gui: bool = False,
    return_as_mesh = True,
) -> Mesh:
    """
    Generate a cube mesh using Gmsh. This is vibe coded, use this as a basic testing example

    Args:
        size: (Lx, Ly, Lz) dimensions of the cube
        divisions: (nx, ny, nz) number of elements along each axis
        cell_type: "hex" | "tet" | "wedge" - type of mesh elements
        mesh_size_factor: Relative mesh size (0.0-1.0), smaller = finer mesh
        show_gui: Whether to show Gmsh GUI during generation

    Returns:
        pde_module.mesh.Mesh object
    """
    gmsh.initialize()

    model = gmsh.model()
    model.add("cube_mesh")
    model.setCurrent("cube_mesh")

    Lx, Ly, Lz = size
    nx, ny, nz = divisions

    lc = min(Lx, Ly, Lz) / max(nx, ny, nz)

    if cell_type == "wedge":
        box_tag = None
    else:
        box_tag = model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
    model.occ.synchronize()

    if cell_type == "hex":
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
        gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        model.mesh.setTransfiniteCurve(1, nx + 1)
        model.mesh.setTransfiniteCurve(2, ny + 1)
        model.mesh.setTransfiniteCurve(3, nx + 1)
        model.mesh.setTransfiniteCurve(4, ny + 1)
        model.mesh.setTransfiniteCurve(5, nx + 1)
        model.mesh.setTransfiniteCurve(6, ny + 1)
        model.mesh.setTransfiniteCurve(7, nx + 1)
        model.mesh.setTransfiniteCurve(8, ny + 1)
        model.mesh.setTransfiniteCurve(9, nz + 1)
        model.mesh.setTransfiniteCurve(10, nz + 1)
        model.mesh.setTransfiniteCurve(11, nz + 1)
        model.mesh.setTransfiniteCurve(12, nz + 1)

        for surf_tag in range(1, 7):
            model.mesh.setTransfiniteSurface(surf_tag)
            model.mesh.setRecombine(2, surf_tag)

        model.mesh.setTransfiniteVolume(box_tag)
        model.mesh.generate(3)

    elif cell_type == "tet":
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
        gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
        model.mesh.generate(3)

    elif cell_type == "wedge":
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
        gmsh.option.setNumber("Mesh.MeshSizeMax", lc)

        bottom_face_tag = model.occ.addRectangle(0, 0, 0, Lx, Ly)
        model.occ.synchronize()

        model.mesh.setTransfiniteCurve(1, nx + 1)
        model.mesh.setTransfiniteCurve(2, ny + 1)
        model.mesh.setTransfiniteCurve(3, nx + 1)
        model.mesh.setTransfiniteCurve(4, ny + 1)
        model.mesh.setTransfiniteSurface(bottom_face_tag)
        model.mesh.generate(2)

        out_dim_tags = model.occ.extrude(
            [(2, bottom_face_tag)], 0, 0, Lz, numElements=[nz], recombine=True
        )
        model.occ.synchronize()

        model.mesh.generate(3)

    node_tags, node_coords, _ = model.mesh.getNodes()

    nodes = np.zeros((len(node_tags), 3), dtype=np.float32)
    for i, tag in enumerate(node_tags):
        idx = int(tag) - 1
        nodes[idx, 0] = node_coords[3 * i]
        nodes[idx, 1] = node_coords[3 * i + 1]
        nodes[idx, 2] = node_coords[3 * i + 2]

    element_types, element_tags, element_node_tags = model.mesh.getElements(dim=3)

    if not element_types or not element_node_tags:
        gmsh.finalize()
        raise ValueError("No 3D elements found in mesh")

    all_cells = []
    all_types = []

    GMSH_TO_VTK = {4: 10, 5: 12, 6: 13}

    for elem_type, tags, node_tags in zip(
        element_types, element_tags, element_node_tags
    ):
        vtk_type = GMSH_TO_VTK.get(elem_type)
        if vtk_type is None:
            continue

        num_nodes_per_elem = {10: 4, 12: 8, 13: 6}.get(vtk_type, 4)

        for i in range(len(tags)):
            start = i * num_nodes_per_elem
            end = start + num_nodes_per_elem
            cell_nodes = (node_tags[start:end] - 1).astype(np.int32)

            cell_array = np.concatenate([[num_nodes_per_elem], cell_nodes])
            all_cells.append(cell_array)
            all_types.append(vtk_type)

    cells_array = (
        np.concatenate(all_cells) if all_cells else np.array([], dtype=np.int32)
    )
    cell_types_array = np.array(all_types, dtype=np.int8)
    if show_gui:
        gmsh.fltk.run()
    gmsh.finalize()

    
    if return_as_mesh:
        mesh = Mesh(
            nodes=nodes,
            cells_connectivity=cells_array,
            cell_types=cell_types_array,
        )
        return mesh
    else:
        return nodes,cells_array,cell_types_array
