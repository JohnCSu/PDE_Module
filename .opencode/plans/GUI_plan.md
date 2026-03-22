# Trame GUI Implementation Plan for PDE Module

## Overview

A web-based GUI using Trame (kitware/kitframe) for configuring and running PDE simulations (initially Navier-Stokes). The GUI is **independent** of the solver - it manages state via numpy arrays and communicates through integer IDs.

---

## Architecture

### Delivery
- **Format**: Web application via `trame server`
- **Access**: Browser-based, runs as local server
- **UI Framework**: Vuetify3 for controls
- **Visualization**: VTK.js via Trame-VTK for 3D rendering

### Pattern
- **ServerBackend** (not FrontendOnly) - enables Warp GPU execution on server
- **VueRouter** for multi-page navigation

```
┌─────────────────────────────────────────────────────────────────┐
│                        Trame Server                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Mesh    │  │  Config  │  │  BC/IC   │  │  Solve   │        │
│  │  View    │→ │  View    │→ │  View    │→ │  View    │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              State Manager (numpy arrays)                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌───────────────────────────▼─────────────────────────────┐   │
│  │              Solver Interface (ID-based)                   │   │
│  └───────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## ID System

All conditions use integer IDs. GUI stores only IDs; solver interprets them.

### Condition Type IDs

| ID | Name | Description |
|----|------|-------------|
| 0 | NONE | No condition |
| 1 | BOUNDARY_CONDITION | BC category |
| 2 | FORCE | Body force category |

### BC Type IDs (within BOUNDARY_CONDITION category)

| ID | Name | Description |
|----|------|-------------|
| 1 | NO_SLIP | Velocity = 0 on wall |
| 2 | IMPERMEABLE | Normal velocity = 0 |
| 3 | DIRICHLET | Prescribed value |
| 4 | VON_NEUMANN | Prescribed flux (dφ/dn = value) |
| 5 | FARFIELD | Far-field/sponge layer |
| 6 | PERIODIC | Periodic boundary |

### Force Type IDs (within FORCE category)

| ID | Name | Description |
|----|------|-------------|
| 1 | NONE | No force |
| 2 | CONSTANT | Constant body force vector |
| 3 | SPATIAL_FUNC | Spatially varying force |

### PDE Type IDs

| ID | Name | Description |
|----|------|-------------|
| 1 | DIFFUSION | Heat/diffusion equation |
| 2 | ADVECTION_DIFFUSION | Advection-diffusion |
| 3 | NAVIER_STOKES | Incompressible Navier-Stokes |

---

## Data Structures (NumPy Arrays)

All state stored as numpy arrays for portability.

### Mesh State

```python
# Mesh geometry (passed to solver)
mesh_nodes: np.ndarray = None       # shape: (N, 3), float32
mesh_cells: np.ndarray = None       # shape: (M, K), int32 (connectivity)
mesh_cell_types: np.ndarray = None  # shape: (M,), int8 (VTK cell types)
mesh_dimensions: np.ndarray = None  # shape: (3,), int32

# Groups: name → numpy array of indices
mesh_groups: dict[str, np.ndarray] = {}
# Example: {"inlet": array([0,1,2,...]), "outlet": array([100,101,...])}
```

### Group Name Mapping

```python
# Separate mapping from group name to integer ID
group_name_to_id: dict[str, np.int32] = {}
# Example: {"inlet": np.int32(0), "outlet": np.int32(1)}

# Reverse mapping for display
group_id_to_name: dict[np.int32, str] = {}
# Example: {np.int32(0): "inlet", np.int32(1): "outlet"}
```

### Boundary Conditions

```python
# BCs stored with group ID reference (one BC per group)
# Shape: (N, 3) where each row is [group_id, bc_type_id, value]
bc_conditions: np.ndarray = None
# Example:
# [[0, NO_SLIP, 0.0],      # group 0 (inlet) → NO_SLIP
#  [1, DIRICHLET, 1.0]]    # group 1 (outlet) → DIRICHLET velocity=1.0

# NO_BC (0) means no BC assigned to that group
```

### Initial Conditions

```python
# Shape: (N, 2) where each row is [field_id, value]
initial_conditions: np.ndarray = None
# Example:
# [[velocity_x, 0.0],
#  [velocity_y, 0.0],
#  [pressure, 1.0]]
```

### Forces

```python
# Shape: (N, 4) where each row is [force_type_id, fx, fy, fz]
forces: np.ndarray = None
# Example (CONSTANT force):
# [[CONSTANT, 1.0, 0.0, 0.0]]  # Body force in +X direction
```

### Simulation Parameters

```python
sim_params: np.ndarray = None  # Shape: (P,) with fields below
# Field names: dtype=[('pde_type','i4'), ('viscosity','f4'), ('dt','f4'),
#                    ('total_steps','i4'), ('current_step','i4')]
```

### Results Buffer

```python
# For time playback - list of numpy arrays, one per saved timestep
results_buffer: list[np.ndarray] = []
# Each array contains full field state for that timestep
```

### Visualization Settings

```python
viz_settings: np.ndarray = None  # Shape: (V,) with fields
# Field names: dtype=[('viz_field','i4'), ('viz_mode','i4'),
#                    ('time_index','i4'), ('show_vectors','i4'),
#                    ('show_scalar','i4'), ('scalar_range','f4',2)]
```

---

## Views/Pages

Four views connected via VueRouter.

| View | Route | Purpose |
|------|-------|---------|
| MeshImport | `/` | Import VTK file or create UniformGridMesh |
| SimConfig | `/config` | Set PDE type, viscosity, dt, simulation params |
| Conditions | `/conditions` | Apply BCs and initial conditions |
| Solve | `/solve` | Run simulation, visualize results |

### View 1: MeshImport (`/`)

**Purpose**: Load or create mesh for simulation

**Components**:
- File upload for VTK files (`.vtu`, `.vtk`, `.msh`)
- UniformGridMesh creator form:
  - Grid dimensions (nx, ny, nz)
  - Domain size (Lx, Ly, Lz)
  - Origin (Ox, Oy, Oz)
- Mesh preview with VTK
- List of available groups extracted from mesh

**Outputs**:
- `mesh_nodes`, `mesh_cells`, `mesh_cell_types`, `mesh_dimensions`
- `mesh_groups`, `group_name_to_id`, `group_id_to_name`

### View 2: SimConfig (`/config`)

**Purpose**: Configure simulation parameters

**Components**:
- PDE type selector (dropdown)
- Viscosity input (float)
- Time step dt (float)
- Total steps (int)
- Initial condition editor:
  - Select field from available fields
  - Enter constant value OR select function
- Force editor:
  - Force type selector
  - Vector inputs for CONSTANT type

**Outputs**:
- `sim_params`
- `initial_conditions`
- `forces`

### View 3: Conditions (`/conditions`)

**Purpose**: Apply boundary conditions to mesh groups

**Components**:
- Group list (from `mesh_groups`)
- For each group:
  - BC type selector (dropdown with ID)
  - Value input (float) - enabled for DIRICHLET, VON_NEUMANN
  - Visual indicator if BC already assigned
- BC preview on mesh (color-coded by BC type)

**Validation**:
- One BC per group (enforced)
- Group with no BC shown with warning

**Outputs**:
- `bc_conditions`

### View 4: Solve (`/solve`)

**Purpose**: Run simulation and visualize results

**Components**:
- **Control Panel**:
  - Step button (single step)
  - Run button (batch to completion)
  - Reset button
  - Progress indicator
- **Time Slider**: Scrub through saved timesteps
- **Visualization Panel**:
  - Scalar contour (color map)
  - Vector glyphs (arrows)
  - Slice/clip controls
- **Field Selector**: Switch between velocity, pressure, etc.

**Modes**:
1. **Interactive**: Step-by-step, live updates
2. **Batch**: Run to completion, then playback

**Outputs**:
- `results_buffer` (updated during simulation)
- `viz_settings`

---

## Backend API Endpoints

All endpoints return/accept numpy arrays.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/mesh/load` | POST | Upload VTK → extract mesh data |
| `/api/mesh/grid` | POST | Create UniformGridMesh from params |
| `/api/mesh/groups` | GET | Get group names and IDs |
| `/api/config/set` | POST | Set simulation parameters |
| `/api/config/get` | GET | Get current parameters |
| `/api/conditions/set` | POST | Set boundary conditions |
| `/api/conditions/get` | GET | Get current BCs |
| `/api/sim/init` | POST | Initialize solver with config |
| `/api/sim/step` | POST | Execute single time step |
| `/api/sim/run` | POST | Run to completion |
| `/api/sim/reset` | POST | Reset to initial conditions |
| `/api/results/frame` | GET | Get specific timestep data |
| `/api/results/fields` | GET | List available result fields |
| `/api/viz/update` | POST | Update visualization settings |

---

## File Structure

```
src/pde_module/GUI/
├── plan.md                    # This file
├── __init__.py
├── app.py                     # Trame app factory
├── constants.py               # ID enums (BCTypeIDs, etc.)
├── state.py                   # State class (numpy arrays)
├── router.py                  # VueRouter configuration
├── mapper.py                  # Maps GUI IDs ↔ solver calls
│
├── views/
│   ├── __init__.py
│   ├── mesh_import.py         # Mesh loading view
│   ├── sim_config.py          # Parameter configuration
│   ├── conditions.py          # BC/IC definition
│   └── solve.py               # Simulation + visualization
│
├── api/
│   ├── __init__.py
│   ├── mesh.py                # Mesh loading endpoints
│   ├── config.py              # Config endpoints
│   ├── conditions.py         # BC/IC endpoints
│   └── simulation.py          # Solve endpoints
│
├── viz/
│   ├── __init__.py
│   ├── renderer.py            # VTK rendering setup
│   ├── scalar_contour.py      # Scalar field rendering
│   └── vector_glyph.py        # Vector field rendering
│
└── utils/
    ├── __init__.py
    ├── mesh_io.py             # VTK → numpy arrays
    └── numpy_helpers.py       # Array utilities
```

---

## Key Implementation Details

### 1. VTK → Mesh Pipeline

```python
def load_vtk_mesh(filename: str) -> dict:
    """Load VTK file and return mesh data as numpy arrays."""
    import meshio
    mesh_data = meshio.read(filename)

    # Extract cells (handle mixed cell types - take first block)
    cells = mesh_data.cells[0]

    return {
        "nodes": mesh_data.points.astype(np.float32),
        "cells": cells.connectivity.astype(np.int32),
        "cell_types": np.array([cells.type.vtk_id] * len(cells.connectivity), dtype=np.int8),
        "dimensions": np.array(mesh_data.points.shape[1], dtype=np.int32),
    }
```

### 2. Group Extraction from Mesh

```python
def extract_groups(mesh_data: dict) -> tuple[dict, dict, dict]:
    """Extract boundary groups from mesh topology."""
    # Groups are extracted based on mesh topology
    # For imported VTK: use meshio metadata or compute from faces
    # For UniformGridMesh: generate from grid geometry

    groups = {}  # name → indices array
    name_to_id = {}  # name → int
    id_to_name = {}  # int → name

    # ... extraction logic ...

    return groups, name_to_id, id_to_name
```

### 3. ID-based BC Application

```python
# In mapper.py - converts GUI state to solver calls
def apply_bc_by_id(mesh: Mesh, bc_conditions: np.ndarray, field: wp.array):
    """Apply BCs to field based on GUI condition arrays."""
    for row in bc_conditions:
        group_id, bc_type_id, value = row

        # Get group name from ID
        group_name = state.group_id_to_name[group_id]
        group = mesh.groups[group_name]

        # Map bc_type_id to solver method
        match bc_type_id:
            case BCTypeIDs.NO_SLIP:
                solver.no_slip(group_name)
            case BCTypeIDs.DIRICHLET:
                solver.dirichlet_BC(group_name, value)
            case BCTypeIDs.VON_NEUMANN:
                solver.vonNeumann_BC(group_name, value)
            # ...
```

### 4. Trame App Factory

```python
# In app.py
from trame.app import Server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify, vtk, router

def create_app():
    app = Server()

    with SinglePageLayout(app) as layout:
        # App bar
        with layout.toolbar:
            vuetify.VAppBarTitle("PDE Module GUI")
            with vuetify.VBtnGroup():
                vuetify.VBtn("Mesh", href="/", router_link=True)
                vuetify.VBtn("Config", href="/config", router_link=True)
                vuetify.VBtn("BCs", href="/conditions", router_link=True)
                vuetify.VBtn("Solve", href="/solve", router_link=True)

        # Main content (router view)
        with layout.content:
            router.RouterView()

    # Register API endpoints
    register_api_routes(app)

    return app
```

### 5. Time Playback State

```python
# Visualization state for playback
viz_settings = np.zeros(1, dtype=[
    ('viz_field', 'i4'),
    ('viz_mode', 'i4'),
    ('time_index', 'i4'),
    ('show_vectors', 'i4'),
    ('show_scalar', 'i4'),
    ('scalar_range', 'f4', 2)
])[0]

# Results buffer - list of field states at each timestep
results_buffer = []  # list[np.ndarray]

# During simulation:
# - results_buffer.append(current_field.copy())
# - viz_settings['time_index'] = len(results_buffer) - 1

# During visualization:
# - slider controls viz_settings['time_index']
# - renderer displays results_buffer[viz_settings['time_index']]
```

---

## Modifications to Existing Code

### 1. New File: `mesh/vtk_importer.py` (optional utility)

```python
# Optional utility for meshio → Mesh conversion
import meshio
from .mesh import Mesh

def from_vtk_file(filename: str) -> Mesh:
    """Create Mesh object from VTK file."""
    data = meshio.read(filename)
    cells = data.cells[0]

    return Mesh(
        nodes=data.points,
        cells_connectivity=cells.connectivity,
        cell_types=np.array([cells.type.vtk_id] * len(cells.connectivity)),
    )
```

### 2. Extend `constants.py` (GUI IDs)

```python
# New file: GUI/constants.py
from enum import IntEnum

# Condition categories
class ConditionTypes(IntEnum):
    NONE = 0
    BOUNDARY_CONDITION = 1
    FORCE = 2

# BC types
class BCTypeIDs(IntEnum):
    NO_SLIP = 1
    IMPERMEABLE = 2
    DIRICHLET = 3
    VON_NEUMANN = 4
    FARFIELD = 5
    PERIODIC = 6

# Force types
class ForceTypeIDs(IntEnum):
    NONE = 1
    CONSTANT = 2
    SPATIAL_FUNC = 3

# PDE types
class PDETypeIDs(IntEnum):
    DIFFUSION = 1
    ADVECTION_DIFFUSION = 2
    NAVIER_STOKES = 3
```

---

## Validation Checklist

Before implementation, confirm:

- [ ] VTK files can be loaded via meshio
- [ ] Mesh.groups contains proper numpy arrays of indices
- [ ] Boundary groups can be extracted from imported meshes
- [ ] Solver can accept config via numpy arrays (not just direct calls)
- [ ] Results can be extracted as numpy arrays for visualization

---

## Dependencies

```python
# Required packages
trame>=3.0.0
trame-vuetify>=3.0.0
trame-vtk>=3.0.0
vuetify>=3.0.0
meshio>=5.0.0
pyvista>=0.40.0
numpy>=1.24.0
```

---

## Next Steps for Implementation

1. **Set up GUI module structure** (directories, __init__.py files)
2. **Implement constants.py** (ID enums)
3. **Implement state.py** (numpy array state class)
4. **Implement mesh_io.py** (VTK → numpy arrays)
5. **Implement API endpoints** (mesh, config, conditions)
6. **Implement views** (MeshImport, SimConfig, Conditions, Solve)
7. **Implement visualization** (VTK rendering)
8. **Implement mapper.py** (GUI IDs → solver calls)
9. **Integrate and test** with existing solver
