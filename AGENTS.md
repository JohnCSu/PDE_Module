# AGENTS.md - PDE_Module Development Guide

## Project Overview
PDE_Module is a PyTorch-inspired framework for GPU-based PDE simulations using the **Warp DSL** (compiles Python to CUDA). The architecture mirrors `torch.nn.Module` with Stencil classes that manage input/output arrays and lifecycle hooks.

**Key Dependencies**: warp-lang, numpy, numba, scipy, matplotlib, meshio, gmsh, pyvista

---

## Build, Lint, and Test Commands

### Package Installation
```bash
# Install in development mode
uv pip install -e .

# Install dev dependencies
uv pip install -G dev .
```

### Running Tests
Tests are located in `tests/` and use pytest as the framework

**Important**: Warp requires initialization before use. Each test should include:
```python
import warp as wp
wp.init()
wp.config.mode = "debug"  # or "launch" for performance
```

### Development Tools
No explicit linting/type-checking commands found in the project. The project uses:
- Python 3.12+
- Warp SDK for GPU kernel compilation
- numpy and numba for CPU operations on numpy arrays
---

## Code Style Guidelines

### Import Conventions

**Package Internal Imports** Use absolute imports:
```python
from pde_module.stencil.utils import (
    create_stencil_op,
    eligible_dims_and_shift,
    create_tensor_divergence_op,
)
```
for any __init__.py you can use relative imports of at most a single . e.g
```python
from .function import foo
```

**External Imports**:
```python
import warp as wp
from warp.types import vector, matrix, type_is_vector, types_equal
import numpy as np
from collections.abc import Iterable
```

**Avoid wildcard imports for external packages** (except `hooks` which is intentional).

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `Stencil`, `ExplicitUniformGridStencil`, `Laplacian` |
| Functions/Methods | snake_case | `create_output_array`, `tuplify`, `forward` |
| Private Methods | snake_case with leading underscore | `_set_registry`, `_copy_to_buffers` |
| Constants | SCREAMING_SNAKE_CASE | `DIRICHLET`, `VON_NEUMANN`, `FLUID_CELL` |
| Instance Variables | snake_case | `self.input_dtypes`, `self.output_array` |
| Type Aliases | wp_Vector, wp_Matrix | Defined in `utils/dummy_types.py` |

### Type Hints
Use type hints for function signatures when types are unambiguous:
```python
def forward(self, input_array: wp.array, dt: float) -> tuple[wp.array]:
def __init__(self, inputs: int|list[int], outputs: int|list[int], dx: float):
```
Ignore LSP errors but the tests should still pass.

For any functions decorated with the @wp.kernel of @wp.func DO NOT modify the type signature or type hints

for function signatures where the outputs are from the warp package use the Type aliases found in utils/dummy_types.py. the convenction is
wp_{VARIABLE_NAME} e.g wp_Vector.

For vectors and matrices use wp_Vector or wp_Matrix regardless of size e.g. if vec3i or mat22f. If the type does not exist, make it


**Warp-specific types**: Use `wp.vec2f`, `wp.vec3i`, `wp.float32`, `wp.float64`, `wp.int8`, etc.

### Class Structure Pattern

All Stencil subclasses follow this hook-based lifecycle:

```python
class MyStencil(Stencil):
    output_array: wp.array | None = None  # Class-level type annotation
    
    def __init__(self, inputs, outputs, dx, ghost_cells=0):
        super().__init__()
        # Initialize parameters
        self._inputs = tuplify(inputs)
        self._outputs = tuplify(outputs)
    
    @property
    def inputs(self) -> tuple[int]:
        return self._inputs
    
    @setup(order = -1)  # Runs first during setup
    def initialize_array(self, input_array, *args, **kwargs):
        self.output_array = self.create_output_array(input_array, self.output_dtype)
    
    @setup(order = 1)  # Runs after initialize_array
    def initialize_kernel(self, input_array, *args, **kwargs):
        self.kernel = create_my_kernel(...)
    
    @before_forward(order = 0)
    def pre_forward(self, *args, **kwargs):
        wp.copy(self.output_array, input_array)
    
    def forward(self, input_array, *args, **kwargs):
        raise NotImplementedError("forward must be implemented by subclass")
    
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
```


### Hooks System (from `stencil/hooks.py`)

| Decorator | Purpose | Execution Time |
|-----------|---------|----------------|
| `@setup(order=N)` | Initialize buffers, kernels, validate inputs | Before first call |
| `@before_forward(order=N)` | Pre-processing before forward | Every call, before forward |
| `@after_forward(order=N)` | Post-processing after forward | Every call, after forward |

Lower `order` values execute first. Negative orders allowed.

### Error Handling

**Assertions** for internal validation (common in this codebase):
```python
assert array_A.shape == array_B.shape, 'input arrays must be the same Shape!'
assert (length % 2) == 1, 'stencil must be odd sized'
assert name not in self.groups.keys(), 'name for group already exists'
```

**Custom Exceptions** (extend ValueError):
```python
class SignatureMismatchError(ValueError):
    """Raised when a provided function does not meet the required input structure."""
    pass
```

**NotImplementedError** for abstract methods:
```python
def forward(self, *args, **kwargs):
    raise NotImplementedError('forward method in Stencil must be implemented')
```

### Docstring Format

Use docstrings for all public classes and methods.
```python
class Laplacian(ExplicitUniformGridStencil):
    '''
    Create Laplacian Stencil of vector field Using Central Based Finite difference.
    
    Args
    ----------
    inputs : int 
        length of vector
    dx : float 
        grid spacing
    ghost_cells : int 
        number of ghost cells on the grid
    stencil : vector | None
        stencil to use for laplacian. If None, 2nd Order stencil is used
    
    Returns
    ---------
    output_array : wp.array3d 
        A 3D array with same vector dtype as input_array representing laplacian
    '''
```

### Warp Kernel Pattern

Define kernels inside functions using `@wp.kernel` decorator:

```python
def create_laplacian_kernel(input_vector, grid_shape, stencil, ghost_cells):
    @wp.kernel
    def laplacian_kernel(
        input_values: wp.array3d(dtype=input_vector),
        alpha: input_vector._wp_scalar_type_,
        output_values: wp.array3d(dtype=input_vector),
    ):
        i, j, k = wp.tid()
        # kernel logic
        output_values[index[0], index[1], index[2]] = laplace
    
    return laplacian_kernel
```

Launch kernels with `wp.launch()`:
```python
wp.launch(kernel, dim=kernel_dim, inputs=[...], outputs=[...])
```
default launch of kernels is gpu unless explicitly declared

### Array Handling

- Use `wp.empty()` or `wp.zeros()` to allocate arrays
- Use `wp.copy()` for array copying (not `=` which creates references)
- Flatten arrays when passing to kernels: `array.flatten()`
- Access shape as tuple: `input_array.shape`

### Project Structure

```
src/pde_module/
├── __init__.py
├── stencil/          # Core Stencil base class and hooks
│   ├── stencil.py     # Base Stencil class
│   ├── hooks.py      # @setup, @before_forward, @after_forward decorators
│   ├── elementWise.py
│   ├── mapWise.py
│   └── utils.py
├── FDM/              # Finite Difference Methods
│   ├── ExplicitUniformGridStencil.py
│   ├── laplacian.py
│   ├── grad.py
│   ├── divergence.py
│   └── boundary/
├── mesh/             # Mesh/Grid structures
│   ├── mesh.py
│   ├── cell/
│   ├── face/
│   ├── edge/
│   └── uniformGridMesh/
├── time_step/        # Time integration schemes
│   ├── rungeKatta.py
│   └── forwardEuler.py
├── utils/            # Utilities and type definitions
│   ├── utils.py
│   ├── constants.py
│   └── dummy_types.py  # wp_Vector, wp_Matrix type aliases
└── experimental/    # Experimental features
```

---

## Common Patterns

### Converting Single Value to Tuple
```python
def tuplify(x: Any):
    """Convert single item into tuple. Strings/bytes treated as single object."""
    if isinstance(x, (str, bytearray, bytes)):
        return (x,)
    elif isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)
```

### Grid Creation
```python
grid = Grid(dx=1/(n-1), num_points=(n,n,1), origin=(0.,0.,0.), ghost_cells=ghost_cells)
u = grid.create_node_field(2)  # 2 = vector dimension
u.fill_(wp.vec2f(0., 0.))
```

### Signature Validation
```python
import inspect
if inspect.signature(f) != inspect.signature(bc):
    raise SignatureMismatchError(f'Signatures do not match!')
```

---

## Testing Guidelines

1. Warp context must be initialized at module top or inside `if __name__ == '__main__'`
2. Use `wp.config.mode = "debug"` during development for better error messages
3. Compare against numpy for validation: `np.all(np.isclose(...))`
4. Test files can serve as usage examples

---

## Architecture Principles

1. **Functional-esque**: Stencils don't modify inputs inplace; output goes to pre-allocated buffer
2. **Fixed Output Array**: Buffers are reused across calls to avoid reallocation
3. **Hook-based Lifecycle**: Initialization separated from execution via decorators
4. **Explicit Memory**: All allocations are explicit, no hidden buffers
