import warp as wp
from pde_module.utils.types import *

def divergence(kernel,input_array: wp_Array, alpha: float,ghost_cells: int,output_array:wp_Array,device = None)-> wp_Array:
    """Compute the divergence of the input field.
    
    Args:
        input_array: 3D array matching input shape (vector or matrix).
        alpha: Scaling factor. Default is 1.0.

    Returns:
        3D array of vectors:
        - If vector input: output is a vector of length 1 (scalar).
        - If matrix input of size (N, D): output is a vector of size (N,).
    """
    dims =  tuple(axis - ghost_cells * 2 if axis > 1 else axis for axis in input_array.shape)
    wp.launch(
        kernel,
        dim= dims,
        inputs=[input_array, alpha],
        outputs=[output_array],
        device= device,
    )
    return output_array

