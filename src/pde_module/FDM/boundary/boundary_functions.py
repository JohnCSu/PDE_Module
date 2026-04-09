import warp as wp
import numpy as np
from typing import Any
from warp.types import vector
from pde_module.utils.types import wp_Function
from dataclasses import dataclass


def get_constant_func(constant: float,input_dtype,output_ids, float_type) -> wp_Function:
    """Create a function that returns a constant value.

    Args:
        constant: The constant value to return.
        float_type: The warp float type.

    Returns:
        A wp.func that returns the constant value.
    """
    constant = float(constant)

    match output_ids:
        case int():
            length = 1
            output_ids = [output_ids]
        case list() | tuple():
            length = len(output_ids)
            
    ids_vector = vector(length,dtype = int)
    ids_to_set =ids_vector(*output_ids)
    
    @wp.func
    def constant_value(
        current_values: wp.array3d(dtype=input_dtype),
        nodeID: wp.vec3i,
        coordinates: wp.array3d(dtype=vector(3, float_type)),
        t: float,
        dx: float,
        params: Any,
    ):
        
        output = current_values[nodeID[0],nodeID[1],nodeID[2]]
        for i in range(wp.static(len(ids_to_set))):
            output[ids_to_set[i]] = float_type(constant)
        return output

    return constant_value