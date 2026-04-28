"""Time integration schemes for PDE solves."""

import warp as wp
from ..stencil.stencil import Stencil
from warp.types import vector, matrix, type_is_float
from ..stencil.hooks import *
from ..utils.types import wp_Array, wp_Kernel, wp_Vector,wp_Matrix

class ForwardEuler(Stencil):
    """Forward Euler time integration.

    Implements: y_{n+1} = y_n + dt * stencil
    """

    def __init__(self,*args,**kwargs) -> None:
        """Initialize Forward Euler integrator.
        
        Args:
            swap_buffers (bool) : If True, after time stepping, set the input array as the output field so the output field can be
                used as the input array in the next time step
        
        """
        # self.input_dtype = input_dtype
        # if type_is_float(input_dtype):
        #     self.float_dtype = input_dtype
        # else:
        #     self.float_dtype = input_dtype._wp_scalar_type_
        super().__init__()

    @setup
    def initialize_kernel(
        self, input_array: wp.array, stencil_values: wp.array, dt: float
    ) -> None:
        """Initialize the forward Euler kernel."""
        
        self.output_array = wp.empty_like(input_array)
        self.kernel = create_forward_euler(input_array.dtype)
        self.size = input_array.size
        assert input_array.shape == stencil_values.shape == self.output_array.shape
        
    def forward(
        self, input_array: wp.array, stencil_values: wp.array, dt: float
    ) -> wp_Array:
        """Perform one forward Euler time step.

        Args:
            input_array: Current field values.
            stencil_values: Field representing time derivative (gradient/forcing).
            dt: Time step size.

        Returns:
            Field values at next time step.
        """
        
        wp.launch(
            kernel=self.kernel,
            dim=self.size,
            inputs=[input_array.flatten(), stencil_values.flatten(), dt],
            outputs=[self.output_array.flatten()],
        )
        return self.output_array

def create_forward_euler(array_dtype):
    
    if type_is_float(array_dtype):
        float_dtype = array_dtype
    else:
        float_dtype = array_dtype._wp_scalar_type_
    
    @wp.kernel
    def forward_euler_kernel(
                        current_values:wp.array1d[array_dtype],
                        stencil_values:wp.array1d[array_dtype],
                        dt:float_dtype,
                        new_values:wp.array1d[array_dtype],
    ):
        
        tid = wp.tid() 
        new_values[tid] = current_values[tid] + dt*stencil_values[tid]
    
    return forward_euler_kernel
