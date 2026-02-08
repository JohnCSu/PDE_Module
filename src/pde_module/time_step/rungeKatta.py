import warp as wp
from ..stencil.stencil import Stencil
import warp as wp
from warp.types import vector,matrix,type_is_matrix,type_is_vector,types_equal,is_array
from ..stencil.hooks import *
from .forwardEuler import ForwardEuler
from ..utils import tuplify,SignatureMismatchError
from typing import Callable
import inspect



class RungeKatta1(Stencil):
    '''
    Runge Katta 1 (Forward Euler) Implementation for multiple fields 
    '''
    def __init__(self, field_dtypes,f:Callable,bc:Callable):
        super().__init__()
        self.input_dtypes = tuplify(field_dtypes)
        self.forwardEulers = tuple(ForwardEuler(input_dtype) for input_dtype in self.input_dtypes)
        self.f = f
        self.bc = bc
        self.num_fields = len(self.input_dtypes)

        if inspect.signature(f) != inspect.signature(bc):
            raise SignatureMismatchError(f'Signature For f {inspect.signature(f)} does not match the signature of bc {inspect.signature(bc)}!')
    
    @setup
    def check_args(self,t,dt,*arrays,**kwargs):
        # For each array, check it is an array and then matches the corresponding Euler Time step input dtype
        for arr,time_step in zip(arrays,self.forwardEulers):
            assert is_array(arr)
            assert types_equal(arr.dtype,time_step.input_dtype), 'arrays and input dtype for euler must match'

        
    def forward(self,t:float,dt:float,*arrays:list[wp.array],**kwargs) -> tuple[wp.array]:
        ''''
        Args
        ---------
            input_array : wp.array3d 
                A 3D array with that matches the input shape (either vector or matrix)
            stencil_values: wp.array3d
                A 3D array that matches the input shape and dtype. represents the gradient or forcing term
            dt: float
                float represeting time increment
        Returns
        ---------
            output_array : wp.array3d 
                A 3D array with same shape and dtype as the input_array
        '''
        #Step 1
        arrays = self.bc(t,*arrays,**kwargs)
        k1s = self.f(t,*arrays,**kwargs)
        return tuple(forwardEuler(arr,k1,dt/2) for forwardEuler,arr,k1 in zip(self.forwardEulers,arrays,k1s))



class RungeKatta2(Stencil):
    '''
    Runge Katta 2 Implementation for multiple fields
    
    $$
    k_1 = f(t_n, y_n)
    $$
    
    $$
    k_2 = f(t_n + \frac{h}{2},
    $$
    
    $$
    y_n + \frac{h}{2}k_1) y_{n+1} = y_n + h k_2
    $$
        
    Args
    ---------
    field_dtypes: wp.vector| wp.matrix | Iterable[ wp.vector| wp.matrix ]
        dtypes of each field to pass into RK2. The order should match the order arrays are passed into when the function is called

    f: Callable
        function that computes the forcing term func. i.e. the f in y_{i+1} = y_i + dt*f(t,y_i)
    
    bc: Callable
        function that applies the BC to each field. 
    
    Both the forcing and boundary condition should have the same signature i.e. f(t,*arrays,**kwargs), bc(t,*arrays,**kwargs)
    here arrays are the fields which are being propagated. The output of func should be a list or tuple of output arrays that
    match shape and dytype as arrays. Any additional parameters or independent fields must be passed by keywords
    
    Usage
    --------
    ```python
        u = grid.create_node_field(2)
        p = grid.create_node_field(1)
        
        
        def f(t,u,p,**kwargs) -> [vector(2,float),vector(1,float)]:
            ...
            return outputs
    
        # dtypes order should match the same given in f
        RK2 = RungeKatta2([u.dtype,p.dtype],func = f)
        
        # input should match the same order as input_dtypes
        u_next,p_next = RK2(t,dt,u,p,**kwargs)
        
    ```
    '''
    def __init__(self, field_dtypes,f:Callable,bc:Callable):
        super().__init__()
        self.input_dtypes = tuplify(field_dtypes)
        self.forwardEulers_1 = tuple(ForwardEuler(input_dtype) for input_dtype in self.input_dtypes)
        self.forwardEulers_2 = tuple(ForwardEuler(input_dtype) for input_dtype in self.input_dtypes)
        self.f = f
        self.bc = bc
        self.num_fields = len(self.input_dtypes)
  
        if inspect.signature(f) != inspect.signature(bc):
            raise SignatureMismatchError(f'Signature For f {inspect.signature(f)} does not match the signature of bc {inspect.signature(bc)}!')
    
    
    
    @setup
    def check_args(self,t,dt,*arrays,**kwargs):
        # For each array, check it is an array and then matches the corresponding Euler Time step input dtype
        for arr,time_step in zip(arrays,self.forwardEulers_1):
            assert is_array(arr)
            assert types_equal(arr.dtype,time_step.input_dtype), 'arrays and input dtype for euler must match'

        
        
    @setup
    def set_buffers(self,t,dt,*arrays,**kwargs):
        self.t0_fields = tuple(wp.empty_like(arr) for arr in arrays)
        
    def _copy_to_buffers(self,arrays):
        for i,arr in enumerate(arrays):
            wp.copy(self.t0_fields[i],arr)
    
    def forward(self,t:float,dt:float,*arrays:list[wp.array],**kwargs) -> tuple[wp.array]:
        ''''
        Args
        ---------
            input_array : wp.array3d 
                A 3D array with that matches the input shape (either vector or matrix)
            stencil_values: wp.array3d
                A 3D array that matches the input shape and dtype. represents the gradient or forcing term
            dt: float
                float represeting time increment
        Returns
        ---------
            output_array : wp.array3d 
                A 3D array with same shape and dtype as the input_array
        '''
        #Step 1
        arrays = self.bc(t,*arrays,**kwargs)
        self._copy_to_buffers(arrays) # We store the ys at the initial increment to self.t0_fields
        k1s = self.f(t,*arrays,**kwargs)
        y1s = [forwardEuler(arr,k1,dt/2) for forwardEuler,arr,k1 in zip(self.forwardEulers_1,arrays,k1s)]
        
        # Step 2
        y1s = self.bc(t,*y1s,**kwargs)
        k2s = self.f(t+dt/2,*y1s,**kwargs)
        
        return tuple(forwardEuler(arr,k2,dt) for forwardEuler,arr,k2 in zip(self.forwardEulers_2,self.t0_fields,k2s))
        