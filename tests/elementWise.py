import numpy as np
import warp as wp
from matplotlib import pyplot as plt
from pde_module.geometry.grid import Grid
from pde_module.FDM.elementWiseOps import scalarVectorMult,scalarVectorMultiply

def create_ElementOp_kernel(element_op,array_A_dtype,array_B_dtype,output_array_dtype):
    @wp.kernel
    def elementWise_kernel(A:wp.array(dtype=array_A_dtype),
                           B:wp.array(dtype=array_B_dtype),
                           output_array:wp.array(dtype = output_array_dtype)):
        i = wp.tid()
        output_array[i] = element_op(A[i],B[i])
        
        
    return elementWise_kernel
        
        


if __name__ == '__main__':
    wp.init()
    wp.config.mode = "debug"

    n = 21
    L = 1
    dx = L/(n-1)
    ghost_cells = 1
    # x,y = np.linspace(0,1,n),np.linspace(0,1,n)
    grid = Grid(dx = 1/(n-1),num_points=(n,n,1),origin= (0.,0.,0.),ghost_cells=ghost_cells)
    
    u = grid.create_node_field(2)
    u.fill_(wp.vec2f(2.,2.))
    p = grid.create_node_field(1)
    p.fill_(3.)
    out = grid.create_node_field(2)
    
    momentum = scalarVectorMult(u.dtype)
    out = momentum(p,u)
    
    # op = scalarVectorMultiply(out.dtype)
    # kernel = create_ElementOp_kernel(op,p.dtype,u.dtype,u.dtype)
    # wp.launch(kernel,dim = u.size,inputs=[p.flatten(),u.flatten()],outputs=[out.flatten()])
    
    print(out.numpy().flatten()[0])
    assert np.all(np.isclose(out.numpy().flatten(),6.))