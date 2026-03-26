import warp as wp
from warp.types import vector,matrix,is_array,is_vector,is_matrix
from pde_module.utils.dummy_types import wp_Array,wp_Vector,wp_Matrix
import numpy as np

def array_to_vec_or_mat(wp_arr:wp_Array) -> wp_Vector|wp_Matrix:
    if is_array(wp_arr):
        arr = wp_arr.numpy()
    else:
        arr = np.array(wp_arr)
        
    if len(arr.shape) == 1:
        return vector(len(arr),dtype = wp.dtype_from_numpy(arr.dtype))(arr)
    elif len(arr.shape) == 2:
        return matrix(arr.shape,dtype = wp.dtype_from_numpy(arr.dtype))(arr)
    else:
        raise ValueError('only arrays that map to 1D or 2D numpy arrays can be used')



def vec_or_mat_to_numpy(x):
    if is_vector(x):
        return np.array(x,dtype= wp.dtype_to_numpy(x._wp_scalar_type_))
    elif is_matrix(x):
        return np.array(x,dtype=wp.dtype_to_numpy(x._wp_scalar_type_)).reshape(x._shape_)
    else:
        raise TypeError('x must be warp vec or mat')