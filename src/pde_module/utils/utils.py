from .dummy_types import *
from warp.types import vector,matrix
from collections.abc import Iterable


def tuplify(x:Any):
    '''Convert single item into tuple of element one or tuple. String and bytes are treated as a single object. All other iterables converted to tuple'''
    if isinstance(x,(str,bytearray,bytes)):
        return (x,)
    elif isinstance(x,Iterable):
        return tuple(x)
    else:
        return (x,)
    

def dtype_from_shape(shape,float_dtype):
    '''Based on the shape provided return a vector or matrix, if int or tuple of a single element return vector, if tuple of size 2 return matrix'''
    if isinstance(shape,Iterable):
            assert all([isinstance(x,int) for x in shape]), 'contents in input/output must be int only'
    else:
        assert isinstance(shape,int)
        shape = tuplify(shape)
        
    if len(shape) == 1:
        return vector(length = shape[0],dtype = float_dtype)
    else:
        return matrix(shape = shape, dtype = float_dtype)
    