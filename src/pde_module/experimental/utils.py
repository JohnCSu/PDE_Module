from .types import *

from collections.abc import Iterable
def tuplify(x:Any):
    '''Convert single item into tuple of element one or tuple. String and bytes are treated as a single object. All other iterables converted to tuple'''
    if isinstance(x,(str,bytearray,bytes)):
        return (x,)
    elif isinstance(x,Iterable):
        return tuple(x)
    else:
        return (x,)