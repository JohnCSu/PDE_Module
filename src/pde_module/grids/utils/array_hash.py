from numba import njit
import numpy as np


@njit
def numba_array_hash(arr:np.ndarray) -> int:
    '''
    Hashes an entire array so it can be used as a key for Numba dict.\\
    Achieves this by flattening the array and then treating it like a tuple and proceding with the tuplehash algorithim
    
    No checks are made on the type of array or on its mutability. integer or fixed string types recommended \\
    It is upto the user to ensure that the array being hashed is effectivily in an immutable state.
    Note that because the array.flatten() is used, This also takes into account the C and F type ordering

    Uses the old hashing algorithim by python. Can be seen in python 3.7 code here:\\
    https://github.com/python/cpython/blob/v3.7.0/Objects/tupleobject.c#L348 

    Newer Python versions use xxHash (https://github.com/Cyan4973/xxHash) for tuplehash\\
    But this should be sufficient to be used with numba dicts
    '''
    x = 0x345678  # Initial Seed Magic value
    mult = 1000003 # Large Prime

    flattened_arr =arr.ravel()
    length =len(flattened_arr)
    for i in flattened_arr:
        y = hash(i)
        x = (x^y) * mult #XOR followed by multiplication
        mult += (82520+2*length) # Magic Number + len of flat array
    
    x += 97531 # Magic Number

    if x == -1: #Avoid exit code of -1 (CPython Artifact)
        x = -2
    
    return x
        