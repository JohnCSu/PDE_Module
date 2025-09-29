import numpy as np
import warp as wp

wp.init()
@wp.kernel
def test(arr:wp.array(dtype=wp.vec(length=1,dtype = float))):
    i,j = wp.tid()
    
    print(i)
    print(j)
    a = wp.vec(length= 1,dtype= float)
    b = wp.vec(length= 1,dtype= float)

arr = wp.ones(shape = 2,dtype=wp.vec(length=1,dtype = float))
wp.launch(test,dim = [1,2,1],inputs = [arr])
