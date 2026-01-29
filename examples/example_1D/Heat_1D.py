
# %%
import warp as wp
import numpy as np
wp.init()

xx = wp.ones(shape =(3,3),dtype=float)
yy = wp.ones(shape =(3,3),dtype=float)

a = wp.types.vector(2,dtype = wp.uint64)()
a[0] = xx.ptr
a[1] = yy.ptr

# a = wp.array(a,dtype=wp.uint64)

@wp.kernel
def test():
    i = wp.tid()
    accum = 0.
    for j in range(wp.static(len(a))):
        ptr = wp.static(wp.uint64(a[j]))
        x = wp.array(ptr=ptr,shape = (9,),dtype=wp.float32)
        accum += x[i]
    
    x[i] = accum
    
    
    




wp.launch(kernel=test, dim = 1,inputs= [])

print(yy.numpy())

print(xx.numpy())
# print(xx.numpy())
# %%
