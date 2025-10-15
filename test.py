import numpy as np
import warp as wp



wp.init()


mat = wp.mat(shape = (3,3),dtype = float)
@wp.kernel
def test(a:wp.array(dtype=mat),
         x:wp.array(dtype=wp.vec3f),
         x1:wp.array(dtype=wp.vec3f)):
    
    i = wp.tid()
    
    # z1  = wp.vec3(1.,1.,1.)
    # z2  = wp.vec3(1.,1.,1.)
    # mat = wp.mat(shape = (3,3),dtype = float)
    # for i in range(3):
    #     mat[i] = z1[i]*z2
    
    a[0][0,0] = 4.
    a[0][1,0] = 3.
    
    # mat = outer(z1,z2)
    c = a[0][:,0]
    
    wp.print(c)
    for i in range(3):
        x[0][i] = c[i]
    
    
    
    
    
x = 1.
b = wp.zeros(shape = 1, dtype=mat)
b.fill_(2.)



x22= wp.zeros(shape = 1,dtype = wp.vec3f)
x22.fill_(2.)

x11= wp.zeros(shape = 1,dtype = wp.vec3f)
x11.fill_(1.)
print(x11.numpy())
wp.launch(test,1,[b,x11,x22])


print('hi')
print(b.numpy())

    
    