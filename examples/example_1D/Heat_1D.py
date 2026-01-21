
# %%
import warp as wp


@wp.kernel
def test(a:wp.array(dtype=wp.mat33)):
    
    
    i = wp.tid()
    v = wp.vec3f(1.,1.,1.)
    # for j in range(3):
    #     a[i][j,0] = v[j]
    
    m = wp.mat33()
    m[0,:] = v
    m *= 2.    
    a[i] = m


a = wp.zeros(shape = 1,dtype=wp.mat33)

wp.launch(kernel=test, dim = 1,inputs= [a])


print(a.numpy())
# %%
