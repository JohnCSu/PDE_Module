
import warp as wp




def central_kernel(wp_func):
    
    @wp.kernel
    def central(grid,values,levels):
        
        wp.static(wp_func)()        
    
    