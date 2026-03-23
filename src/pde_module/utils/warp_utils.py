import warp as wp

@wp.func
def ijk_to_global_c(i:int,j:int,k:int,Nx:int,Ny:int,Nz:int):
    '''
    Map back to global index for 3D array in C memory format
    '''
    return i*(Ny*Nz) +j*Nz + k

@wp.func
def xijk_to_global_c(x:int,i:int,j:int,k:int,X:int,Nx:int,Ny:int,Nz:int):
    '''
    Map back to global index for 4D array in C memory format
    '''
    return x*(Nx*Ny*Nz) +i*(Ny*Nz) + j*Nz + k
