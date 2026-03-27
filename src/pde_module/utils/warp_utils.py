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

@wp.func
def global_to_ijk_c(global_id:int,Nx:int,Ny:int,Nz:int):
    '''
    Map a global ID to i,j,k coords of a structured Grid
    '''
    # How many 2D planes (Nj * Nk) fit into the global_id?
    i = global_id // (Ny * Nz)
    # What's left over after removing those planes?
    # remainder = global_id % (Ny * Nz)
    # Within that plane, how many rows (Nk) fit?
    j = (global_id // Nz )% Ny
    # What's left over is the position in the current row
    k = global_id % Nz
    return i,j,k