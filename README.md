# PDE_Module
PDE_Module is a pytorch inspired framework (namely `torch.nn`) to create a solve PDEs upto 3 dimensions on unifrom grids (both nodal and voxel/cell centered grids).

A key plan for this framework is to allow users to easily create and construct batched based physics based simulations and write effecient GPU level code via [Nvidia Warp]()

# Features

## Module syle formation
Similar to pytorch's nn.Module, you can create a simulation by via the `StencilModule`. kernels are naturally batched allowing 

## Python Only
Leveraging Nvidia Warp, users can write GPU kernels without having users need to learn CUDA. This allows users to easily create their own GPU kernels without leaving python.

## Memory Control
User have more direct control over memory via nvidia warp. Most tensor based frameworks implicitly allocate new arrays each time a function/module is called. This can be turned off or on using the `dynamic_alloc` flag. Not needing to reallocated memory can significantly speed up runtimes and reduce memory footprint especially for runtimes not requiring autodifferentiation

## Cuda Graphs
direct memory control via Nvidia Warp allows for Cuda Graph implementation which can further increase runtimes


# To Add
- Finite Volume on Uniform Grid
- Differentiation is not implemented yet (as requires memory control)
- Implicit Modules
- Multi GPU framework
- Unstructured Grids/Meshes
- FEM


# FAR AWAY
- leverage Mojo Lang for hardware independent kernels
