# PDE_Module
PDE_Module is a pytorch inspired framework (namely `torch.nn`) but instead aimed for quickly developing simulations

From a high level simulations turn out to be very similar to Deep Learning networks in that simulations are really just a composing of several blocks together

# Why Warp?

There are lots of DSL's availiable such as CuPy, taichi, pytorch and so on but warp has the following advantages:
- unlike cupy, users write python code that is turned into Cuda code instead of simply writing cuda code as a string.
- Warp IMO provides much stricter typing requirements and semantics which makes it easier to reason about kernel behaviour
- Compared to Tensor based programming like pytorch, provides a lot more control over memory and operations as usesr have control over individual threads

This makes it a lot easier for user to begin developing GPU-based simulations quickly

# Features

## Module syle formation
Similar to pytorch's nn.Module, you can create a simulation by via the `Stencil` class. The `Stencil` class provides several hooks and conviences for creating your own kernel

## Memory Control
Before the run, `Stencil` moduls first allocate  


## Python Only
Leveraging Nvidia Warp, users can write GPU kernels without having users need to learn CUDA. This allows users to easily create their own GPU kernels without leaving python.



## Cuda Graphs
direct memory control via Nvidia Warp allows for Cuda Graph implementation which can further increase runtimes


# To Add
- Finite Volume on Uniform Grid
- LBM
- SoA style kernels (currently AoS kernels for simplicity)
- 
- Differentiation is not implemented yet (as requires memory control)
- Unstructured Grids/Meshes
- Multi GPU framework




# FAR AWAY
- leverage Mojo Lang for hardware independent kernels
