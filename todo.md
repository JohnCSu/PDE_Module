# Project To Do List

## Key:
- [] Not Completed
- [X] Completed
- [~] Partial/In-Progress

## UniformGridMesh
- [X] Move Boundary Conditions groups and flag status to mesh so can be used by different stencils easily. The boundary stencils handles the interpretation and any associated fields. 
- [] Add stl import capability. Use numpy STL?

## Stencil
- [] Add dynamic flag option where if true, create_output array is set to before_forward hook
## FDM
- [] Clean up Boundary Conditions
- [~] Change formulation to first pull data into an array so we can do fused ops
## LBM
- [X] Add Equilibrium BC (inlet and outlet BC)
- [X] Add Midside Bounceback stencils (Will be fused with BC)
## Finite Volume
- [] make explicit method and handle unstructrued

## 

## Visualization
- [] Add object to help clean up animation code