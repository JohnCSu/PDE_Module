import warp as wp
from warp.types import vector
from typing import Any


@wp.func
def linear_interpolation(
    owner_value:Any,
    neighbor_value:Any,
    mass_flow:Any,
    owner_centroid:Any,
    neighbor_centroid:Any,
    face_centroid:Any,
):
    '''
    Calulating Face Value using central differencing
    '''
    dist = wp.length(neighbor_centroid - owner_centroid)
    owner_to_face = wp.length(face_centroid - owner_centroid)
    ratio = owner_to_face/dist
    return ratio*owner_value + (1-ratio)*neighbor_value
    
@wp.func
def upwind(
    owner_value:Any,
    neighbor_value:Any,
    mass_flow:Any,
    owner_centroid:Any,
    neighbor_centroid:Any,
    face_centroid:Any,
):
    '''
    Calcluating face value using first order upwind
    '''
    return wp.where(mass_flow > 0.0, owner_value, neighbor_value)

@wp.func
def central_difference(
    owner_value:Any,
    neighbor_value:Any,
    face_normal:Any,
    owner_centroid:Any,
    neighbor_centroid:Any,
    face_centroid:Any):
    '''
    Calcluating the gradient Flux for diffusion term
    '''
    
    dist = wp.length(neighbor_centroid - owner_centroid)
    return (neighbor_value - owner_value)/dist


    