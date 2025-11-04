import warp as wp
from typing import Any
'''
Functions to calculate first order derivatives
'''

@wp.func
def central_difference(value_h:Any,value:Any,value__h:Any,dx:float):
    return (value_h - value__h)/(2.*dx)

@wp.func
def forward_difference(value_h:Any,value:Any,value__h:Any,dx:float):
    return (value_h - value)/(dx)

@wp.func
def backward_difference(value_h:Any,value:Any,value__h:Any,dx:float):
    return (value - value__h )/(dx)
