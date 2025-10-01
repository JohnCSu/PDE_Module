import warp as wp
from typing import Any
'''
Functions to calculate first order derivatives
'''

@wp.func
def central_difference(x_h:Any,x:Any,x__h:Any,value_h:Any,value:Any,value__h:Any):
    return (value_h - value__h)/(x_h - x__h)

@wp.func
def forward_difference(x_h:Any,x:Any,value_h:Any,value:Any):
    return (value_h - value)/(x_h- x)

@wp.func
def backward_difference(x:Any,x__h:Any,value:Any,value__h:Any):
    return (value - value__h )/(x- x__h)
