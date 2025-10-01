import warp as wp
from typing import Any
'''
Functions to calculate second order derivatives
'''

@wp.func
def central_difference(x_h:float,x:float,x__h:float,value_h:Any,value:Any,value__h:Any):
    return (value_h - 2.*value + value__h)/wp.pow((x_h-x),2.)
