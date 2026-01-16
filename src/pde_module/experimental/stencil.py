import warp as wp
from warp import types
from .dummy_types import *

from collections.abc import Iterable
def tuplify(x:Any):
    '''Convert single item into tuple of element one or tuple. String and bytes are treated as a single object. All other iterables converted to tuple'''
    if isinstance(x,(str,bytearray,bytes)):
        return (x,)
    elif isinstance(x,Iterable):
        return tuple(x)
    else:
        return (x,)

class Stencil:
    '''
    Array of Structure Stencil Base Module.
    
    Think of it like nn.Module from Pytorch does the following:
    - Identify the appropriate dtype for input and output
    - Initialise and store the output array
    
    TODO:
    - Register stencil like nn.Module (Far into future)
    - Add grad for autodiff
    '''
    step:int = 0
    float_dtype: wp.float32 | wp.float64 
    def __init__(self,input_dtype,output_dtype,float_dtype):        
        self.initial = True
        self._iniput_dtype = input_dtype
        self._output_dtype = output_dtype
        self.float_dtype = float_dtype
    
    @property
    def output_dtype(self) -> wp_Matrix | wp_Vector:
        return self._output_dtype
    
    @property
    def input_dtype(self) ->wp_Matrix | wp_Vector :
        return self._iniput_dtype
    
    def create_output_array(self,input_array:wp.array):
        shape = input_array.shape
        return wp.empty(shape = shape, dtype= self.output_dtype)
    
    
    def forward(self,*args,**kwargs):
        '''
        Method to overide. This method should contain all the code neccesary to caclulate the stencil output. Default, this method raises an exception
        '''
        raise NotImplementedError('forward method in Stencil Object must be implemented by the user')
    
    
    def _set_registry(self,call:str):
        call_list_name = f'_{call}_list'
        if hasattr(self,call_list_name) is False:
            call_list = []    
            
            for attr in dir(self):
                method = getattr(self,attr)
                if hasattr(method,f'_{call}_order') and callable(method):
                    call_list.append(method)
                    
            setattr(self,call_list_name,sorted(call_list,key = lambda x: getattr(x,f'_{call}_order')))
            
        return call_list_name
    
    
    def setup(self,*args,**kwargs):
        '''
        Call this function to safely call initialize outside of call function to gurantee setting the initial flag
        '''
        # reigister before and after forward methods but dont execute
        _ = self._set_registry('before_forward')
        _ = self._set_registry('after_forward')
        
        call_list_name = self._set_registry('setup')
        
        for method in getattr(self,call_list_name):
            method(*args,**kwargs)
        
        self.initial = False
    
    def before_forward(self,*args,**kwargs):
        '''
        Hook to do operation before forward pass is called. By Default this will zero the output array. If this is undesirable, overide this method
        '''
        call_list_name = self._set_registry('before_forward')
        for method in getattr(self,call_list_name):
            method(*args,**kwargs)
        
        
    def after_forward(self,*args,**kwargs):
        '''
        Hook to do operation after forward pass is called. By default, does nothing
        '''
        call_list_name = self._set_registry('after_forward')
        for method in getattr(self,call_list_name):
            method(*args,**kwargs)
    
    
    def __call__(self,*args,**kwargs):
        '''
        Call Function to Stencil. Does the following:
        1. Setup (if step == 0)(default: create output array and create kernel.)
        2. Before Forward Call (default: zero output array)
        3. Forward call - Do your calcs here (MUST BE IMPLEMENTED)
        4. After Forward Call (default does nothing)
        5. increment step counter by 1
        '''
        if self.initial:
            self.setup(*args,**kwargs)
        self.before_forward(*args,**kwargs)
        output = self.forward(*args,**kwargs)
        self.after_forward(*args,**kwargs)
        self.step += 1
        return output
        
    
if __name__ == '__main__':
    x = wp.array()
    wp.types.vector()