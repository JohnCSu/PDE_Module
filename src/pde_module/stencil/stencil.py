import warp as wp
from warp import types
from ..utils.dummy_types import *

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
    initial:bool = True
    
    def __init__(self,*args,**kwargs):        
        pass
    
    @staticmethod
    def create_output_array(input_array:wp.array,output_dtype = None) -> wp.array:
        '''
        Create Output array based on incoming input_array and target output dtype. If output_dtype is None, the input_array dtype is used
        Note that the array returned is not zeroed
        '''
        shape = input_array.shape
        output_dtype = output_dtype if output_dtype is not None else input_array.dtype
        return wp.empty(shape = shape, dtype= output_dtype)
    
    
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
                    # if getattr(method,'_debug') and self.debug is False:
                    #     #Skip append if self.debug is False but method debug is set to true    
                    #     continue 
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
        1. Setup (if initial == True else skip)
        2. Before Forward Call
        3. Forward call - Do your calcs here (MUST BE IMPLEMENTED)
        4. After Forward Call
        5. Return contents from forward call
        '''
        if self.initial:
            self.setup(*args,**kwargs)
        self.before_forward(*args,**kwargs)
        output = self.forward(*args,**kwargs)
        self.after_forward(*args,**kwargs)
        return output
        
    
if __name__ == '__main__':
    x = wp.array()
    wp.types.vector()