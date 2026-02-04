'''
    Hooks to register methods on stencil objects
    
    Arguments:
        fn (Callable | None, optional) :
            function to wrap. With decorator notation this fn can be ignored
        
        order (int) :
            order of hooked function. lower value (-ve allowed) means the function is run earlier
        debug (bool) :
            If set to true, the wrapped function is registered if the debug flag in the stencil is set to true,
            otherwise the function is ignored. Useful for debugging or for setting safer 
            but less performant code (such as array and dtype checking)
''' 

def setup(fn=None,*,order = 0,debug = False):
    '''
    Hook to register this method to run during setup.
    
    Arguments:
        fn (Callable | None, optional) :
            function to wrap. With decorator notation this fn can be ignored
        
        order (int) :
            order of hooked function. lower value (-ve allowed) means the function is run earlier
        debug (bool) :
            If set to true, the wrapped function is registered if the debug flag in the stencil is set to true,
            otherwise the function is ignored. Useful for debugging or for setting safer 
            but less performant code (such as array and dtype checking)
    '''
    assert isinstance(order,int), 'order must be integer'
    def decorator(func):
        assert all([hasattr(func,attr) is False for attr in ['_setup_order' ,'_after_forward','_before_forward']]), 'Method already registered!'
        func._setup_order = order
        func._debug = debug
        return func
    if fn is None:
        return decorator
    else:
        return decorator(fn)
    
def before_forward(fn=None,*,order = 0,debug = False):
    '''
    Hook to register this method to run before_forward.
    
    Arguments:
        fn (Callable | None, optional) :
            function to wrap. With decorator notation this fn can be ignored
        
        order (int) :
            order of hooked function. lower value (-ve allowed) means the function is run earlier
        debug (bool) :
            If set to true, the wrapped function is registered if the debug flag in the stencil is set to true,
            otherwise the function is ignored. Useful for debugging or for setting safer 
            but less performant code (such as array and dtype checking)
    '''
    assert isinstance(order,int), 'order must be integer'
    def decorator(func):
        assert all([hasattr(func,attr) is False for attr in ['_setup_order' ,'_after_forward','_before_forward']]), 'Method already registered!'
        func._before_forward_order = order
        func._debug = debug
        return func
    
    if fn is None:
        return decorator
    else:
        return decorator(fn)

def after_forward(fn=None,*,order = 0,debug = False):
    '''
    Hook to register this method to run after_forward.
    s
    Arguments:
        fn (Callable | None, optional) :
            function to wrap. With decorator notation this fn can be ignored
        
        order (int) :
            order of hooked function. lower value (-ve allowed) means the function is run earlier
        debug (bool) :
            If set to true, the wrapped function is registered if the debug flag in the stencil is set to true,
            otherwise the function is ignored. Useful for debugging or for setting safer 
            but less performant code (such as array and dtype checking)
    '''
    assert isinstance(order,int), 'order must be integer'
    def decorator(func):
        assert all([hasattr(func,attr) is False for attr in ['_setup_order' ,'_after_forward','_before_forward']]), 'Method already registered!'
        func._after_forward_order = order
        func._debug = debug
        return func
    
    if fn is None:
        return decorator
    else:
        return decorator(fn)

