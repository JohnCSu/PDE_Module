 
def setup(fn=None,*,order = 0):
    assert isinstance(order,int), 'order must be integer'
    def decorator(func):
        assert all([hasattr(func,attr) is False for attr in ['_setup_order' ,'_after_forward','_before_forward']]), 'Method already registered!'
        func._setup_order = order
        return func
    if fn is None:
        return decorator
    else:
        return decorator(fn)
    
def before_forward(fn=None,*,order = 0):
    assert isinstance(order,int), 'order must be integer'
    def decorator(func):
        assert all([hasattr(func,attr) is False for attr in ['_setup_order' ,'_after_forward','_before_forward']]), 'Method already registered!'
        func._before_forward = order
        return func
    
    if fn is None:
        return decorator
    else:
        return decorator(fn)

def after_forward(fn=None,*,order = 0):
    assert isinstance(order,int), 'order must be integer'
    def decorator(func):
        assert all([hasattr(func,attr) is False for attr in ['_setup_order' ,'_after_forward','_before_forward']]), 'Method already registered!'
        func._before_forward = order
        return func
    
    if fn is None:
        return decorator
    else:
        return decorator(fn)
