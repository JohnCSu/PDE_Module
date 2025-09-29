import warnings
import warp as wp
from typing import Any
class SolverModule():
    t:float = 0.
    global_time:float
    dt:float
    end_time:float
    start_time:float
    values:list
    name:str 
    '''Buffer list to store any intermediate states needed for computation'''
    
    initial_value:Any
    final_value:Any
    
    def __init__(self,name:str = None,**kwargs):
        if name is None:
            name = 'Solver'
        
    
    def calculate_dt(self,*args, **kwargs):
        '''Overide this method to create custom time incrementation loops, default is fixed incrementation'''    
        return self.dt
    
    def __call__(self,grid,initial_value,*args, **kwargs):
        self.initial_value = initial_value
        self.dt = dt
        
        self.before_iteration_loop()
        while self.t < self.end_time:
            iteration_output = self.forward(*args, **kwargs)
            dt = self.calculate_dt(iteration_output,*args, **kwargs)
            self.t += dt
            
            if self.early_exit(iteration_output,*args, **kwargs):
                break
            
        self.after_iteration_loop()
        
        
        return grid,self.final_value
            
    def forward(self,*args,**kwargs):
        '''
        Like pytorch write the code for one iteration of your solver loop here:
        '''
        return None
    
    
    def capture_graph(self):
        with wp.ScopedCapture(device="cuda") as iteration_loop:
            self.forward()
            # swap
            self.forward()
    
    
    def early_exit(self,*args, **kwargs) -> bool:
        '''
        Early exit criteria, Overide this function, should return True if you need to exit early otherwise return False
        '''
        
        return False
    
    def before_iteration_loop(self,*args,**kwargs):
        pass
    
    def after_iteration_loop(self,*args,**kwargs):
        pass