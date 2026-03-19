import numpy as np
from dataclasses import dataclass,field

@dataclass
class Group:
    name:str
    cell_ids: np.ndarray = field(default_factory=lambda: np.array([],dtype= np.int32))
    face_ids: np.ndarray = field(default_factory=lambda: np.array([],dtype= np.int32))
    edge_ids: np.ndarray = field(default_factory=lambda: np.array([],dtype= np.int32))
    node_ids:np.ndarray = field(default_factory=lambda: np.array([],dtype= np.int32))
    int_dtype:np.dtype = np.int32
    keys:tuple[str] = field(init=False,repr=False)
    
    def __post_init__(self):
        self.keys = ('cell_ids','face_ids','edge_ids','vertex_ids')
        if self.int_dtype != np.int32:
            for key in self.keys:
                val = getattr(self,key)
                setattr(self,key,np.astype(val,self.int_dtype))
        
    def __getitem__(self,key):
        return getattr(self,key)
    
    @staticmethod
    def union(group_1,group_2,name =None,int_dtype:np.dtype = np.int32):
        if name is None:
            name = f'Union_{group_1}_{group_2}'
        return Group(name,*{key: np.union1d(group_1[key],group_2[key]) for key in group_1.keys},int_dtype=int_dtype)
    
    @staticmethod
    def intersection(group_1,group_2,name =None,int_dtype:np.dtype = np.int32):
        if name is None:
            name = f'Intersection_{group_1}_{group_2}'
        return Group(name,*{key: np.intersect1d(group_1[key],group_2[key]) for key in group_1.keys},int_dtype=int_dtype)
    
    