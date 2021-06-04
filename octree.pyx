from libcpp cimport bool as bool_t
from scipy import spatial.cKDTree
cdef class Octant:  
    """
        branch: 0 1 2 3 4 5 6 7
        x:      - - - - + + + +
        y:      - - + + - - + +
        z:      - + - + - + - +
    """
    def __init__(self):
        self.children = [None,None,None,None,None,None,None,None]
        self.isLeafNode = True
