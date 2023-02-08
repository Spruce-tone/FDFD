from enum import Enum, auto, unique, IntEnum



@unique
class BC(IntEnum):
    DBC = 0 # Dirichlet boundary condition
    PBC = 1 # periodic boundary condition

def sparse_div(a, b):
    inv_b = b.copy()
    inv_b.data = 1 / inv_b.data
    return a.multiply(inv_b)