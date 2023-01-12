from enum import Enum, auto, unique, IntEnum



@unique
class BC(IntEnum):
    DBC = 0 # Dirichlet boundary condition
    PBC = 1 # periodic boundary condition