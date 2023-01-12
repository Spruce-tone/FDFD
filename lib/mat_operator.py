import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from typing import Union, Tuple, List
from lib.utils import BC

def yeeder2d(Ns: Tuple, Res: Tuple, bc: Tuple=(BC.DBC, BC.DBC), kinc: Union[Tuple, bool]=False) -> Tuple:
    '''
    Make 2D derivative mtrix for Yee grid 
    Note: For normalized grids use k0*RES and kinc/k0

    Parameters
    ----------
    Ns : Union[Tuple]
       [Nx, Ny] grid size of each axis
    Res : Union[Tuple]
        [dx, dy] Grid resolution of each axis
    BC : Union[Tuple]
        [xbc, ybc] boundary condition
        0 (DBC) : Dirichlet boundary condition
        1 (PBC) : Periodic boundary condition
    kinc : Union[Tuple, bool], optional
        incident wave vector
        this is only needed for periodic boundary condition (PBC)
    
    Returns
    -------
    Tuple
        Return derivative matrix
        (DeX, DeY, DhX, DhY)        
    ''' 
    dim = 2 # dimension
    assert len(Ns)==dim, f'Length of the "Ns" should be {dim}'
    assert len(Res)==dim, f'Length of the "Res" should be {dim}'
    assert len(bc)==dim, f'Length of the "bc" should be {dim}'

    # grid parameters
    Nx, Ny = Ns
    dx, dy = Res

    # default kinc
    if not kinc:
        kinc = [0, 0]

    # calculate matrix size
    M = Nx * Ny

    
    ##################
    # Build DeX
    ##################
    # if grid size is 1
    if Nx==1:
        DeX = -1j * kinc[0] * sp.eye(m=M, n=M, k=0, dtype = 'complex_') # maxtrix size (m, n) = (M, M)
    
    # other cases
    else:

        # main diagonal
        d0 = -np.ones((M, ))

        # upper diagonal
        d1 = np.ones((M, ))
        d1[Nx-1 : M : Nx] = 0

        # Derivative matrix with Dirichlet Boundary condition
        DeX = sp.diags(np.array([d0, d1])/dx, [0, 1], shape=(M, M), dtype = 'complex_')

        # Incorporate Periodic Boundary Conditions
        if (bc[0].value==1) or (bc[0].name=='PBC'):
            d1 = np.zeros((M,), dtype = 'complex_')
            d1[: M : Nx] = np.exp(-1j * kinc[0] * Nx * dx) / dx
            DeX.setdiag(d1, k= 1 - Nx)

    ##################
    # Build DeY
    ##################
    if Ny==1:
        DeY = -1j * kinc[1] * sp.eye(m=M, n=M, k=0, dtype = 'complex_')  # maxtrix size (m, n) = (M, M)
    else:

        # main diagonal
        d0 = -np.ones((M,))

        # upper diagonal
        d1 = np.ones((M,))

        # Derivative matrix with Drichlet Boundary condition
        DeY = sp.diags(np.array([d0, d1])/dy, [0, Nx], dtype = 'complex_')

        # Incorporate Periodic Boundary Conditions
        if (bc[1].value==1) or (bc[1].name=='PBC'):
            d1 = np.exp(-1j * kinc[1] * Ny * dy) / dy * np.ones((M,))
            DeY.setdiag(d1, k=Nx-M)
    
    DhX = - (DeX.T.conjugate())
    DhY = - (DeY.T.conjugate())
    return DeX, DeY, DhX, DhY

def yeeder3d(Ns: Tuple, Res: Tuple, bc: Tuple=(BC.DBC, BC.DBC, BC.DBC), kinc: Union[Tuple, bool]=False) -> Tuple:
    '''
    Make 3D derivative mtrix for Yee grid 
    Note: For normalized grids use k0*RES and kinc/k0

    Parameters
    ----------
    Ns : Union[Tuple]
       [Nx, Ny, Nz] grid size of each axis
    Res : Union[Tuple]
        [dx, dy, dz] Grid resolution of each axis
    BC : Union[Tuple]
        [xbc, ybc, zbc] boundary condition
        0 (DBC) : Dirichlet boundary condition
        1 (PBC) : Periodic boundary condition
    kinc : Union[Tuple, bool], optional
        incident wave vector
        this is only needed for periodic boundary condition (PBC)
    
    Returns
    -------
    Tuple
        Return derivative matrix
        (DeX, DeY, DhX, DhY)        
    ''' 
    dim = 3 # dimension
    assert len(Ns)==dim, f'Length of the "Ns" should be {dim}'
    assert len(Res)==dim, f'Length of the "Res" should be {dim}'
    assert len(bc)==dim, f'Length of the "bc" should be {dim}'

    # grid parameters
    Nx, Ny, Nz = Ns
    dx, dy, dz = Res

    # default kinc
    if not kinc:
        kinc = [0, 0, 0]

    # calculate matrix size
    M = Nx * Ny * Nz

    
    ##################
    # Build DeX
    ##################
    # if grid size is 1
    if Nx==1:
        DeX = -1j * kinc[0] * sp.eye(m=M, n=M, k=0, dtype = 'complex_') # maxtrix size (m, n) = (M, M)
    
    # other cases
    else:

        # main diagonal
        d0 = -np.ones((M,))

        # upper diagonal
        d1 = np.ones((M,))
        d1[Nx-1 : M : Nx] = 0

        # Derivative matrix with Dirichlet Boundary condition
        DeX = sp.diags(np.array([d0, d1])/dx, [0, 1], shape=(M, M), dtype = 'complex_')

        # Incorporate Periodic Boundary Conditions
        if (bc[0].value==1) or (bc[0].name=='PBC'):
            d1 = np.zeros((M,), dtype = 'complex_')
            d1[: M : Nx] = np.exp(-1j * kinc[0] * Nx * dx) / dx
            DeX.setdiag(d1, k= 1 - Nx)

    ##################
    # Build DeY
    ##################
    if Ny==1:
        DeY = -1j * kinc[1] * sp.eye(m=M, n=M, k=0, dtype = 'complex_')  # maxtrix size (m, n) = (M, M)
    else:

        # main diagonal
        d0 = -np.ones((M,))

        # upper diagonal
        d1 = np.hstack([np.ones((Ny-1)*Nx), np.zeros(Nx)])
        d1 = np.tile(d1, Nz)

        # Derivative matrix with Drichlet Boundary condition
        DeY = sp.diags(np.array([d0, d1])/dy, [0, Nx], dtype = 'complex_')

        # Incorporate Periodic Boundary Conditions
        if (bc[1].value==1) or (bc[1].name=='PBC'):
            ph = np.exp(-1j * kinc[1] * Ny * dy) / dy # phase

            d1 = np.hstack([np.ones(Nx), np.zeros((Ny-1)*Nx)])
            d1 = np.tile(d1, Nz)

            DeY.setdiag(ph * d1, k=-Nx*(Ny-1))

    ##################
    # Build DeZ
    ##################
    if Nz==1:
        DeZ = -1j * kinc[2] * sp.eye(m=M, n=M, k=0, dtype = 'complex_')  # maxtrix size (m, n) = (M, M)
    else:
        d0 = np.ones((M,))

        DeZ = sp.diags(np.array([-d0, d0])/dz, [0, Nx*Ny], dtype = 'complex_')

        if (bc[2].value==1) or (bc[2].name=='PBC'):
            d0 = np.exp(-1j * kinc[2] * Nz * dz) / dz  * np.ones((M,))
            DeZ.setdiag(d0, -Nx*Ny*(Nz-1))


    DhX = - (DeX.T.conjugate())
    DhY = - (DeY.T.conjugate())
    DhZ = - (DeZ.T.conjugate())
    return DeX, DeY, DeZ, DhX, DhY, DhZ