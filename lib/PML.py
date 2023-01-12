import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from typing import Union, Tuple, List, Union
from lib.utils import BC

def aPML(n: Union[int, np.ndarray], N: int, amax: float=4, p: float=3) -> Union[int, np.ndarray]:
    '''
    amplitude profile function for UPML (uniaxial perfectly matched layer)

    Parameters
    ----------
    n : Union[int, np.ndarray]
        grid index
    N : int
        UPML size
    amax : float, optional
        maximum valum of amplitude profile, by default 4
    p : float, optional
        power of the profile, by default 3

    Returns
    -------
    Union[int, np.ndarray]
        UPML amplitude profile
    '''    
    return 1 + (amax - 1) * (n / N)**p

def cPML(n: Union[int, np.ndarray], N: int, cmax: float=1) -> Union[int, np.ndarray]:
    '''
    conductivity profile function for UPML (uniaxial perfectly matched layer)

    Parameters
    ----------
    n : Union[int, np.ndarray]
        grid index
    N : int
        UPML size
    cmax : float, optional
        maximum value of conductivity, by default 1

    Returns
    -------
    Union[int, np.ndarray]
        conductivity profile of UPML
    '''    
    return cmax * np.sin(np.pi/2 * n/N)**2

def sPML(n: Union[int, np.ndarray], N: int, amax: float=4, p: float=3, cmax: float=1) -> Union[int, np.ndarray]:
    '''
    UPML (uniaxial perfectly matched layer) parameters

    Parameters
    ----------
    n : Union[int, np.ndarray]
        grid index
    N : int
        UPML size
    amax : float, optional
        maximum valum of amplitude profile, by default 4
    p : float, optional
        power of the profile, by default 3
    cmax : float, optional
        maximum value of conductivity, by default 1

    Returns
    -------
    Union[int, np.ndarray]
        UPML parameters
    '''    
    a = aPML(n=n, N=N, amax=amax, p=p)
    c = cPML(n=n, N=N, cmax=cmax)
    return a * (1 - 1j * 60 * c)

def uplm2d(ER2: np.ndarray, UR2: np.ndarray, NPML: Tuple, amax: float=4, cmax: float=1, p: float=3) -> Tuple[np.ndarray]:
    '''
    add UPMLto a 2d Yee grid 

    Parameters
    ----------
    ER2 : np.ndarray
        relative permittivity on 2x grid
    UR2 : np.ndarray
        relative permeability on 2x grid
    NPML : Tuple
        size of UPML on 1x grid
        (NXLO, NXHI, NYLO, NYHI)
    amax : float, optional
        maximum valum of amplitude profile, by default 4
    p : float, optional
        power of the profile, by default 3
    cmax : float, optional
        maximum value of conductivity, by default 1

    Returns
    -------
    Tuple[np.ndarray]
        ERxx      xx Tensor Element for Relative Permittivity
        ERyy      yy Tensor Element for Relative Permittivity
        ERzz      zz Tensor Element for Relative Permittivity
        URxx      xx Tensor Element for Relative Permeability
        URyy      yy Tensor Element for Relative Permeability
        URzz      zz Tensor Element for Relative Permeability
    '''   
    # extract grid parameters
    Nx2, Ny2 = ER2.shape 

    NXLO, NXHI, NYLO, NYHI = map(lambda x: x*2, NPML)

    ##########################
    # calculate PML parameters
    ##########################
    sx = np.ones((Nx2, Ny2), dtype='complex_')
    sy = np.ones_like(sx, dtype='complex_')

    sx_LO = np.tile(np.arange(NXLO, 0, -1), (Ny2, 1)).T
    sx[:NXLO, :] = sPML(n=sx_LO, N=NXLO, amax=amax, cmax=cmax, p=p)

    sx_HI = np.tile(np.arange(1, NXHI+1), (Ny2, 1)).T
    sx[-NXHI:, :] = sPML(n=sx_HI, N=NXHI, amax=amax, cmax=cmax, p=p)

    sy_LO = np.tile(np.arange(NYLO, 0, -1), (Nx2, 1))
    sy[:, :NYLO] = sPML(n=sy_LO, N=NYLO, amax=amax, cmax=cmax, p=p)

    sy_HI = np.tile(np.arange(1, NYHI+1), (Nx2, 1))
    sy[:, -NYHI:] = sPML(n=sy_HI, N=NYHI, amax=amax, cmax=cmax, p=p)

    ERxx = ER2 / sx * sy
    ERyy = ER2 * sx / sy
    ERzz = ER2 * sx * sy

    URxx = UR2 / sx * sy
    URyy = UR2 * sx / sy
    URzz = UR2 * sx * sy

    ERxx = ERxx[1::2,0::2]
    ERyy = ERyy[0::2,1::2]
    ERzz = ERzz[0::2,0::2]

    URxx = URxx[0::2,1::2]
    URyy = URyy[1::2,0::2]
    URzz = URzz[1::2,1::2]

    return (ERxx, ERyy, ERzz, URxx, URyy, URzz)


def plm3d(NGRID: Tuple, NPML: Tuple, amax: float=4, cmax: float=1, p: float=3) -> Tuple[np.ndarray]:
    '''
    calculate 3D PML parameters  

    Parameters
    ----------
    NGRID : Tuple
        The number of cells on grid for each axis
        (Nx, Ny, Nz)
    NPML : Tuple
        size of PML on 1x grid
        (NXLO, NXHI, NYLO, NYHI, NZLO, NZHI)
    amax : float, optional
        maximum valum of amplitude profile, by default 4
    p : float, optional
        power of the profile, by default 3
    cmax : float, optional
        maximum value of conductivity, by default 1

    Returns
    -------
    Tuple[np.ndarray]
        3D PML parameters
        (sx, sy, sz)
    '''   
    # extract grid size
    Nx, Ny, Nz = NGRID 

    # extract PML size
    NXLO, NXHI, NYLO, NYHI, NZLO, NZHI = NPML

    ##########################
    # calculate PML parameters
    ##########################
    sx = np.ones((Nx, Ny, Nz), dtype='complex_')
    sy = np.ones_like(sx, dtype='complex_')
    sz = np.ones_like(sx, dtype='complex_')

    sx_LO = np.expand_dims(np.arange(NXLO, 0, -1), axis=1).repeat(Ny, axis=1)
    sx_LO = np.expand_dims(sx_LO, axis=2).repeat(Nz, axis=2)
    sx[:NXLO, :, :] = sPML(n=sx_LO, N=NXLO, amax=amax, cmax=cmax, p=p)

    sx_HI = np.expand_dims(np.arange(1, NXHI+1), axis=1).repeat(Ny, axis=1)
    sx_HI = np.expand_dims(sx_HI, axis=2).repeat(Nz, axis=2)
    sx[-NXHI:, :, :] = sPML(n=sx_HI, N=NXHI, amax=amax, cmax=cmax, p=p)

    # sy_LO = np.expand_dims(np.arange(NYLO, 0, -1), axis=0).repeat(Ny, axis=0)
    # sy_LO = np.expand_dims(sy_LO, axis=2).repeat(Nz, axis=2)
    # sy[:, :NYLO, :] = sPML(n=sy_LO, N=NYLO, amax=amax, cmax=cmax, p=p)

    # sy_HI = np.expand_dims(np.arange(1, NYHI+1), axis=0).repeat(Ny, axis=0)
    # sy_HI = np.expand_dims(sy_HI, axis=2).repeat(Nz, axis=2)
    # sy[:, -NYHI:, :] = sPML(n=sy_HI, N=NYHI, amax=amax, cmax=cmax, p=p)

    # sz_LO = np.zeros((Nx, Ny, len(np.arange(NZLO, 0, -1))))
    # sz_LO[:, :, :] = np.arange(NZLO, 0, -1)
    # sz[:, :, :NZLO] = sPML(n=sz_LO, N=NZLO, amax=amax, cmax=cmax, p=p)

    # sz_HI = np.zeros((Nx, Ny, len(np.arange(1, NZHI+1))))
    # sz_HI[:, :, :] = np.arange(1, NZHI+1)
    # sz[:, :, -NZHI:] = sPML(n=sz_HI, N=NZHI, amax=amax, cmax=cmax, p=p)

    sx_LO = np.zeros((len(np.arange(NXLO, 0, -1)), Ny, Nz))
    sx_LO[:, :, :] = np.arange(NXLO, 0, -1).reshape(len(np.arange(NXLO, 0, -1)), 1, 1)
    sx[:NXLO, :, :] = sPML(n=sx_LO, N=NXLO, amax=amax, cmax=cmax, p=p)

    sx_HI = np.zeros((len(np.arange(1, NXHI+1)), Ny, Nz))
    sx_HI[:, :, :] = np.arange(1, NXHI+1).reshape(len(np.arange(1, NXHI+1)), 1, 1)
    sx[-NXHI:, :, :] = sPML(n=sx_HI, N=NXHI, amax=amax, cmax=cmax, p=p)

    sy_LO = np.zeros((Nx, len(np.arange(NYLO, 0, -1)), Nz))
    sy_LO[:, :, :] = np.arange(NYLO, 0, -1).reshape(1, len(np.arange(NYLO, 0, -1)), 1)
    sy[:, :NYLO, :] = sPML(n=sy_LO, N=NYLO, amax=amax, cmax=cmax, p=p)

    sy_HI = np.zeros((Nx, len(np.arange(1, NYHI+1)), Nz))
    sy_HI[:, :, :] = np.arange(1, NYHI+1).reshape(1, len(np.arange(1, NYHI+1)), 1)
    sy[:, -NYHI:, :] = sPML(n=sy_HI, N=NYHI, amax=amax, cmax=cmax, p=p)

    sz_LO = np.zeros((Nx, Ny, len(np.arange(NZLO, 0, -1))))
    sz_LO[:, :, :] = np.arange(NZLO, 0, -1).reshape(1, 1, len(np.arange(NZLO, 0, -1)))
    sz[:, :, :NZLO] = sPML(n=sz_LO, N=NZLO, amax=amax, cmax=cmax, p=p)

    sz_HI = np.zeros((Nx, Ny, len(np.arange(1, NZHI+1))))
    sz_HI[:, :, :] = np.arange(1, NZHI+1).reshape(1, 1, len(np.arange(1, NZHI+1)))
    sz[:, :, -NZHI:] = sPML(n=sz_HI, N=NZHI, amax=amax, cmax=cmax, p=p)

    return sx, sy, sz