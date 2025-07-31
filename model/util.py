import numpy as np
import pandas as pd
from scipy.optimize import root
from typing import Callable
import os as os

# load fiducial_network.csv
def load_fiducial_network(in_figure_folder:bool=False):
    """
    Load the fiducial network parameters from the CSV file.
    If in_figure_folder is True, it assumes the script is run from a figure folder
    and adjusts the path accordingly.
    
    Parameters:
    in_figure_folder (bool): 
        If True, adjusts the path to load the CSV file correctly.
    
    Returns:
    --------
    Wji (np.ndarray): 
        The weight matrix.
    pset (np.ndarray): 
        The parameter set.
    amp (float): 
        The amplitude.
    dur (float): 
        The duration.
    l_kernel (np.ndarray): 
        The left kernel.
    r_kernel (np.ndarray): 
        The right kernel.
    
    """
    # original directory
    original_dir = os.getcwd()

    # move to the correct directory
    if in_figure_folder:
        path_to_main = os.path.join(os.getcwd(), '..', '..')
        os.chdir(os.path.abspath(path_to_main))

    # check if can move into model directory, then do it
    if os.path.exists('model'):
        os.chdir(os.path.join(os.getcwd(), 'model'))
    
    id, _, pset, amp, dur, l_kernel, r_kernel = tuple(pd.read_csv('fiducial_network.csv'))
    Wji = np.load(f'Wji_{id}.npy')

    # convert strings to appropriate types
    str_to_arr = lambda s, d: np.array([float(x) for x in s.removeprefix('[').removesuffix(']').split(d)])
    pset = str_to_arr(pset, None)
    amp = float(amp)
    dur = float(dur)
    l_kernel = str_to_arr(l_kernel, ',')
    r_kernel = str_to_arr(r_kernel, ',')

    # move back to the original directory
    os.chdir(original_dir)

    return Wji, pset, amp, dur, l_kernel, r_kernel

def make_data_folder(path:str, name:str='data'):
    """
    Create a directory if it does not already exist.
    
    Parameters:
    -----------
    path (str): 
        The path to create a data directory in.
    
    Returns:
    --------
    data_dir (str): 
        The path to the created data directory.
    """
    from pathlib import Path

    if Path(path + f'/{name}').exists():
        id = 1
        while Path(path + f'/{name}_{id}').exists():
            id += 1
        data_dir = path + f'/{name}_{id}'
        Path(data_dir).mkdir(parents=True, exist_ok=True)
    else:
        data_dir = path + f'/{name}'
        Path(data_dir).mkdir(parents=True, exist_ok=True)
    return data_dir

# root solver -----------------------------------------------------------------
# a constructor that makes a function to be used in the root finding algorithm
# solutions are of the form [WEE, WEI, WIE, WII]
def make_target(rE:float, rI:float, tr:float, det:float, thetaE:float, thetaI:float, tauE:float=10e-3, tauI:float=10e-3):
    def target(x):
        # enforce sign and minimum
        x = x ** 2 + 1e-3
        x[2], x[3] = -x[2], -x[3]
        WEE, WEI, WIE, WII = x
        
        # calculate the function
        F = np.zeros(4)
        d = (WEE-1)*(WII-1) - WEI*WIE
        F[0] = rE + ((1-WII)*thetaE + WIE*thetaI)/d
        F[1] = rI + ((1-WEE)*thetaI + WEI*thetaE)/d
        F[2] = tr - ((WEE-1)/tauE + (WII-1)/tauI)
        F[3] = det - ((WEE-1)*(WII-1) - WEI*WIE)/(tauE*tauI)
        return F
    return target

def get_solution(target: Callable, maxiter:int=100, **kwargs):
    for i in range(maxiter):
        # generate random initial conditions
        x0 = np.random.rand(4) * (6*(i/maxiter)**2 + 1) # bias towards low solutions
        # make sure the initial conditions are valid
        x0[2] = -np.abs(x0[2])
        x0[3] = -np.abs(x0[3])
        # solve
        sol = root(target, x0, **kwargs)
        x = sol.x
        ier = sol.success
        valid = x[0] > 0 and x[1] > 0 and x[2] < 0 and x[3] < 0
        if ier == 1 and valid:
            break
    valid &= ier == 1
    # enforce sign and minimum
    x = x ** 2 + 1e-3
    x[2], x[3] = -x[2], -x[3]
    return x, valid