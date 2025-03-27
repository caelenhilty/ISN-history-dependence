from numba import prange, njit, vectorize, float64
import numpy as np
from os import urandom
import matplotlib.pyplot as plt

# ----------------- Network Equations ------------------ #
# rate equation (nonlinearity enforced by solver)
@vectorize([float64(float64, float64, float64, float64)])
def drdt(r, tau, theta, I):
    return (-r + I - theta)/tau

# for short-term synaptic depression
@vectorize([float64(float64, float64, float64, float64, float64, float64)])
def dsdt(s, D, r, alpha0, pr, tauS):
    return -s/tauS + alpha0*pr*D*r*(1-s)

@vectorize([float64(float64, float64, float64, float64)])
def dDdt(D, r, tauD, pr):
    return (1-D)/tauD - pr*D*r

# ----------------- Root Solver for N = 1 ----------------- #
# a constructor that makes a function to be used in the root finding algorithm
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

from scipy.optimize import root
def get_solution(target: callable, maxiter:int=100, **kwargs):
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

# ------------------ One Pair Functions ------------------ #

@njit
def simulateISP(dt: float, duration: float, rmax: float,
            tauE: float, tauI: float,
            WEE: float, WEI: float, WIE: float, WII: float, thetaE: float, thetaI: float, 
            IappI: np.array, IappE: np.array,
            rE0 = 0., rI0=0.) ->  tuple:
    """ Simulate the firing rates of an inhibition stabilized pair of neurons with short term depression
    
    Parameters
    ----------
    dt : float
        Time step of the simulation
    duration : float
        Duration of the simulation
    rmax : float
        Maximum firing rate of the neurons
    tauE : float
        Time constant of the excitatory unit
    tauI : float
        Time constant of the inhibitory unit
    pr : float
        Release probability
    alpha0 : float
        Maximum synaptic strength
    WEE : float
        Weight of the excitatory to excitatory connection
    WEI : float
        Weight of the excitatory to inhibitory connection
    WIE : float
        Weight of the inhibitory to excitatory connection
    WII : float
        Weight of the inhibitory to inhibitory connection
    thetaE : float
        Threshold of the excitatory unit
    thetaI : float
        Threshold of the inhibitory unit
    IappI : np.array
        Applied current to the inhibitory unit
    IappE : np.array
        Applied current to the excitatory unit
    rE0 : float, default 0.
        Initial firing rate of the excitatory unit
    rI0 : float, default 0.
        Initial firing rate of the inhibitory unit
        
    Returns
    -------
    tuple
        Tuple containing the firing rates of the excitatory and inhibitory units, and the synaptic gating variables of the excitatory and inhibitory units.
        (rE, rI, DE, DI, sE, sI)
    """
    
    # vectors
    rE = np.ones(int(duration/dt))*rE0                        # firing rate of the excitatory unit
    rI = np.ones(int(duration/dt))*rI0                        # firing rate of the inhibitory unit
    
    for t in range(1, int(duration/dt)):
        # calculate k1 for all variables
        rE_k1 = dt*drdt(rE[t-1], tauE, thetaE, WEE*rE[t-1] + WIE*rI[t-1] + IappE[t-1])
        rI_k1 = dt*drdt(rI[t-1], tauI, thetaI, WEI*rE[t-1] + WII*rI[t-1] + IappI[t-1])
        
        # calculate k2 for all variables
        rE_k2 = dt*drdt(rE[t-1] + rE_k1/2, tauE, thetaE, WEE*rE[t-1] + WIE*rI[t-1] + IappE[t-1])
        rI_k2 = dt*drdt(rI[t-1] + rI_k1/2, tauI, thetaI, WEI*rE[t-1] + WII*rI[t-1] + IappI[t-1])
        
        # calculate k3 for all variables
        rE_k3 = dt*drdt(rE[t-1] + rE_k2/2, tauE, thetaE, WEE*rE[t-1] + WIE*rI[t-1] + IappE[t-1])
        rI_k3 = dt*drdt(rI[t-1] + rI_k2/2, tauI, thetaI, WEI*rE[t-1] + WII*rI[t-1] + IappI[t-1])
        
        # calculate k4 for all variables
        rE_k4 = dt*drdt(rE[t-1] + rE_k3, tauE, thetaE, WEE*rE[t-1] + WIE*rI[t-1] + IappE[t-1])
        rI_k4 = dt*drdt(rI[t-1] + rI_k3, tauI, thetaI, WEI*rE[t-1] + WII*rI[t-1] + IappI[t-1])

        # update variables
        rE[t] = rE[t-1] + (rE_k1 + 2*rE_k2 + 2*rE_k3 + rE_k4)/6
        rI[t] = rI[t-1] + (rI_k1 + 2*rI_k2 + 2*rI_k3 + rI_k4)/6
        
        # make sure the variables are within the bounds
        if rE[t] < 0:
            rE[t] = 0
        if rI[t] < 0:
            rI[t] = 0
        if rE[t] > rmax:
            rE[t] = rmax
        if rI[t] > rmax:
            rI[t] = rmax

    return rE, rI
