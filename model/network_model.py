from operator import ge
from numba import njit, vectorize, float64
import numpy as np
from os import urandom
import matplotlib.pyplot as plt
import time
import networkx as nx

# ----------------- Network Equation ------------------ #
# rate equation (nonlinearity enforced by solver)
@vectorize([float64(float64, float64, float64, float64)])
def drdt(r, tau, theta, I):
    return (-r + I - theta)/tau

# ------------------ One Pair "Network" ------------------ #
@njit
def simulateISP(dt: float, duration: float, rmax: float,
            tauE: float, tauI: float,
            WEE: float, WEI: float, WIE: float, WII: float, thetaE: float, thetaI: float, 
            IappI: np.ndarray, IappE: np.ndarray,
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


# ------------------ Size-N Network ------------------ #
def pack_parameters(tauE: float, tauI: float, tauDE: float, tauDI: float, tauSE: float, tauSI: float,
            pr: float, alpha0: float,
            WEE: float, WEI: float, WIE: float, WII: float, thetaE: float, thetaI: float
            ) -> np.array:
    """ Pack network parameters into a numpy array
    
    Parameters
    ----------
    tauE : float
        Time constant of the excitatory unit
    tauI : float
        Time constant of the inhibitory unit
    tauDE : float
        Time constant of the depression of the excitatory unit
    tauDI : float
        Time constant of the depression of the inhibitory unit
    tauSE : float
        Time constant of the synaptic gating variable of the excitatory unit
    tauSI : float
        Time constant of the synaptic gating variable of the inhibitory unit
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
        
    Returns
    -------
    np.array
        Array containing all the parameters \n
        [tauE, tauI, tauDE, tauDI, tauSE, tauSI, pr, alpha0, WEE, WEI, WIE, WII, thetaE, thetaI]
    """
    return np.array([tauE, tauI, tauDE, tauDI, tauSE, tauSI, pr, alpha0, WEE, WEI, WIE, WII, thetaE, thetaI], dtype=np.float64)

@njit
def unpack_parameters(pset: np.array) -> tuple:
    """ Unpack network parameters from a numpy array
    
    Parameters
    ----------
    pset : np.array
        Array containing all the parameters \n
        [tauE, tauI, tauDE, tauDI, tauSE, tauSI, pr, alpha0, WEE, WEI, WIE, WII, thetaE, thetaI]
        
    Returns
    -------
    tuple
        Tuple containing all the parameters \n
        (tauE, tauI, tauDE, tauDI, tauSE, tauSI, pr, alpha0, WEE, WEI, WIE, WII, thetaE, thetaI)
    """
    return pset[0], pset[1], pset[2], pset[3], pset[4], pset[5], pset[6], pset[7], pset[8], pset[9], pset[10], pset[11], pset[12], pset[13]

# model setup
def getRNG() -> tuple:
    """ Get a random number generator and seed it with urandom.
    
    Returns
    -------
    rng: np.random.default_rng
        The random number generator.
    seed: int
        The seed used to seed the random number generator.
    """
    random_data = urandom(8)
    seed = int.from_bytes(random_data, byteorder="big")
    rng = np.random.default_rng(seed)   
    return rng, seed

def makeWji_all_types(rng: np.random.default_rng, numPairs: float, 
                      meanStrengths: np.array, stdStrengths: np.array,
                      timeout=np.inf, mean_tol=None, std_tol=None, verbose=False,
                      timeout_limit=np.inf) -> np.array:
    """ Create a random connection weight matrix for the network, 
    
    Parameters
    ----------
    rng : np.random.default_rng
        The random number generator to use.
    numPairs : int
        The number of pairs in the network.
    meanStrengths : np.array
        The mean of the connection weights for each type of connection.
        Must have 4 elements.
    stdStrengths : np.array
        The standard deviation of the connection weights for each type of connection.
        Must have 4 elements.
    timeout : float, default np.inf
        Max time spent in normalization & sign enforcement loop before giving up
    
    Returns
    -------
    np.array
        The (4 x numPairs x numPairs) matrix of connection weights between the pairs.
        (0, :, :) is the I-to-E connection weights
        (1, :, :) is the I-to-I connection weights
        (2, :, :) is the E-to-E connection weights
        (3, :, :) is the E-to-I connection weights    
    
    """
    
    if meanStrengths.size != stdStrengths.size:
        raise ValueError("meanStrengths and stdStrengths must be the same size")
    if meanStrengths.size != 4:
        raise ValueError("meanStrengths and stdStrengths must have 4 elements")
    
    # check that the first two elements of meanStrengths are <= 0
    if not np.all(meanStrengths[:2] <= 0):
        raise ValueError("The first two elements of meanStrengths must be <= 0")
    # check that the last two elements of meanStrengths are >= 0
    if not np.all(meanStrengths[2:] >= 0):
        raise ValueError("The last two elements of meanStrengths must be >= 0")
    
    return np.array([make_Wji(rng, numPairs, strength, std, timeout=timeout,
                             mean_tol=mean_tol, std_tol=std_tol, verbose=verbose, timeout_limit=timeout_limit) for strength, std in zip(meanStrengths, stdStrengths)])

def make_Wji(rng:np.random.default_rng, numPairs:int, mean:float, std:float,
                 mean_tol = None, std_tol =None, timeout=np.inf, timeout_limit=np.inf,
                 verbose=False):
    """
    Create a weight matrix for the connections between pairs of neurons.

    Parameters
    ----------
    rng : np.random.default_rng
        The random number generator to use.
    numPairs : int
        The number of pairs of neurons.
    mean : float
        The mean weight for the connections.
    std : float
        The standard deviation of the weights.
    mean_tol : float, optional
        The tolerance for the mean weight (default is None).
    std_tol : float, optional
        The tolerance for the standard deviation (default is None).
    timeout : float, optional
        The timeout for the weight matrix generation (default is np.inf).
    timeout_limit : int, optional
        The maximum number of timeouts allowed (default is np.inf).
    verbose : bool, optional
        Whether to print verbose output (default is False).

    Returns
    -------
    np.ndarray
        The weight matrix for the connections between pairs of neurons. Shape: (numPairs x numPairs)
    """
    
    assert mean != 0, "Mean cannot be zero"
    assert std  >= 0, "Standard deviation must be non-negative"
    

    if mean_tol is None:
        mean_tol = np.abs(0.05 * mean)
    else:
        mean_tol = np.abs(mean_tol * mean) # convert to absolute
    if std_tol is None:
        std_tol = np.abs(0.05 * std)
    else:
        std_tol = np.abs(std_tol * std)

    if std == 0:
        Wji = np.ones((numPairs, numPairs)) * mean
        np.fill_diagonal(Wji, 0)
        return Wji

    sign = -1 if mean < 0 else 1
    abs_mean = np.abs(mean)

    mu_log = np.log(abs_mean**2 / np.sqrt(std**2 + abs_mean**2))
    sigma_log = np.sqrt(np.log(1 + (std**2 / abs_mean**2)))
    Wji = _log_normal_helper(rng, numPairs, mu_log, sigma_log, abs_mean, mean_tol, timeout)
    # now enforce std
    get_whole_std = lambda Wji: np.std(Wji[np.eye(Wji.shape[0], dtype=bool) == False])
    timeouts = 0
    d_closest = np.inf
    closest_val = get_whole_std(Wji)
    std_boost = 0
    tries = 0
    while np.abs(get_whole_std(Wji) - std) > std_tol:
        try:
            std_boost += 1e-3 * std * np.sign(std - get_whole_std(Wji)) 
            # small boost to sigma to help convergence: if sample std is too low, we need to increase it a bit
            # if std is too high, we need to decrease it a bit
            # helps a lot for higher stds, where we need to get samples from the tail to match the mean and std simultaneously

            # recompute log-normal parameters
            mu_log = np.log(abs_mean**2 / np.sqrt(max(std + std_boost, 1e-10)**2 + abs_mean**2))
            sigma_log = np.sqrt(np.log(1 + (max(std + std_boost, 1e-10)**2 / abs_mean**2)))
            Wji = _log_normal_helper(rng, numPairs, mu_log, sigma_log, abs_mean, mean_tol, timeout)
            tries += 1
            current_std = get_whole_std(Wji)
            if np.abs(current_std - std) < d_closest:
                d_closest = np.abs(current_std - std)
                closest_val = current_std
            if verbose and tries%1000==0: print(f"Retry {tries}: closest = {closest_val:.3f}, target = {std:.3f}, boost = {std_boost:.5f}", end='\r')
        except TimeoutError:
            timeouts += 1
            if timeouts > timeout_limit:
                raise TimeoutError(f"Timeout count exceeded while generating Wji with mean {mean} and std {std}")
            continue
    if verbose: print()
    return Wji * sign

def _log_normal_helper(rng, numPairs, mu, sigma, mean, mean_tol, timeout=np.inf):
    """Generates an array where each row has the desired mean"""
    Wji = np.zeros((numPairs, numPairs))
    for i in range(numPairs):
        start_time = time.time()
        curr_time = start_time
        values = rng.lognormal(mu, sigma, numPairs-1)
        current_mean = np.mean(values)
        while abs(current_mean - mean) > mean_tol and \
                abs(curr_time - start_time) < timeout:
            values = rng.lognormal(mu, sigma, numPairs-1)
            current_mean = np.mean(values)
            curr_time = time.time()
        if np.abs(current_mean - mean) <= mean_tol:
            Wji[i] = np.insert(values, i, 0)  # Insert zero on the diagonal
        if abs(curr_time - start_time) >= timeout:
            raise TimeoutError(f"Timeout exceeded while generating Wji with mean {mean} and std {sigma}")
    return Wji

@njit(cache=True)
def simulateISN(Wji: np.array, numPairs: int, r0: np.array, pset: np.array, 
                 IappE: np.array = np.empty((1,1)), IappI: np.array = np.empty((1,1)), dt: float = 1e-3, duration: float = 6
                 ) -> tuple:
    """Use RK4 to simulate a homogenous network of inhibition stabilized pairs.
    
    Simulate the behavior of the network for the given duration. A threshold linear model is used for the firing rates of the units.
    
    Parameters
    ----------
    Wji : np.array
        The (4 x numPairs x numPairs) matrix of connection weights between the pairs
        Use makeWji_allTypes to generate this matrix.
        (I-to-E, I-to-I, E-to-E, E-to-I)
    numPairs : int
        The number of pairs in the network.
    r0 : np.array
        The (numPairs x 2) array of initial firing rates of the units.
    pset : np.array
        The parameter set for the network.
    IappI : np.array
        The (numPairs x int(duration/dt)) array of applied input to the inhibitory units.
    IappE : np.array
        The (numPairs x int(duration/dt)) array of applied input to the excitatory units.
    
    Returns 
    -------
    rates : np.array
        A 3D array of the firing rates of the units, with the first dimension being the pair, the second dimension being the unit, and the third dimension being time
    
    """
    # validate Wji
    assert Wji.ndim == 3, "Wji must be a 3D array"
    assert Wji.shape[0] == 4, "Wji must have 4 elements in the first dimension"
    assert Wji.shape[1] == Wji.shape[2] and Wji.shape[1] == numPairs, "Wji must be a (4 x numPairs x numPairs) array"
    
    # unpack parameters
    tauE, tauI, _, _, _, _, _, _, WEE, WEI, WIE, WII, thetaE, thetaI = unpack_parameters(pset)

    # add within-pair connections to Wji's diagonal
    for i in range(numPairs):
        Wji[0,i,i] = WIE
        Wji[1,i,i] = WII
        Wji[2,i,i] = WEE
        Wji[3,i,i] = WEI

    # initialize Iapp arrays
    if IappE.size == 1: IappE = np.zeros((numPairs, int(duration/dt)))
    if IappI.size == 1: IappI = np.zeros((numPairs, int(duration/dt)))
    
    # output arrays
    rates = np.zeros((numPairs, 2, int(duration/dt))) # column 0 is excitatory, column 1 is inhibitory
    rates[:, :, 0] = r0
    
    # pre-allocate temporary arrays for RK4
    rEk1, rEk2, rEk3, rEk4 = np.empty(numPairs), np.empty(numPairs), np.empty(numPairs), np.empty(numPairs)
    rIk1, rIk2, rIk3, rIk4 = np.empty(numPairs), np.empty(numPairs), np.empty(numPairs), np.empty(numPairs)
    
    for t in range(1, int(duration/dt)):
        # calculate k1 for all variables
        rEk1 = dt*drdt(rates[:,0,t-1], tauE, thetaE,
                        # I-to-E
                         np.dot(np.ascontiguousarray(Wji[0,:,:]), np.ascontiguousarray(rates[:,1,t-1])) + 
                        # E-to-E
                         np.dot(np.ascontiguousarray(Wji[2,:,:]), np.ascontiguousarray(rates[:,0,t-1])) +
                        # applied current
                         IappE[:,t-1])
        rIk1 = dt*drdt(rates[:,1,t-1], tauI, thetaI,
                        # I-to-I
                         np.dot(np.ascontiguousarray(Wji[1,:,:]), np.ascontiguousarray(rates[:,1,t-1])) +
                        # E-to-I
                         np.dot(np.ascontiguousarray(Wji[3,:,:]), np.ascontiguousarray(rates[:,0,t-1])) +
                        # applied current
                         IappI[:,t-1])
        
        # k2
        rEk2 = dt * drdt(rates[:,0,t-1] + rEk1/2, tauE, thetaE,
                         np.dot(np.ascontiguousarray(Wji[0,:,:]), np.ascontiguousarray(rates[:,1,t-1] + rIk1/2)) + 
                         np.dot(np.ascontiguousarray(Wji[2,:,:]), np.ascontiguousarray(rates[:,0,t-1] + rEk1/2)) + IappE[:,t-1])
        rIk2 = dt * drdt(rates[:,1,t-1] + rIk1/2, tauI, thetaI,
                         np.dot(np.ascontiguousarray(Wji[1,:,:]), np.ascontiguousarray(rates[:,1,t-1] + rIk1/2)) +
                         np.dot(np.ascontiguousarray(Wji[3,:,:]), np.ascontiguousarray(rates[:,0,t-1] + rEk1/2)) + IappI[:,t-1])
        
        # k3
        rEk3 = dt * drdt(rates[:,0,t-1] + rEk2/2, tauE, thetaE,
                         np.dot(np.ascontiguousarray(Wji[0,:,:]), np.ascontiguousarray(rates[:,1,t-1] + rIk2/2)) + 
                         np.dot(np.ascontiguousarray(Wji[2,:,:]), np.ascontiguousarray(rates[:,0,t-1] + rEk2/2)) + IappE[:,t-1])
        rIk3 = dt * drdt(rates[:,1,t-1] + rIk2/2, tauI, thetaI,
                         np.dot(np.ascontiguousarray(Wji[1,:,:]), np.ascontiguousarray(rates[:,1,t-1] + rIk2/2)) +
                         np.dot(np.ascontiguousarray(Wji[3,:,:]), np.ascontiguousarray(rates[:,0,t-1] + rEk2/2)) + IappI[:,t-1])
        
        # k4
        rEk4 = dt * drdt(rates[:,0,t-1] + rEk3, tauE, thetaE,
                         np.dot(np.ascontiguousarray(Wji[0,:,:]), np.ascontiguousarray(rates[:,1,t-1] + rIk3)) + 
                         np.dot(np.ascontiguousarray(Wji[2,:,:]), np.ascontiguousarray(rates[:,0,t-1] + rEk3)) + IappE[:,t])
        rIk4 = dt * drdt(rates[:,1,t-1] + rIk3, tauI, thetaI,
                         np.dot(np.ascontiguousarray(Wji[1,:,:]), np.ascontiguousarray(rates[:,1,t-1] + rIk3)) +
                         np.dot(np.ascontiguousarray(Wji[3,:,:]), np.ascontiguousarray(rates[:,0,t-1] + rEk3)) + IappI[:,t])
        
        # update variables
        rates[:,0,t] = rates[:,0,t-1] + (rEk1 + 2*rEk2 + 2*rEk3 + rEk4)/6.
        rates[:,1,t] = rates[:,1,t-1] + (rIk1 + 2*rIk2 + 2*rIk3 + rIk4)/6.
        
        # make sure the firing rates are within the bounds
        rates[:, :, t] = np.clip(rates[:, :, t], 0, 100)
    
    return rates

@njit(cache=True)
def simulateISN_noisy(Wji: np.array, numPairs: int, r0: np.array, pset: np.array, 
                 IappE: np.array = np.empty((1,1)), IappI: np.array = np.empty((1,1)), dt: float = 1e-3, duration: float = 6,
                 sigma_noise: np.array = np.empty((1,1,1))) -> tuple:
    """Use RK4 to simulate a homogenous network of inhibition stabilized pairs.
    
    Simulate the behavior of the network for the given duration. A threshold linear model is used for the firing rates of the units.
    
    Parameters
    ----------
    Wji : np.array
        The (4 x numPairs x numPairs) matrix of connection weights between the pairs
        Use makeWji_allTypes to generate this matrix.
        (I-to-E, I-to-I, E-to-E, E-to-I)
    numPairs : int
        The number of pairs in the network.
    r0 : np.array
        The (numPairs x 2) array of initial firing rates of the units.
    pset : np.array
        The parameter set for the network.
    IappI : np.array
        The (numPairs x int(duration/dt)) array of applied input to the inhibitory units.
    IappE : np.array
        The (numPairs x int(duration/dt)) array of applied input to the excitatory units.
    noise : float
        The standard deviation of the noise in the input to the excitatory units.
    
    Returns 
    -------
    rates : np.array
        A 3D array of the firing rates of the units, with the first dimension being the pair, the second dimension being the unit, and the third dimension being time
    
    """
    # validate Wji
    assert Wji.ndim == 3, "Wji must be a 3D array"
    assert Wji.shape[0] == 4, "Wji must have 4 elements in the first dimension"
    assert Wji.shape[1] == Wji.shape[2] and Wji.shape[1] == numPairs, "Wji must be a (4 x numPairs x numPairs) array"
    
    # unpack parameters
    tauE, tauI, _, _, _, _, _, _, WEE, WEI, WIE, WII, thetaE, thetaI = unpack_parameters(pset)

    # add within-pair connections to Wji's diagonal
    for i in range(numPairs):
        Wji[0,i,i] = WIE
        Wji[1,i,i] = WII
        Wji[2,i,i] = WEE
        Wji[3,i,i] = WEI

    # initialize Iapp arrays
    if IappE.size == 1: IappE = np.zeros((numPairs, int(duration/dt)))
    if IappI.size == 1: IappI = np.zeros((numPairs, int(duration/dt)))
    
    # initialize noise array
    if sigma_noise.size == 1: sigma_noise = np.zeros((numPairs, 2, int(duration/dt)))
    
    # output arrays
    rates = np.zeros((numPairs, 2, int(duration/dt))) # column 0 is excitatory, column 1 is inhibitory
    rates[:, :, 0] = r0
    
    # pre-allocate temporary arrays for RK4
    rEk1, rEk2, rEk3, rEk4 = np.empty(numPairs), np.empty(numPairs), np.empty(numPairs), np.empty(numPairs)
    rIk1, rIk2, rIk3, rIk4 = np.empty(numPairs), np.empty(numPairs), np.empty(numPairs), np.empty(numPairs)
    
    for t in range(1, int(duration/dt)):
        # noise for each unit, to be scaled by the square root of the firing rate and connection weight
        noise_vec = np.random.normal(0, 1, size=(numPairs, 2)) * sigma_noise[:, :, t-1]
        
        # calculate k1 for all variables
        rEk1 = dt*drdt(rates[:,0,t-1], tauE, thetaE,
                        # I-to-E
                         np.dot(np.ascontiguousarray(Wji[0,:,:]), np.ascontiguousarray(rates[:,1,t-1])) + 
                         np.dot(np.ascontiguousarray(Wji[0,:,:])*np.sqrt(np.abs(np.ascontiguousarray(rates[:,1,t-1]))), noise_vec[:, 1]) + # noise
                        # E-to-E
                         np.dot(np.ascontiguousarray(Wji[2,:,:]), np.ascontiguousarray(rates[:,0,t-1])) +
                         np.dot(np.ascontiguousarray(Wji[2,:,:])*np.sqrt(np.abs(np.ascontiguousarray(rates[:,0,t-1]))), noise_vec[:, 0]) + # noise
                        # applied current
                         IappE[:,t-1])
        rIk1 = dt*drdt(rates[:,1,t-1], tauI, thetaI,
                        # I-to-I
                         np.dot(np.ascontiguousarray(Wji[1,:,:]), np.ascontiguousarray(rates[:,1,t-1])) +
                         np.dot(np.ascontiguousarray(Wji[1,:,:])*np.sqrt(np.abs(np.ascontiguousarray(rates[:,1,t-1]))), noise_vec[:, 1]) + # noise
                        # E-to-I
                         np.dot(np.ascontiguousarray(Wji[3,:,:]), np.ascontiguousarray(rates[:,0,t-1])) +
                         np.dot(np.ascontiguousarray(Wji[3,:,:])*np.sqrt(np.abs(np.ascontiguousarray(rates[:,0,t-1]))), noise_vec[:, 0]) + # noise
                        # applied current
                         IappI[:,t-1])
        
        # k2
        adjusted_E_rates = np.ascontiguousarray(rates[:,0,t-1] + rEk1/2)
        adjusted_I_rates = np.ascontiguousarray(rates[:,1,t-1] + rIk1/2)
        rEk2 = dt * drdt(rates[:,0,t-1] + rEk1/2, tauE, thetaE,
                         np.dot(np.ascontiguousarray(Wji[0,:,:]), adjusted_I_rates) + np.dot(np.ascontiguousarray(Wji[0,:,:])*np.sqrt(np.abs(adjusted_I_rates)), noise_vec[:, 1]) +
                         np.dot(np.ascontiguousarray(Wji[2,:,:]), adjusted_E_rates) + np.dot(np.ascontiguousarray(Wji[2,:,:])*np.sqrt(np.abs(adjusted_E_rates)), noise_vec[:, 0]) +
                         IappE[:,t-1])
        rIk2 = dt * drdt(rates[:,1,t-1] + rIk1/2, tauI, thetaI,
                         np.dot(np.ascontiguousarray(Wji[1,:,:]), adjusted_I_rates) + np.dot(np.ascontiguousarray(Wji[1,:,:])*np.sqrt(np.abs(adjusted_I_rates)), noise_vec[:, 1]) +
                         np.dot(np.ascontiguousarray(Wji[3,:,:]), adjusted_E_rates) + np.dot(np.ascontiguousarray(Wji[3,:,:])*np.sqrt(np.abs(adjusted_E_rates)), noise_vec[:, 0]) +
                         IappI[:,t-1])
        
        # k3
        adjusted_E_rates = np.ascontiguousarray(rates[:,0,t-1] + rEk2/2)
        adjusted_I_rates = np.ascontiguousarray(rates[:,1,t-1] + rIk2/2)
        rEk3 = dt * drdt(rates[:,0,t-1] + rEk2/2, tauE, thetaE,
                         np.dot(np.ascontiguousarray(Wji[0,:,:]), adjusted_I_rates) + np.dot(np.ascontiguousarray(Wji[0,:,:])*np.sqrt(np.abs(adjusted_I_rates)), noise_vec[:, 1]) +
                         np.dot(np.ascontiguousarray(Wji[2,:,:]), adjusted_E_rates) + np.dot(np.ascontiguousarray(Wji[2,:,:])*np.sqrt(np.abs(adjusted_E_rates)), noise_vec[:, 0]) +
                         IappE[:,t-1])
        rIk3 = dt * drdt(rates[:,1,t-1] + rIk2/2, tauI, thetaI,
                         np.dot(np.ascontiguousarray(Wji[1,:,:]), adjusted_I_rates) + np.dot(np.ascontiguousarray(Wji[1,:,:])*np.sqrt(np.abs(adjusted_I_rates)), noise_vec[:, 1]) +
                         np.dot(np.ascontiguousarray(Wji[3,:,:]), adjusted_E_rates) + np.dot(np.ascontiguousarray(Wji[3,:,:])*np.sqrt(np.abs(adjusted_E_rates)), noise_vec[:, 0]) +
                         IappI[:,t-1])
        
        # k4
        adjusted_E_rates = np.ascontiguousarray(rates[:,0,t-1] + rEk3)
        adjusted_I_rates = np.ascontiguousarray(rates[:,1,t-1] + rIk3)
        rEk4 = dt * drdt(rates[:,0,t-1] + rEk3, tauE, thetaE,
                         np.dot(np.ascontiguousarray(Wji[0,:,:]), adjusted_I_rates) + np.dot(np.ascontiguousarray(Wji[0,:,:])*np.sqrt(np.abs(adjusted_I_rates)), noise_vec[:, 1]) +
                         np.dot(np.ascontiguousarray(Wji[2,:,:]), adjusted_E_rates) + np.dot(np.ascontiguousarray(Wji[2,:,:])*np.sqrt(np.abs(adjusted_E_rates)), noise_vec[:, 0]) +
                         IappE[:,t])
        rIk4 = dt * drdt(rates[:,1,t-1] + rIk3, tauI, thetaI,
                         np.dot(np.ascontiguousarray(Wji[1,:,:]), adjusted_I_rates) + np.dot(np.ascontiguousarray(Wji[1,:,:])*np.sqrt(np.abs(adjusted_I_rates)), noise_vec[:, 1]) +
                         np.dot(np.ascontiguousarray(Wji[3,:,:]), adjusted_E_rates) + np.dot(np.ascontiguousarray(Wji[3,:,:])*np.sqrt(np.abs(adjusted_E_rates)), noise_vec[:, 0]) +
                         IappI[:,t])
        
        # update variables
        rates[:,0,t] = rates[:,0,t-1] + (rEk1 + 2*rEk2 + 2*rEk3 + rEk4)/6.
        rates[:,1,t] = rates[:,1,t-1] + (rIk1 + 2*rIk2 + 2*rIk3 + rIk4)/6.
        
        # make sure the firing rates are within the bounds
        rates[:, :, t] = np.clip(rates[:, :, t], 0, 100)
    
    return rates

@njit
def b10ToBinary(n: int):
    """ Convert a base 10 integer to binary.
    
    Parameters
    ----------
    n : int
        The base 10 integer to convert.
        n must be less than 2^32.
    
    Returns
    -------
    out : np.array
        The binary representation of n.
        Can hold up to 32 bits -- the array is padded with zeros to the left.
    """
    # convert a base 10 integer to binary
    out = np.zeros(32, dtype=np.int64)
    for i in range(31, -1, -1):
        k = n >> i
        if k & 1:
            out[31-i] = 1
        
    return out

def get_all_states(Wji: np.array, pset, numPairs: int, Eactive: float, Iactive:float,
                     IappI:np.array=np.empty((1,1)), IappE:np.array=np.empty((1,1)), 
                     dt:float=1e-5, duration:float=7) -> tuple:
    assert Wji.ndim == 3, "Wji must be a 3D array"
    assert duration > 2, "Duration must be greater than 2 seconds"
    
    # create a set of initial firing rates
    activeRates = np.clip(np.array([[Eactive/2, Iactive/2],
                                [Eactive, Iactive], 
                                [Eactive + (100-Eactive)/2, Iactive * ((Eactive + (100-Eactive)/2)/Eactive)],
                                ], dtype=np.float64), 0, 100)
    
    # initialize Iapp arays
    if IappE.size == 1: IappE = np.zeros((numPairs, int(duration/dt))) # size == 1 is used as a placeholder for numba compatibility
    if IappI.size == 1: IappI = np.zeros((numPairs, int(duration/dt))) # input arrays should never be size 1 other than by default
    
    # add small amount of noise to the input currents from t = -1.5 to t = -1
    noise_mg = 0.001
    start_idx = int((duration-1.5)/dt)
    end_idx = int((duration-1)/dt)
    num_t_steps = end_idx - start_idx
    IappE[:, start_idx:end_idx] += noise_mg * np.random.randn(numPairs, num_t_steps)

    # output vectors
    steady_states = np.ones(((2**numPairs)*len(activeRates), numPairs, 2))*-1
    
    for n in range(0, 2**numPairs):
        # uses binary counting to get all possible combinations of active pairs
        binaryActive = b10ToBinary(n)[-numPairs:] # handles left padding
        for i in range(len(activeRates)):
            # set initial conditions
            r0 = np.zeros((numPairs,2), dtype=np.float64)
            for pairID in range(numPairs):
                if binaryActive[pairID] == 1:
                    r0[pairID,0] = activeRates[i,0]
                    r0[pairID,1] = activeRates[i,1]
                    
            # run simulation
            rates = simulateISN(Wji, numPairs, r0, pset, IappE, IappI, dt, duration)
            
            # check if the rates are stable within 0.1 and below 100 Hz
            final_rates = rates[:, :, -1]
            stable = np.allclose(rates[:, :, int((duration-0.5)/dt):-1], final_rates[..., np.newaxis], 
                                 rtol=0, atol=0.1) and np.all(final_rates < 100)
            
            if stable:
                steady_states[(n * len(activeRates)) + i] = final_rates
        
    return steady_states[steady_states[:, 0, 0] != -1]    # filter out the -1 values which indicate that the state was not stable

def count_homogeneous_ISN_states(Wji: np.array, pset, numPairs: int, Eactive: float, Iactive:float,
                     IappI:np.array=np.empty((1,1)), IappE:np.array=np.empty((1,1)), 
                     dt:float=1e-5, duration:float=7, ON_threshold=1) -> tuple:
    assert Wji.ndim == 3, "Wji must be a 3D array"
    assert duration > 2, "Duration must be greater than 2 seconds"
    
    # create a set of initial firing rates for the ON units (OFF units are 0 Hz)
    active_rates = np.clip(np.array([[Eactive/2, Iactive/2],
                                [Eactive, Iactive], 
                                [Eactive + (100-Eactive)/2, Iactive * ((Eactive + (100-Eactive)/2)/Eactive)],
                                ], dtype=np.float64), 0, 100)
    
    # initialize Iapp arays
    if IappE.size == 1: IappE = np.zeros((numPairs, int(duration/dt))) # size == 1 is used as a placeholder for numba compatibility
    if IappI.size == 1: IappI = np.zeros((numPairs, int(duration/dt))) # input arrays should never be size 1 other than by default
    
    # add small amount of noise to the input currents from t = -1.5 to t = -1
    noise_mg = 0.001
    start_idx = int((duration-1.5)/dt)
    end_idx = int((duration-1)/dt)
    num_t_steps = end_idx - start_idx
    IappE[:, start_idx:end_idx] += noise_mg * np.random.randn(numPairs, num_t_steps)

    possible_on_states = np.ones(numPairs * len(active_rates), dtype=np.int64) * -1

    for n in range(0, numPairs):
        for i in range(len(active_rates)):
            r0 = np.zeros((numPairs,2), dtype=np.float64)
            # set initial conditions
            r0[:n, 0] = active_rates[i,0]
            r0[:n, 1] = active_rates[i,1]
            
            # run simulation
            rates = simulateISN(Wji, numPairs, r0, pset, IappE, IappI, dt, duration)
            
            # check for stability
            final_rates = rates[:, :, -1]
            stable = np.allclose(rates[:, :, int((duration-0.5)/dt):-1], final_rates[..., np.newaxis], 
                                 rtol=0, atol=0.1) and np.all(final_rates < 100)
            
            if stable:
                # count the number of ON units (considering only E units)
                on_count = np.sum(final_rates[:, 0] > ON_threshold)
                possible_on_states[(n * len(active_rates)) + i] = on_count

    possible_on_states = possible_on_states[possible_on_states != -1]  # filter out the -1 values
    # count the number of unique ON states
    return np.sum([__nCr(numPairs, r) for r in np.unique(possible_on_states)])
    
@njit
def __factorial(n:int) -> int:
    """ Calculate the factorial of a number n. njit version.

    Parameters
    ----------
    n : int
        The number to calculate the factorial of.
        
    Returns
    -------
    int
        The factorial of n.
    """
    if n == 0 or n == 1:
        return 1
    else:
        return n * __factorial(n - 1)

@njit
def __nCr(n:int, r:int) -> int:
    """ Calculate the number of combinations of n items taken r at a time. njit version."""
    if r > n:
        return 0
    if r == 0 or r == n:
        return 1
    # calculate n choose r (combinations), round to nearest integer
    return __factorial(n) // (__factorial(r) * __factorial(n - r))

def get_state_transition_graph(Wji, pset, stim_amplitude, stim_duration, equil_duration = 2, dt = 1e-5,
                               Eactive = 5, Iactive = 10, max_duration=12):
    """ Creates a graph of the state transitions elicited by the specified stimulus.
    First, finds all states, then performs one simulation per state to find all edges.
    
    Parameters
    ----------
    Wji : np.ndarray
        The weight matrix of the network.
    pset : dict
        The parameter set for the simulation.
    stim_amplitude : float
        The amplitude of the stimulus.
    stim_duration : float
        The duration of the stimulus.
    equil_duration : float, optional
        The duration of the equilibration period (default is 2 seconds).
    dt : float, optional
        The time step for the simulation (default is 1e-5 seconds).
    Eactive : float, optional
        The initial firing rate of the excitatory population (default is 5 Hz).
    Iactive : float, optional
        The initial firing rate of the inhibitory population (default is 10 Hz).
    max_duration : float, optional
        The duration spent allowing the network to reach a steady state during the initial state search.

    Returns
    -------
    nx.DiGraph
        A directed graph representing the state transitions of the network.
    """
    # get all states
    numPairs = Wji.shape[1]
    states = get_all_states(Wji, pset, numPairs, Eactive, Iactive, duration = max_duration, dt=dt)
    states = np.unique(states, axis=0)  # remove duplicates
    states = [np.array(state) for state in states]  # convert to list

    # edge simulation setup
    duration = 2*equil_duration + stim_duration
    IappE = np.zeros((numPairs, int(duration/dt)))
    IappE[::, int(equil_duration/dt):int((equil_duration + stim_duration)/dt)] = stim_amplitude  # apply stimulus to all pairs
    IappI = np.copy(IappE)  # same for inhibitory units

    # construct graph
    G = nx.DiGraph()
    unstable_states = False
    for i, state in enumerate(states):
        G.add_node(i+1) # won't add repeated states

        rates = simulateISN(Wji, numPairs, state, pset,
                            IappE, IappI,
                            dt=dt, duration=duration)
        final_state = rates[:, :, -1]
        
        # no self loops (instead indicated by out-degree zero)
        if np.allclose(final_state, state, rtol=0, atol=0.5):
            continue
        
        # identify final_state
        for match_idx, match_state in enumerate(states):
            found = np.allclose(final_state, match_state, rtol=0, atol=0.5)
            if found:
                G.add_edge(i+1, match_idx+1) # adds nodes if necessary
                break
        if not found:
            # check stability
            stable = np.allclose(rates[:, :, int((duration-1)/dt):-1], rates[:, :, -1][..., np.newaxis], rtol=0, atol=0.1) and np.all(final_state < 100)
            if stable:
                G.add_edge(i+1, len(states) + 1)
                states.append(final_state)  # add new state if not found in the existing states
            else:
                unstable_states = True
                G.add_edge(i+1, -1) # always use -1 for unstable states (will always end up as a sink)
                # don't add to the states list -- don't want to simulate unstable states

        return G, unstable_states
    
def longest_path(G):
    """Find length of the longest path in G.
    Only works for graphs where every node has out-degree of 1 (as in the state transition graph).
    """
    involved_nodes = get_longest_path(G)
    G2 = G.subgraph(involved_nodes)
    try:
        cycle = nx.find_cycle(G2, orientation='original')
    except nx.exception.NetworkXNoCycle:
        return nx.dag_longest_path_length(G2)
    
    lengths = np.zeros(G2.number_of_nodes())
    for i, node in enumerate(G2.nodes):
        lengths[i] = len(nx.dfs_successors(G2, source = node, depth_limit=G2.number_of_nodes())) + 1
    return np.max(lengths)

def get_longest_path(G):
    """ Return all nodes involved in the longest path in G.
    Only works for graphs where every node has out-degree of 1 (as in the state transition graph).
    """
    max_length = 0
    involved_nodes = []
    for node in G.nodes:
        successors = nx.dfs_successors(G, source = node, depth_limit=G.number_of_nodes())
        length = len(successors) + 1
        if length > max_length:
            max_length = length
            involved_nodes = set(successors.keys()).union({val for row in successors.values() for val in row})
        if length == max_length:
            involved_nodes = involved_nodes.union(set(successors.keys()).union({val for row in successors.values() for val in row}))
    
    return involved_nodes