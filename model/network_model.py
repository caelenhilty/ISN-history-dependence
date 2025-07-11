from numba import prange, njit, vectorize, float64
import numpy as np
from os import urandom
import matplotlib.pyplot as plt
import time

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
@njit
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
    IappI : np.array
        Applied current to the inhibitory unit
    IappE : np.array
        Applied current to the excitatory unit
        
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

def makeWji(rng: np.random.default_rng, numPairs: float, meanStrength: float, stdStrength: float, timeout=np.inf) -> np.array:
    """Create a random connection weight matrix for the network.
    
    Create a random (numPairs x numPairs) matrix for the network, with the weights being drawn from a normal distribution 
    with mean meanStrength and standard deviation stdStrength. 
    The diagonal is set to zero, and all values that don't match the sign of meanStrength are regenerated.
    Regeneration means that the truncated normal distribution doesn't match the mean and standard deviation exactly.
    
    Parameters
    ----------
    rng : np.random.default_rng
        The random number generator to use.
    numPairs : int
        The number of pairs in the network.
    meanStrength : float
        The target mean of the matrix.
    stdStrength : float
        The target standard deviation of the matrix.
    timeout : float, default 0.5
        Max time spent in normalization & sign enforcement loop before giving up
    
    """
    Wji = rng.normal(meanStrength, stdStrength, (numPairs, numPairs))
    # normalize
    Wji = normalize_Wji(Wji, meanStrength, stdStrength)
    t0 = time.time()
    if meanStrength < 0:
        # for all values that are greater than zero, regenerate those values until they are negative
        while np.any(Wji > 0) and abs(time.time() - t0) <= timeout:
            pos_vals = Wji > 0
            Wji[pos_vals] = rng.normal(meanStrength, stdStrength, pos_vals.sum())
            Wji = normalize_Wji(Wji, meanStrength, stdStrength)
        if abs(time.time() - t0) > timeout:
            pos_vals = Wji > 0
            Wji[pos_vals] = meanStrength
    elif meanStrength > 0:
        # for all values that are less than zero, regenerate those values until they are positive
        while np.any(Wji < 0) and abs(time.time() - t0) <= timeout:
            neg_vals = Wji < 0
            Wji[neg_vals] = rng.normal(meanStrength, stdStrength, neg_vals.sum())
            Wji = normalize_Wji(Wji, meanStrength, stdStrength)
        if abs(time.time() - t0) > timeout:
            neg_vals = Wji < 0
            Wji[neg_vals] = meanStrength
    # make the diagonal zero
    np.fill_diagonal(Wji, 0)
    return Wji

def makeWji_all_types(rng: np.random.default_rng, numPairs: float, meanStrengths: np.array, stdStrengths: np.array,
                      timeout=np.inf) -> np.array:
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
    
    return np.array([makeWji(rng, numPairs, strength, std, timeout=timeout) for strength, std in zip(meanStrengths, stdStrengths)])

def normalize_Wji(Wji: np.array, mean: float, std: float) -> np.array:
    """ Normalize the mean and standard devation of the connection weights to a given value.
    
    Parameters
    ----------
    Wji : np.array
        The (numPairs x numPairs) matrix of connection weights between the pairs.
    meanStrength : float
        The desired mean of the connection weights.
    
    Returns
    -------
    np.array
        The normalized connection weights.
    """
    # get current mean and std, ignoring diagonals
    current_std = np.std(Wji[np.eye(Wji.shape[0], dtype=bool) == False])
    current_mean = np.mean(Wji[np.eye(Wji.shape[0], dtype=bool) == False])
    if current_std == 0:
        Wji = (Wji - current_mean) + mean
        return Wji
    Wji = (Wji - current_mean)/current_std * std + mean
    # set diagonal to zero
    np.fill_diagonal(Wji, 0)
    return Wji

def normalize_Wji_inputs(Wji: np.array, enforce_sign=True, timeout=np.inf) -> np.array:
    """ Normalize each row of each Wji to have the same mean as the entire matrix.

    Parameters
    ----------
    Wji : np.array
        The (numPairs x numPairs) matrix of connection weights between the pairs.
    enforce_sign : bool, default True
        Whether to enforce that all values are the same sign as the mean of the matrix.
    timeout : float, default np.inf
        Max time spent in normalization & sign enforcement loop before giving up
    
    Returns
    -------
    np.array
        The normalized connection weights.
    """
    assert Wji.ndim == 2, "Wji must be a 2D array"
    mean = np.mean(Wji[np.eye(Wji.shape[0], dtype=bool) == False]) # mean of the entire matrix
    Wji = _normalize_rows(Wji, mean)
    # check that all values are the same sign as the mean
    if enforce_sign:
        t0 = time.time()
        if mean < 0:
            while np.any(Wji > 0) and abs(time.time() - t0) <= timeout:
                pos_vals = Wji > 0
                Wji[pos_vals] = -np.abs(Wji[pos_vals])
                Wji = _normalize_rows(Wji, mean)
            if abs(time.time() - t0) > timeout: # crude fix
                pos_vals = Wji > 0
                Wji[pos_vals] = mean
        elif mean > 0:
            while np.any(Wji < 0) and abs(time.time() - t0) <= timeout:
                neg_vals = Wji < 0
                Wji[neg_vals] = np.abs(Wji[neg_vals])
                Wji = _normalize_rows(Wji, mean)
            if abs(time.time() - t0) > timeout:
                neg_vals = Wji < 0
                Wji[neg_vals] = mean
            
    return Wji

def _normalize_rows(Wji, mean):
    """ a helper function for normalize_Wji_inputs """
    row_means = np.mean(Wji[np.eye(Wji.shape[0], dtype=bool) == False].reshape((Wji.shape[0], Wji.shape[1]-1)), axis=1) # mean of each row
    for i in range(Wji.shape[0]):
        # normalize each row to have the same mean as the entire matrix
        Wji[i, :] = Wji[i, :] - row_means[i] + mean
    np.fill_diagonal(Wji, 0)
    return Wji

@njit(cache=True)
def simulateISN(Wji: np.array, numPairs: int, r0: np.array, pset: np.array, 
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
        rates[:,0,t] = rates[:,0,t-1] + (rEk1 + 2*rEk2 + 2*rEk3 + rEk4)/6. + sigma_noise[:,0,t-1] * np.random.normal(0, 1, size=numPairs) * np.sqrt(dt)
        rates[:,1,t] = rates[:,1,t-1] + (rIk1 + 2*rIk2 + 2*rIk3 + rIk4)/6. + sigma_noise[:,1,t-1] * np.random.normal(0, 1, size=numPairs) * np.sqrt(dt)
        
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