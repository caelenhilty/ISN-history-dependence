import numpy as np
import multiprocessing as mp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import util, network_model

# search parameters ---------------------------------------------------------------
rE_target = 5
rI_target = 10
dt = 1e-5

# utility --------------------------------------------------------------------------

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

# construct random pset
def generate_valid_pset(i):
    rng = np.random.default_rng(i)
    
    valid = 0 # bit tracks if pset satisfies constraints
    while not valid:
        # select firing thresholds
        thetaE_range = np.logspace(0, 2, 100).round(2)
        thetaE = rng.choice(thetaE_range)
        thetaI_range = np.logspace(np.log(thetaE), 3, 100).round(2)
        thetaI = rng.choice(thetaI_range)
        
        # find weights
        tr_range = np.logspace(0.5, 3, 100).round(2) * -1
        tr_target = rng.choice(tr_range)
        det_range = np.logspace(np.log((tr_target**2 / 4)*1.1), 6, 100).round(2)
        det_target = rng.choice(det_range)
        target = make_target(rE_target, rI_target, tr_target, det_target, thetaE, thetaI)
        x, valid = get_solution(target, method='hybr')
    WEE, WEI, WIE, WII = x
    
    return network_model.pack_parameters(10e-3, 10e-3, 0, 0, 0, 0, 0, 0, WEE, WEI, WIE, WII, thetaE, thetaI)

def bistable_no_depression(pset, stimulus_timing, stimulus_duration, stimulus_amplitude, duration, dt, initial_conditions=[0, 0]):
    rmax = 100
    tauE, tauI, _, _, _, _, _, _, WEE, WEI, WIE, WII, thetaE, thetaI = network_model.unpack_parameters(pset)

    IappI = np.zeros(int(duration/dt))
    IappE = np.zeros(int(duration/dt))
    IappE[int(stimulus_timing/dt):int((stimulus_timing+stimulus_duration)/dt)] = stimulus_amplitude
    IappI[int(stimulus_timing/dt):int((stimulus_timing+stimulus_duration)/dt)] = stimulus_amplitude

    rE, rI = network_model.simulateISP(dt, duration, rmax, tauE, tauI, 
                            WEE, WEI, WIE, WII, thetaE, thetaI,
                            IappI, IappE, rE0=initial_conditions[0], rI0=initial_conditions[1])   
    
    return rE, rI

def run_trial(i, trial_duration=6, timing=0.5):
    pset = generate_valid_pset(i)
    stim_dur_range = np.logspace(-3, -1, n_dur).round(5)
    stim_amp_range = np.logspace(np.log(pset[0] + 1), 2, n_amp).round(2)
    amp_dur_pairs = []
    
    for amp in stim_amp_range:
        for dur in stim_dur_range:
            off_to_on = bistable_no_depression(pset, timing, dur, amp, trial_duration, 1e-5)
            on_to_off = bistable_no_depression(pset, timing, dur, amp, trial_duration, 1e-5, initial_conditions=[rE_target, rI_target])
    
            # check stability of all nodes
            stably_on = np.all(off_to_on[0][int(trial_duration-1):-1] - rE_target < 0.1)
            stably_off = np.all(on_to_off[0][int(trial_duration-1):-1] < 0.1)
            
            if stably_on and stably_off:
                amp_dur_pairs.append((amp, dur))

    return pset, amp_dur_pairs 


# run search ----------------------------------------------------------------
if __name__ == "__main__":
    from tqdm import tqdm    

    n_psets = 1000
    n_amp = 20
    n_dur = 20
    results = []
    for i in tqdm(range(n_psets)):
        results.append(run_trial(i))
        
    # unpack results
    psets = []
    amp_durs = []
            
    # set up csv
    import csv
    fname = f'data.csv'
    header = ['pset', 'amp-dur']
    with open(fname, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for (pset, pairs) in results:
            if pairs:
                writer.writerow({'pset':pset, 'amp-dur':pairs})