import numpy as np
import multiprocessing as mp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import util, left_right_task as lrt, network_model

# core parameters
rE_target = 5
rI_target = 10
Wji, pset, _, _, _, _ = util.load_fiducial_network()
thetaE, thetaI = pset[-2], pset[-1]
numPairs = Wji.shape[1]
max_duration = 6
dt = 1e-5

def trial(amp, dur, states):
    if dur == 0:
        return 0
    G, _ = network_model.get_state_transition_graph(Wji, pset, amp, dur, Eactive=rE_target, Iactive=rI_target, max_duration=max_duration, dt=dt, states=states)
    return network_model.longest_path(G)
