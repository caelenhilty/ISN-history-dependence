import numpy as np
import matplotlib.pyplot as plt
import csv as csv
import networkx as nx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import network_model, util

# make an example trace
# find all states
Wji, pset, _, _, _, _ = util.load_fiducial_network()
rE_target = 5
rI_target = 10
max_duration = 6
dt =1e-5

print("Finding states...")
numPairs = Wji.shape[1]
states = network_model.get_all_states(Wji, pset, numPairs, rE_target, rI_target, duration = max_duration, dt=dt)
_, unique_idxs = np.unique(np.round(states, 0), axis=0, return_index=True)  # remove duplicates
states = [np.array(state) for state in states[unique_idxs]]  # convert to list
print("Found states")

np.save('figures/figure4/data/states.npy', np.array(states), allow_pickle=True)