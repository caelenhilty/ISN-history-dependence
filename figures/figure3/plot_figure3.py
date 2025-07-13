import numpy as np
import matplotlib.pyplot as plt
import csv as csv
import networkx as nx
import pandas as pd
from tqdm import tqdm

import csv

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import left_right_task as lrt, network_model, util

Wji, pset, amp, dur, l_kernel, r_kernel = util.load_fiducial_network()

# task parameters
numPairs = 5
dt = 1e-5
seq_len = 6
sequences = lrt.make_all_sequences(seq_len, ['L', 'R'])
equil_duration = 2

l_stim = np.ones((numPairs)) * amp * l_kernel
l_stim = np.repeat(l_stim[:, np.newaxis], int(dur/dt), axis=1)
r_stim = np.ones((numPairs)) * amp * r_kernel
r_stim = np.repeat(r_stim[:, np.newaxis], int(dur/dt), axis=1)
stim_map = {'L': l_stim, 'R': r_stim}
FSM = lrt.make_FSM(numPairs, pset, Wji, stim_map, 2, dt=dt)

# TODO (when figure is ready)