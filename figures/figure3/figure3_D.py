import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import util, left_right_task as lrt

Wji, pset, _, _, l_kernel, r_kernel = util.load_fiducial_network()

# task parameters
numPairs = 5
dt = 1e-5
seq_len = 6
sequences = lrt.make_all_sequences(seq_len, ['L', 'R'])
equil_duration = 2

# parameter sweep
amp_range = np.logspace(0, 2, 50)
dur_range = np.linspace(0, 0.1, 50)
dur_mesh, amp_mesh = np.meshgrid(dur_range, amp_range) # amp on y-axis, duration on x-axis
dur_flat, amp_flat = dur_mesh.ravel(), amp_mesh.ravel()
reliabilities = np.zeros(dur_flat.shape)

for i, (dur, amp) in tqdm(enumerate(zip(dur_flat, amp_flat)), total=len(dur_flat)):
    if dur == 0:
        reliabilities[i] = 0.5
        continue

    # define the stimulus
    stim_map = lrt.make_stim_map(numPairs, amp, dur, l_kernel, r_kernel, dt)

    # run the model
    FSM = lrt.make_FSM(numPairs, pset, Wji, stim_map, 2, dt=dt)
    reliability = lrt.FSM_reliability(sequences, FSM)
    reliabilities[i] = reliability

# count the number of reliabilities.npy in the figures/figure3 directory
if Path('figures/figure3/reliabilities.npy').exists():
    id = 1
    while Path(f'figures/figure3/reliabilities_{id}.npy').exists():
        id += 1
    np.save(f'figures/figure3/reliabilities_{id}.npy', reliabilities)
else:
    np.save('figures/figure3/reliabilities.npy', reliabilities)

# quick plot of results
fig, ax = plt.subplots(layout='constrained')
c = ax.pcolormesh(dur_mesh, amp_mesh, reliabilities.reshape(dur_mesh.shape), cmap='viridis', shading='auto')
ax.set_yscale('log')
ax.set_xlabel(r"$\tau_{dur}$")
ax.set_ylabel(r"$I_{app}$")
cbar = fig.colorbar(c, ax=ax)
cbar.set_label('Accuracy')

plt.show()