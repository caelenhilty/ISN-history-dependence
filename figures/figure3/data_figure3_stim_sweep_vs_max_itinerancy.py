import numpy as np
import multiprocessing as mp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import util, left_right_task as lrt, network_model

# core parameters
rE_target = 10
rI_target = 5
thetaE = 5.34
thetaI = 82.43
max_duration = 12
dt = 1e-5