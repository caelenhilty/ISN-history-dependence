import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import util, left_right_task as lrt, network_model

if __name__ == '__main__':
    Wji, pset, _, _, _, _ = util.load_fiducial_network()