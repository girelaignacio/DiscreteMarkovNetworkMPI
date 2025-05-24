import os
import numpy as np
import pandas as pd
from discrete_gm_nonpos import discrete_graphical_model
print(os.getcwd())
data = np.genfromtxt('../simulations/sim_data1.txt', delimiter=' ')

ncores = 10
ci = discrete_graphical_model(np.geomspace(.1, 100,10000),ncores= ncores).estimate_stable_CI(
                                data, PFER=1., pi_min = 0.6, pi_max=0.9, npartitions = 50)
print(ci)
