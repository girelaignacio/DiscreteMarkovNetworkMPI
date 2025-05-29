import os
print(os.getcwd())

import numpy as np
import pandas as pd
from discrete_gm_nonpos import discrete_graphical_model

ncores = 40
for n_sim in [1,5]:
    print("Estimating SIMULATION NUMBER ",str(n_sim))

    data = np.genfromtxt("./code/simulations/sim"+str(n_sim)+"_data.txt", delimiter=' ')


    print("estimating unstable...")
    ci = discrete_graphical_model(np.linspace(.1,3,2), ncores = ncores).estimate_CI(data)
    conserv = ci['conserv']
    np.savetxt("./code/simulations/sim"+str(n_sim)+"_conserv_c0.1.txt", 
                       conserv[0,:,:], fmt="%5i")
    np.savetxt("./code/simulations/sim"+str(n_sim)+"_conserv_c3.txt", 
                       conserv[1,:,:], fmt="%5i")
    nconserv = ci['nconserv']
    np.savetxt("./code/simulations/sim"+str(n_sim)+"_nconserv_c0.1.txt", 
                       nconserv[0,:,:], fmt="%5i")
    np.savetxt("./code/simulations/sim"+str(n_sim)+"_nconserv_c3.txt", 
                       nconserv[1,:,:], fmt="%5i")

    print("estimating stable...")
    ci_stable = discrete_graphical_model(np.geomspace(.1, 10,10000),ncores= ncores).estimate_stable_CI(
                                data, PFER=1., pi_min = 0.5, pi_max=0.9, npartitions = 50)
    np.savetxt("./code/simulations/sim"+str(n_sim)+"_stable_conserv.txt", 
                       ci_stable['conserv'], fmt="%5i")
    np.savetxt("./code/simulations/sim"+str(n_sim)+"_stable_nconserv.txt", 
                       ci_stable['nconserv'], fmt="%5i")
