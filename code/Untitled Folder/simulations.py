import os
import numpy as np
import pandas as pd
from discrete_gm_nonpos import discrete_graphical_model
print(os.getcwd())

ncores = 40
for n_sim in range(1,6):

    data = np.genfromtxt("./code/sim"+str(n_sim)+"_data.txt", delimiter=' ')


    print("estimating unstable...")
    ci = discrete_graphical_model(np.linspace(.1,3,2), ncores = ncores).estimate_CI(data)
    print(ci)
    conserv = ci['conserv']
    np.savetxt("./code/sim1_conserv_c0.1.txt", 
                       conserv[0,:,:], fmt="%5i")
    np.savetxt("./code/sim1_conserv_c3.txt", 
                       conserv[1,:,:], fmt="%5i")
    nconserv = ci['nconserv']
    np.savetxt("./code/sim1_nconserv_c0.1.txt", 
                       nconserv[0,:,:], fmt="%5i")
    np.savetxt("./code/sim1_nconserv_c3.txt", 
                       nconserv[1,:,:], fmt="%5i")

    print("estimating stable..."
    ci_stable = discrete_graphical_model(np.geomspace(.1, 10,10000),ncores= ncores).estimate_stable_CI(
                                data, PFER=1., pi_min = 0.6, pi_max=0.9, npartitions = 50)
    np.savetxt("./code/sim1_stable_conserv.txt", 
                       ci_stable['conserv'], fmt="%5i")
    np.savetxt("./code/sim1_stable_nconserv.txt", 
                       ci_stable['nconserv'], fmt="%5i")
