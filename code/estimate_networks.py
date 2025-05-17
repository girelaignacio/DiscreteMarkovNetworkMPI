### Import libraries ###

import os
import pandas as pd
import numpy as np
from discrete_gm_nonpos import discrete_graphical_model
import time

# Record the start time
start_time = int(time.time())

print(os.getcwd())

### Multidimensional Poverty Measurement ###


# Define dimensions and indicators in dictionary
dimensions_indicators = {
    "hl" : ["d_cm","d_nutr"],
    "ed" : ["d_satt","d_educ"],
    "ls" : ["d_elct","d_wtr","d_sani","d_hsg","d_ckfl","d_asst"]
}


def calculate_weights(mpi_indicators):
    """
    Parameters:
    mpi_indicators: dictionary
    Computation of each dimension and indicators weigths following capability approach criterion

    Returns two dictionaries: one with dimensions weigths and another with indicators weights
    """
    dim_weights  = {}
    indic_weights = {}
    for key in mpi_indicators.keys():
        weight = 1/len(mpi_indicators.keys())
        dim_weights[key] = weight
        for value in mpi_indicators[key]:
            indic_weights[value] = weight / len(mpi_indicators[key])
            
    return dim_weights, indic_weights

def deprivation_score(mpi_indicators,data):
    dimensions_weights, indicators_weights = calculate_weights(mpi_indicators)
    indicators_ = list(indicators_weights.keys())
    mpi_data = data[indicators_]
    #mpi_data = mpi_data.to_numpy()
    for indicator in mpi_data.columns:
        mpi_data[indicator] *= indicators_weights[indicator]
    score = mpi_data.sum(axis=1)
    
    return score

def censored_deprivation_score(deprivation_score, k):
    censored_deprivation_score = np.where(deprivation_score >= k, deprivation_score, 0)
    
    return censored_deprivation_score

### Start estimation ###
total_number_of_cases = len(os.listdir("./processed_data/"))
print("### Start estimating stable graphs ###")
counter = 0

for filename in os.listdir("./processed_data/")[35:50]:
    counter += 1
    print("# Working on", filename, "\t"+str(counter)+"/"+str(total_number_of_cases))
    
    estimated = False
    for s in os.listdir("./results_stable2/"):
        if filename in s:
            estimated = True
            break
    if estimated:
        print('Estimated. Continue with the next case.')
    else:
        # read data
        df = pd.read_csv("./processed_data/"+ filename, index_col=0)
        # clean data
        df = df.dropna()
        df = df.astype(int)
        # calculate censored deprivation scores
        c_k = censored_deprivation_score(deprivation_score(dimensions_indicators, df), 33/100)

        # Prepare data
        raw = np.zeros(df.shape[0]).reshape(-1,1)
        mpi_poor =  np.where(c_k > 0, 1, 0).reshape(-1,1)

        data = {'X':df.to_numpy(),
                'raw': raw,
                'mpi_poor' : mpi_poor}
    
        # Run models
        inner_counter = 0
        inner_start_time = int(time.time())
        for i in ["raw","mpi_poor"]:
            inner_counter = inner_counter + 1
            Y = data[i]
            X = data['X']
            indx_nan=np.isnan(X).any(1)|np.isnan(Y).any(1)
            Xclean = X[~indx_nan,:]
            Yclean = Y[~indx_nan,:]
            
            ncores = 10
            ci=discrete_graphical_model(np.geomspace(.1, 100,10000),ncores= ncores).estimate_stable_CI(
                                        Xclean>0, Yclean>0, PFER=1., pi_min = 0.6, pi_max=0.9,
                                        npartitions = 100)
            print("Networks estimated proceed to save data (",str(inner_counter),")")
            np.savetxt("./results_stable2/"+filename+"_"+i+"_conserv"+".txt", 
                       ci['conserv'], fmt="%5i")
            np.savetxt("./results_stable2/"+filename+"_"+i+"_nconserv"+".txt", 
                       ci['nconserv'], fmt="%5i")
        # End of country estimation
            # calculate time of excution
        inner_end_time = int(time.time())
        inner_elapsed_time = inner_end_time - inner_start_time
        h = divmod(inner_elapsed_time,3600)  # hours
        m = divmod(h[1],60)  # minutes
        s = m[1]  # seconds
        print('Code in', filename[:3] ,
              'took %d hours, %d minutes, %d seconds' % (h[0],m[0],s))
        
        
        
# Record the end time
end_time = int(time.time()) 

# Calculate the difference
elapsed_time = end_time - start_time

# Print the elapsed time
d = divmod(elapsed_time,86400)  # days
h = divmod(d[1],3600)  # hours
m = divmod(h[1],60)  # minutes
s = m[1]  # seconds

print('All the code took %d days, %d hours, %d minutes, %d seconds' % (d[0],h[0],m[0],s))