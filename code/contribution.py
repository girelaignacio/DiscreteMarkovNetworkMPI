import os
import pandas as pd
import numpy as np
from discrete_gm_nonpos import sdr_discrete_graphical_model
import time

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
counter = 0
total_number_of_cases = len(os.listdir("./processed_data/"))

for filename in os.listdir("./processed_data/")[48:]:
    counter += 1
    print("# Working on", filename, "\t"+str(counter)+"/"+str(total_number_of_cases))
    # read data
    df = pd.read_csv("./processed_data/" + filename, index_col=0)
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

    adj_matrix = np.loadtxt("./results_stable2/"+filename+"_mpi_poor_conserv"+".txt").astype(int)
    Y = data['mpi_poor'].reshape(-1,1).astype(int)
    X = data['X'].astype(int)

    sdr = sdr_discrete_graphical_model(X, Y, adj_matrix)
    importance_results = sdr.evaluate_importance(method="kfold", kfolds=10, random_state=42)

    np.savetxt("./contributions/"+filename+"_auc"+".txt", importance_results['auc'], fmt="%5i")