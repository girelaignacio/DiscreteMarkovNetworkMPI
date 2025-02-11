#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from discrete_gm_nonpos import discrete_graphical_model


# In[2]:


os.getcwd()


# In[3]:


dimensions_indicators = {
    "hl" : ["d_cm","d_nutr"],
    "ed" : ["d_satt","d_educ"],
    "ls" : ["d_elct","d_wtr","d_sani","d_hsg","d_ckfl","d_asst"]
}


# In[4]:


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


# In[5]:


def deprivation_score(mpi_indicators,data):
    dimensions_weights, indicators_weights = calculate_weights(mpi_indicators)
    indicators_ = list(indicators_weights.keys())
    mpi_data = data[indicators_]
    #mpi_data = mpi_data.to_numpy()
    for indicator in mpi_data.columns:
        mpi_data[indicator] *= indicators_weights[indicator]
    score = mpi_data.sum(axis=1)
    
    return score


# In[6]:


def censored_deprivation_score(deprivation_score, k):
    censored_deprivation_score = np.where(deprivation_score >= k, deprivation_score, 0)
    
    return censored_deprivation_score


# In[12]:


for filename in os.listdir("./processed_data/"):
    print("Working on", filename)
    if filename not in os.listdir("./results/"):
        
        # read data
        df = pd.read_csv(os.getcwd() + "/processed_data/"+ filename, index_col=0)
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
        # save results
        os.mkdir("./results/"+ filename)
        
        # Run models
        
        for i in ["raw","mpi_poor"]:
            Y = data[i]
            X = data['X']
            indx_nan=np.isnan(X).any(1)|np.isnan(Y).any(1)
            Xclean = X[~indx_nan,:]
            Yclean = Y[~indx_nan,:]
            ci = discrete_graphical_model(np.linspace(1, 10,10), ncores = 20).estimate_CI(Xclean>0, Yclean>0)# only binary data allowed
            print("Networks estimated ... proceed to save data")
            for ic in range(len(ci['conserv'])):
                np.savetxt("./results/"+filename+"/"+filename+"_"+i+"_"+"conservative_c"+str(ic)+".txt", ci['conserv'][ic] , fmt="%5i")
                np.savetxt("./results/"+filename+"/"+filename+"_"+i+"_"+"nconservative_c"+str(ic)+".txt", ci['nconserv'][ic] , fmt="%5i")

