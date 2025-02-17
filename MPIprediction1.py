#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
from discrete_gm_nonpos import *


# In[ ]:


os.path.exists(os.getcwd() + "/data")


# In[ ]:


mpi2023 = pd.read_csv(os.getcwd() + "/data/MPI2023_microdata.csv")


# In[ ]:


dimensions_indicators = {
    "ed" : ["hh_d_ni_noasis","hh_d_esc_retardada","hh_d_logro_min"],
    "em" : ["hh_d_destotalmax","hh_d_subocup_max","hh_d_10a17_ocup","hh_d_no_afil","hh_d_jubi_pens"],
    "vi" : ["hh_d_materialidad","hh_d_hacinamiento","hh_d_sin_basur"],
    "sa" : ["hh_d_sin_salud","hh_d_agua_mejor","hh_d_san_mejor","hh_d_combus"]
}


# In[ ]:


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


# In[ ]:


def deprivation_score(mpi_indicators,data):
    dimensions_weights, indicators_weights = calculate_weights(mpi_indicators)
    indicators_ = list(indicators_weights.keys())
    mpi_data = data[indicators_]
    #mpi_data = mpi_data.to_numpy()
    for indicator in mpi_data.columns:
        mpi_data[indicator] *= indicators_weights[indicator]
    score = mpi_data.sum(axis=1)
    
    return score


# In[ ]:


def censored_deprivation_score(deprivation_score, k):
    censored_deprivation_score = np.where(deprivation_score >= k, deprivation_score, 0)
    
    return censored_deprivation_score


# In[ ]:


c_k = censored_deprivation_score(deprivation_score(dimensions_indicators, mpi2023), 26/100)


# In[ ]:


mpi2023['mpi_poor'] =  np.where(c_k > 0, 1, 0)


# In[ ]:


pd.crosstab(mpi2023['pobnopoi'], mpi2023['mpi_poor'])


# In[ ]:


nested_indicators = list(dimensions_indicators.values())

indicators = []
for i in nested_indicators:
    indicators.extend(i)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection


# In[ ]:


X = mpi2023[indicators]
Y = mpi2023['pobnopoi']

Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X,Y,train_size = 0.8, random_state=912)


# In[ ]:


# Discrete model
kfolds = 5
sdr = sdr_discrete_graphical_model(c=np.linspace(.1,1,10),ncores=10)
#cross_validation_in_prediction(sdr,X,Y,kfolds,AUC,bigger_is_better=True).learn_1fold(0)


# In[ ]:


ytrain = (ytrain > 0).to_numpy().reshape(-1,1).astype(int)>0
Xtrain = (Xtrain).to_numpy().astype(int)>0


# In[ ]:


cross_validation_in_prediction(sdr,Xtrain,ytrain,kfolds,AUC,bigger_is_better=True).learn()# update sdr object


# In[ ]:


import pickle

with open("./models/sdr.pkl", "wb") as f:
    pickle.dump(sdr, f)


# In[ ]:


#with open("./models/sdr.pkl", "rb") as f:
#    sdr = pickle.load(f)


# In[ ]:


print(sdr.c,sdr.ne,np.hstack((ytest, sdr.predict(Xtest))))

