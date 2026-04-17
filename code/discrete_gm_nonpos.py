#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 19:33:07 2024

@author: eric
"""

import numpy as np
#from sklearn.metrics import roc_auc_score as AUC
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def int2bin(x, bits):
    x = int(x)
    return np.array([int(i) for i in bin(x)[2:].zfill(bits)])

class discrete_graphical_model:
    def __init__(self,c=np.geomspace(1e-9,1e1,10000),ncores=1):
        self.c = np.sort(c.reshape(-1))[::-1].reshape(-1, 1)# column in decreasing direction
        self.ncores = ncores
    def _lpl_bic(self,X,indx_v,indx_w,ne_size):
        Xnz = np.zeros_like(X)
        Xnz[:,indx_v+indx_w]=X[:,indx_v+indx_w]
        n,p = X.shape
        # find all configurations presented in the data (applied to subvector)        
        Xint = Xnz.dot(np.power(2,np.arange(p-1,0-1,-1)))
        keys = np.unique(Xint)
        N_av_aw = dict()
        for key in keys:
            N_av_aw[key] = sum(Xint==key)
        N_aw = dict()
        logP_av_given_aw = dict()
        for key1 in keys:
            if len(indx_w)>0:
                N_aw[key1]=N_av_aw[key1]
                for key2 in keys:
                   if (key1!=key2) and all(int2bin(key1, p)[indx_w]==int2bin(key2, p)[indx_w]):
                       N_aw[key1]+=N_av_aw[key2]
            else:
                N_aw[key1]=n
            logP_av_given_aw[key1]=np.log(N_av_aw[key1])-np.log(N_aw[key1])
        
        lpl = 0
        for key in keys:
            lpl += logP_av_given_aw[key]*N_av_aw[key]
        
        #bic = lpl - self.c*pow(len(np.unique(X)),len(indx_w))*np.log(n)
        bic = lpl - self.c*pow(len(np.unique(X)),ne_size)*np.log(n)
        return((lpl,bic))
    def compute_ne_i(self,i,X,Y):
        YX = np.hstack((Y,X))
        q = Y.shape[1]# covariates
        p = X.shape[1]
        
        indx_v = [i]
        indx_all = set(range(p)) 
        indx_no_v =indx_all.difference(indx_v)
        
        bic_v = []
        ne_v  = []
        for ne in range(len(indx_all)):
            for indx_w in combinations(indx_no_v, ne):
                ne_v.append(indx_w)
                bic_v.append(self._lpl_bic(YX,[q+i for i in indx_v],list(range(q))+[q+j for j in indx_w],len(indx_w))[1])
        #ne_v_optim = ne_v[max(enumerate(bic_v), key=lambda x: x[1])[0]]
        #NE[i,ne_v_optim]=True
        ne_v_optim_indx=np.argmax(np.hstack(bic_v),axis=1,keepdims=True)
        ne_v_optim     = np.zeros((len(self.c),p),dtype=bool)
        for ic in range(len(self.c)):
            ne_v_optim[ic,ne_v[int(ne_v_optim_indx[ic])]]=True 
        #NElst.append(ne_v_optim)
        return(ne_v_optim)
    def estimate_CI(self,X,Y=None):
        # estimate the neighbourhood of every index
        # X predictors
        # Y covariates
        # c positive constant (regularization)
        if Y is None:
            Y = np.zeros((X.shape[0],0))
        
        #NElst = list()
        if (self.ncores>1):
            with ProcessPoolExecutor(max_workers=self.ncores) as executor:
                NElst=list(executor.map(partial(self.compute_ne_i,X=X,Y=Y), range(X.shape[1])))
        else:
           NElst = [self.compute_ne_i(i=i, X=X, Y=Y) for i in range(X.shape[1])]
        
        NE=np.stack(NElst,2)# |c|xpxp <=> c,neighbor, variable
        # simetrization
        NE_conserv  = NE & np.moveaxis(NE,-1,-2)
        NE_nconserv = NE | np.moveaxis(NE,-1,-2)
        return({'conserv' : np.split(NE_conserv, 1, 0)[0], 'nconserv' : np.split(NE_nconserv, 1, 0)[0]})
        # if self.conservative:
        #     NE = NE & np.transpose(NE)
        # else:
        #     NE = NE | np.transpose(NE)
        # return(NE)
    def _estimate_CI_subsample_i(self,i,X,Y,index_list):
        signle_core_self = discrete_graphical_model(self.c,1)
        cihat = signle_core_self.estimate_CI(X=X[index_list[i],:],Y=Y[index_list[i],:])                
        cihat_combined = np.stack((cihat['conserv'], cihat['nconserv']), axis=1)
        return cihat_combined
    def _evaluate_c_i(self, i, Eqhat, NE, p, PFER, q_min, q_max):
        #lambda_index = np.argmin(np.abs(np.cumsum(accepted_q,axis =0)/np.sum(accepted_q,axis=0)-.5) , axis=0)
        q_c  = Eqhat[i,0]
        q_nc = Eqhat[i,1]   
        
        if (q_c>q_min) & (q_c<q_max):
            CI_c  = np.mean(np.cumsum(NE,axis=1)>0,axis=0)[i,0,:,:]>(1+q_c**2/p/PFER)/2
            num_elements_c = np.sum(CI_c)
        else:
            num_elements_c = np.nan
        if (q_nc>q_min) & (q_nc<q_max):
            CI_nc = np.mean(np.cumsum(NE,axis=1)>0,axis=0)[i,1,:,:]>(1+q_nc**2/p/PFER)/2
            num_elements_nc = np.sum(CI_nc)
        else:
            num_elements_nc = np.nan
            
        return i, num_elements_c, num_elements_nc
    
    def estimate_stable_CI(self,X,Y=None,PFER=.1, npartitions=100, pi_min =.5, pi_max = .7, seed = None):
        if Y is None:
            Y = np.zeros((X.shape[0],0))
        
        # data partition
        rkf = RepeatedKFold(n_splits=2, n_repeats=int(npartitions/2), random_state=seed)
        index_list = list([train_index for (train_index, test_index) in rkf.split(X, Y)])
        
        fun_i = partial(self._estimate_CI_subsample_i, X=X, Y=Y, index_list=index_list)
        with ProcessPoolExecutor(max_workers=self.ncores) as executor:
            NElst=list(executor.map(fun_i, range(len(index_list))))
        
        NE = np.stack(NElst,axis = 0)
        qhat = np.sum(np.sum(np.cumsum(NE,axis=1)>0,axis=-1),axis=-1)/2 
        
        Eqhat = np.mean(qhat,axis=0)
        
        p = X.shape[1] * (X.shape[1] - 1)/2 # num of arrows
        # lambdamin st q^2<pv
        q_max = np.sqrt(p*PFER*(2*pi_max-1))
        q_min = np.sqrt(p*PFER*(2*pi_min-1)) 
        
        assert q_min >= 0, f"Invalid range: q_min = {q_min} < 0. Increase PFER, or pi_min"
        assert q_max < p, f"Invalid range: q_max = {q_max} > {p}. Decrease PFER, or pi_max"
        
        accepted_q = (Eqhat>q_min) & (Eqhat<q_max)
        assert np.all(np.any(accepted_q,axis=0)), f"Not encounter any c in self.c such that the expected number of discovery {q_min} < q < {q_max}. q in {Eqhat}"
        
        # lambda_index = np.argmin(np.abs(np.cumsum(accepted_q,axis =0)/np.sum(accepted_q,axis=0)-.5) , axis=0)
        # q_c  = Eqhat[lambda_index[0],0]
        # q_nc = Eqhat[lambda_index[1],1]   
        
        # assert (q_c>q_min) & (q_c<q_max), f"conserv did not find a c in self.c such that q_min={q_min} < q={q_c} < q_max={q_max}"
        # assert (q_nc>q_min) & (q_nc<q_max), f"nconserv did not find a c in self.c such that q_min={q_min} < q={q_nc} < q_max={q_max}"
        
        # CI_c  = np.mean(np.cumsum(NE,axis=1)>0,axis=0)[lambda_index[0],0,:,:]>(1+q_c**2/p/PFER)/2
        # CI_nc = np.mean(np.cumsum(NE,axis=1)>0,axis=0)[lambda_index[1],1,:,:]>(1+q_nc**2/p/PFER)/2
        
        fun_i = partial(self._evaluate_c_i, Eqhat=Eqhat, NE=NE, p=p, PFER=PFER, q_min=q_min, q_max=q_max)
        with ProcessPoolExecutor(max_workers=self.ncores) as executor:
            discoveriesLst=list(executor.map(fun_i,list(np.where(np.any(accepted_q,axis=1))[0])))
        discoveries = np.stack(discoveriesLst)
        index_max_discoveries_c = discoveries[np.where(discoveries[:,1] == np.nanmax(discoveries[:,1]))[0],0]
        index_max_discoveries_nc = discoveries[np.where(discoveries[:,2] == np.nanmax(discoveries[:,2]))[0],0]
        
        index_selected_c  = int(np.median(index_max_discoveries_c))
        index_selected_nc = int(np.median(index_max_discoveries_nc))
        
        q_c  = Eqhat[index_selected_c,0]
        q_nc = Eqhat[index_selected_nc,1]   
        CI_c  = np.mean(np.cumsum(NE,axis=1)>0,axis=0)[index_selected_c,0,:,:]>(1+q_c**2/p/PFER)/2
        CI_nc = np.mean(np.cumsum(NE,axis=1)>0,axis=0)[index_selected_nc,1,:,:]>(1+q_nc**2/p/PFER)/2
        
        return  ({'conserv' : CI_c, 'nconserv' : CI_nc})
    def estimate_stable_CI_multiple_datasets(self,X_Y_list, ncores_outer= 1, PFER=.1, npartitions=100, pi_min =.5, pi_max = .7, seed = None):
        # assumes X_Y_list = [(X1, Y1), (X2, Y2), ... ]
        func = partial(self.estimate_stable_CI, PFER=PFER, npartitions=npartitions, pi_min=pi_min, pi_max=pi_max, seed=seed)
        with ThreadPoolExecutor(max_workers=self.ncores) as executor:
            return list(executor.map(lambda data: func(*data), X_Y_list))

    def compute_interaction_logOR(self, X, Y, ne, smoothing=1.0, symmetrize=True):
        """
        Computes the expected empirical log-Odds Ratios (logOR) non-parametrically.

        Parameters:
        -----------
        X : np.array of shape (n, p), binary predictors
        Y : np.array of shape (n, q), binary covariates/outcomes (or None for marginal)
        ne : np.array of shape (p, p), boolean adjacency matrix (e.g., from estimate_stable_CI)
        smoothing : float, Laplace smoothing parameter to avoid log(0).
        symmetrize: bool, whether to average the directed logORs symmetrically.

        Returns:
        --------
        logOR_matrix : np.array of shape (p, p)
                       logOR_matrix[i, j] is the expected logOR of X_i and X_j given X_W and Y.
                       Non-edges are set to np.nan.
        logOR_Y :      np.array of shape (p,)
                       logOR_Y[i] is the expected logOR of X_i and Y given X_W.
                       (Assumes Y is univariate; evaluates the first column of Y).
        """
        if Y is None:
            Y = np.zeros((X.shape[0], 0))

        n, p = X.shape
        q = Y.shape[1]
        YX = np.hstack((Y, X))

        # 1. Initialize Outputs
        logOR_matrix = np.full((p, p), np.nan)
        logOR_Y = np.full(p, np.nan)

        # ==========================================================
        # 2. Compute interaction of X_i with X_j (given Y and X_W\j)
        # ==========================================================
        for i in range(p):
            for j in range(p):
                if i == j or not ne[i, j]:
                    continue

                # W is the neighborhood of i, excluding j
                W_indices = np.where(ne[i, :])[0]
                W_indices = [w for w in W_indices if w != j]

                Z_indices = list(range(q)) + [q + w for w in W_indices]
                V1 = X[:, i]
                V2 = X[:, j]

                if len(Z_indices) == 0:
                    strata = np.zeros(n, dtype=int)
                    num_strata = 1
                else:
                    Z = YX[:, Z_indices]
                    _, strata = np.unique(Z, axis=0, return_inverse=True)
                    num_strata = strata.max() + 1

                state = V1 * 2 + V2
                combined_index = strata * 4 + state

                counts = np.bincount(combined_index, minlength=num_strata * 4).reshape(num_strata, 4)
                stratum_weights = np.sum(counts, axis=1)

                counts = counts.astype(float) + smoothing
                d_k = counts[:, 0]  # X_i=0, X_j=0
                c_k = counts[:, 1]  # X_i=0, X_j=1
                b_k = counts[:, 2]  # X_i=1, X_j=0
                a_k = counts[:, 3]  # X_i=1, X_j=1

                log_or_k = np.log(a_k) + np.log(d_k) - np.log(b_k) - np.log(c_k)
                valid = stratum_weights > 0

                if np.sum(stratum_weights[valid]) > 0:
                    weighted_log_or = np.sum(log_or_k[valid] * stratum_weights[valid]) / np.sum(stratum_weights[valid])
                    logOR_matrix[i, j] = weighted_log_or

        # Symmetrize logOR_matrix (Arithmetic Mean)
        if symmetrize:
            logOR_matrix_sym = np.full((p, p), np.nan)
            for i in range(p):
                for j in range(i, p):
                    if ne[i, j] or ne[j, i]:
                        l_ij = logOR_matrix[i, j]
                        l_ji = logOR_matrix[j, i]

                        if not np.isnan(l_ij) and not np.isnan(l_ji):
                            sym_val = (l_ij + l_ji) / 2.0
                            logOR_matrix_sym[i, j] = sym_val
                            logOR_matrix_sym[j, i] = sym_val
                        elif not np.isnan(l_ij):
                            logOR_matrix_sym[i, j] = l_ij
                            logOR_matrix_sym[j, i] = l_ij
                        elif not np.isnan(l_ji):
                            logOR_matrix_sym[i, j] = l_ji
                            logOR_matrix_sym[j, i] = l_ji
            logOR_matrix = logOR_matrix_sym

        # ==========================================================
        # 3. Compute interaction of X_i with Y (given X_W)
        # ==========================================================
        if q > 0:
            Y_col = Y[:, 0]  # Assume Y is univariate target
            for i in range(p):
                # W is the exact neighborhood of i
                W_indices = np.where(ne[i, :])[0]
                W_indices = [w for w in W_indices if w != i]

                if len(W_indices) == 0:
                    strata = np.zeros(n, dtype=int)
                    num_strata = 1
                else:
                    Z = X[:, W_indices]
                    _, strata = np.unique(Z, axis=0, return_inverse=True)
                    num_strata = strata.max() + 1

                V1 = X[:, i]
                V2 = Y_col

                state = V1 * 2 + V2
                combined_index = strata * 4 + state

                counts = np.bincount(combined_index, minlength=num_strata * 4).reshape(num_strata, 4)
                stratum_weights = np.sum(counts, axis=1)

                counts = counts.astype(float) + smoothing
                d_k = counts[:, 0]  # X_i=0, Y=0
                c_k = counts[:, 1]  # X_i=0, Y=1
                b_k = counts[:, 2]  # X_i=1, Y=0
                a_k = counts[:, 3]  # X_i=1, Y=1

                log_or_k = np.log(a_k) + np.log(d_k) - np.log(b_k) - np.log(c_k)
                valid = stratum_weights > 0

                if np.sum(stratum_weights[valid]) > 0:
                    weighted_log_or = np.sum(log_or_k[valid] * stratum_weights[valid]) / np.sum(stratum_weights[valid])
                    logOR_Y[i] = weighted_log_or

        return logOR_matrix, logOR_Y

# class cross_validated_discrete_graphical_model:
#     def __init__(self,c=np.linspace(.1,1,10),ncores=None):
#         self.c = c.reshape(-1,1)# column
#         self.ncores = ncores
#     def cross_validation(self,X,Y,kfolds=10):
#         YX  = np.hstack((Y,X))
#         n,p = X.shape
#         q   = Y.shape[1]# covariates
        
#         dgm = discrete_graphical_model(self.c,self.ncores)
        
#         kf = KFold(n_splits=kfolds)
#         kf.get_n_splits(X)
            
#         ll = np.zeros((kfolds,len(self.c),2))
#         for k, (train_index, test_index) in enumerate(kf.split(X)):
#             # estimate ne(v)
#             nehat = dgm.estimate_CI(X[train_index,:], Y[train_index,:])# conserv or nconserv and is a list given [ic] of length |c|
            
#             # for each c, conserv compute the conditional-likelihood in test given ne(v)
#             for iconserv,conserv in enumerate(('conserv' , 'nconserv')):
#                 for ic,c in enumerate(self.c):
#                     myne = nehat[conserv][ic]# neighbourhood matrix 
                    
#                     # compute conditional likelihood of x_v given Y,X_w
#                     for i in range(p):
#                         indx_v = [i]
#                         indx_w = list(np.where(myne[i,:])[0]) #np.delete(myne[i,:],i)# ver esto en el codigo de antes porque no tiuene qu ser binario
                        
#                         indx_yx_v = [q+i for i in indx_v]
#                         indx_yx_w = list(range(q))+[q+j for j in indx_w]
                        
#                         # only evaluate keys in test
#                         YXnz_test = np.zeros_like(YX[test_index,])
#                         YXnz_test[:,indx_yx_v+indx_yx_w]=YX[test_index,:][:,indx_yx_v+indx_yx_w]
#                         YXint_test = YXnz_test.dot(np.power(2,np.arange(YX.shape[1]-1,0-1,-1)))
#                         keys_test = np.unique(YXint_test)
                        
#                         # find int representation of train data
#                         YXnz_train = np.zeros_like(YX[train_index,])
#                         YXnz_train[:,indx_yx_v+indx_yx_w]=YX[train_index,:][:,indx_yx_v+indx_yx_w]
#                         YXint_train = YXnz_train.dot(np.power(2,np.arange(YX.shape[1]-1,0-1,-1)))
                        
                        
#                         # computed in train data
#                         ntrain = len(train_index)
#                         N_av_aw = dict()# in train
#                         for key in keys_test:
#                             N_av_aw[key] = sum(YXint_train==key)
#                         N_aw = dict()# in train
#                         logP_av_given_aw = dict()
#                         for key1 in keys_test:
#                             if len(indx_w)>0:
#                                 N_aw[key1]=N_av_aw[key1]
#                                 for key2 in keys_test:
#                                    if (key1!=key2) and all(int2bin(key1, p+q)[indx_yx_w]==int2bin(key2, p+q)[indx_yx_w]):
#                                        N_aw[key1]+=N_av_aw[key2]
#                             else:
#                                 N_aw[key1]=ntrain
#                             logP_av_given_aw[key1]=np.log(N_av_aw[key1])-np.log(N_aw[key1])
                        
#                         lpl = 0
#                         for key in keys_test:
#                             # computed in test data
#                             N_av_aw_test = sum(YXint_test==key)
#                             # combine and compute likelihood 
#                             lpl += logP_av_given_aw[key]*N_av_aw_test
#                         # save result
#                         ll[k,ic,iconserv] = lpl
                        
#         # mean across k-folds
#         llmean = np.mean(ll,0) # |c| x 2
#         # select the largest
#         ll_best_c = np.argmax(llmean,0) # 2 
#         ll_best_conserv = np.argmax(ll_best_c) # 1
#         ll_best_c_conserv = ll_best_c[ll_best_conserv]
        
#         ll_best_c_conserv_value = self.c[ll_best_c_conserv] 
#         ll_best_conserv_str = ['conserv' , 'nconserv'][ll_best_conserv]
        
#         return (ll_best_c_conserv,ll_best_c_conserv_value,ll_best_conserv_str)
        
                        

class sdr_discrete_graphical_model:
    def __init__(self, X, Y, ne):
        # X: (n, p), Y: (n, 1), ne: (p, p) boolean array (adjacency/neighborhood matrix)
        assert Y.ndim == 2 and Y.shape[1] == 1, "Y must be univariate and 2D"
        assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"
        self.X = X.astype(int)
        self.Y = Y.astype(int).reshape(-1, 1)
        self.ne = ne
        self.p = X.shape[1]
    def learn(self, X, Y):
        # Deprecated: learning is now done at initialization
        raise NotImplementedError("Use __init__(X, Y, ne) to initialize the model with neighborhood structure.")
    def predict(self, X):
        """
        Returns (Ri, R): 
        - Ri[s, i] = log-ratio for variable i at sample s
        - R[s] = sum_i Ri[s, i]
        Uses the encoding-based approach of predict_difference to compute per-variable log-ratios.
        Applies Laplace smoothing to conditional probability estimates.
        """
        n_samples, p = X.shape
        Ri = np.zeros((n_samples, p))
        X_train = self.X
        Y_train = self.Y[:, 0].astype(int)
        index_y1 = (Y_train == 1)
        index_y0 = (Y_train == 0)

        for i in range(p):
            ne_i = self.ne[i] > 0
            ne_i_with_self = ne_i.copy()
            ne_i_with_self[i] = True

            # Neighbor encodings for train and test
            if np.any(ne_i):
                Xwint_train = X_train[:, ne_i].dot(np.power(2, np.arange(np.sum(ne_i)-1, -1, -1)))
                Xwint_test = X[:, ne_i].dot(np.power(2, np.arange(np.sum(ne_i)-1, -1, -1)))
            else:
                Xwint_train = np.zeros(X_train.shape[0], dtype=int)
                Xwint_test = np.zeros(n_samples, dtype=int)

            # Neighbor+variable encodings for train and test
            Xvwint_train = X_train[:, ne_i_with_self].dot(np.power(2, np.arange(np.sum(ne_i_with_self)-1, -1, -1)))
            Xvwint_test = X[:, ne_i_with_self].dot(np.power(2, np.arange(np.sum(ne_i_with_self)-1, -1, -1)))

            for s in range(n_samples):
                # Find training samples where neighbors match and Y=1 or Y=0
                idx_match_w_y1 = (Xwint_train == Xwint_test[s]) & index_y1
                idx_match_w_y0 = (Xwint_train == Xwint_test[s]) & index_y0
                idx_match_vw_y1 = (Xvwint_train == Xvwint_test[s]) & index_y1
                idx_match_vw_y0 = (Xvwint_train == Xvwint_test[s]) & index_y0

                denom_y1 = np.sum(idx_match_w_y1)
                numer_y1 = np.sum(idx_match_vw_y1)
                denom_y0 = np.sum(idx_match_w_y0)
                numer_y0 = np.sum(idx_match_vw_y0)

                # Laplace smoothing: add 1 to numerator, 2 to denominator
                prob_y1 = (numer_y1 + 1) / (denom_y1 + 2)
                prob_y0 = (numer_y0 + 1) / (denom_y0 + 2)

                Ri[s, i] = np.log(prob_y1 / prob_y0)

        R = Ri.sum(axis=1)
        return (Ri, R)
    def predict_difference(self, X):
        """
        Compute SDR using pseudolikelihood difference:
        R(X) = P(X|Y=1) - P(X|Y=0).
        Returns: R_diff of shape (n_samples,)
        """
        n_samples, p = X.shape
        X_train = self.X
        Y_train = self.Y[:, 0]
        index_y1 = (Y_train != 0)

        pl_y1 = np.ones(n_samples)
        pl_y0 = np.ones(n_samples)

        for i in range(p):
            ne_i = self.ne[i] > 0
            ne_i_with_self = ne_i.copy()
            ne_i_with_self[i] = True

            # Encodings of neighbors and neighbors+variable
            Xwint_test = (X[:, ne_i]).dot(np.power(2, np.arange(np.sum(ne_i)-1, -1, -1))) if np.any(ne_i) else np.zeros(n_samples)
            Xwint_train = (X_train[:, ne_i]).dot(np.power(2, np.arange(np.sum(ne_i)-1, -1, -1))) if np.any(ne_i) else np.zeros(X_train.shape[0])

            Xvwint_test = (X[:, ne_i_with_self]).dot(np.power(2, np.arange(np.sum(ne_i_with_self)-1, -1, -1)))
            Xvwint_train = (X_train[:, ne_i_with_self]).dot(np.power(2, np.arange(np.sum(ne_i_with_self)-1, -1, -1)))

            for s in range(n_samples):
                index_match_w = (Xwint_train == Xwint_test[s])
                index_match_vw = (Xvwint_train == Xvwint_test[s])

                # For Y=1
                Nwy = np.sum(index_match_w & index_y1)
                pl_y1[s] *= 0 if Nwy == 0 else np.sum(index_match_vw & index_y1) / Nwy

                # For Y=0
                Nwny = np.sum(index_match_w & ~index_y1)
                pl_y0[s] *= 0 if Nwny == 0 else np.sum(index_match_vw & ~index_y1) / Nwny

        R_diff = pl_y1 - pl_y0
        return R_diff

    def compute_metrics(self, Ri, R, Y_true, R_diff=None):
        from sklearn.metrics import roc_curve
        # Per-variable
        importance_unsigned = np.mean(np.abs(Ri), axis=0)
        preds_var = (Ri > 0).astype(int)
        error_rate_cut0 = np.mean(preds_var != Y_true[:, None], axis=0)

        aucs = []
        error_rate_opt = []
        for j in range(Ri.shape[1]):
            try:
                aucs.append(roc_auc_score(Y_true, Ri[:, j]))
            except ValueError:  # only one class present
                aucs.append(np.nan)
            # Compute optimal error rate for this variable (threshold sweep)
            try:
                fpr, tpr, thresholds = roc_curve(Y_true, Ri[:, j])
                # error = 1 - accuracy = min{FP+FN}/N = min(fpr*neg + (1-tpr)*pos)/N
                n_pos = np.sum(Y_true == 1)
                n_neg = np.sum(Y_true == 0)
                errors = fpr * n_neg / len(Y_true) + (1 - tpr) * n_pos / len(Y_true)
                error_rate_opt.append(np.min(errors))
            except Exception:
                error_rate_opt.append(np.nan)
        aucs = np.array(aucs)
        error_rate_opt = np.array(error_rate_opt)

        # Global additive (log-ratio)
        global_importance_unsigned = np.mean(np.abs(R))
        preds_global = (R > 0).astype(int)
        global_error_rate = np.mean(preds_global != Y_true)
        try:
            global_auc = roc_auc_score(Y_true, R)
        except ValueError:
            global_auc = np.nan
        # Global optimal error rate
        try:
            fpr_g, tpr_g, thresholds_g = roc_curve(Y_true, R)
            n_pos_g = np.sum(Y_true == 1)
            n_neg_g = np.sum(Y_true == 0)
            errors_g = fpr_g * n_neg_g / len(Y_true) + (1 - tpr_g) * n_pos_g / len(Y_true)
            global_error_rate_opt = np.min(errors_g)
        except Exception:
            global_error_rate_opt = np.nan

        results = dict(
            importance_unsigned=importance_unsigned,
            error_rate_cut0=error_rate_cut0,
            error_rate_opt=error_rate_opt,
            auc=aucs,
            global_importance_unsigned=global_importance_unsigned,
            global_error_rate=global_error_rate,
            global_error_rate_opt=global_error_rate_opt,
            global_auc=global_auc
        )

        # Global difference (if provided)
        if R_diff is not None:
            preds_diff = (R_diff > 0).astype(int)
            global_diff_error_rate = np.mean(preds_diff != Y_true)
            try:
                global_diff_auc = roc_auc_score(Y_true, R_diff)
            except ValueError:
                global_diff_auc = np.nan
            results["global_diff_error_rate"] = global_diff_error_rate
            results["global_diff_auc"] = global_diff_auc

        return results
    
    def evaluate_importance(self, method="insample", kfolds=5, random_state=None):
        """
        Evaluate variable importance, signed/unsigned, error rate, and AUC.
        Also reports the same metrics for the global SDR score.
        Returns: dict with keys for per-variable and global metrics.
        """
        if method == "insample":
            Ri, R = self.predict(self.X)
            R_diff = self.predict_difference(self.X)
            return self.compute_metrics(Ri, R, self.Y[:, 0], R_diff=R_diff)

        elif method == "kfold":
            X, Y, ne = self.X, self.Y, self.ne
            #kf = KFold(n_splits=kfolds, shuffle=True, random_state=random_state)
            kf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=random_state)

            metrics_list = []
            for train_idx, test_idx in kf.split(X,Y[:,0]):
                model = sdr_discrete_graphical_model(X[train_idx], Y[train_idx], ne)
                Ri_test, R_test = model.predict(X[test_idx])
                R_diff = model.predict_difference(X[test_idx])
                metrics = self.compute_metrics(Ri_test, R_test, Y[test_idx, 0], R_diff=R_diff)
                metrics_list.append(metrics)

            # Average across folds
            agg = {}
            for key in metrics_list[0].keys():
                vals = [m[key] for m in metrics_list]
                # Use isinstance for type checking, and aggregate accordingly
                if isinstance(vals[0], np.ndarray):
                    agg[key] = np.nanmean(np.vstack(vals), axis=0)
                else:
                    agg[key] = np.nanmean(vals)
            return agg

        else:
            raise ValueError(f"Unknown method: {method}")
        
    def _connected_components_indices(self):
        """Return list of connected components (each as a 1D np.array of indices)
        for a dense, binary, symmetric adjacency self.ne.
        """
        ne = (np.asarray(self.ne) != 0)
        p = ne.shape[0]
        visited = np.zeros(p, dtype=bool)
        comps = []
        for i in range(p):
            if not visited[i]:
                stack = [i]
                visited[i] = True
                comp = []
                while stack:
                    u = stack.pop()
                    comp.append(u)
                    # neighbors of u (no self-edge assumed)
                    for v in np.flatnonzero(ne[u]):
                        if not visited[v]:
                            visited[v] = True
                            stack.append(v)
                comps.append(np.array(comp, dtype=int))
        return comps

    def _aggregate_Ri_by_components(self, Ri, components):
        """Sum per-variable Ri within each connected component."""
        if len(components) == 0:
            return np.zeros((Ri.shape[0], 0))
        # Each column j is the sum over variables in component j
        return np.stack([Ri[:, idx].sum(axis=1) for idx in components], axis=1)

    def evaluate_importance_connected_component(self, method="insample", kfolds=5, random_state=None, include_null=True):
        """
        Same metrics as `evaluate_importance`, but computed per connected component.
        We aggregate Ri over each component, then call `compute_metrics` so that
        AUC and the two error estimates are reported in the same style.

        Returns
        -------
        dict
            {"full": metrics_dict, "null": metrics_dict} if include_null,
            otherwise {"full": metrics_dict}.
            Per-component arrays (e.g., 'auc', 'error_rate_cut0', 'error_rate_opt',
            'importance_unsigned') now have length = number of connected components.
        """
        components = self._connected_components_indices()
        component_labels = np.full(self.p, -1, dtype=int)
        for cid, idx in enumerate(components):
            component_labels[idx] = cid
        if method == "insample":
            # Full model: reuse the current object
            Ri, R = self.predict(self.X)
            R_diff = self.predict_difference(self.X)
            Ri_cc = self._aggregate_Ri_by_components(Ri, components)
            full = self.compute_metrics(Ri_cc, R, self.Y[:, 0], R_diff=R_diff)

            if not include_null:
                return {
                    "full": full,
                    "components": components,               # list of arrays with node indices
                    "component_labels": component_labels,   # length p: node -> component id
                }

            # Null / independence model
            ne_null = np.zeros_like(self.ne)
            model_null = sdr_discrete_graphical_model(self.X, self.Y, ne_null)
            Ri_n, R_n = model_null.predict(self.X)
            R_diff_n = model_null.predict_difference(self.X)
            Ri_cc_n = self._aggregate_Ri_by_components(Ri_n, components)
            null = self.compute_metrics(Ri_cc_n, R_n, self.Y[:, 0], R_diff=R_diff_n)

            return {
                    "full": full,
                    "null": null,
                    "components": components,               # list of arrays with node indices
                    "component_labels": component_labels,   # length p: node -> component id
            }

        elif method == "kfold":
            X, Y, ne = self.X, self.Y, self.ne
            #kf = KFold(n_splits=kfolds, shuffle=True, random_state=random_state)
            kf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=random_state)

            full_list = []
            null_list = [] if include_null else None

            for train_idx, test_idx in kf.split(X, Y[:,0]):
                # Full model
                model = sdr_discrete_graphical_model(X[train_idx], Y[train_idx], ne)
                Ri_t, R_t = model.predict(X[test_idx])
                R_diff_t = model.predict_difference(X[test_idx])
                Ri_cc_t = self._aggregate_Ri_by_components(Ri_t, components)
                full_list.append(self.compute_metrics(Ri_cc_t, R_t, Y[test_idx, 0], R_diff=R_diff_t))

                # Null / independence model
                if include_null:
                    model_n = sdr_discrete_graphical_model(X[train_idx], Y[train_idx], np.zeros_like(ne))
                    Ri_nt, R_nt = model_n.predict(X[test_idx])
                    R_diff_nt = model_n.predict_difference(X[test_idx])
                    Ri_cc_nt = self._aggregate_Ri_by_components(Ri_nt, components)
                    null_list.append(self.compute_metrics(Ri_cc_nt, R_nt, Y[test_idx, 0], R_diff=R_diff_nt))

            # Average across folds (same style as evaluate_importance)
            def _avg(metrics_list):
                agg = {}
                for key in metrics_list[0].keys():
                    vals = [m[key] for m in metrics_list]
                    if isinstance(vals[0], np.ndarray):
                        agg[key] = np.nanmean(np.vstack(vals), axis=0)
                    else:
                        agg[key] = np.nanmean(vals)
                return agg

            full = _avg(full_list)
            if include_null:
                null = _avg(null_list)
                return {
                    "full": full,
                    "null": null,
                    "components": components,               # list of arrays with node indices
                    "component_labels": component_labels,   # length p: node -> component id
                }
            else:
                return {
                    "full": full,
                    "components": components,               # list of arrays with node indices
                    "component_labels": component_labels,   # length p: node -> component id
                }

        else:
            raise ValueError(f"Unknown method: {method}")
        
        
        
        
    # def select_c(self,kfolds=10):
    #     kf = KFold(n_splits=kfolds)
    #     kf.get_n_splits(self.X)
        
    #     ll = np.zeros(kfolds,len(self.c))
    #     for i, (train_index, test_index) in enumerate(kf.split(X)):
    #         ll[i] = self.structure_in_train_likelihood_in_test(train_index, test_index)
            
    # def structure_in_train_likelihood_in_test(self,train_index,test_index):
    #     # structure of X in train
    #     ne = self.dgm.estimate_CI(self.X[train_index,:],self.Y[train_index,:])#['conserv' if self.conservative else 'nconserv']
    #     # log pseudolikelihood of X in test
    #     pl =  np.ones((len(test_index),self.c.shape[0]))
        
    #     for ic,c in enumerate(self.c):
    #         neic=ne[ic]# neighbourhood matrix 
    #         for indx_v in range(self.p):
    #             indx_w = neic[indx_v]# row i of incidence matrix
    #             indx_vw = np.array(indx_w)
    #             indx_vw[indx_v]=True
                
    #             # project X(test) into ne_v
    #             Xwint_test  = (self.X[test_index,:]*indx_w).dot(np.power(2,np.arange(self.p-1,0-1,-1)))
    #             Xwint_train = (self.X[train_index,:]*indx_w).dot(np.power(2,np.arange(self.p-1,0-1,-1)))
                
    #             Xvwint_test  = (self.X[test_index,:]*indx_vw).dot(np.power(2,np.arange(self.p-1,0-1,-1)))
    #             Xvwint_train = (self.X[train_index,:]*indx_vw).dot(np.power(2,np.arange(self.p-1,0-1,-1)))
    #             for s in test_index:# sample
    #                 # filter training data to match neighbour values
    #                 index_match_w  = Xwint_train==Xwint_test[s]
    #                 index_match_vw = Xvwint_train==Xvwint_test[s]
                    
    #                 Nwy=sum(np.bitwise_and(index_match_w,index_y1))
    #                 pl[s,ic] *= 0 if Nwy==0 else sum(np.bitwise_and(index_match_vw,index_y1))/Nwy
    #                 #Nwny=sum(np.bitwise_and(index_match_w,~index_y1))
    #                 #pl_y1_y0[s,ic,0] *= 0 if Nwny==0 else sum(np.bitwise_and(index_match_vw,~index_y1))/Nwny
    #     return(pl)
        

class direct_ci_model:
    def __init__(self,c=np.linspace(.1,1,10)):
        self.c = c
    def learn(self, X,Y):
        # learn ne(Y)
        # Y|X = Y|R(X) <=> Y _||_ X_i if Xi not in R(X)
        assert Y.shape[1]==1,'Y must be univariate'
        assert Y.shape[0]==X.shape[0],'n'
        self.p = X.shape[1]
        self.q = Y.shape[1]
        
        self.X=X
        self.Y=Y
        
        self.dgm = discrete_graphical_model(self.c,1)
        YX = np.hstack((Y,X))
        self.ne = self.dgm.compute_ne_i(0, YX>0, np.zeros((YX.shape[0],0))>0) # c x p+1
    def predict(self,X):
        #  P(Y=1 given X=x)
        py =  np.ones((X.shape[0],self.c.shape[0]))
               
        index_y1 = self.Y[:,0]!=0
        for ic,c in enumerate(self.c):
            indx_w = self.ne[ic][1:]# the neighbours of v (dim p+1), where v =0
            
            # project X(test) into ne_v
            Xwint_test  = (X*indx_w).dot(np.power(2,np.arange(self.p-1,0-1,-1)))
            Xwint_train = (self.X*indx_w).dot(np.power(2,np.arange(self.p-1,0-1,-1)))
            for s in range(X.shape[0]):# sample
                # filter training data to match neighbour values
                index_match = Xwint_train==Xwint_test[s]
                if (sum(index_match)>0):
                    index_match_y1 = np.bitwise_and(index_match,index_y1)
                    Nw= sum(index_match)
                    py[s,ic] *= 0 if Nw==0 else sum(index_match_y1)/Nw
        return(py)
    

class cross_validation_in_prediction:
    def __init__(self,predObj,X,Y,kfolds,perfMeasure,bigger_is_better=True,ncores=None):
        self.predObj = predObj#predObj hast train and predict method and .c atribute
        self.perfMeasure = perfMeasure#arguments y_true, y_score like roc_auc_score
        self.bigger_is_better = bigger_is_better
        self.X = X
        self.Y = Y
        self.kf = KFold(n_splits=kfolds)
        self.kfsplit = list(self.kf.split(X))
        self.ncores = ncores#parallelize across kfolds
    def learn(self):
        pf = np.zeros((self.kf.get_n_splits(),len(self.predObj.c)))
        for i, (train_index, test_index) in enumerate(self.kf.split(self.X)):
            self.predObj.learn(self.X[train_index,:], self.Y[train_index,:])
            Yhat = self.predObj.predict(self.X[test_index,:])
            assert Yhat.shape[1]==len(self.predObj.c),"dimension missmatch"
            for ic in range(Yhat.shape[1]):
                pf[i,ic]=self.perfMeasure(self.Y[test_index,:],Yhat[:,ic])
        pfmean = np.mean(pf,0)# len of c
        if self.bigger_is_better:
            icstar = np.argmax(pfmean)
        else:
            icstar = np.argmin(pfmean)
        # train with full data and update predObj
        self.predObj.c = self.predObj.c[icstar,None]
        self.predObj.learn(self.X,self.Y)
    # def learn_1fold(self,ifold):
    #     train_index,test_index = self.kfsplit[ifold]
    #     self.predObj.learn(self.X[train_index,:], self.Y[train_index,:])
    #     Yhat = self.predObj.predict(self.X[test_index,:])
    #     assert Yhat.shape[1]==len(self.predObj.c),"dimension missmatch"
    #     pfi = np.zeros((1,len(self.predObj.c)))
    #     for ic in range(Yhat.shape[1]):
    #          pfi[0,ic] = self.perfMeasure(self.Y[test_index,:],Yhat[:,ic])
    #     return pfi
    # def learn(self):
    #     with multiprocessing.Pool(self.ncores) as pool:
    #         pf=pool.map(self.learn_1fold, range(self.kf.get_n_splits()))
    #     pfmean = np.mean(pf,0)# len of c
    #     if self.bigger_is_better:
    #         icstar = np.argmax(pfmean)
    #     else:
    #         icstar = np.argmin(pfmean)
    #     # train with full data and update predObj
    #     self.predObj.c = self.predObj.c[icstar,None]
    #     self.predObj.learn(self.X,self.Y)
if __name__ == "__main__": # test
    np.random.seed(111)
    # generate data
    p=6
    n=100
    beta = (np.random.rand(p,1)>.5).astype(int)
    print(beta.T)

    # Generate X and Xtest using multivariate normal, then threshold to binary
    rho = 0.3# reforcement correlation
    gamma = -0. # noisy coorrelation
    cov = 1*np.eye(p) + rho* beta @ beta.T + gamma * (np.ones((p,p))-np.eye(p))
    mean = np.zeros(p)
    X_real = np.random.multivariate_normal(mean, cov, size=n)
    Xtest_real = np.random.multivariate_normal(mean, cov, size=n)
    X = (X_real > 0).astype(int)
    Xtest = (Xtest_real > 0).astype(int)
    
    
    Y     = ((X @ beta)>0).astype(int)
    Ytest = ((Xtest @ beta)>0).astype(int)
    
    # # graphical model
    # ci=discrete_graphical_model(np.linspace(1, 10,10),10).estimate_CI(X>0, Y>0)# only binary data allowed
    
    # # direct model, predicts Y based on its neighborhood
    # ci = direct_ci_model(c=np.linspace(.1,1,3))
    # ci.learn(X>0, Y>0)
    # Yhat=ci.predict(Xtest>0)
    

    # # sdr inverse model (here the orediction is not balanced)
    # sdr=sdr_discrete_graphical_model(c=np.linspace(.1,1,3),ncores=10)
    # sdr.learn(X>0, Y>0)
    # Yhatsdr = sdr.predict(Xtest)
    # # the conditional graphical model neighborhood matrix (interactions) given Y
    # print(sdr.ne)    
    
    # # print predictions
    # print(np.hstack((Ytest,Yhat,Yhatsdr)))
    
    
    
    # # cross validated graphical model
    # #cvdgm = cross_validated_discrete_graphical_model(np.logspace(-20,-10,10),4)
    # #result = cvdgm.cross_validation(X, Y)


    # # cross validation in prediction
    # kfolds = 10
    # sdr=sdr_discrete_graphical_model(c=np.linspace(.1,1,10),ncores=None)
    # #cross_validation_in_prediction(sdr,X,Y,kfolds,AUC,bigger_is_better=True).learn_1fold(0)
    # cross_validation_in_prediction(sdr,X,Y,kfolds,AUC,bigger_is_better=True).learn()# update sdr object
    # print(sdr.c,sdr.ne,np.hstack((Ytest, sdr.predict(Xtest))))
    
    # # same for direct model
    # ci=direct_ci_model(c=np.linspace(.1,1,10))
    # cross_validation_in_prediction(ci,X,Y,kfolds,AUC,bigger_is_better=True).learn()# update ci object
    # print(ci.c,ci.ne,np.hstack((Ytest, ci.predict(Xtest))))


    # stable graph
    
    
    dgm = discrete_graphical_model(np.geomspace(1e3, 1e-9,1000),ncores=11)
    #cihat = dgm.estimate_CI(X>0, Y>0)
    CI_stable =dgm.estimate_stable_CI(X,Y=Y,PFER=1,npartitions=100,seed=1)
    print(CI_stable['conserv'])
    
    # sdr
    # --- Demonstration of sdr_discrete_graphical_model usage ---
    # Initialize SDR model with neighborhood from CI_stable
    sdr = sdr_discrete_graphical_model(X, Y, CI_stable['conserv'])

    # Call predict on test data (here, reuse Xtest)
    Ri, R = sdr.predict(Xtest)
    print("Per-sample per-variable contributions Ri (shape):", Ri.shape)
    print("Global SDR scores R (first 10):", R[:10])

    # Call evaluate_importance
    importance_results = sdr.evaluate_importance(method="kfold", kfolds=5, random_state=42)
    print("Importance / error / AUC results:")
    for k, v in importance_results.items():
        print(k, v)
   
    # connected components
    # In-sample (both full and null)
    cc_metrics = sdr.evaluate_importance_connected_component(method="insample", include_null=True)
    print(cc_metrics["full"]["auc"])   # per-component AUC
    print(cc_metrics["null"]["auc"])
    print(1-cc_metrics["full"]["error_rate_opt"])  
    print(1-cc_metrics["null"]["error_rate_opt"])
    print(cc_metrics["components"])  # sizes of each connected component
    print(cc_metrics["component_labels"])  # mapping of variable index to component ID
    # Or k-fold:
    cc_metrics_kf = sdr.evaluate_importance_connected_component(method="kfold", kfolds=5, random_state=42)
    print(cc_metrics_kf["full"]["auc"])   # per-component AUC
    print(cc_metrics_kf["null"]["auc"])
    print(1-cc_metrics_kf["full"]["error_rate_opt"])  
    print(1-cc_metrics_kf["null"]["error_rate_opt"])
    print(cc_metrics_kf["components"])  # sizes of each connected component
    print(cc_metrics_kf["component_labels"])  # mapping of variable index to component ID
    print(cc_metrics_kf)

    # logOR
    logOR_matrix, logOR_Y = dgm.compute_interaction_logOR(X, Y, ne=CI_stable['conserv'])
    logOR_matrix_NP_marginal, logOR_Y_marginal = dgm.compute_interaction_logOR(X, Y=None, ne=CI_stable['conserv'])
    print("Marginal NP:", logOR_matrix_NP_marginal, logOR_Y_marginal)
    print("Conditional NP:", logOR_matrix, logOR_Y)