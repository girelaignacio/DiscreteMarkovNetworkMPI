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
from sklearn.model_selection import KFold, RepeatedKFold


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
    def __init__(self,conservative=True,c=np.linspace(.1,1,10),ncores=None):
        self.c = c
        self.ncores = ncores
        self.conservative = conservative
    def learn(self, X,Y):
        assert Y.shape[1]==1,'Y must be univariate'
        assert Y.shape[0]==X.shape[0],'n'
        self.p = X.shape[1]
        self.q = Y.shape[1]
        
        self.X=X
        self.Y=Y
        
        self.dgm = discrete_graphical_model(self.c,self.ncores)
        self.ne = self.dgm.estimate_CI(X,Y)['conserv' if self.conservative else 'nconserv']
    def predict(self,X):
        # sdr of Y given X
        pl_y1_y0 =  np.ones((X.shape[0],self.c.shape[0],2))
        
        index_y1 = self.Y[:,0]!=0
        for ic,c in enumerate(self.c):
            neic=self.ne[ic]# neighbourhood matrix 
            for indx_v in range(self.p):
                indx_w = neic[indx_v]# row i of incidence matrix
                indx_vw = np.array(indx_w)
                indx_vw[indx_v]=True
                
                # project X(test) into ne_v
                Xwint_test  = (X*indx_w).dot(np.power(2,np.arange(self.p-1,0-1,-1)))
                Xwint_train = (self.X*indx_w).dot(np.power(2,np.arange(self.p-1,0-1,-1)))
                
                Xvwint_test  = (X*indx_vw).dot(np.power(2,np.arange(self.p-1,0-1,-1)))
                Xvwint_train = (self.X*indx_vw).dot(np.power(2,np.arange(self.p-1,0-1,-1)))
                for s in range(X.shape[0]):# sample
                    # filter training data to match neighbour values
                    index_match_w  = Xwint_train==Xwint_test[s]
                    index_match_vw = Xvwint_train==Xvwint_test[s]
                    
                    Nwy=sum(np.bitwise_and(index_match_w,index_y1))
                    pl_y1_y0[s,ic,1] *= 0 if Nwy==0 else sum(np.bitwise_and(index_match_vw,index_y1))/Nwy
                    Nwny=sum(np.bitwise_and(index_match_w,~index_y1))
                    pl_y1_y0[s,ic,0] *= 0 if Nwny==0 else sum(np.bitwise_and(index_match_vw,~index_y1))/Nwny
        sdr = pl_y1_y0[:,:,1] - pl_y1_y0[:,:,0]
        return(sdr)
        
        
        
        
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
    p=5
    n=1000
    beta = (np.random.rand(p,1)>.5).astype(int)
    
    X     = np.random.randint(0,2,(n,p)).astype(int)>0
    Xtest = np.random.randint(0,2,(n,p)).astype(int)>0
    
    
    Y     = ((X @ beta)>0).astype(int)>0
    Ytest = ((Xtest @ beta)>0).astype(int)>0
    
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
    
    
    # parallelize multiple trainigs:
   
