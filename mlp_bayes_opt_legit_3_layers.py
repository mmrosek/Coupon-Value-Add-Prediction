#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:52:17 2017

@author: miller
"""
from sklearn.cross_validation import cross_val_score
from sklearn.neural_network import MLPClassifier
from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import datetime
import sys
import time

def train_3_layer_mlp(features_std, y, num_iter, init_points, max_layer_1, max_layer_2, max_layer_3):
    
    '''Nesting mlp_cv function because it trains on features_std but can't take features_std as input /
       directly in order to be compatible with bayesian optimization function. 
       
       Function runs bayesian optimization to find optimal
       parameters then trains and returns an mlp using those parameters.'''
       
    early_stopping_bool = True
    
    def mlp_cv(max_iter, num_nodes_1, num_nodes_2, num_nodes_3):
    
        # Determining architecture
        num_nodes_1 = int(num_nodes_1)    
        num_nodes_2 = int(num_nodes_2)     
        num_nodes_3 = int(num_nodes_3) 
        num_nodes_list = [num_nodes_1,num_nodes_2, num_nodes_3]
        
        layer_sizes = []
        for i in range(3):
            layer_sizes.append(num_nodes_list[i])
            
        hidden_layer_array = np.array(layer_sizes)
        
        print('')
        print('Hidden layer architecture: ' + str(hidden_layer_array))
        print('')
                            
        val = cross_val_score(
            MLPClassifier(learning_rate_init = .001,
                max_iter = int(max_iter),
                early_stopping = early_stopping_bool,
                hidden_layer_sizes = hidden_layer_array,
                random_state=2,
            ),
            features_std, y, 'roc_auc', cv=5, n_jobs = -1
        ).mean()
            
        return val
    
    
    ### Bayesian Optimization ###
    
    gp_params = {"alpha": 1e-5}
    
    mlpBO = BayesianOptimization(
        mlp_cv,
        {'max_iter':(200,800),
        'num_nodes_1':(2,max_layer_1), 
        'num_nodes_2':(2,max_layer_2),
        'num_nodes_3':(2,max_layer_3)
        }
    )
    
    mlpBO.maximize(init_points = init_points, n_iter=num_iter, acq = 'ucb', kappa = 3, **gp_params)
    
    print('Final Results')
    print('MLP Cross-Validated AUC: %f' % mlpBO.res['max']['max_val'])
    
    num_nodes_1 = int(mlpBO.res['max']['max_params']['num_nodes_1'])
    num_nodes_2 = int(mlpBO.res['max']['max_params']['num_nodes_2'])
    num_nodes_3 = int(mlpBO.res['max']['max_params']['num_nodes_3'])
    max_iter = int(mlpBO.res['max']['max_params']['max_iter'])
    
    ### Training MLP using params learned from Bayesian Optimization ###
    mlp = MLPClassifier(max_iter = max_iter, hidden_layer_sizes=np.array([num_nodes_1,num_nodes_2, num_nodes_3]), random_state=2, learning_rate_init = .001, early_stopping = True)
    mlp.fit(features_std, y) # Fitting model
    
    return mlp


        


    