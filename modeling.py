#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 19:48:13 2018

@author: miller
"""
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import *
from sklearn.cross_validation import KFold

def split_feats_label(data):
    
    cols_not_to_train = ['household_key', 'DAY', 'WEEK_NO', 'label', "BASKET_ID", 
                         'COUPON_DISC', 'QUANTITY', 'SALES_VALUE', 'STORE_ID', 
                         'CUSTOMER_PAID', 'PROD_PURCHASE_COUNT', "AGE_DESC"]
    
    training_cols = [col for col in data if col not in cols_not_to_train]
    
    print("Training cols: " + str(training_cols))
    
    X = np.array(data.loc[:,np.array(training_cols)])
    
    Y = np.array(data.loc[:,'label'])
    
    return X,Y, data['household_key']

def train_mod(X,Y,k=5):
        
	#Report the mean accuracy and mean auc of all the folds for logistic regression model
                
    kf = KFold(X.shape[0],k)
    
    acc_dict = {}
    auc_dict = {}
    
    for regularization_wt in [1,10,0.1]:
    
        accuracy_list = []
        auc_list = []
    
        for train_index, test_index in kf:
            
            print(len(train_index))
                    
            # Separating train/test sets
            x_train = X[np.array(train_index)]
            x_test = X[np.array(test_index)]
            
            y_train = Y[np.array(train_index)]
            y_test = Y[np.array(test_index)]
            
            # Training model
            lr = LogisticRegression(C = regularization_wt, penalty="l2")
            lr.fit(x_train, y_train)
            y_pred = lr.predict(x_test)  
                        
            acc = accuracy_score(y_test, y_pred) 
            auc = roc_auc_score(y_test, y_pred)
            
            print("AUC, lambda = {}: ".format(regularization_wt) + str(auc))
            
            accuracy_list.append(acc)
            auc_list.append( auc )
            
        acc_dict[regularization_wt] = np.mean(accuracy_list)
        auc_dict[regularization_wt] =  np.mean(auc_list)
        
    min_auc = 100000
    min_auc_reg = 0
    
    for key in auc_dict:
        
        print("AUC for all folds, lambda = {}: ".format(key) + str(auc_dict[key]))
        
        if auc_dict[key] < min_auc:
            
            min_auc = auc_dict[key]
            min_auc_reg = key
            
    lr = LogisticRegression(C = min_auc_reg, penalty = "l2")
    lr.fit(X, Y) 
        
    return lr


if __name__ == "__main__":
    
    ### Train logistic regression model
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    Y_pred = lr.predict(X_train)
    
    

