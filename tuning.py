# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 18:50:41 2016

@author: r
"""
import numpy as np

class tuning(object):
    def __init__(self, u_tune, y_tune, pred_H, cont_H, u_low, u_high, y_low, y_high, u_roc_up, u_roc_down, eps_high, eps_low): 
        self.u_tune = u_tune
        self.y_tune = y_tune
        
        self.pred_H = pred_H
        self.cont_H = cont_H
        
        self.u_low = u_low
        self.u_high = u_high

        self.y_low = y_low
        self.y_high = y_high
        
        self.u_roc_up = u_roc_up
        self.u_roc_down = u_roc_down

        self.eps_high = eps_high
        self.eps_low = eps_low
        
        # Tuning matrices
        self.mat_U, self.mat_Y = tuning_matrices(self)
        
def tuning_matrices(tuning_obj):
    mat_U = create_tuning_mat(tuning_obj.cont_H, tuning_obj.u_tune)
    mat_Y = create_tuning_mat(tuning_obj.pred_H, tuning_obj.y_tune)
    return mat_U, mat_Y

def create_tuning_mat(horizon, matrix):
    variables = len(matrix)    
    
    mat = np.matrix( np.zeros([ horizon*variables, horizon*variables]) )
    
    diagonal = [] 
    for i in range(0, variables):    
        diagonal.append( np.tile(matrix[i], horizon) )
    
    np.fill_diagonal(mat, diagonal)
    return mat
