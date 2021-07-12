# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 20:10:14 2016

@author: r
"""
import numpy as np
from numpy import matlib

# Stepresponse matrix functions
###############################################################################
def create_Su(stepresponse_mat):
    r =  np.array( np.shape(stepresponse_mat) )

    # get height, width of first matrix in order to construct Su
    height, width = np.shape( stepresponse_mat[0][0] )    
    Su = np.matrix( np.zeros( [height*r[0], width*r[1]] ) )
    
    for i in range(0, r[0]):
        for j in range(0, r[1]):
            Su[i*height:(i+1)*height, j*width:(j+1)*width] = stepresponse_mat[i][j]
            
    return Su
    
def step_response_list(model_stack, stepresponse, cont_H):
    r =  np.array( np.shape(model_stack) )
    step_list = matlib.repmat(None, r[0], r[1])
    
    for i in range(0, r[0]):
        for j in range(0, r[1]):
            step_list[i][j] = build_stepmat(stepresponse[i][j], cont_H)
    
    return step_list
    
def build_stepmat(stepresponse, cont_H):
    StepMat = np.matrix( np.zeros( (len(stepresponse), cont_H) ) )
    for i in range(0, cont_H):
        StepMat[i:, i] = np.matrix(stepresponse[0:(len(stepresponse)-i)]).T
    return StepMat
###############################################################################
    
# Vector handling functions
###############################################################################
def vector_appending( val, horizon):
    # tiles val over the entire horizon
    vec = []
    for i in range(0, len(val)):
        vec.append( np.tile( val[i], horizon) )
    return np.matrix( np.ndarray.flatten( np.array(vec) ) ).T
    
def vector_one_mv_move( val, horizon):
    # Insers val, at intevals corresponding to horizon, the rest zeros
    vec = np.zeros( [len(val)*horizon, 1] )
    for i in range(0, len(val)):
        vec[i*horizon] = val[i]
    return np.matrix( np.ndarray.flatten( np.array(vec) ) ).T
    
def extract_mv( val, horizon):
    # Extract val at the interval - horizon
    vec = []
    for i in range(0, np.int(len(val)/horizon), ):
        vec.append( val[i*horizon] )
    return np.matrix( np.ndarray.flatten( np.array(vec) ) ).T
###############################################################################