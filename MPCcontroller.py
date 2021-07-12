# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:48:37 2016

@author: r
"""

import numpy as np
import scipy.signal as sig
from numpy import matlib
import matplotlib.pyplot as plt
from Model import get_step as stepper
import CommonUtils as Tools

class Control(object):
    def __init__(self, Model, delta_T, pred_H, cont_H, tuning, initial_cond):  
        self.Model = Model  # The model internal to the controller
          
        self.Tuning = tuning # The tuning constants
        #self.Ranges = ranges # The normalisation ranges -> Normalisation will be done external to the MPC

        self.initial_cond  = initial_cond # initial conditions of the MPC
        
        self.delta_T = delta_T # Controller configs
        self.pred_H = pred_H # Controller configs
        self.cont_H = cont_H # Controller configs
        
        # The stepresponses used in the Hessian
        self.stepresponse = stepper(self.Model.model_stack, self.delta_T, self.pred_H)
        self.stepresponse_mat = step_response_list(self.Model.model_stack, self.stepresponse, self.cont_H)
        self.Su = create_Su(self.stepresponse_mat)
	        	
        # Calculate the Hessian and Gradient matrices
        self.Hessian = update_Hessian(self.Su, self.Tuning)
        
        self.Gradient = update_Gradient(self.Su, self.Tuning, self.initial_cond.Epred)
                
        # Calculate constraint matrices
        self.U_lhs, self.U_rhs = update_Constraints(self.Su, self.initial_cond.Ypred, self.cont_H, self.pred_H,
                                                    self.initial_cond.U_init,
                                                    self.Tuning.u_low, self.Tuning.u_high,
                                                    self.Tuning.u_roc_up, self.Tuning.u_roc_down,
                                                    self.Tuning.y_low, self.Tuning.y_high)
        
    def update(self, state, u_current, SP):
        self.Y_openloop = state
        self.Epred = state - SP
        self.Gradient = update_Gradient(self.Su, self.Tuning, self.Epred)
        self.U_lhs, self.U_rhs = update_Constraints(self.Su, self.Y_openloop, self.cont_H, self.pred_H,
                                                    u_current,
                                                    self.Tuning.u_low, self.Tuning.u_high,
                                                    self.Tuning.u_roc_up, self.Tuning.u_roc_down,
                                                    self.Tuning.y_low, self.Tuning.y_high)
        
def update_Constraints(Su, Y_openloop, cont_H, pred_H , u_current, u_low, u_high, u_roc_up, u_roc_down, y_low, y_high):
    no_mvs = len(u_current)
    u_rhs_low = np.matrix( np.zeros([cont_H*no_mvs, 1]) )
    u_rhs_high = np.matrix( np.zeros([cont_H*no_mvs, 1]) )
    
    # Calculate RHS of mv limit constraints
    #--------------------------------------------------------------------------
    tmp_high = u_high - u_current
    tmp_low = u_current - u_low
    
    for i in range(0, no_mvs):
        u_rhs_high[i*cont_H:cont_H*(i+1), :] = tmp_high[i]
        u_rhs_low[i*cont_H:cont_H*(i+1), :] = tmp_low[i]
    
    u_rhs_lims = np.concatenate( [u_rhs_high, u_rhs_low], axis = 0 )
    #--------------------------------------------------------------------------
    
    # Calculate LHS of mv limit constraints
    #--------------------------------------------------------------------------
    Il = np.matrix( np.tril(np.ones([cont_H, cont_H])), dtype="float64")
 
    u_lhs_high = np.matrix( np.zeros([cont_H*no_mvs, cont_H*no_mvs]), dtype="float64")
    u_lhs_low = np.matrix( np.zeros([cont_H*no_mvs, cont_H*no_mvs]), dtype="float64")
    
    for i in range(0, no_mvs):
        u_lhs_high[i*cont_H:cont_H*(i+1), i*cont_H:cont_H*(i+1)] = Il
        u_lhs_low[i*cont_H:cont_H*(i+1), i*cont_H:cont_H*(i+1)] = -Il
        
    u_lhs_lims = np.concatenate( [u_lhs_high, u_lhs_low], axis = 0 )
    #--------------------------------------------------------------------------    
    
    # Calculate ROC rhs constraints
    u_rhs_roc_up = np.matrix( np.zeros([cont_H*no_mvs, 1]) )
    u_rhs_roc_down = np.matrix( np.zeros([cont_H*no_mvs, 1]) )
    #--------------------------------------------------------------------------
    for i in range(0, no_mvs):
        u_rhs_roc_up[i*cont_H:cont_H*(i+1), :] = u_roc_up[i]
        u_rhs_roc_down[i*cont_H:cont_H*(i+1), :] = u_roc_down[i]
        
    u_rhs_roc = np.concatenate( [u_rhs_roc_up, u_rhs_roc_down], axis = 0 )
    #--------------------------------------------------------------------------

    # Calculate ROC lhs constraints
    #--------------------------------------------------------------------------        
    u_lhs_roc_up = np.matrix( np.zeros([cont_H*no_mvs, cont_H*no_mvs]), dtype="float64")
    u_lhs_roc_down = np.matrix( np.zeros([cont_H*no_mvs, cont_H*no_mvs]), dtype="float64")
    I = np.matrix( np.eye(cont_H), dtype="float64")
    
    for i in range(0, no_mvs):
        u_lhs_roc_up[i*cont_H:cont_H*(i+1), i*cont_H:cont_H*(i+1)] = I
        u_lhs_roc_down[i*cont_H:cont_H*(i+1), i*cont_H:cont_H*(i+1)] = -I
        
    u_lhs_roc = np.concatenate( [u_lhs_roc_up, u_lhs_roc_down], axis = 0 )
    #--------------------------------------------------------------------------

    no_y_const = len( y_high ) + len( y_low ) 
    # Calculate eps rhs constraints
    #--------------------------------------------------------------------------
    # Upper Y constraints
    Y_eps_y_high = - Y_openloop + Tools.vector_appending( y_high, pred_H) 
    # lower Y constraints
    Y_eps_y_low = Y_openloop - Tools.vector_appending( y_low, pred_H) 
    # epsilon > 0 
    Y_eps = np.zeros([no_y_const, 1])

    eps_RHS = np.concatenate([Y_eps_y_high, Y_eps_y_low, Y_eps])
    #--------------------------------------------------------------------------

    # Calculate eps lhs constraints
    #--------------------------------------------------------------------------
    lst = []
    for i in range(0, no_y_const):
        I_zeros = np.zeros([pred_H, no_y_const])
        I_zeros[:,i] = 1
        lst.append( I_zeros )

    Y_eps = np.matrix(lst[0])
    for i in range(1, len(lst)):
        Y_eps = np.concatenate( [Y_eps, lst[i]] )

    Su_pad = np.concatenate( [Su, -Su ])
    Su_comb_pad = np.concatenate([Su_pad, Y_eps], axis = 1)
    #--------------------------------------------------------------------------

    # Construct final U_RHS
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    U_RHS = np.concatenate( [u_rhs_lims, u_rhs_roc, eps_RHS], axis = 0 )
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    # Construct final U_LHS
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    U_LHS = np.concatenate( [u_lhs_lims, u_lhs_roc], axis = 0 )
    [no_all_u_const, no_all_u] = np.shape( U_LHS )
    
    U_LHS_right_eps = np.concatenate( [U_LHS, np.matrix(np.zeros([np.int(no_all_u_const) , no_y_const]))], axis = 1 ) 
    Y_pad_eps = np.concatenate([np.zeros([ no_y_const, no_all_u]), np.diag( np.ravel( np.ones([no_y_const, 1]) ) )], axis=1)

    U_LHS = np.concatenate( [U_LHS_right_eps, Su_comb_pad, Y_pad_eps] ) 
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    return U_LHS, U_RHS
        
def update_Hessian(Su, tuning):
    Hessian = Su.T*tuning.mat_Y.T*tuning.mat_Y*Su + tuning.mat_U.T*tuning.mat_U
    lambda_tune = tuning.pred_H*np.ravel( np.array( [tuning.eps_high, tuning.eps_low]) )
    
    # Add slack variables to make CV constrained solution tracktable
    no_y_const = len( tuning.y_high ) + len( tuning.y_low )
    no_cvs = len( tuning.y_high )
    no_mvs = len( tuning.u_high )

    H_pad_right = np.zeros( [tuning.cont_H*no_mvs, no_y_const] )
    H_pad_below = np.zeros( [no_y_const, tuning.cont_H*no_mvs] )
    H_pad_below = np.concatenate([H_pad_below, np.diag( np.ravel(np.ones([no_y_const, 1]))*lambda_tune )], axis = 1)
    
    Hessian_new = np.concatenate([ Hessian, H_pad_right ], axis = 1)
    Hessian_new = np.concatenate([ Hessian_new, H_pad_below ])
    return Hessian_new
    
def update_Gradient(Su, tuning, Error):
    Gradient = Su.T*tuning.mat_Y.T*tuning.mat_Y*Error

    # Add the slack variables to the gradient - zero!
    no_y_const = len( tuning.y_high ) + len( tuning.y_low )
    Gradient_new = np.concatenate( [Gradient, np.zeros([no_y_const, 1])])
    return Gradient_new
    
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
