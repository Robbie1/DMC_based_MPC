# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 20:14:57 2016

@author: r

To Dos:
1 - include disturbance model
2 - handle integrators
"""
import numpy as np
import CommonUtils as Tools
from Model import get_step as stepper

class Predictor(object):
    def __init__(self, Model, init_state):
        self.Model = Model  # The model internal to the state predictor
        #self.Dist_Model = Model # The disturbance model
        
        # state predictor configs
        self.delta_T = Model.delt_T
        self.pred_H = Model.pred_H
        self.cont_H = 1
        
        self.state = init_state # Initialise state
        
        # Get stepresponse matrices - Model only
        self.stepresponse = stepper(self.Model.model_stack, self.delta_T, self.pred_H)
        self.stepresponse_mat = Tools.step_response_list(self.Model.model_stack, self.stepresponse, self.cont_H)
        self.Su = Tools.create_Su(self.stepresponse_mat)

        # Get stepresponse matrices - Disturbance Model
        #self.Gd_stepresponse = stepper(self.Dist_Model.model_stack, self.delta_T, self.pred_H)
        #self.Gd_stepresponse_mat = Tools.step_response_list(self.Dist_Model.model_stack, self.Gd_stepresponse, self.cont_H)
        #self.Gd_Su = Tools.create_Su(self.Gd_stepresponse_mat)
        
    def update_state(self, y_meas, u_meas):
    #def update_state(self, y_meas, u_meas):
        # u_meas - 1d array of the current MV measurements
        # y_meas - 1d array of the current CV measurements
        no_cvs, no_mvs = self.Model.size
        #no_cvs1, no_dvs = self.Model.dist_size
        
        # update state with previous mv and shift
        self.state = shift_Yhat(self.state, self.Model.pred_H, no_cvs) + self.Su*Tools.vector_one_mv_move(u_meas, self.cont_H)# + self.Gd_Su*Tools.vector_one_mv_move(d_meas, self.cont_H)
        
        # get next step prediction
        y_preds = self.state[ range(0, no_cvs*self.Model.pred_H, self.Model.pred_H) ]
        
        # calc error 
        self.error = (y_meas - np.ravel( y_preds ))
        
        # Feedback error - > Feedback Law can be improved, investigate literature
        self.state = self.state + Tools.vector_appending(self.error, self.pred_H)
        
        return self.state
    
def shift_Yhat(Y_hat, pred_H, no_cvs):
    
    new_Yhat = []
    for i in range(0, no_cvs):
        tmp = np.append( np.array(Y_hat[(i*pred_H+1):((i+1)*pred_H)]), Y_hat[(i+1)*pred_H-1] )
        new_Yhat.append( tmp )
    
    return np.matrix( np.ndarray.flatten( np.array(new_Yhat) ) ).T
