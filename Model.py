# -* coding: utf-8 -*-
"""
Created on Wed Jun 08 15:59:14 2016

@author: r
"""
import numpy as np
import scipy.signal as sig
from numpy import matlib
import matplotlib.pyplot as plt

class Mod(object):
    def __init__(self, model_stack, delta_T, pred_H):      
        self.size = np.shape(model_stack)
        self.model_stack = model_stack # No Disturbances
        #self.dist_stack = dist_stack # Disturbances
        #self.dist_size = np.shape(dist_stack)
        self.delt_T = delta_T
        self.pred_H = pred_H
        
        self.stepresponse = get_step(model_stack, delta_T, pred_H) # biuld Model.stepresponse
        #self.dist_stepresponse = get_step(dist_stack, delta_T, pred_H) # biuld Model.stepresponse
        
    def plot_stepresponse(self):
        r, c =  self.size
        cnt = 1
        time = np.linspace(self.delt_T, len(self.stepresponse[0][0])*self.delt_T,  self.pred_H)
        
        for i in range(0,r):
            for j in range(0,c):
                plt.subplot(r,c,cnt)
                plt.plot(time, self.stepresponse[i][j])
                cnt = cnt + 1
        plt.show()

    """
    def plot_dist_stepresponse(self):
        r, c =  self.dist_size
        cnt = 1
        time = np.linspace(self.delt_T, len(self.dist_stepresponse[0][0])*self.delt_T,  self.pred_H)
        
        for i in range(0,r):
            for j in range(0,c):
                plt.subplot(r,c,cnt)
                plt.plot(time, self.dist_stepresponse[i][j])
                cnt = cnt + 1
        plt.show()
    """
    
    def simulate(self, u):
        # u - rows are time, columns are MVs
        r, c = np.shape(self.model_stack)
        #r1, d_n = np.shape(self.dist_stack)
        
        y_out = np.zeros( [r, len(u[:,0])] )
        
        for i in range(0, r):
            for j in range(0, c):
                t, y, u_in = sig.lsim(self.model_stack[i][j], u[:, j], range(0, self.delt_T*len(u[:,0]), self.delt_T) )
                y_out[i,:] = y_out[i,:] + y
        """
        for i in range(0, r):
            for j in range(0, d_n):
                t, y, d_in = sig.lsim(self.dist_stack[i][j], d[:, j], range(0, self.delt_T*len(d[:,0]), self.delt_T) )
                y_out[i,:] = y_out[i,:] + y
        """
        
        return t, y_out.T
                
def get_step(model_stack, delt_T, pred_H):
    # get rows, columns of model_stack
    r, c =  np.shape(model_stack) 

    stepresponse = matlib.repmat(None, r, c)
                
    for i in range(0, r):
        for j in range(0, c):
            stepresponse[i][j] = step_me(model_stack[i][j], pred_H, delt_T)          
            
    return stepresponse
    
def step_me(model, pred_H, del_T):
    x = np.zeros((pred_H + 1,1))
    x[range( 1,len(x) )] = 1.0
    
    sim_time = np.linspace(0, len(x)*del_T,  pred_H+2)
    
    t, y, u = sig.lsim2(model, x, sim_time[0:-1])
    
    y = y[1:]
    return y  
    

