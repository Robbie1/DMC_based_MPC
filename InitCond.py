# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:40:58 2016

@author: r
"""
import numpy as np

class init_cond(object):
    def __init__(self, pred_H, cont_H, sp_val, y_val, u_val, d_val):
        self.Ypred = vector_appending( y_val, pred_H )
        self.Epred = vector_appending( y_val - sp_val, pred_H ) # (y - sp)
        self.U_init = u_val
        self.Y_init = y_val
        self.D_init = d_val
        self.SP = sp_val
        
def vector_appending( val, horizon):
    vec = []
    for i in range(0, len(val)):
        vec.append( np.tile( val[i], horizon) )
    return np.matrix( np.ndarray.flatten( np.array(vec) ) ).T
