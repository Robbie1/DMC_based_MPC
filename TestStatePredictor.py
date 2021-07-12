# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 20:06:45 2016

@author: r
"""

import os
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
from numpy import matlib


import Model as LTImodel # dynamic models
import StatePredictor as StatePred # state predictor
import CommonUtils as Tools # few usefull tools

# testing
from StatePredictor import shift_Yhat as t

clear = lambda: os.system('cls')
clear()

# Create the model
G = matlib.repmat(None,2,2)

G[0][0] = sig.lti([2], [20, 1.0])
G[0][1] = sig.lti([0.0], [1.0, 0.0])
G[1][0] = sig.lti([0.0], [20.0, 1.7, 1.0])
G[1][1] = sig.lti([1.0, -0.5], [5.0, 3.2, 1.0])

# state predictor config
pred_H = 100
cont_H = 20
deltaT = 2

# Create the model object
G_model = LTImodel.Mod(G, deltaT, pred_H)

###############################################################################
init_state = Tools.vector_appending([0.0, 0.0], pred_H)
State = StatePred.Predictor(G_model, init_state)
###############################################################################

# Simulate the state predictor
len_test = 30
u = np.random.rand(len_test, 2)
#u[0,:] = 1.0
#u[1:,:] = 0.0

y_hist = []
for i in range(0, len_test):
    State.update_state(np.array([0.0, 0.0]), u[i,:])
    np.disp(State.state)
    y_hist.append( np.array(State.state)[ np.array([0, pred_H]) ] )
    
y_hist = np.ndarray.flatten( np.array(y_hist) ) 

# simulate the plant
T, yout, xout = sig.lsim(G_model.model_stack[0][0], np.cumsum(u[:,0]), range(0, 2*len_test, 2))
T1, yout1, xout1 = sig.lsim(G_model.model_stack[1][1], np.cumsum(u[:,1]), range(0, 2*len_test, 2))

plt.subplot(2,1,1)
plt.plot( range(0, 2*len_test, 2), y_hist[range(0, 2*len_test, 2)] )
plt.plot( T, yout )
plt.plot( range(0, 2*len_test, 2), y_hist[range(1, 2*len_test, 2)] )
plt.plot( T1, yout1 )
plt.legend( ('state','actual', 'state1','actual1') )

plt.subplot(2,1,2)
plt.plot( u[:,1] )
plt.plot( y_hist[range(1, 2*len_test, 2)] )
plt.show()
