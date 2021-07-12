# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:06:09 2016

@author: r
"""

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

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")
        
###############################################################################
clear = lambda: os.system('cls')
clear()

# Create the model
G = matlib.repmat(None, 2, 2)

G[0][0] = sig.lti([1], [30, 1.0])
G[0][1] = sig.lti([-1], [40, 1.0])

G[1][0] = sig.lti([0.4], [50.0, 3.0, 1.0])
G[1][1] = sig.lti(100.0*np.array([0.5, -0.01]), [80.0, 15.0, 3.0])

# state predictor config
pred_H = 100
cont_H = 20
deltaT = 2

# Create the model object
G_model = LTImodel.Mod(G, deltaT, pred_H)

u = np.random.rand( 20, 2 )
y_plant = [[]]
for i in range(1, len(u[:,1])):
    tic()
    t, y = G_model.simulate(u[0:i,:])
    if i == 1:
        y_plant = np.append(y_plant, [y[:,-1]], axis=1)
        
    else:
        y_plant = np.append(y_plant, [y[:,-1]], axis=0)
        
    toc()
    plt.plot(y_plant[:,0])
    plt.plot(y_plant[:,1])
    plt.show()
    
##############################################################################





