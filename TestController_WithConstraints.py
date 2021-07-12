# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:40:58 2016

@author: r


Works in environment spyder_env
"""
import os
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
from numpy import matlib
#from cvxopt.solvers import qp
#from cvxopt import matrix

# controller classes
import Model as LTImodel # dynamic models
import InitCond as initial # init conditions for controller
import tuning as tune # tuning matrices for controller
import ContRanges as c_ranges # tuning matrices for controller
import MPCcontroller as MPC # MPC controller
import StatePredictor as StatePred # state predictor

# Tools
import CommonUtils as Tools # some tools to make handling things easier

# Testing
from StatePredictor import shift_Yhat as my
import time as clock


clear = lambda: os.system('cls')
clear()

"""
Global Config Parameters
-------------------------------------------------------------------------
"""
pred_H = 30
cont_H = 20
deltaT = 2

"""
Modelling
-------------------------------------------------------------------------
"""
# Create the model
G = matlib.repmat(None,2,2)

G[0][0] = sig.lti([0.01], [10.0, 0.0])
G[0][1] = sig.lti([-2.0], [3.0, 2.0, 1.0])

G[1][0] = sig.lti([-5.0], [20.0, 1.7, 1.0])
G[1][1] = sig.lti([1.0], [5.0, 1.0])

# Create the model object
G_model = LTImodel.Mod(G, deltaT, pred_H)

"""
State predictor Configuration
-------------------------------------------------------------------------
"""
init_PV = np.array([0.0, 0.0])
init_MV = np.array([0.0, 0.0])

# state predictor config
init_state = Tools.vector_appending(init_PV, pred_H)
State = StatePred.Predictor(G_model, deltaT,  pred_H, cont_H, init_state)

"""
Control Configuration
-------------------------------------------------------------------------
"""
# MV and CV ranges
u_range = np.array([[0.0, 5.0],
                    [0.0, 1.0]])

y_range = np.array([[0.5, 2.0],
                    [0.0, 4.0]])

# MV and CV weights
u_tune = np.array([1.0, 1.0])
y_tune = np.array([1.0, 1.0])

# MV limits
u_low = np.array([-5.0, -5.0])
u_high = np.array([5.0, 5.0])

# MV roc limits - not being used now
u_roc_up = np.array([3.0, 3.0])
u_roc_down = -np.array([3.0, 3.0])

# CV Setpoints
init_SP = np.array([10.0, 5.0])

# Create object
#--------------
# Create ranges object - u_tune, y_tune
CRanges = c_ranges.cont_ranges(u_range, y_range)

# Create tuning object - u_tune, y_tune
Tuning = tune.tuning(u_tune, y_tune, pred_H, cont_H, u_low, u_high, u_roc_up, u_roc_down)

# Initial conditions - pred_H, cont_H, SP, PV
BeginCond = initial.init_cond(pred_H, cont_H, init_SP, init_PV, init_MV)

# Create the MPC control object
Controller = MPC.Control(G_model, deltaT, pred_H, cont_H, Tuning, CRanges, BeginCond)




"""
Run Simulation
-------------------------------------------------------------------------
"""

"""
# initialise MVs
u_meas = np.zeros( len( G_model.model_stack[0,:] ) ) # 1d array
y_meas = np.zeros( len( G_model.model_stack[0,:] ) ) # 1d array
u_prev = np.zeros( len( G_model.model_stack[0,:] ) ) # 1d array

u_all = np.matrix(u_meas) # u_all -> matrix, rows = time, cols = vars
y_all = np.matrix(y_meas) # y_all -> matrix, rows = time, cols = vars
time = np.array([0])

optimal_mv_pre = qp( matrix(Controller.Hessian), matrix(Controller.Gradient), matrix(Controller.U_lhs), matrix(Controller.U_rhs) )
tmp1 = np.array(optimal_mv_pre['x']) # Extract x from qp atributes
tpm1_y = Controller.Su*tmp1

for i in range(0, 80):
    # update states
    print(i)
    tic_timer = clock.time()
    
    ystate = State.update_state( y_meas, u_meas - u_prev)
        
    # update controller 
    Controller.update( my(ystate, Controller.pred_H, 2), u_meas, Tools.vector_appending(init_SP, Controller.pred_H) )
    
    # Solve QP - i.e. implement control move       
    optimal_mv = qp( matrix(Controller.Hessian), matrix(Controller.Gradient), matrix(Controller.U_lhs), matrix(Controller.U_rhs) )
    tmp = np.array(optimal_mv['x']) # Extract x from qp atributes
    u_current = np.ravel( Tools.extract_mv( tmp, Controller.cont_H ) ) # Extract only mv moves that will be implemented
    
    # calculate closed loop prediction    
    Y_CL = Controller.Su*tmp + Controller.Y_openloop
        
    # save all the mv movements - past till now    
    u_all = np.concatenate( [u_all, (u_all[i,:] + u_current)], axis = 0)
    
    # implement move
    t, y = G_model.simulate( u_all )

    # measure and save data    
    u_meas = np.ravel( u_all[-1,:] )
    u_prev = np.ravel( u_all[i,:] )
    
    y_meas = np.ravel( y[-1,:] )
    y_all = np.concatenate( [y_all, ( y_all[0,:] + y_meas)], axis = 0)
    
#    print "Closed loop error"
#    print np.ravel( Y_CL[np.array([0, pred_H])] ) - y_meas
    time = np.append(time, (i+1)*deltaT)
    elapsed = clock.time() - tic_timer
    print('Iteration', i, 'time elapsed (ms):', elapsed*1000)
  
plt.subplot(211)
plt.plot(np.array(np.cumsum( np.concatenate( [np.matrix([0.0]), tmp1[0:cont_H]], axis=0)  ))[0], 'g.-')
plt.plot(u_all[:, 0], 'k.--')
plt.plot(np.tile(init_SP[0], cont_H), 'r')

plt.plot(np.append( [0.0], tpm1_y[0:cont_H] ), 'g.-')
plt.plot(y_all[:,0], 'k.--')

plt.subplot(212)
plt.plot(np.array(np.cumsum( np.concatenate( [np.matrix([0.0]), tmp1[cont_H:]], axis=0)  ))[0], 'g')
plt.plot(u_all[:, 1], 'k.--')

plt.plot(np.append( [0.0], tpm1_y[pred_H:pred_H+cont_H] ), 'g.-')
plt.plot(y_all[:, 1], 'k.--')
plt.plot(np.tile(init_SP[1], cont_H))
plt.show()
"""
