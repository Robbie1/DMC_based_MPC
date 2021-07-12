"""
Created on Tue Jun 14 14:40:58 2016

@author: r


Works in environment spyder_python python 3.5 required
"""
import os
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
from numpy import matlib
from gurobipy import *
# from cvxopt.solvers import qp
# from cvxopt import matrix

# controller classes
import Model as LTImodel # dynamic models
import InitCond as initial # init conditions for controller
import tuning as tune # tuning matrices for controller
import MPCcontroller as MPC # MPC controller
import StatePredictor as StatePred # state predictor

# Tools
import CommonUtils as Tools # some tools to make handling things easier

# Testing
from StatePredictor import shift_Yhat as my

import pandas as pd

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
# MV and CV weights
u_tune = np.array([1.0, 1.0])
y_tune = np.array([5.0, 5.0])

# MV limits
u_low = np.array([-1.0, -1.0])
u_high = np.array([1.0, 1.0])

# MV roc limits
u_roc_up = np.array([5.0, 3.0])
u_roc_down = np.array([5.0, 3.0])

# CV Setpoints
init_SP = np.array([1.0, 1.5])

# Create tuning object - u_tune, y_tune
Tuning = tune.tuning(u_tune, y_tune, pred_H, cont_H, u_low, u_high, u_roc_up, u_roc_down)

# Initial conditions - pred_H, cont_H, SP, PV
BeginCond = initial.init_cond(pred_H, cont_H, init_SP, init_PV, init_MV)

# Create the MPC control object
Controller = MPC.Control(G_model, deltaT, pred_H, cont_H, Tuning, BeginCond)


"""
Run Simulation
-------------------------------------------------------------------------
"""
# initialise MVs
u_meas = np.zeros( len( G_model.model_stack[0,:] ) ) # 1d array
y_meas = np.zeros( len( G_model.model_stack[0,:] ) ) # 1d array
u_prev = np.zeros( len( G_model.model_stack[0,:] ) ) # 1d array

u_all = np.matrix(u_meas) # u_all -> matrix, rows = time, cols = vars
y_all = np.matrix(y_meas) # y_all -> matrix, rows = time, cols = vars
time = np.array([0])

# Solve using CVXopt
# -------------------------------------------------------------------------
# optimal_mv_pre = qp( matrix(Controller.Hessian), matrix(Controller.Gradient), matrix(Controller.U_lhs), matrix(Controller.U_rhs) )
# -------------------------------------------------------------------------

# Solve using gurobi
# -------------------------------------------------------------------------
m = Model("MPC")
mv_define = Tools.vector_appending(init_MV, cont_H)
mv_gurobi = m.addVars(range(0, len(mv_define)), lb = Tools.vector_appending(-u_roc_down, cont_H), ub = Tools.vector_appending(u_roc_up, cont_H))

obj_func = 0.5*Controller.Hessian.dot(pd.Series(mv_gurobi)).dot(pd.Series(mv_gurobi))[0].tolist()[0][0] + np.sum(np.ravel(Controller.Gradient)*pd.Series(mv_gurobi))

m.setObjective(obj_func, GRB.MINIMIZE)
m.setParam("OutputFlag", 0)
m.optimize()

#mv_optimal = m.getVars()
X = np.zeros(len(m.getVars()))
i = 0
for v in m.getVars():
    X[i] = v.x
    i = i + 1

# plot first MV
'''
plt.subplot(2,1,1)
plt.plot( (Controller.Su*np.matrix(X).T)[0:pred_H] ) 
plt.plot( np.matrix(X).T[0:cont_H] ) 
plt.plot(np.tile(init_SP[0], pred_H), 'r')

plt.subplot(2,1,2)
plt.plot( (Controller.Su*np.matrix(X).T)[pred_H:] ) 
plt.plot(np.tile(init_SP[1], pred_H), 'r')
plt.plot( np.matrix(X).T[cont_H:] ) 
plt.show()
'''

# -------------------------------------------------------------------------
#tmp1 = np.array(optimal_mv_pre['x']) # Extract x from qp atributes
tmp1 = np.ravel(X)
tpm1_y = Controller.Su*np.matrix(tmp1).T


for i in range(0, 2):
    # update states
    tic_timer = clock.time()
    ystate = State.update_state( y_meas, u_meas - u_prev)
        
    # update controller 
    Controller.update( my(ystate, Controller.pred_H, 2), u_meas, Tools.vector_appending(init_SP, Controller.pred_H) )
    
    # Solve QP - i.e. implement control move       
    # Solve using cvxopt
    # -------------------------------------------------------------------------
    #optimal_mv = qp( matrix(Controller.Hessian), matrix(Controller.Gradient), matrix(Controller.U_lhs), matrix(Controller.U_rhs) )
    # tmp = np.array(optimal_mv['x']) # Extract x from qp atributes
    mv_gurobi.vType = GRB.INTEGER
    
    obj_func = 0.5*Controller.Hessian.dot(pd.Series(mv_gurobi)).dot(pd.Series(mv_gurobi))[0].tolist()[0][0] + np.sum(np.ravel(Controller.Gradient)*pd.Series(mv_gurobi))   
    
    m.setObjective(obj_func, GRB.MINIMIZE)
    m.setParam("OutputFlag", 0)
    m.optimize()

    #mv_optimal = m.getVars()
    X = np.zeros(len(m.getVars()))
    j = 0
    for v in m.getVars():
        X[j] = v.x
        j = j + 1   
    tmp = np.ravel(X)
    # -------------------------------------------------------------------------
    u_current = np.ravel( Tools.extract_mv( tmp, Controller.cont_H ) ) # Extract only mv moves that will be implemented
    
    # calculate closed loop prediction    
    # cvxopt: #Y_CL = Controller.Su*tmp + Controller.Y_openloop
    Y_CL = Controller.Su*np.matrix(tmp).T + Controller.Y_openloop
        
    # save all the mv movements - past till now    
    # u_all = np.concatenate( [u_all, (u_all[i,:] + u_current)], axis = 0)
    u_all = np.concatenate( [u_all, (u_all[i,:] + np.matrix(u_current))], axis = 0)
    
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
'''
plt.plot(np.array(np.cumsum( np.concatenate( [np.matrix([0.0]), tmp1[0:cont_H]], axis=0)  ))[0], 'g.-')
'''
plt.plot(u_all[:, 0], 'k.--')
plt.plot(np.tile(init_SP[0], cont_H), 'r')

plt.plot(np.append( [0.0], tpm1_y[0:cont_H] ), 'g.-')
plt.plot(y_all[:,0], 'k.--')

plt.subplot(212)
'''
plt.plot(np.array(np.cumsum( np.concatenate( [np.matrix([0.0]), tmp1[cont_H:]], axis=0)  ))[0], 'g')
'''
plt.plot(u_all[:, 1], 'k.--')

plt.plot(np.append( [0.0], tpm1_y[pred_H:pred_H+cont_H] ), 'g.-')
plt.plot(y_all[:, 1], 'k.--')
plt.plot(np.tile(init_SP[1], cont_H))
plt.show()

