# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 12:33:47 2016

@author: r
"""

import os
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
from numpy import matlib

from cvxopt.solvers import qp
from cvxopt import matrix

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

clear = lambda: os.system('cls')
clear()

"""
Global Config Parameters
-------------------------------------------------------------------------
"""
pred_H = 100
cont_H = 20
deltaT = 2

"""
Modelling
-------------------------------------------------------------------------
"""

# Create the model
G = matlib.repmat(None,2,2)

G[0][0] = sig.lti([0.01], [11.0, 1.0])
G[0][1] = sig.lti([-2.0], [3.0, 2.0, 1.0])

G[1][0] = sig.lti([5.0], [20.0, 1.7, 1.0])
G[1][1] = sig.lti([-1.0], [5.0, 1.0])

# Create the model object
G_model = LTImodel.Mod(G, deltaT, pred_H)

#G_model.plot_stepresponse()

"""
State predictor Configuration
-------------------------------------------------------------------------
"""
init_PV = np.array([0.0, 0.0])

# state predictor config

init_state = Tools.vector_appending(init_PV, pred_H)
State = StatePred.Predictor(G_model, init_state)

"""
Control Configuration
-------------------------------------------------------------------------
"""
u_tune = [10.0, 10.0]
u_low = -np.array([10.0, 10.0])
u_high = np.array([10.0, 10.0])
u_roc_up = np.array([1.0, 1.0])
u_roc_down = np.array([1.0, 1.0])

y_tune = [1.0, 1.0]
y_low = -np.array([10.0, 10.0])
y_high = np.array([10.0, 10.0])
eps_high = np.array([0.1, 0.1])
eps_low = np.array([0.1, 0.1])

init_SP = np.array([1.0, -0.5])

# Create tuning object - u_tune, y_tune
Tuning = tune.tuning(u_tune, y_tune, pred_H, cont_H, u_low, u_high, y_low, y_high, u_roc_up, u_roc_down, eps_high, eps_low)

# Initial conditions - pred_H, cont_H, SP, PV
u_val = np.array([0.0, 0.0])
d_val = np.array([0.0])
BeginCond = initial.init_cond(pred_H, cont_H, init_SP, init_PV, u_val, d_val)

# Create the MPC control object
Controller = MPC.Control(G_model, deltaT, pred_H, cont_H, Tuning, BeginCond)

"""
Run Simulation
-------------------------------------------------------------------------
"""
# initialise MVs
u_meas = np.zeros( 2 )

y_meas = np.zeros( 2 )

u_prev = np.zeros( 2 )
sp_prev = init_SP


hist = 50
u_all = np.matrix(np.matlib.repmat(u_meas, hist, 1)) 
y_all = np.matrix(np.matlib.repmat(y_meas, hist, 1))

sp_all = np.matrix(np.matlib.repmat(sp_prev, hist, 1))
time = np.array([0])

optimal_mv_pre = qp( matrix(Controller.Hessian), matrix(Controller.Gradient) )
tmp1 = np.array(optimal_mv_pre['x']) # Extract x from qp atributes

tpm1_y = Controller.Su*tmp1[0:-4]

print(u_meas.T - np.matrix(u_prev).T)
print(y_meas)

# Simulation settings
sim_no = 100

#solvers.options['show_progress'] = False
Sim_SP = np.matlib.repmat(np.array([1.0, 0.0]), sim_no, 1)
Sim_SP[10:50, :] = np.matrix([1.0, -1.0])


# initial plots
line_lstCVs_past = []
line_lstCVs_future = []
line_lstSPs_past = []
line_lstSPs_future = []
line_lstMVs_past = []
line_lstMVs_future = []

no_cvs = 2
no_mvs = 2

for i in range(no_cvs):
    plt.subplot(2, np.max([no_mvs, no_cvs]), i+1)
    line1, = plt.plot(y_all[:,i], 'k')
    line2, = plt.plot(range(hist, hist+pred_H), State.state[0:pred_H], 'r')
    line3, = plt.plot(sp_all[:,i], 'b:')
    line4, = plt.plot(range(hist, hist+pred_H), np.repeat(sp_prev[i], pred_H), 'b:')
    plt.plot([hist, hist], [-3, 3], 'k--')
    plt.xlim([0, hist+pred_H])
    plt.ylim([-3,  3])
    line_lstCVs_past.append(line1)
    line_lstCVs_future.append(line2)
    line_lstSPs_past.append(line3)
    line_lstSPs_future.append(line4)
    plt.grid()
    
for j in range(no_mvs):
    plt.subplot(2, np.max([no_mvs, no_cvs]), np.max([no_mvs, no_cvs]) + j+1)
    line1, = plt.plot(u_all[:,0], 'k')
    line2, = plt.plot(range(hist, hist+cont_H), np.zeros(cont_H), 'b')
    plt.plot(range(hist+pred_H), np.repeat( Tuning.u_low[j], hist+pred_H), 'g')
    plt.plot(range(hist+pred_H), np.repeat( Tuning.u_high[j], hist+pred_H), 'g')
    plt.plot([hist, hist], [-3, 3], 'k--')
    plt.xlim([0, hist+pred_H])
    plt.ylim([-3,  3])
    line_lstMVs_past.append(line1)
    line_lstMVs_future.append(line2)    
    plt.grid()


for i in range(0, sim_no):
#     print(i)
   # tic_timer = clock.time()
    
    # update states - MV readback 
    ystate = State.update_state(y_meas, np.concatenate([u_meas]) - np.concatenate([u_prev]))
        
    # update controller 
    #---------------------------------------------
    Controller.update( my(ystate, Controller.pred_H, 2), u_meas, Tools.vector_appending(Sim_SP[i,:], Controller.pred_H) )
    
    # Solve QP - i.e. implement control move       
    optimal_mv = qp( matrix(Controller.Hessian), matrix(Controller.Gradient), matrix(Controller.U_lhs), matrix(Controller.U_rhs) )
    tmp = np.array(optimal_mv['x'])[0:-4] # Extract x from qp atributes
    u_current = np.ravel( Tools.extract_mv( tmp, Controller.cont_H ) ) # Extract only mv moves that will be implemented   
    
    # calculate closed loop prediction - Strictly this is not correct as disturbance is not included
    Y_CL = Controller.Su*tmp + Controller.Y_openloop
    #---------------------------------------------
            
    # save all the mv movements - past till now    
    u_all = np.concatenate( [u_all, (u_all[-1,:] + u_current)], axis = 0)
    
    # Perform simulation - Implement move on plant 
    #---------------------------------------------
    t, y = G_model.simulate( np.concatenate( [ u_all ], axis=1 ) )
    #---------------------------------------------

    # measure and save data    
    u_meas = np.ravel( u_all[-1,:] )
    u_prev = np.ravel( u_all[-2,:] )
        
    # save all cv movements
    y_meas = np.ravel( y[-1,:] )
    y_all = np.concatenate( [y_all, ( y_all[0,:] + y_meas)], axis = 0)
    sp_all = np.concatenate( [sp_all, np.matrix(Sim_SP[i,:])], axis=0)
    # print data
    #---------------------------------------------
    #time = np.append(time, (i+1)*deltaT)
    #elapsed = clock.time() - tic_timer
    #print('Iteration', i, 'time elapsed (ms):', elapsed*1000, 'prediction error', np.sum(State.error) )
    #---------------------------------------------
    
    # update plot
    #---------------------------------------------
    for j in range(no_cvs):
        line_lstCVs_past[j].set_ydata(y_all[-hist:,j]) 
        line_lstCVs_future[j].set_ydata( Y_CL[(j*pred_H):( (j+1)*pred_H )] ) 
        line_lstSPs_past[j].set_ydata(sp_all[-hist:,j]) 
        line_lstSPs_future[j].set_ydata( np.repeat(Sim_SP[i,j], pred_H) )
    
    for k in range(no_mvs):
        line_lstMVs_past[k].set_ydata(u_all[-hist:,k]) 
        line_lstMVs_future[k].set_ydata( np.cumsum( tmp[(k*cont_H):( (k+1)*cont_H )] ) + u_meas[k] )
        
    plt.pause(1e-6)
    #---------------------------------------------

"""
for i in range(0, 20):
    # update states
    
    ystate = State.update_state(y_meas, np.concatenate([u_meas]) - np.concatenate([u_prev]))
    print(i)
#    print y_meas
#    print u_meas
#    print init_SP    
    
    # update controller - Need to incorporate into object
    Controller.update( my(ystate, Controller.pred_H, 2), u_meas, Tools.vector_appending(init_SP, Controller.pred_H) )
    
    # Solve QP - i.e. implement control move       
    optimal_mv = qp( matrix(Controller.Hessian), matrix(Controller.Gradient) )
    tmp = np.array(optimal_mv['x'])[0:-4] # Extract x from qp atributes
    u_current = Tools.extract_mv( tmp, Controller.cont_H ).T # Extract only mv moves that will be implemented
    
    # calculate closed loop prediction    
    Y_CL = Controller.Su*tmp + Controller.Y_openloop
    
    
    # save all the mv movements - past till now    
    u_all = np.concatenate( [u_all, (u_all[-1,:] + u_current)], axis = 0)
    
    # implement move
    t, y = G_model.simulate( np.array(u_all) )
    y = y.T

    # measure and save data    
    u_meas = np.ravel( u_all[-1,:] )
    u_prev = np.ravel( u_all[-2,:] )
    
    yy_meas = np.ravel( y[-1,:] )
    y_all = np.concatenate( [y_all, ( y_all[0,:] + y_meas)], axis = 0)
    
    print("Closed loop error")
    print(Y_CL[np.array([0, pred_H])].T - y_meas)
    time = np.append(time, (i+1)*deltaT)
   
plt.subplot(211)
plt.plot(np.array(np.cumsum( np.concatenate( [np.matrix([0.0]), tmp1[0:cont_H]], axis=0)  ))[0], 'g.-')
plt.plot(u_all[0:cont_H, 0], 'k.--')
plt.plot(np.tile(init_SP[0], cont_H), 'r')

plt.plot(np.append( [0.0], tpm1_y[0:cont_H] ), 'g.-')
plt.plot(y_all[0:cont_H,0], 'k.--')

plt.subplot(212)
plt.plot(np.array(np.cumsum( np.concatenate( [np.matrix([0.0]), tmp1[cont_H:]], axis=0)  ))[0], 'g')
plt.plot(u_all[0:cont_H, 1], 'k.--')

plt.plot(np.append( [0.0], tpm1_y[pred_H:pred_H+cont_H] ), 'g.-')
plt.plot(y_all[0:cont_H,1], 'k.--')
plt.plot(np.tile(init_SP[1], cont_H))
plt.show()
"""
