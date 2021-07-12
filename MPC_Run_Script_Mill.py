# %matplotlib notebook
#%matplotlib qt 

# Current status and things to do
#----------------------------------
# 1) Done(Basic) - Feedback - See how to deal with this, state estimator
# 2) Done(One timestep out, only YCL plotting) Disturbances - State estimator 
# 3) Soft contstraints on CVs
# 4) make it nicer to run in a loop - (Models from other function/text file, Controller Tuning, Initialisation -> Check XML read)
# 5) Nice plotting environment to check simulation during runtime (PyQT?)

import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
from numpy import matlib
from cvxopt import solvers
from cvxopt.solvers import qp
from cvxopt import matrix

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

# to be removed (Why?)
import scipy.signal as sig

#---------------------------------------------------
# Create the model
no_mvs = 3
no_cvs = 3
no_dvs = 1

# Model
G = matlib.repmat(None, no_cvs, no_mvs + no_dvs)

G[0][0] = sig.lti([0.007], [980.0, 1.0])
G[0][1] = sig.lti([785, 0.15], [32050, 1050, 1])
G[0][2] = sig.lti([-0.065], [1200, 1])

G[1][0] = sig.lti([0.5], [950, 1.0])
G[1][1] = sig.lti([-26], [1450.0, 1.0])
G[1][2] = sig.lti([-15], [900, 1.0])

G[2][0] = sig.lti([0.03], [950.0, 1.0])
G[2][1] = sig.lti([620, -0.7], [32050, 1050, 1])
G[2][2] = sig.lti([-0.35], [980.0, 1.0])

# Disturbance
G[0][3] = sig.lti([0.008], [450.0, 1.0])
G[1][3] = sig.lti([0.04], [450.0, 1.0])
G[2][3] = sig.lti([0.003], [450.0, 1.0])

# Create the model object
pred_H = 100
deltaT = 30
G_model_state = LTImodel.Mod(G, deltaT, pred_H)
G_model_cont = LTImodel.Mod(G[:,0:no_mvs], deltaT, pred_H)
G_model_cont_dist = LTImodel.Mod(G[:,no_mvs:], deltaT, pred_H)

# Plot stepresponse
#---------------------------------------------------
#G_model_state.plot_stepresponse()
#---------------------------------------------------
# state predictor config
init_PV = np.array([0.0, 0.0, 0.0])
init_MV = np.array([0.0, 0.0, 0.0])

init_state = Tools.vector_appending(init_PV, pred_H)
State = StatePred.Predictor(G_model_state, init_state)
#---------------------------------------------------

#---------------------------------------------------
# DMC config
# Horizons
# pred_H -  Same as state predictor
cont_H = 75

# Tuning
#-----------------------------
# MV and CV weights
u_tune = 2.5*np.array([1.0, 80.0, 60])
y_tune = 0.1*np.array([70.0, 1.0, 0.00001])

# MV limits
u_low = np.array([-1500.0, -10.0, -5])
u_high = np.array([500.0, 5.0, 1])

# MV roc limits - not being used now
u_roc_up = np.array([150.0, 0.3, 0.3])
u_roc_down = np.array([330.0, 0.3, 0.3])

# CV limits
y_low = np.array([-3.0, -250.0, -5.0])
y_high = np.array([4.0, 80.0, 5.0])

eps_high = np.array([5000, 100, 100000])
eps_low = np.array([100, 100, 100])
#-----------------------------
#---------------------------------------------------

#---------------------------------------------------
# Initial Conditions
#-----------------------------
init_SP = np.array([0.0, 0.0, 0.0])
init_PV = np.array([0.0, 0.0, 0.0])
init_MV = np.array([0.0, 0.0, 0.0])
init_DV = np.array([0.0])
#-----------------------------

# Create object
#-----------------------------
# Create tuning object - u_tune, y_tune
Tuning = tune.tuning(u_tune, y_tune, pred_H, cont_H, u_low, u_high, y_low, y_high, u_roc_up, u_roc_down, eps_high, eps_low)
#Tuning = tune.tuning(u_tune, y_tune, pred_H, cont_H, u_low, u_high, u_roc_up, u_roc_down)

# Initial conditions - pred_H, cont_H, SP, PV
BeginCond = initial.init_cond(pred_H, cont_H, init_SP, init_PV, init_MV, init_DV)

# Create the MPC control object
Controller = MPC.Control(G_model_cont, deltaT, pred_H, cont_H, Tuning, BeginCond)
#-----------------------------
#---------------------------------------------------

#---------------------------------------------------
hist = 250
# initialise MVs
u_meas = init_MV # 1d array - The current measurement
d_meas = np.zeros( no_dvs ) # 1d array
y_meas = init_PV # 1d array

u_prev = init_MV # 1d array
d_prev = np.zeros( no_dvs ) # 1d array
sp_prev = init_SP

u_all = np.matrix(np.matlib.repmat(u_meas, hist, 1)) # u_all -> matrix, rows = time, cols = vars
d_all = np.matrix(np.matlib.repmat(d_meas, hist, 1)) # y_all -> matrix, rows = time, cols = vars
y_all = np.matrix(np.matlib.repmat(y_meas, hist, 1)) # y_all -> matrix, rows = time, cols = vars

sp_all = np.matrix(np.matlib.repmat(sp_prev, hist, 1))
time = np.array([0])

# Simulation settings
#---------------------------------------------------
#---------------------------------------------------
sim_no = 30

plant_d = np.asmatrix( np.cumsum( 10*( np.random.rand( sim_no, no_dvs) - 0.5) ) ).T
plant_d[50:54,0] = 180.0 + plant_d[50:54,0]
plant_d[120:124,0] = 180.0 + plant_d[120:124,0]
plant_d[180:210,0] = 330.0 + plant_d[180:210,0]
plant_d[240:244,0] = 180.0 + plant_d[240:244,0]
plant_d[300:330,0] = 330.0 + plant_d[300:330,0]

solvers.options['show_progress'] = False
Sim_SP = np.matlib.repmat(np.array([4.0, 80.0, 5.0]), sim_no, 1)
#---------------------------------------------------
#---------------------------------------------------

# initial plots
line_lstCVs_past = []
line_lstCVs_future = []
line_lstSPs_past = []
line_lstSPs_future = []
line_lstMVs_past = []
line_lstMVs_future = []

for i in range(no_cvs):
    plt.subplot(2, np.max([no_mvs, no_cvs]), i+1)
    line1, = plt.plot(y_all[:,i], 'k')
    line2, = plt.plot(range(hist, hist+pred_H), State.state[((i)*pred_H):((i+1)*pred_H)], 'r')
    line3, = plt.plot(sp_all[:,i], 'b:')
    line4, = plt.plot(range(hist, hist+pred_H), np.repeat(sp_prev[i], pred_H), 'b:')
    plt.plot([hist, hist], [1.2*y_low[i],  1.2*y_high[i]], 'k--')
    plt.xlim([0, hist+pred_H])
    plt.ylim([1.2*y_low[i],  1.2*y_high[i]])
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
    plt.plot([hist, hist], [1.2*u_low[j],  1.2*u_high[j]], 'k--')
    plt.xlim([0, hist+pred_H])
    plt.ylim([1.2*u_low[j],  1.2*u_high[j]])
    line_lstMVs_past.append(line1)
    line_lstMVs_future.append(line2)    
    plt.grid()

for i in range(0, sim_no):
#     print(i)
    tic_timer = clock.time()
    
    # update states - MV readback 
    ystate = State.update_state(y_meas, np.concatenate([u_meas , d_meas]) - np.concatenate([u_prev, d_prev]))
  
    # update controller 
    #---------------------------------------------
    Controller.update( my(ystate, Controller.pred_H, no_cvs), u_meas, Tools.vector_appending(Sim_SP[i,:], Controller.pred_H) )
    
    # Solve QP - i.e. implement control move       
    optimal_mv = qp( matrix(Controller.Hessian), matrix(Controller.Gradient), matrix(Controller.U_lhs), matrix(Controller.U_rhs) )
    #tmp = np.array(optimal_mv['x']) # Extract x from qp atributes
    tmp = np.array(optimal_mv['x'])[0:-6] # TO DO: Only extract MVs not Epsilons

    u_current = np.ravel( Tools.extract_mv( tmp, Controller.cont_H ) ) # Extract only mv moves that will be implemented
    d_current = plant_d[i,:]      
    
    # calculate closed loop prediction - Strictly this is not correct as disturbance is not included
    Y_CL = Controller.Su*tmp + Controller.Y_openloop
    #---------------------------------------------
            
    # save all the mv movements - past till now    
    u_all = np.concatenate( [u_all, (u_all[-1,:] + u_current)], axis = 0)
    d_all = np.concatenate( [d_all, d_current ], axis = 0)   
    
    # Perform simulation - Implement move on plant 
    #---------------------------------------------
    t, y = G_model_state.simulate( np.concatenate( [ u_all, d_all ], axis=1 ) )
    #---------------------------------------------

    # measure and save data    
    u_meas = np.ravel( u_all[-1,:] )
    u_prev = np.ravel( u_all[-2,:] )
    d_meas = np.ravel( d_all[-1,:] )
    d_prev = np.ravel( d_all[-2,:] )
        
    # save all cv movements
    y_meas = np.ravel( y[-1,:] )
    y_all = np.concatenate( [y_all, ( y_all[0,:] + y_meas)], axis = 0)
    sp_all = np.concatenate( [sp_all, np.matrix(Sim_SP[i,:])], axis=0)
    # print data
    #---------------------------------------------
    time = np.append(time, (i+1)*deltaT)
    elapsed = clock.time() - tic_timer
    print('Iteration', i, 'time elapsed (ms):', elapsed*1000, 'prediction error', np.sum(State.error) )
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
    #plt.pause(1e-16)
    #---------------------------------------------
     
