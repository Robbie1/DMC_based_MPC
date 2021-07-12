# -*- coding: utf-8 -*-
"""
Created on Wed Jun 08 16:17:39 2016

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

clear = lambda: os.system('cls')
clear()

# Create the model
G = matlib.repmat(None,2,2)

G[0][0] = sig.lti([2], [20, 1.0])
G[0][1] = sig.lti([0.0], [1.0, 0.0])
G[1][0] = sig.lti([0.0], [20.0, 1.7, 1.0])
G[1][1] = sig.lti([1.0, -0.5], [5.0, 3.2, 1.0])

pred_H = 100
cont_H = 50
deltaT = 2

# Create the model object
G_model = LTImodel.Mod(G, deltaT, pred_H)

# Create tuning object - u_tune, y_tune
Tuning = tune.tuning([10.0, 10.0], [5.0, 5.0], pred_H, cont_H)

# Initial conditions - pred_H, cont_H, SP, PV
BeginCond = initial.init_cond(pred_H, cont_H, np.array([1.0, 0.5]), np.array([0.0, 0.0]) )

# Create the MPC control object
Controller = MPC.Control(G_model, 1.5, pred_H, cont_H, Tuning, BeginCond)

# Test QP
optimal_mv = qp( matrix(Controller.Hessian), matrix(Controller.Gradient) )
tmp = np.array(optimal_mv['x'])

y = Controller.Su*tmp

#plt.figure()
#G_model.plot_stepresponse()
#plt.show()

# Plot the response
plt.subplot(2,1,1)
plt.plot(y[0:pred_H])
plt.plot(np.tile(1.0, pred_H))
plt.plot(y[pred_H:])
plt.plot(np.tile(0.5, pred_H))
plt.ylim(-1, 2)
plt.xlim(0, pred_H)
plt.legend( ['y1','y1-sp', 'y2','y2-sp'] )


plt.subplot(2,1,2)
plt.plot( np.cumsum( np.append(np.array([0]), tmp[0:cont_H]) ) )
plt.plot( np.cumsum( np.append(np.array([0]), tmp[cont_H:]) ) )
plt.ylim(-2, 2)
plt.xlim(0, pred_H)
plt.legend( ['u1','u2'] )

plt.show()






