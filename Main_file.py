# -*- coding: utf-8 -*-
"""
Created on Tue Jun 07 16:05:31 2016

@author: r
"""
import os
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

clear = lambda: os.system('cls')
clear()

Pred_H = 100
Cont_H = 50

x = np.zeros((Pred_H + 1,1))
x[range(1,len(x))] = 1

T = range(0, len(x))

model = sig.lti([2],[20, 1.0])
t, y, u = sig.lsim2(model, x, T)
y = y[1:]

Su = np.zeros( (Pred_H, Cont_H) )

for i in range(0, Cont_H):
    r = len( Su[i:, i] )
    Su[i:, i] = y[0:r]


U = 0.1*np.array( np.random.rand(Cont_H) )
#U = x[1:] - x[0:-1]
#U =  np.vstack( ([0], U[0:49]))

plt.plot(t[1:], y)
plt.plot(range(0, Pred_H), np.matrix(Su)*np.matrix(U).T)
plt.plot(range(0, len(U)), np.cumsum( U ) )
plt.show()

