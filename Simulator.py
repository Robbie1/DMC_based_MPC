import numpy as np
import matplotlib.pyplot as plt

import CommonUtils as Tools # some tools to make handling things easier

from cvxopt import solvers
from cvxopt.solvers import qp
from cvxopt import matrix

# controller classes
import MPCcontroller as MPC # MPC controller
import StatePredictor as StatePred # state predictor
from StatePredictor import shift_Yhat as my

class Simulator(object):
    def __init__(self, State_pred, Controller, Begin_cond):
        self.hist = 50
        self.State_pred = State_pred
        self.Controller = Controller
        
        # Initial MVs, CVS, DVs
        #----------------------------------
        self.no_mvs = len(Begin_cond.U_init)
        self.no_dvs = len(Begin_cond.D_init)
        self.no_cvs = len(Begin_cond.Y_init)
        
        self.u_prev = Begin_cond.U_init
        self.d_prev = Begin_cond.D_init # Need to make this variable, with and without disturbance
        self.y_prev = Begin_cond.Y_init
        self.sp_prev = Begin_cond.SP
        
        self.u_meas = self.u_prev
        self.d_meas = self.d_prev # same same
        self.y_meas = self.y_prev
        self.sp_meas = self.sp_prev
        
        # Initial U predictions
        self.u_predict = Tools.vector_appending(Begin_cond.U_init, Controller.cont_H)
        #----------------------------------
        
        # Hisory for all variables
        #----------------------------------
        # u_all -> matrix, rows = time, cols = vars
        self.u_all = np.matrix(np.matlib.repmat(self.u_meas, self.hist, 1)) 
        self.d_all = np.matrix(np.matlib.repmat(self.d_meas, self.hist, 1)) 
        self.y_all = np.matrix(np.matlib.repmat(self.y_meas, self.hist, 1))
        self.sp_all = np.matrix(np.matlib.repmat(self.sp_prev, self.hist, 1))
        #----------------------------------
        
        
        # Initialise plots        
        self.line_lstCVs_past = []
        self.line_lstCVs_future = []
        self.line_lstSPs_past = []
        self.line_lstSPs_future = []
        self.line_lstMVs_past = []
        self.line_lstMVs_future = []
        self.line_lstEps_high =[]
        self.line_lstEps_low =[]
        self.line_lstCV_high =[]
        self.line_lstCV_low =[]
        
        self.init_plots()
        
    def init_plots(self):
        # initial plots       
        hist = self.hist
        
        State = self.State_pred
        Controller = self.Controller
        Tuning = self.Controller.Tuning
        
        pred_H = self.Controller.pred_H
        cont_H = self.Controller.cont_H
 
        no_mvs = self.no_mvs
        no_dvs = self.no_dvs
        no_cvs = self.no_cvs
        
        sp_prev = self.sp_prev
        y_all = self.y_all
        u_all = self.u_all
        sp_all = self.sp_all
        
        for i in range(no_cvs):
            plt.subplot(2, np.max([no_mvs, no_cvs]), i+1)
            line1, = plt.plot(y_all[:,i], 'k')
            line2, = plt.plot(range(hist, hist+pred_H), State.state[0:pred_H], 'r')
            line3, = plt.plot(sp_all[:,i], 'b:')
            line4, = plt.plot(range(hist, hist+pred_H), np.repeat(sp_prev[i], pred_H), 'b:')
            line5, = plt.plot(range(hist, hist+pred_H), np.repeat(Tuning.y_high[i], pred_H), 'r--')
            line6, = plt.plot(range(hist, hist+pred_H), np.repeat(Tuning.y_low[i], pred_H), 'r--')
            line7, = plt.plot(range(hist, hist+pred_H), np.repeat(Tuning.y_low[i], pred_H), 'g--')
            line8, = plt.plot(range(hist, hist+pred_H), np.repeat(Tuning.y_low[i], pred_H), 'g--')
            
            plt.plot([hist, hist], [-3, 3], 'k--')
            plt.xlim([0, hist+pred_H])
            plt.ylim([-3,  3])
            self.line_lstCVs_past.append(line1)
            self.line_lstCVs_future.append(line2)
            self.line_lstSPs_past.append(line3)
            self.line_lstSPs_future.append(line4)
            self.line_lstEps_high.append(line5)
            self.line_lstEps_low.append(line6)
            self.line_lstCV_high.append(line7)
            self.line_lstCV_low.append(line8)
            plt.grid()

        for j in range(no_mvs):
            plt.subplot(2, np.max([no_mvs, no_cvs]), np.max([no_mvs, no_cvs]) + j+1)
            line1, = plt.plot(u_all[:,0], 'k')
            line2, = plt.plot(range(hist, hist+cont_H), self.u_predict[(i*cont_H):((i+1)*cont_H) ], 'b')
            plt.plot(range(hist+pred_H), np.repeat( Tuning.u_low[j], hist+pred_H), 'g')
            plt.plot(range(hist+pred_H), np.repeat( Tuning.u_high[j], hist+pred_H), 'g')
            plt.plot([hist, hist], [-3, 3], 'k--')
            plt.xlim([0, hist+pred_H])
            plt.ylim([-3,  3])
            self.line_lstMVs_past.append(line1)
            self.line_lstMVs_future.append(line2)    
            plt.grid()
        
    
    def Controller_execute(self, ystate, sim_sp):
        pred_H = self.Controller.pred_H
        no_eps = len( self.Controller.Tuning.y_high ) + len( self.Controller.Tuning.y_low ) 
        # update controller 
        #---------------------------------------------
        self.Controller.update( my(ystate, pred_H, 2), self.u_meas, Tools.vector_appending(sim_sp, pred_H) )

        # Solve QP - i.e. implement control move       
        optimal_mv = qp( matrix(self.Controller.Hessian), matrix(self.Controller.Gradient), matrix(self.Controller.U_lhs), matrix(self.Controller.U_rhs) )
        self.u_predict = np.array(optimal_mv['x'])[:-no_eps] # Extract x from qp atributes
        self.eps = np.array(optimal_mv['x'])[-no_eps:]
        
        self.u_current = np.ravel( Tools.extract_mv( self.u_predict, self.Controller.cont_H ) ) # Extract only mv moves that will be implemented  

        # calculate closed loop prediction - Strictly this is not correct as disturbance is not included
        self.Y_CL = self.Controller.Su*self.u_predict + self.Controller.Y_openloop
        #---------------------------------------------
    
    def update_plot(self, sp_meas, it):
        hist = self.hist
        pred_H = self.Controller.pred_H
        cont_H = self.Controller.cont_H
        no_cvs = self.no_cvs
        no_mvs = self.no_mvs
        Tuning = self.Controller.Tuning
        eps = self.eps
        
        # update plot
        #---------------------------------------------
        for j in range(no_cvs):
            self.line_lstCVs_past[j].set_ydata(self.y_all[-hist:,j]) 
            self.line_lstCVs_future[j].set_ydata( self.Y_CL[(j*pred_H):( (j+1)*pred_H )] ) 
            self.line_lstSPs_past[j].set_ydata(self.sp_all[-hist:,j]) 
            self.line_lstSPs_future[j].set_ydata( np.repeat(sp_meas[j], pred_H) )
            self.line_lstEps_high[j].set_ydata( np.repeat(Tuning.y_high[j] - eps[j], pred_H) )
            self.line_lstEps_low[j].set_ydata( np.repeat(Tuning.y_low[j] + eps[no_cvs + j], pred_H) )
            self.line_lstCV_high[j].set_ydata( np.repeat(Tuning.y_high[j], pred_H) )
            self.line_lstCV_low[j].set_ydata( np.repeat(Tuning.y_low[j], pred_H) )
            
        for k in range(no_mvs):
            self.line_lstMVs_past[k].set_ydata(self.u_all[-hist:,k]) 
            self.line_lstMVs_future[k].set_ydata( np.cumsum( self.u_predict[(k*cont_H):( (k+1)*cont_H )] ) + self.u_meas[k] - self.u_predict[(k*cont_H)])
        #---------------------------------------------
        plt.pause(1e-5)
        plt.savefig('test' + str(it) + '.png',dpi=120)
        plt.draw
    
    def update_inputs(self, u_current, sim_sp, plant_d):
        # save all the mv movements - past till now    
        self.u_all = np.concatenate( [self.u_all, (self.u_all[-1,:] + u_current)], axis = 0)
        self.u_meas = np.ravel( self.u_all[-1,:] )
        self.u_prev = np.ravel( self.u_all[-2,:] )
        
        self.d_all = np.concatenate( [self.d_all, plant_d ], axis = 0)         
        self.d_meas = np.ravel( self.d_all[-1,:] )
        self.d_prev = np.ravel( self.d_all[-2,:] )

        self.sp_all = np.concatenate( [self.sp_all, np.matrix(sim_sp)], axis=0)
        self.sp_meas = np.ravel( self.sp_all[-1,:] )
        
    def update_outputs(self, y):
        self.y_meas = np.ravel( y[-1,:] )
        self.y_all = np.concatenate( [self.y_all, ( self.y_all[0,:] + self.y_meas)], axis = 0)
            
    def run_sim(self, sim_no, sim_sp, sim_dist):
        pred_H = self.Controller.pred_H
        
        for i in range(0, sim_no):
            # update state
            ystate = self.State_pred.update_state(self.y_meas, np.concatenate([self.u_meas , self.d_meas]) - np.concatenate([self.u_prev, self.d_prev]))
            
            self.Controller_execute(ystate, sim_sp[i,:])  
            
            self.update_inputs(self.u_current, sim_sp[i,:], sim_dist[i,:])
            
            # Perform simulation - Implement move on plant 
            #---------------------------------------------
            t, y = self.State_pred.Model.simulate( np.concatenate( [ self.u_all, self.d_all ], axis=1 ) )
            #---------------------------------------------           
            
            self.update_outputs(y)                    
            
            self.update_plot(sim_sp[i,:], i)
             
