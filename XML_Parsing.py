# -*- coding: utf-8 -*-
"""
Created on who cares

@author: r
"""

import xml.etree.ElementTree as ET

import numpy as np
from numpy import matlib

import scipy.signal as sig

import Model as LTImodel
import tuning as tune
import MPCcontroller as MPC # MPC controller
import InitCond as initial # init conditions for controller

# Parsing Conroller
#------------------------------------
def Controller_from_XML(path, Begin_cond):
    tree = ET.parse(path)
    root = tree.getroot()

    pred_H = int( root.findall('pred_H')[0].text )
    cont_H = int( root.findall('cont_H')[0].text )
    deltaT = int( root.findall('deltaT')[0].text )

    G_model = Model_from_root(root, 'Process_Model')

    Tuning = Tuning_from_root(root)
         
    Controller = MPC.Control(G_model, deltaT, pred_H, cont_H, Tuning, Begin_cond)
    
    return Controller

def Tuning_from_root(root):
    Tuning_el = root.findall('Tuning')[0]

    # Function to extract u_tune from root element
    u_tune = get_array_from_element(Tuning_el.findall('Move_Suppression'))
    y_tune = get_array_from_element(Tuning_el.findall('SP_Tracking'))
    pred_H = int( root.findall('pred_H')[0].text )
    cont_H = int( root.findall('cont_H')[0].text )

    MV_Lim_el = Tuning_el.findall('MV_Limits')[0]
    u_high = get_array_from_element(MV_Lim_el.findall('MV_High'))
    u_low = get_array_from_element(MV_Lim_el.findall('MV_Low'))

    MV_ROC_el = Tuning_el.findall('MV_ROC')[0]
    u_roc_up = get_array_from_element(MV_ROC_el.findall('ROC_Up'))
    u_roc_down = get_array_from_element(MV_ROC_el.findall('ROC_Down'))

    CV_Lim_el = Tuning_el.findall('CV_Limits')[0]
    y_high = get_array_from_element(CV_Lim_el.findall('CV_High'))
    y_low = get_array_from_element(CV_Lim_el.findall('CV_Low'))

    eps_el = Tuning_el.findall('CV_Slack')[0]
    eps_high = get_array_from_element(eps_el.findall('High'))
    eps_low = get_array_from_element(eps_el.findall('Low'))
    
    Tuning = tune.tuning(u_tune, y_tune, pred_H, cont_H, u_low, u_high, y_low, y_high, u_roc_up, u_roc_down, eps_high, eps_low)
    return Tuning

def get_array_from_element( el ):
    D = list( el )[0]
    U = np.array( [float(a.text) for a in D.iter('Value')] )     
    return U
#------------------------------------

# Initial Conditions
#------------------------------------
def Init_cond_XML(Cont_path, Init_path):
    tree = ET.parse(Cont_path)
    root = tree.getroot()

    tree1 = ET.parse(Init_path)
    root1 = tree1.getroot()
    
    pred_H = int( root.findall('pred_H')[0].text )
    cont_H = int( root.findall('cont_H')[0].text )
    init_SP = get_array_from_element( root1.findall('SP') )
    init_PV = get_array_from_element( root1.findall('PV') )
    init_MV = get_array_from_element( root1.findall('MV') )
    init_DV = get_array_from_element( root1.findall('DV') )
    
    BeginCond = initial.init_cond(pred_H, cont_H, init_SP, init_PV, init_MV, init_DV)
    return BeginCond
#------------------------------------

# Parsing Model
#------------------------------------
def Model_from_XML(path, model_type):
    # Read Model from XML
    tree = ET.parse(path)
    root = tree.getroot()

    mdl_el = root.findall(model_type)
    mdl_lst = get_process_model( mdl_el[0] )

    pred_H = int( root.findall('pred_H')[0].text )
    deltaT = int( root.findall('deltaT')[0].text )
    
    G_model = LTImodel.Mod(mdl_lst, deltaT, pred_H)
    return G_model

def Model_from_root(root, model_type):
    # Root is directly passed here as the controller already red it
    mdl_el = root.findall(model_type)
    mdl_lst = get_process_model( mdl_el[0] )

    pred_H = int( root.findall('pred_H')[0].text )
    deltaT = int( root.findall('deltaT')[0].text )
    
    G_model = LTImodel.Mod(mdl_lst, deltaT, pred_H)
    return G_model

def get_tf(tf_element):
    num = list(tf_element)[0]
    den = list(tf_element)[1]
    
    num_vec = np.array( [float(a.text) for a in num.iter('Value')] )
    den_vec = np.array( [float(a.text) for a in den.iter('Value')] )
    return sig.lti(num_vec, den_vec)

def get_cv_tf(CV_element):
    tf_lst = CV_element.findall('tf')
    
    Model_lst = []
    
    for i in range(len(tf_lst)):
        Model_lst.append( get_tf(tf_lst[i]) )
    return Model_lst

def get_process_model(Proces_mdl_element):
    CV_list = Proces_mdl_element.findall('CV')
    
    tmp_model = []
    
    for i in range(len(CV_list)):
        tmp_model.append( get_cv_tf(CV_list[i]) )
        
    row, col = np.shape( tmp_model )
    Process_model = matlib.repmat(None, row, col)
    
    for i in range(row):
        for j in range(col):
            Process_model[i][j] = tmp_model[i][j]
    
    return Process_model # list containing all the process models
#------------------------------------
