# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:40:58 2016

@author: r

NOTE ----- THIS HAS BEEN DISCONTINUED, SCALING MUST BE DONE EXTERNALLY
"""
import numpy as np

class cont_ranges(object):
    def __init__(self, u_range, y_range):
        self.u_range = u_range
        self.y_range = y_range

        self.u_factor = scale(u_range)
        self.y_factor = scale(y_range)
        self.scale_mat()

    def scale_mat(self):
        r_u = len(self.u_factor)
        r_y = len(self.y_factor)

        scale_matrix = np.zeros([r_y, r_u])
        for col in range(0, r_u):
            for row in range(0, r_y):
                scale_matrix[row][col] = self.y_factor[row]/self.u_factor[col]

        self.scale_matrix = scale_matrix
		
def scale(range_array):
    r, c = np.shape(range_array)

    scale_array = np.zeros(r)
    for i in range(0, r):
        scale_array[i] = range_array[i][1] - range_array[i][0]

    return np.ravel(scale_array)
