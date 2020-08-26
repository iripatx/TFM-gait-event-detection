# -*- coding: utf-8 -*-
"""
    This file implements the functions required to visualize data.
"""

# Required imports
import numpy as np
import matplotlib.pyplot as plt

def view_sample(signals, labels = np.zeros(1), length = 500, start = 0):
    
    plt.figure()
    
    plt.subplot(211)
    plt.plot(signals[start:start + length, 0], 'b', label = 'X axis acceleration')
    if not np.sum(labels) == 0:
        for xc in np.where(labels[start:start + length, 0] == 1)[0]:
            plt.axvline(x = xc, color = 'r')
        for xc in np.where(labels[start:start + length, 1] == 1)[0]:
            plt.axvline(x = xc, color = 'g')
    plt.title('Left foot')
    plt.legend()
        
    plt.subplot(212)
    plt.plot(signals[start:start + length, 1], 'b', label = 'X axis acceleration')
    if not np.sum(labels) == 0:
        for xc in np.where(labels[start:start + length, 2] == 1)[0]:
            plt.axvline(x = xc, color = 'r')
        for xc in np.where(labels[start:start + length, 3] == 1)[0]:
            plt.axvline(x = xc, color = 'g')
    plt.title('Right foot')
    plt.legend()
    
    plt.show()