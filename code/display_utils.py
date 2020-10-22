# -*- coding: utf-8 -*-
"""
    This file implements the functions required to visualize data.
"""

# Required imports
import numpy as np
import matplotlib.pyplot as plt

def view_sample(signals, labels = np.zeros(1), length = 500, start = 0):
    """
        view_sample
        Plots a cropped sample of the x-axis accelerations of both feet along with the given events

    Parameters
    ----------
    signals : n x 2 numpy array
        a-axis accelerations of both feet.
    labels : numpy array, optional
        Event labels. The default is np.zeros(1).
    length : int, optional
        length of the sample. The default is 500.
    start : int, optional
        Starting point of the sample. The default is 0.

    Returns
    -------
    None.

    """
    plt.figure()
    
    # Left foot
    plt.subplot(211)
    # Accelerations
    plt.plot(signals[start:start + length, 0], 'b', label = 'X axis acceleration')
    # Vertical lines for the events
    if not np.sum(labels) == 0:
        for xc in np.where(labels[start:start + length, 0] == 1)[0]:
            plt.axvline(x = xc, color = 'r')
        for xc in np.where(labels[start:start + length, 1] == 1)[0]:
            plt.axvline(x = xc, color = 'g')
    plt.title('Left foot')
    plt.legend()
        
    # Right foot
    plt.subplot(212)
    # Accelerations
    plt.plot(signals[start:start + length, 1], 'b', label = 'X axis acceleration')
    # Vertical lines for the events
    if not np.sum(labels) == 0:
        for xc in np.where(labels[start:start + length, 2] == 1)[0]:
            plt.axvline(x = xc, color = 'r')
        for xc in np.where(labels[start:start + length, 3] == 1)[0]:
            plt.axvline(x = xc, color = 'g')
    plt.title('Right foot')
    plt.legend()
    
    plt.show()
    
def compare_labels(y, out):
    
    signal =  np.zeros(y.shape[1])
    
    plt.figure()
    
    # Left foot
    plt.subplot(211)
    # Sample signal
    plt.plot(signal, 'w')
    # Vertical lines for the events
    for xc in np.where(y[0, :] == 1)[0]:
        plt.axvline(x = xc, color = 'b')
    for xc in np.where(y[1, :] == 1)[0]:
        plt.axvline(x = xc, color = 'c')
    for xc in np.where(y[2, :] == 1)[0]:
        plt.axvline(x = xc, color = 'r')
    for xc in np.where(y[3, :] == 1)[0]:
        plt.axvline(x = xc, color = 'y')
    plt.title('Ground Truth')
    plt.legend()
        
    # Right foot
    plt.subplot(212)
    # Sample signal
    plt.plot(signal, 'w')
    # Vertical lines for the events
    for xc in np.where(out[0, :] == 1)[0]:
        plt.axvline(x = xc, color = 'b')
    for xc in np.where(out[1, :] == 1)[0]:
        plt.axvline(x = xc, color = 'c')
    for xc in np.where(out[2, :] == 1)[0]:
        plt.axvline(x = xc, color = 'r')
    for xc in np.where(out[3, :] == 1)[0]:
        plt.axvline(x = xc, color = 'y')
    plt.title('Predictions')
    plt.legend()
    
    plt.show()