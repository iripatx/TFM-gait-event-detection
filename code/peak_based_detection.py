# -*- coding: utf-8 -*-
"""
This file implements the peak-based event detection using linear accelerometers 
by valley detection.
"""

# Required imports
from pathlib import Path
from math import isnan
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import io_utils
import display_utils
from preprocess_utils import amplify_windows

# Defining global paths
root_path = Path.cwd().parent
data_path = root_path / 'data' / 'MAREA_dataset'

# Defining indoor tests
indoor_tests = ['treadmill_flat', 'treadmill_slope', 'flat_space']


def detect_events(window_size = 500):
    """
        detect_events
        Detects peaks is movement signals and and classifies them as events
    """
    
    # Retrieving data
    data = extract_data()
    
    for instance in data:
        
        # Declaring predictions matrix
        predictions = np.zeros([instance.shape[0], 4])
        
        # Finding valleys and setting them as predictions
        for i, foot in enumerate(['left', 'right']):
            
            # Storing signal
            signal = instance[:, i]
            
            # Valleys are peaks in the inverse signals
            valleys = find_peaks(signal*(-1), distance = 30, width = 4)[0]
            
            # HS are the deep valleys, TOs are the shallow ones, using the mean of all valleys to pick them out
            # Setting up a search by windows
            for start in np.arange(0, instance.shape[0], window_size):
                
                end = min(start + window_size, instance.shape[0])

                current_valleys = valleys[(start <= valleys) & (valleys <= end)]
                
                hs = current_valleys[signal[current_valleys] < np.mean(signal[current_valleys])]
                to = current_valleys[signal[current_valleys] >= np.mean(signal[current_valleys])]
                
                # Marking predictions
                predictions[hs, 2 * i] = 1
                predictions[to, 2 * i + 1] = 1
            
        # Amplifying event windows
        predictions = amplify_windows(predictions)
        
        # Storing predictions
        data[data.index(instance)] = np.concatenate((instance, predictions), axis = 1)
        
    return data
            


def extract_data():
    """
        extract_data
        Extracts only the necessary data and arranges it for the detection.
        Each instace will be stored as a n x 6 numpy array where:
            - n is the signal's length
            - Column 1 is the left foot's acceleration signal in the X axis
            - Column 2 is the right foot's acceleration signal in the X axis
            - Columns 3 and 4 are the ground thruth labels for the left foot's 
                heel strike and toe off respectively
            - Columns 5 and 6 are the ground thruth labels for the left foot's 
                heel strike and toe off respectively
    """
    
    # Obtain database
    db = io_utils.get_database(preprocessed = True)
    
    # Declarating list to store the data
    data = []
    
    # Extracting data
    # INDOORS DATA
    for i in np.arange(1, db.shape[0] + 1):
        
        # Skipping NaN values
        if pd.isna(db['indoors'][i]): continue
        
    # Extracting and storing foot accelerations on the X axis and its labels
        for test in indoor_tests:
            
            left_foot = db['indoors'][i][test]['LF'][:, 0].reshape(-1, 1)
            right_foot = db['indoors'][i][test]['RF'][:, 0].reshape(-1, 1)
            labels = db['indoors'][i][test]['labels']
            data.append(np.concatenate((left_foot, right_foot, labels), axis = 1))
            
    # OUTDOORS DATA
    for i in np.arange(1, db.shape[0] + 1):
        
        # Skipping NaN values
        if pd.isna(db['outdoors'][i]): continue
        
        # Extracting and storing foot accelerations on the X axis and its labels
        left_foot = db['outdoors'][i]['street']['LF'][:, 0].reshape(-1, 1)
        right_foot = db['outdoors'][i]['street']['RF'][:, 0].reshape(-1, 1)
        labels = db['outdoors'][i]['street']['labels']
        data.append(np.concatenate((left_foot, right_foot, labels), axis = 1))
        
    return data
    