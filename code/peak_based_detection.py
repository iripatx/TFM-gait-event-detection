# -*- coding: utf-8 -*-
"""
This file implements the peak-based event detection using linear accelerometers 
described by Jan M. Jasiewicz and his colleagues  in "Gait event detection 
using linear accelerometers or angular velocity transducers in able-bodied and 
spinal-cord injured individuals" at
https://www.sciencedirect.com/science/article/pii/S0966636206000129
"""

# required imports
from pathlib import Path
from math import isnan
import numpy as np
import pandas as pd
import io_utils
import display_utils



# Defining global paths
root_path = Path.cwd().parent
data_path = root_path / 'data' / 'MAREA_dataset'

# indoor tests
indoor_tests = ['treadmill_flat', 'treadmill_slope', 'flat_space']

"""
    detect_events
    Detects peaks is movement signals and and classifies them as events
    
    Arguments:
        data(list): a list containing a numpy array with the data.
"""
#def detect_events():
    
#    for instance in data:
        
        

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
    db = io_utils.get_database()
    
    # Declarating list to store the data
    data = []
    
    # Extracting data
    # INDOORS DATA
    for i in np.arange(1, db.shape[0] + 1):
        print(i)
        
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
        print(i)
        
        # Skipping NaN values
        if pd.isna(db['outdoors'][i]): continue
        
        # Extracting and storing foot accelerations on the X axis and its labels
        left_foot = db['outdoors'][i]['street']['LF'][:, 0].reshape(-1, 1)
        right_foot = db['outdoors'][i]['street']['RF'][:, 0].reshape(-1, 1)
        labels = db['outdoors'][i]['street']['labels']
        data.append(np.concatenate((left_foot, right_foot, labels), axis = 1))
        
    display_utils.view_sample(data[0][:, 0:2], data[0][:, 2:6])
    return data
    