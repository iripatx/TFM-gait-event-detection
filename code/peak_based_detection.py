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



# Defining global paths
root_path = Path.cwd().parent
data_path = root_path / 'data' / 'MAREA_dataset'

indoor_tests = ['treadmill_flat', 'treadmill_slope', 'flat_space']

def extract_data():
    """
        extract_data
        Extracts only the necessary data and arranges it for the detection.
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
                
    