# -*- coding: utf-8 -*-
"""
    This file implements preprocessing functions on the data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sys import exit

# indoor tests
indoor_tests = ['treadmill_flat', 'treadmill_slope', 'flat_space']
# Accelerometer positions
acc_locations = ['LF', 'RF', 'Waist', 'Wrist']

def preprocess_data(db):
    
    # INDOORS DATA
    for i in np.arange(1, db.shape[0] + 1):
        
        # Skipping NaN values
        if pd.isna(db['indoors'][i]): continue
        
        for test in indoor_tests:
            
            # Normalize signals
            # for location in acc_locations:
            #     db['indoors'][i][test][location] = normalize(amplify_windows(db['indoors'][i][test][location]), axis = 0)
            
            # Amplify event windows
            db['indoors'][i][test]['labels'] = amplify_windows(db['indoors'][i][test]['labels'])
            
    # OUTDOORS DATA
    for i in np.arange(1, db.shape[0] + 1):
        
        # Skipping NaN values
        if pd.isna(db['outdoors'][i]): continue
    
        # Normalize signals
        # for location in acc_locations:
        #     db['outdoors'][i]['street'][location] = normalize(amplify_windows(db['outdoors'][i]['street'][location]), axis = 0)
        
        # Amplify event windows
        db['outdoors'][i]['street']['labels'] = amplify_windows(db['outdoors'][i]['street']['labels'])
        
    return db
    
    
def amplify_windows(labels, window_size = 9):
    
    # If the window size isn't even, end script
    if (window_size % 2 == 0):
        exit('ERROR: Window size must be an even number')
    
    # Computing window margin's size
    margin = int(window_size / 2)
    
    for i in np.arange(labels.shape[1]):
        
        for event in np.where(labels[:, i] == 1)[0]:
            
            # Checking margins to avoid dimension errors
            left_margin = max(event - margin, 0)
            right_margin = min(event + margin + 1, labels.shape[0])
            
            # Amplifying window
            labels[left_margin:right_margin, i] = 1
            
    return labels