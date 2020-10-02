# -*- coding: utf-8 -*-
"""
    This script implements an event detection method based on a TCN neural
    network architecture.
"""

# Required imports
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import io_utils
import matplotlib.pyplot as plt
from TCN_model import TCN
import random

# Defining global paths
root_path = Path.cwd().parent
data_path = root_path / 'data' / 'MAREA_dataset'

# Defining indoor tests
indoor_tests = ['treadmill_flat', 'treadmill_slope', 'flat_space']

def detect_events(trained_model = None):
    
    # Number of epochs
    epochs = 30
    
    # Input dimension
    input_dim = 12
    
    # Checking if ther's a previously trained model
    if trained_model is not None:
        
        # Defining paths
        model_path = root_path / 'models' / 'TCN' / trained_model
        
        # Load previously trained model
        with open(model_path, 'rb') as f:
            checkpoint = torch.load(f)
     
        net = TCN(batch_size=checkpoint['batch_size'],seq_length=checkpoint['sequence_length'],
                      epochs=epochs, input_dim = input_dim, drop_prob=0.3, num_labels=4)
        
        net.load_state_dict(checkpoint['state_dict'])
        
        print('Trained model loaded')
        
    else:
        
        net = TCN(batch_size = 20, seq_length = 250, epochs = 30, 
                   input_dim = input_dim, num_labels = 4)
        
    data, val_data = extract_data()  
    net.trainloop(data, val_data)

    
def extract_data():
    """
    extract_data
        Extracts only the necessary data and arranges it for the detection.
        Each instace will be stored as a n x 8 numpy array where:
            - Columns 0-3 are the accelerations
            - Columns 4-7 are the labels


    Returns
    -------
    data : Python list of numpy array
        List containing all instances of data

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
            
            left_foot = np.concatenate((db['indoors'][i][test]['LF'][:, 0].reshape(-1, 1),
                                       db['indoors'][i][test]['LF'][:, 1].reshape(-1, 1),
                                       db['indoors'][i][test]['LF'][:, 2].reshape(-1, 1)), axis = 1)
            right_foot = np.concatenate((db['indoors'][i][test]['RF'][:, 0].reshape(-1, 1),
                                       db['indoors'][i][test]['RF'][:, 1].reshape(-1, 1),
                                       db['indoors'][i][test]['RF'][:, 2].reshape(-1, 1)), axis = 1)
            waist = np.concatenate((db['indoors'][i][test]['Waist'][:, 0].reshape(-1, 1),
                                       db['indoors'][i][test]['Waist'][:, 1].reshape(-1, 1),
                                       db['indoors'][i][test]['Waist'][:, 2].reshape(-1, 1)), axis = 1)
            wrist = np.concatenate((db['indoors'][i][test]['Wrist'][:, 0].reshape(-1, 1),
                                       db['indoors'][i][test]['Wrist'][:, 1].reshape(-1, 1),
                                       db['indoors'][i][test]['Wrist'][:, 2].reshape(-1, 1)), axis = 1)
            labels = db['indoors'][i][test]['labels']
            data.append(np.concatenate((left_foot, right_foot, waist, wrist, labels), axis = 1))
            
    # OUTDOORS DATA
    for i in np.arange(1, db.shape[0] + 1):
        
        # Skipping NaN values
        if pd.isna(db['outdoors'][i]): continue
        
        # Extracting and storing foot accelerations on the X axis and its labels
        left_foot = np.concatenate((db['outdoors'][i]['street']['LF'][:, 0].reshape(-1, 1),
                                   db['outdoors'][i]['street']['LF'][:, 1].reshape(-1, 1),
                                   db['outdoors'][i]['street']['LF'][:, 2].reshape(-1, 1)), axis = 1)
        right_foot = np.concatenate((db['outdoors'][i]['street']['RF'][:, 0].reshape(-1, 1),
                                   db['outdoors'][i]['street']['RF'][:, 1].reshape(-1, 1),
                                   db['outdoors'][i]['street']['RF'][:, 2].reshape(-1, 1)), axis = 1)
        waist = np.concatenate((db['outdoors'][i]['street']['Waist'][:, 0].reshape(-1, 1),
                                   db['outdoors'][i]['street']['Waist'][:, 1].reshape(-1, 1),
                                   db['outdoors'][i]['street']['Waist'][:, 2].reshape(-1, 1)), axis = 1)
        wrist = np.concatenate((db['outdoors'][i]['street']['Wrist'][:, 0].reshape(-1, 1),
                                   db['outdoors'][i]['street']['Wrist'][:, 1].reshape(-1, 1),
                                   db['outdoors'][i]['street']['Wrist'][:, 2].reshape(-1, 1)), axis = 1)
        labels = db['outdoors'][i]['street']['labels']
        data.append(np.concatenate((left_foot, right_foot, waist, wrist, labels), axis = 1))
        
        
    # One out for validation
    val_data = data[3:5]
    del data[3:5]
    val_data.append(data[34])
    del data[34]
        
    return data, val_data