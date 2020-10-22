# -*- coding: utf-8 -*-
"""
    This script implements an event detection method based on a TCN neural
    network architecture.
"""

# Required imports
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from TCN_model import TCN
import random
from io_utils import extract_data

# Defining global paths
root_path = Path.cwd().parent
data_path = root_path / 'data' / 'MAREA_dataset'

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
        
        net = TCN(batch_size = 40, seq_length = 125, hidden_dim = 150, levels = 4, 
                drop_prob = 0.5, kernel_size = 7, pos_weight = 10, epochs = 20)

        
    data, val_data, test_data = extract_data('Run', 9)  
    net.trainloop(data, val_data)