# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 10:37:27 2020

@author: PCyax
"""

from pathlib import Path
from io_utils import extract_data
from LSTM_model import LSTM
import torch

root_path = Path.cwd().parent
model_name = '1W.net'
model_path = root_path / 'models' / 'LSTM' / 'saved' / model_name

# Load previously trained model
with open(model_path, 'rb') as f:
    checkpoint = torch.load(f)
     
net = LSTM(batch_size = checkpoint['batch_size'], seq_length = checkpoint['sequence_length'],
          hidden_dim = checkpoint['hidden_dim'], n_layers = checkpoint['n_layers'], 
          drop_prob = checkpoint['drop_prob'], pos_weight = checkpoint['pos_weight'])

net.load_state_dict(checkpoint['state_dict'])

# Extract test data

_, _, test_data = extract_data('Walk', 0)

net.eval_model(test_data)