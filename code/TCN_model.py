# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:51:07 2020

@author: PCyax
"""

import numpy as np
import torch
from torch import nn
from torch import optim
import time
import matplotlib.pyplot as plt
from pathlib import Path
from TCN.tcn import TemporalConvNet
# import torch.nn.functional as F
    
class TCN_model(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x).transpose(1, 2)
        output = self.linear(output).double()
        return output

class TCN(TCN_model):
    
    def __init__(self, batch_size, seq_length, epochs, input_dim, num_labels,  lr = 0.001, clip = 5, 
                 hidden_dim = 256, levels = 5, drop_prob=0.5, kernel_size = 5, 
                 pos_weight = 5, starting_epoch = 0):
        
        self.num_channels = [hidden_dim] * levels
        
        super().__init__(input_dim, num_labels, self.num_channels, kernel_size, drop_prob)
        
        # Storing arguments as class attributes
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.input_dim = input_dim
        self.lr = lr
        self.clip = clip
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.pos_weight = pos_weight
        self.starting_epoch = starting_epoch
        
        # Defining the remaining attributes
        self.optim = optim.Adam(self.parameters(), self.lr)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight = torch.ones(4) * pos_weight)
        
        # Lists to store losses
        self.loss_during_training = []
        self.val_loss_during_training = []
        
        # Setting device on GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print()
        print('Using device:', self.device)
        
        #Additional Info when using cuda
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        
        print()
        
        self.to(self.device)
        
        # Training mode by default
        self.train()
        
        
    def get_batches(self, instance):
        """
        Generates batches of single time samples for the network

        Parameters
        ----------
        instance : numpy array

        Returns
        -------
        batches: list of numpy array

        """
        
        # Trim signal length to adjust to batch size
        total_batch_size = self.batch_size * self.seq_length
        total_batches = instance.shape[0] // total_batch_size
        instance = instance[:total_batches * total_batch_size, :]
        
        # Splitting trimmed array into batches
        batches = np.split(instance, total_batches)
        
        # 
        for batch in batches:
            
            reshaped = np.zeros([self.batch_size, self.input_dim + self.num_labels, self.seq_length])
            
            for i, start in enumerate(np.arange(0, total_batch_size, self.seq_length)):
                
                reshaped[i, :, :] = batch[start:start + self.seq_length, :].T
                
            x = reshaped[:, 0:self.input_dim, :]
            y = reshaped[:, self.input_dim:, :]
            
            yield x, y
            
    def save_model(self,epoch):

        # Defining paths
        root_path = Path.cwd().parent
        model_name = 'tcn_' + str(epoch + 1) + '_epoch.net'
        model_path = root_path / 'models' / 'TCN' / model_name
        
        checkpoint = {'hidden_dim': self.hidden_dim,
                      'input_dim': self.input_dim,
                      'batch_size':self.batch_size,
                      'sequence_length': self.seq_length,
                      'state_dict': self.state_dict()}
 
        with open(model_path, 'wb') as f:
            torch.save(checkpoint, f)
            
            
    def trainloop(self, data, val_data):
        """
        Network training method

        Parameters
        ----------
        data : list of numpy array
            Data to be fed to the network.

        Returns
        -------
        None.

        """
        
        for e in range(self.epochs):
            
            # Storing current time
            epoch_start_time = time.time()
            
            # TRAINING
            
            # For each instance of data
            for number, instance in enumerate(data):
                
                running_loss = 0.
                
                counter = 0.
                
                # Storing current time
                batch_start_time = time.time()
                
                # Each instance has multiple batches
                for x, y in self.get_batches(instance):
    
                    
                    counter = counter + 1.
                    
                    # Convert data to tensor
                    x, y = torch.from_numpy(x).float().to(self.device), torch.from_numpy(y).float().to(self.device)
                    
                    # Resetting gradients
                    self.optim.zero_grad()
                    
                    # Compute output
                    out = self.forward(x)
                    
                    # Compute loss
                    loss = self.criterion(out, y.transpose(1, 2))
                    
                    # Compute gradients
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    nn.utils.clip_grad_norm_(self.parameters(), self.clip)
                    
                    self.optim.step() 
                    
                    # Storing current loss
                    running_loss += loss.item()
                    
                self.loss_during_training.append(running_loss/counter)
                            
                print("Instance %d processed. Training loss: %f, Time: %f seconds" 
                  %(number,self.loss_during_training[-1], (time.time() - batch_start_time))) 
            
            # VALIDATION
            with torch.no_grad():
                
                # Switching to evaluation mode
                self.eval()
                
                for number, instance in enumerate(val_data):
                    
                    val_running_loss = 0.
                    val_counter = 0.
                 
                    for x, y in self.get_batches(instance):
                        
                        val_counter = val_counter + 1.
                        
                        # Convert data to tensor
                        x, y = torch.from_numpy(x).float().to(self.device), torch.from_numpy(y).float().to(self.device)
                        
                        # Resetting gradients
                        self.optim.zero_grad()
                        
                        # Compute output
                        out = self.forward(x)
                        
                        # Compute loss
                        loss = self.criterion(out, y.transpose(1, 2))
                        
                        # Storing current loss
                        val_running_loss += loss.item()
                            
                    self.val_loss_during_training.append(val_running_loss/val_counter)
                
                # Switch back to train mode
                self.train()
                
            print("Epoch %d processed. Training loss: %f, Validation loss : %F, Time: %f seconds" 
                    %(e + 1, sum(self.loss_during_training[-len(data):])/len(data), 
                      sum(self.val_loss_during_training[-len(val_data):])/len(val_data),
                      (time.time() - epoch_start_time))) 
            
            if e % 1 == 0:
                self.save_model(e)
                print('Model saved')