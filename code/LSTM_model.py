# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
from torch import optim
import time
import matplotlib.pyplot as plt
from pathlib import Path

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
# torch.backends.cudnn.enabled = False

class LSTM_model(nn.Module):
    
    def __init__(self, seq_length, input_dim, num_labels, hidden_dim=256, n_layers=2, 
                 drop_prob=0.5, bidirectional = True):      
        
        super().__init__()
        
        # Storing arguments as class attributes
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.bidirectional = bidirectional
             
        # LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, dropout=self.drop_prob, 
                            batch_first=True, bidirectional = self.bidirectional)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=self.drop_prob)
        
        # Fully connected layer (size 2 * hidden_dim because it's bidirectional)
        self.fc = nn.Linear(self.hidden_dim * 2, self.num_labels)
        
        # Sigmoid for predictions
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, h = None):
    
        if (h==None):
            # If no initial hidden_state and memory are provided, they are set to 0
            r_output, hidden = self.lstm(x)  
        
        else:
            r_output, hidden = self.lstm(x,h)     
        
        # Pass through a dropout layer
        out = self.dropout(r_output)
        
        # Put x through the fully-connected layer
        out = self.fc(out)
        
        # Return the final output and the hidden state
        return out, hidden
    
    def predict(self, x, h = None):
    
        if (h==None):
            # If no initial hidden_state and memory are provided, they are set to 0
            r_output, hidden = self.lstm(x)  
        
        else:
            r_output, hidden = self.lstm(x,h)     
        
        # Pass through a dropout layer
        out = self.dropout(r_output)
        
        # Put x through the fully-connected layer
        out = self.fc(out)
        
        out = self.Sigmoid(out)
        
        # Return the final output and the hidden state
        return round(out)
    
    
class LSTM(LSTM_model):
    
    def __init__(self, batch_size, seq_length, hidden_dim, n_layers, pos_weight, drop_prob,
                 num_labels = 4, input_dim = 12, lr = 0.001, clip = 5,  epochs = 30, bidirectional = True,
                 starting_epoch = 0):

        super().__init__(seq_length, input_dim, num_labels, hidden_dim, n_layers, drop_prob, bidirectional)
        
        # Storing arguments as class attributes
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.epochs = epochs
        self.lr = lr
        self.clip = clip
        self.num_labels = num_labels
        self.n_layers = n_layers
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
            
            batch =  batch.reshape(self.batch_size, self.seq_length, -1)
            
            x = batch[:, :, 0:self.input_dim]
            y = batch[:, :, self.input_dim:]
            
            yield x, y
            
    def save_model(self,epoch):

        # Defining paths
        root_path = Path.cwd().parent
        model_name = 'lstm_' + str(epoch + 1) + '_epoch.net'
        model_path = root_path / 'models' / 'LSTM' / model_name
        
        checkpoint = {'hidden_dim': self.hidden_dim,
              'n_layers': self.n_layers,
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
            
            # Shuffling instances of data
            random.shuffle(data)
            
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
                    out, _ = self.forward(x)
                    
                    # Compute loss
                    loss = self.criterion(out, y)
                    
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
                        out, _ = self.forward(x)
                        
                        # Compute loss
                        loss = self.criterion(out, y)
                        
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
                