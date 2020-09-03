# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:00:01 2020

@author: PCyax
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

def compute_accuracy(data, num_labels = 4):
    
    accuracies = []
    
    for instance in data:
        
        instance_accuracies = []
        
        for i in np.arange(num_labels):
            
            instance_accuracies.append(accuracy_score(instance[:, 2 + i], instance[:, 2 + i + 4]))
            
        accuracies.append(np.mean(instance_accuracies))
        
    return np.mean(accuracies)

def compute_precision(data, num_labels = 4):
    
    precisions = []
    
    for instance in data:
        
        instance_precisions = []
        
        for i in np.arange(num_labels):
            
            instance_precisions.append(precision_score(instance[:, 2 + i], instance[:, 2 + i + 4]))
            
        precisions.append(np.mean(instance_precisions))
        
    return np.mean(precisions)

def compute_recall(data, num_labels = 4):
    
    recalls = []
    
    for instance in data:
        
        instance_recalls = []
        
        for i in np.arange(num_labels):
            
            instance_recalls.append(recall_score(instance[:, 2 + i], instance[:, 2 + i + 4]))
            
        recalls.append(np.mean(instance_recalls))
        
    return np.mean(recalls)

