# -*- coding: utf-8 -*-
"""
    This script implements some metrics to check various method's performance
    Implemnted metrics:
        - accuracy
        - precision
        - recall
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

def compute_batch_metrics(y_true, y_pred, num_labels = 4):
    """
    compute_accuracy
    Computes the mean accuracy of the predictions for the given dataset
    
    Arguments:
        data (list of numpy array): List of instances of data, including ground
        truth and predictions
        num_labels(int, default = 4): Number of classes
    """ 
    
    # Declarating list to store results
    acc = []
    pre = []
    rec = []
    
    for batch in np.arange(y_true.shape[0]):
        
        # Declarating list to store individual results
        batch_acc = []
        batch_pre = []
        batch_rec = []
        
        for label in np.arange(num_labels):
            
            # Computing and storing metrics for each class
            batch_acc.append(accuracy_score(y_true[batch, label, :], y_pred[batch, label, :]))
            batch_pre.append(precision_score(y_true[batch, label, :], y_pred[batch, label, :]))
            batch_rec.append(recall_score(y_true[batch, label, :], y_pred[batch, label, :]))
        
        # Storing mean results of the instance
        acc.append(np.mean(batch_acc))
        pre.append(np.mean(batch_pre))
        rec.append(np.mean(batch_rec))
        
    # Returning mean of all results
    return np.mean(acc), np.mean(pre), np.mean(rec)

def compute_accuracy(data, num_labels = 4):
    """
    compute_accuracy
    Computes the mean accuracy of the predictions for the given dataset
    
    Arguments:
        data (list of numpy array): List of instances of data, including ground
        truth and predictions
        num_labels(int, default = 4): Number of classes
    """ 
    
    # Declarating list to store results
    accuracies = []
    
    for instance in data:
        
        # Declarating list to store individual results
        instance_accuracies = []
        
        for i in np.arange(num_labels):
            
            # Computing and storing accuracy for each class
            instance_accuracies.append(accuracy_score(instance[:, 2 + i], instance[:, 2 + i + 4]))
        
        # Storing mean results of the instance
        accuracies.append(np.mean(instance_accuracies))
        
    # Returning mean of all results
    return np.mean(accuracies)

def compute_precision(data, num_labels = 4):
    """
    compute_precision
    Computes the mean precision of the predictions for the given dataset
    
    Arguments:
        data (list of numpy array): List of instances of data, including ground
        truth and predictions
        num_labels(int, default = 4): Number of classes
    """ 
    
    # Declarating list to store results
    precisions = []
    
    for instance in data:
        
        # Declarating list to store individual results
        instance_precisions = []
        
        for i in np.arange(num_labels):
            
            # Computing and storing precision for each class
            instance_precisions.append(precision_score(instance[:, 2 + i], instance[:, 2 + i + 4]))
            
        # Storing mean results of the instance
        precisions.append(np.mean(instance_precisions))
    
    # Returning mean of all results
    return np.mean(precisions)

def compute_recall(data, num_labels = 4):
    """
    compute_recall
    Computes the mean recall of the predictions for the given dataset
    
    Arguments:
        data (list of numpy array): List of instances of data, including ground
        truth and predictions
        num_labels(int, default = 4): Number of classes
    """ 
    
    # Declarating list to store results
    recalls = []
    
    for instance in data:
        
        # Declarating list to store individual results
        instance_recalls = []
        
        for i in np.arange(num_labels):
            
            # Computing and storing accuracy for each class
            instance_recalls.append(recall_score(instance[:, 2 + i], instance[:, 2 + i + 4]))
            
        # Storing mean results of the instance
        recalls.append(np.mean(instance_recalls))
    
    # Returning mean of all results
    return np.mean(recalls)

