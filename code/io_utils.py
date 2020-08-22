# -*- coding: utf-8 -*-

# required imports
from pathlib import Path
from scipy.io import loadmat
import numpy as np
import pandas as pd
import time

# Defining global paths
root_path = Path.cwd().parent
data_path = root_path / 'data' / 'MAREA_dataset'
timings_path = data_path / 'Activity Timings'
subjects_path = data_path / 'Subject Data_txt format'

def get_database(verbose = True):
    """
    get_database
    Imports database as a pandas DataFrame. If the data has not been converted yet,
    read_data will be called.
    
    Arguments:
        verbose (boolean, default = True): If true, the function will display information
            about the process
            
    Returns: 
        db (pandas.DataFrame): Database
    """
    
    # Check if database.pkl exists
    if not (data_path / 'database').exists():
        # Database has not been yet read
        if verbose: print('Database not found. Reading data...')
        read_all_data(verbose = verbose)
        
    # Import database
    db = pd.read_pickle(data_path / 'database')
    if verbose: print('Database sucessfully loaded')
    
    # Sorting dataframe
    db.index = db.index.astype(int)
    db = db.sort_index()
    
    return db
    

def read_all_data(verbose = True):
    """
    read_all_data
    Reads the database, converts and stores it as a pandas dataframe.
    The dataframe is stored as a pickle (.pkl) file
    
    Arguments:
        verbose (boolean, default = True): If true, the function will display information
            about the process
    """ 
    
    # Take current time
    start = time.time()
        
    # Creating dictionary
    data = {"indoors":{} , "outdoors":{}}
    
    # INDOORS DATA
    if verbose: print('Reading indoors data...')
    indoors_dic = read_indoors_data(verbose = verbose)
    data['indoors'] = indoors_dic
    
    # OUTDOORS DATA
    if verbose: print('Reading outdoors data...')
    outdoors_dic= read_outdoors_data(verbose = verbose)
    data['outdoors'] = outdoors_dic
    
    # Converting dictionary to dataframe
    db = pd.DataFrame.from_dict(data)
    
    # Sorting dataframe
    db.index = db.index.astype(int)
    db = db.sort_index()
    
    # LABELS
    db = read_labels(db, verbose = verbose)
    
    # Storing database as a pickle file
    db.to_pickle(data_path / 'database')
    
    # Take current time and measure computation time
    end = time.time()
    if verbose: print('Tiempo total empleado: ' + str(end - start))
    
    end



def read_indoors_data(test = False, verbose = True):
    """
    read_indoors_data
    Returns data taken from indoors experiments
    
    Arguments:
        test (boolean, default = False): If true, only the first subject will be read
        verbose (boolean, default = True): If true, the function will display information
            about the process
    """
    
    # Reading indoor timings file to find the number of subjects
    with (timings_path / 'Indoor Experiment Timings.txt').open() as f_timings:
        num_subjects = sum(1 for line in f_timings)
        
    # If test is true, only one subject will be processed
    if test:
        num_subjects = 1
    
    # Opening indoor timings file
    f_timings = (timings_path / 'Indoor Experiment Timings.txt').open()
        
    # Creating dictionary for all indoor subjects
    indoor_dic = {}
    
    # For each subject
    for i in np.arange(num_subjects):
        
        # Subject 4 lacks wrist data and will be skipped
        if (i == 3):
            continue
        
        # Take current time
        start = time.time()
        
        # Read timings line
        string = f_timings.readline()
        # Extract timings
        timings = string.split(',')
        # Remove trailing newline characters
        timings[-1] = timings[-1].rstrip('\n')
        # Converting to int
        timings = [int(i) for i in timings]
        
        # Creating dictionary for the subject
        dic = {'treadmill_flat':{'LF':[], 'RF':[], 'Waist':[], 'Wrist':[]}, \
               'treadmill_slope':{'LF':[], 'RF':[], 'Waist':[], 'Wrist':[]}, \
               'flat_space':{'LF':[], 'RF':[], 'Waist':[], 'Wrist':[]}}
        
        # Reading subject's data
        LF_filename = 'Sub' + str(i + 1) + '_LF.txt'
        RF_filename = 'Sub' + str(i + 1) + '_RF.txt'
        Waist_filename = 'Sub' + str(i + 1) + '_Waist.txt'
        Wrist_filename = 'Sub' + str(i + 1) + '_Wrist.txt'
        LF = np.loadtxt(subjects_path / LF_filename, delimiter = ',', skiprows = 1)
        RF = np.loadtxt(subjects_path / RF_filename, delimiter = ',', skiprows = 1)
        Waist = np.loadtxt(subjects_path / Waist_filename, delimiter = ',', skiprows = 1)
        Wrist = np.loadtxt(subjects_path / Wrist_filename, delimiter = ',', skiprows = 1)
        
        # Storing subject's data
        # Treadmill (flat) walk and run
        dic['treadmill_flat']['LF'] = LF[0:timings[2]-1, :]
        dic['treadmill_flat']['RF'] = RF[0:timings[2]-1, :]
        dic['treadmill_flat']['Waist'] = Waist[0:timings[2]-1, :]
        dic['treadmill_flat']['Wrist'] = Wrist[0:timings[2]-1, :]
        # Treadmill (slope) walk
        dic['treadmill_slope']['LF'] = LF[0:timings[4]-1, :]
        dic['treadmill_slope']['RF'] = RF[0:timings[4]-1, :]
        dic['treadmill_slope']['Waist'] = Waist[0:timings[4]-1, :]
        dic['treadmill_slope']['Wrist'] = Wrist[0:timings[4]-1, :]
        # Indoor flat space walk and run
        dic['flat_space']['LF'] = LF[0:timings[7]-1, :]
        dic['flat_space']['RF'] = RF[0:timings[7]-1, :]
        dic['flat_space']['Waist'] = Waist[0:timings[7]-1, :]
        dic['flat_space']['Wrist'] = Wrist[0:timings[7]-1, :]
        
        # Take current time and measure computation time
        end = time.time()
        if verbose: print('Sujeto ' + str(i + 1) + ' procesado. Tiempo empleado: ' + str(end - start))
        
        # Storing subject
        indoor_dic[str(i + 1)] = dic
    
    return indoor_dic


def read_outdoors_data(test = False, verbose = True):
    """
    read_outdoors_data
    Returns data taken from outdoors experiments
    
    Arguments:
        test(boolean, default = False): If true, only the first subject will be read
        verbose (boolean, default = True): If true, the function will display information
            about the process
    """
    
    # Reading outdoor timings file to find the number of subjects
    with (timings_path / 'Outdoor Experiment Timings.txt').open() as f_timings:
        num_subjects = sum(1 for line in f_timings)
        
    # If test is true, only one subject will be processed
    if test:
        num_subjects = 1
            
    # Creating dictionary for all indoor subjects
    outdoor_dic = {}
    
    # For each subject
    for i in np.arange(num_subjects):
        
        # Take current time to measure computation time
        start = time.time()
        
        # Creating dictionary for the subject
        dic = {'street':{'LF':[], 'RF':[], 'Waist':[], 'Wrist':[]}}
        
        # Reading subject's data
        LF_filename = 'Sub' + str(i + 12) + '_LF.txt'
        RF_filename = 'Sub' + str(i + 12) + '_RF.txt'
        Waist_filename = 'Sub' + str(i + 12) + '_Waist.txt'
        Wrist_filename = 'Sub' + str(i + 12) + '_Wrist.txt'
        LF = np.loadtxt(subjects_path / LF_filename, delimiter = ',', skiprows = 1)
        RF = np.loadtxt(subjects_path / RF_filename, delimiter = ',', skiprows = 1)
        Waist = np.loadtxt(subjects_path / Waist_filename, delimiter = ',', skiprows = 1)
        Wrist = np.loadtxt(subjects_path / Wrist_filename, delimiter = ',', skiprows = 1)
        
        # Storing subject's data
        dic['street']['LF'] = LF
        dic['street']['RF'] = RF
        dic['street']['Waist'] = Waist
        dic['street']['Wrist'] = Wrist
        
        # Take current time to measure computation time
        end = time.time()
        if verbose: print('Sujeto ' + str(i + 12) + ' procesado. Tiempo empleado: ' + str(end - start))
        
        # Storing subject
        outdoor_dic[str(i + 1)] = dic
    
    return outdoor_dic

def read_labels(db, verbose = True):
    """
        read_labels
        Imports event labels. Notation is:
            1: Left Foot Heel Strike
            2: Left Foot Toe Off
            3: Right Foot Heel Strike
            4: Right Foot Toe Off
    """
    
    # import MATLAB file
    mat = loadmat(data_path / 'GroundTruth.mat')
    gt = mat['GroundTruth']
    
    # Indoor labels
    for i in np.arange(gt.size):
        
        # Skipping 4th subject
        if i == 3: continue
        
        # Creating label matrices
        treadmill_flat_labels = np.zeros([db['indoors'][i+1]['treadmill_flat']['LF'].size, 4])
        treadmill_slope_labels = np.zeros([db['indoors'][i+1]['treadmill_slope']['LF'].size, 4])
        flat_space_labels = np.zeros([db['indoors'][i+1]['flat_space']['LF'].size, 4])
        
        # Assigning labels
        treadmill_flat_labels[gt[0, i]['treadWalknRun']['LF_HS'].item(), 0] = 1
        treadmill_flat_labels[gt[0, i]['treadWalknRun']['LF_TO'].item(), 1] = 1
        treadmill_flat_labels[gt[0, i]['treadWalknRun']['RF_HS'].item(), 2] = 1
        treadmill_flat_labels[gt[0, i]['treadWalknRun']['RF_TO'].item(), 3] = 1
        
        treadmill_slope_labels[gt[0, i]['treadIncline']['LF_HS'].item(), 0] = 1
        treadmill_slope_labels[gt[0, i]['treadIncline']['LF_TO'].item(), 1] = 1
        treadmill_slope_labels[gt[0, i]['treadIncline']['RF_HS'].item(), 2] = 1
        treadmill_slope_labels[gt[0, i]['treadIncline']['RF_TO'].item(), 3] = 1
        
        flat_space_labels[gt[0, i]['indoorWalknRun']['LF_HS'].item(), 0] = 1
        flat_space_labels[gt[0, i]['indoorWalknRun']['LF_TO'].item(), 1] = 1
        flat_space_labels[gt[0, i]['indoorWalknRun']['RF_HS'].item(), 2] = 1
        flat_space_labels[gt[0, i]['indoorWalknRun']['RF_TO'].item(), 3] = 1
        
        # Storing label in the dataframe
        db['indoors'][i+1]['treadmill_flat']['labels'] = treadmill_flat_labels
        db['indoors'][i+1]['treadmill_slope']['labels'] = treadmill_slope_labels
        db['indoors'][i+1]['flat_space']['labels'] = flat_space_labels
        
    # Outdoor labels
    for i in np.arange(gt.size - 2):
        
        # Creating label matrix
        street_labels = np.zeros([db['outdoors'][i+1]['street']['LF'].size, 4])
        
        # Assigning labels
        street_labels[gt[0, i]['outdoorWalknRun']['LF_HS'].item(), 0] = 1
        street_labels[gt[0, i]['outdoorWalknRun']['LF_TO'].item(), 1] = 1
        street_labels[gt[0, i]['outdoorWalknRun']['RF_HS'].item(), 2] = 1
        street_labels[gt[0, i]['outdoorWalknRun']['RF_TO'].item(), 3] = 1
        
        db['outdoors'][i+1]['street']['labels'] = street_labels
        
    
    return db
    
        