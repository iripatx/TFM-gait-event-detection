# -*- coding: utf-8 -*-
"""
    This file implements the functions required to implement the database
    into a pandas dataframe.
"""

# required imports
from pathlib import Path
from scipy.io import loadmat
import numpy as np
import pandas as pd
import time
import preprocess_utils
import random

# Defining global paths
root_path = Path.cwd().parent
data_path = root_path / 'data' / 'MAREA_dataset'
timings_path = data_path / 'Activity Timings'
subjects_path = data_path / 'Subject Data_txt format'

# Defining indoor tests
indoor_tests = ['treadmill_flat_walk', 'treadmill_flat_run', 'treadmill_slope', 'flat_space_walk', 'flat_space_run']

def get_database(verbose = True, preprocessed = False):
    """
    get_database
    Imports database as a pandas DataFrame. If the data has not been converted yet,
    read_data will be called.
    
    Arguments:
        verbose (boolean, default = True): If true, the function will display information
            about the process
        preprocessed (boolean, default = True): If True, the function returns the 
            database already preprocessed.
            
    Returns: 
        db (pandas.DataFrame): Database
    """
    
    # If the preprocessed data is required
    if preprocessed:
        # Check if preprocessed database (prep_database.pkl) exists
        if (data_path / 'prep_database').exists():
            # Import database
            db = pd.read_pickle(data_path / 'prep_database')
            if verbose: print('Preprocessed database sucessfully loaded')
            return db
        else:
            if verbose: print('Preprocessed database not found. Loading databse...')
    
    # Check if database already exists
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
    
    # If the preprocessed data is required
    if preprocessed:
        if verbose: print('Preprocessing database...')
        db = preprocess_utils.preprocess_data(db)
        if verbose: print('Database sucessfully preprocessed')
        # Storing database as a pickle file
        db.to_pickle(data_path / 'prep_database')
    
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
        
        # Read timings line
        string = f_timings.readline()
        # Extract timings
        timings = string.split(',')
        # Remove trailing newline characters
        timings[-1] = timings[-1].rstrip('\n')
        # Converting to int
        timings = [int(i) for i in timings]
        
        # Subject 4 lacks wrist data and will be skipped
        if (i == 3):
            continue
        
        # Take current time
        start = time.time()
        
        # Creating dictionary for the subject
        dic = {'treadmill_flat_walk':{'LF':[], 'RF':[], 'Waist':[], 'Wrist':[]}, \
               'treadmill_flat_run':{'LF':[], 'RF':[], 'Waist':[], 'Wrist':[]}, \
               'treadmill_slope':{'LF':[], 'RF':[], 'Waist':[], 'Wrist':[]}, \
               'flat_space_walk':{'LF':[], 'RF':[], 'Waist':[], 'Wrist':[]}, \
               'flat_space_run':{'LF':[], 'RF':[], 'Waist':[], 'Wrist':[]}}
        
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
        
        for test in indoor_tests:
            # Treadmill (flat) walk
            dic['treadmill_flat_walk']['LF'] = LF[0:timings[1], :]
            dic['treadmill_flat_walk']['RF'] = RF[0:timings[1], :]
            dic['treadmill_flat_walk']['Waist'] = Waist[0:timings[1], :]
            dic['treadmill_flat_walk']['Wrist'] = Wrist[0:timings[1], :]
            # Treadmill (flat) walk
            dic['treadmill_flat_run']['LF'] = LF[timings[1]:timings[2], :]
            dic['treadmill_flat_run']['RF'] = RF[timings[1]:timings[2], :]
            dic['treadmill_flat_run']['Waist'] = Waist[timings[1]:timings[2], :]
            dic['treadmill_flat_run']['Wrist'] = Wrist[timings[1]:timings[2], :]
            # Treadmill (slope) walk
            dic['treadmill_slope']['LF'] = LF[timings[3]:timings[4], :]
            dic['treadmill_slope']['RF'] = RF[timings[3]:timings[4], :]
            dic['treadmill_slope']['Waist'] = Waist[timings[3]:timings[4], :]
            dic['treadmill_slope']['Wrist'] = Wrist[timings[3]:timings[4], :]
            # Indoor flat space run
            dic['flat_space_walk']['LF'] = LF[timings[5]:timings[6], :]
            dic['flat_space_walk']['RF'] = RF[timings[5]:timings[6], :]
            dic['flat_space_walk']['Waist'] = Waist[timings[5]:timings[6], :]
            dic['flat_space_walk']['Wrist'] = Wrist[timings[5]:timings[6], :]
            # Indoor flat space run
            dic['flat_space_run']['LF'] = LF[timings[6]:timings[7], :]
            dic['flat_space_run']['RF'] = RF[timings[6]:timings[7], :]
            dic['flat_space_run']['Waist'] = Waist[timings[6]:timings[7], :]
            dic['flat_space_run']['Wrist'] = Wrist[timings[6]:timings[7], :]
        
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
        
    # Opening indoor timings file
    f_timings = (timings_path / 'Outdoor Experiment Timings.txt').open()
            
    # Creating dictionary for all indoor subjects
    outdoor_dic = {}
    
    # For each subject
    for i in np.arange(num_subjects):
        
        # Take current time to measure computation time
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
        dic = {'street_walk':{'LF':[], 'RF':[], 'Waist':[], 'Wrist':[]}, \
               'street_run':{'LF':[], 'RF':[], 'Waist':[], 'Wrist':[]}}
        
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
        # Outdoors walk
        dic['street_walk']['LF'] = LF[0:timings[1]-1, :]
        dic['street_walk']['RF'] = RF[0:timings[1]-1, :]
        dic['street_walk']['Waist'] = Waist[0:timings[1]-1, :]
        dic['street_walk']['Wrist'] = Wrist[0:timings[1]-1, :]
        # Outdoors run
        dic['street_run']['LF'] = LF[timings[1]:timings[2]-1, :]
        dic['street_run']['RF'] = RF[timings[1]:timings[2]-1, :]
        dic['street_run']['Waist'] = Waist[timings[1]:timings[2]-1, :]
        dic['street_run']['Wrist'] = Wrist[timings[1]:timings[2]-1, :]
        
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
    
    # Opening indoor timings file
    f_timings = (timings_path / 'Indoor Experiment Timings.txt').open()
    
    for i in np.arange(gt.size):
        
        # Read timings line
        string = f_timings.readline()
        # Extract timings
        timings = string.split(',')
        # Remove trailing newline characters
        timings[-1] = timings[-1].rstrip('\n')
        # Converting to int
        timings = [int(i) for i in timings]
        
        # Skipping 4th subject
        if i == 3: continue
        
        # Creating label matrice
        treadmill_flat_labels = np.zeros([timings[2], 4])
        treadmill_slope_labels = np.zeros([timings[4] - timings[3], 4])
        flat_space_labels = np.zeros([timings[7] - timings[5], 4])
        
        # Assigning labels
        treadmill_flat_labels[gt[0, i]['treadWalknRun']['LF_HS'].item() - 1, 0] = 1
        treadmill_flat_labels[gt[0, i]['treadWalknRun']['LF_TO'].item() - 1, 1] = 1
        treadmill_flat_labels[gt[0, i]['treadWalknRun']['RF_HS'].item() - 1, 2] = 1
        treadmill_flat_labels[gt[0, i]['treadWalknRun']['RF_TO'].item() - 1, 3] = 1
        
        treadmill_slope_labels[gt[0, i]['treadIncline']['LF_HS'].item() - 1, 0] = 1
        treadmill_slope_labels[gt[0, i]['treadIncline']['LF_TO'].item() - 1, 1] = 1
        treadmill_slope_labels[gt[0, i]['treadIncline']['RF_HS'].item() - 1, 2] = 1
        treadmill_slope_labels[gt[0, i]['treadIncline']['RF_TO'].item() - 1, 3] = 1
        
        flat_space_labels[gt[0, i]['indoorWalknRun']['LF_HS'].item() - 1, 0] = 1
        flat_space_labels[gt[0, i]['indoorWalknRun']['LF_TO'].item() - 1, 1] = 1
        flat_space_labels[gt[0, i]['indoorWalknRun']['RF_HS'].item() - 1, 2] = 1
        flat_space_labels[gt[0, i]['indoorWalknRun']['RF_TO'].item() - 1, 3] = 1
        
        # Storing label in the dataframe
        db['indoors'][i+1]['treadmill_flat_walk']['labels'] = treadmill_flat_labels[0:timings[1]]
        db['indoors'][i+1]['treadmill_flat_run']['labels'] = treadmill_flat_labels[timings[1]:timings[2]]
        db['indoors'][i+1]['treadmill_slope']['labels'] = treadmill_slope_labels
        db['indoors'][i+1]['flat_space_walk']['labels'] = flat_space_labels[0:timings[6] - timings[5]]
        db['indoors'][i+1]['flat_space_run']['labels'] = flat_space_labels[timings[6] - timings[5]:]
        
        
    # Outdoor labels
    
    # Opening indoor timings file
    f_timings = (timings_path / 'Outdoor Experiment Timings.txt').open()
    
    for i in np.arange(gt.size - 2):
        
        # Read timings line
        string = f_timings.readline()
        # Extract timings
        timings = string.split(',')
        # Remove trailing newline characters
        timings[-1] = timings[-1].rstrip('\n')
        # Converting to int
        timings = [int(i) for i in timings]
        
        # Creating label matrix
        street_labels = np.zeros([timings[2], 4])
        
        # Assigning labels
        street_labels[gt[0, i]['outdoorWalknRun']['LF_HS'].item() - 1, 0] = 1
        street_labels[gt[0, i]['outdoorWalknRun']['LF_TO'].item() - 1, 1] = 1
        street_labels[gt[0, i]['outdoorWalknRun']['RF_HS'].item() - 1, 2] = 1
        street_labels[gt[0, i]['outdoorWalknRun']['RF_TO'].item() - 1, 3] = 1
        
        db['outdoors'][i+1]['street_walk']['labels'] = street_labels[0:timings[1] - 1]
        db['outdoors'][i+1]['street_run']['labels'] = street_labels[timings[1]: - 1]
        
    
    return db

def extract_data(trial = 'All', pair = 1):
    """
    extract_data
        Extracts the necessary data and arranges it for the detection.
        
        Options: 'Walk', 'Run' and 'All'

    Returns
    -------
    data : Python list of numpy array
        List containing all instances of data

    """
    
    # Obtain database
    db = get_database(preprocessed = True)
    
    # Declarating list to store the data
    data = []
    # Setting up counters to select validation and test subjects later 
    indoor_count = 0
    outdoor_count = 0
    
    # Selecting trials
    if trial == 'Walk':
        indoor_tests = ['treadmill_flat_walk', 'treadmill_slope', 'flat_space_walk']
        outdoor_tests = ['street_walk']
    elif trial == 'Run':
        indoor_tests = ['treadmill_flat_run', 'flat_space_run']
        outdoor_tests = ['street_run']
    else:
        print('Trial not selected')
        
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
            indoor_count += 1
            
            
    # OUTDOORS DATA
    for i in np.arange(1, db.shape[0] + 1):
        
        # Skipping NaN values
        if pd.isna(db['outdoors'][i]): continue
    
        for test in outdoor_tests:
        
            # Extracting and storing foot accelerations on the X axis and its labels
            left_foot = np.concatenate((db['outdoors'][i][test]['LF'][:, 0].reshape(-1, 1),
                                       db['outdoors'][i][test]['LF'][:, 1].reshape(-1, 1),
                                       db['outdoors'][i][test]['LF'][:, 2].reshape(-1, 1)), axis = 1)
            right_foot = np.concatenate((db['outdoors'][i][test]['RF'][:, 0].reshape(-1, 1),
                                       db['outdoors'][i][test]['RF'][:, 1].reshape(-1, 1),
                                       db['outdoors'][i][test]['RF'][:, 2].reshape(-1, 1)), axis = 1)
            waist = np.concatenate((db['outdoors'][i][test]['Waist'][:, 0].reshape(-1, 1),
                                       db['outdoors'][i][test]['Waist'][:, 1].reshape(-1, 1),
                                       db['outdoors'][i][test]['Waist'][:, 2].reshape(-1, 1)), axis = 1)
            wrist = np.concatenate((db['outdoors'][i][test]['Wrist'][:, 0].reshape(-1, 1),
                                       db['outdoors'][i][test]['Wrist'][:, 1].reshape(-1, 1),
                                       db['outdoors'][i][test]['Wrist'][:, 2].reshape(-1, 1)), axis = 1)
            labels = db['outdoors'][i][test]['labels']
            
            data.append(np.concatenate((left_foot, right_foot, waist, wrist, labels), axis = 1))
            outdoor_count +=1
        
    # One out for test
    test_data = data[(pair -1) * len(indoor_tests):(pair -1) * len(indoor_tests) + len(indoor_tests)]
    del data[(pair -1) * len(indoor_tests):(pair -1) * len(indoor_tests) + len(indoor_tests)]
    test_data.append(data[-(outdoor_count - pair + 1)])
    del data[-(outdoor_count - pair + 1)]
        
    # Random One out for validation
    indoor_subject = random.randrange(0, indoor_count - len(indoor_tests)*2, len(indoor_tests))
    outdoor_subject = random.randrange(0, outdoor_count - 1)
    val_data = data[indoor_subject:indoor_subject + len(indoor_tests)]
    del data[indoor_subject:indoor_subject + len(indoor_tests)]
    val_data.append(data[- outdoor_subject])
    del data[- outdoor_subject]
    
    
    return data, val_data, test_data
    
        