# -*- coding: utf-8 -*-

# required imports
from pathlib import Path
import numpy as np
import time

# Defining global paths
root_path = Path.cwd().parent
data_path = root_path / 'data' / 'MAREA_dataset'
timings_path = data_path / 'Activity Timings'
subjects_path = data_path / 'Subject Data_txt format'
    

def read_all_data():
    """
    read_all_data
    Returns all of the database in a dictionary.
    The dictionary splits in other two dictionaries for indoors and outdoors data
    Inside of them, each entry represents a subject, 
    containing the corresponding activities.
    """ 
    
    # Creating dictionary
    data = {"indoors":{} , "outdoors":{}}
    
    # INDOORS DATA
    indoors_dic = read_indoors_data(test = True)
        
    # OUTDOORS DATA
    outdoors_dic= read_outdoors_data()



def read_indoors_data(test = False):
    """
    Returns data taken from indoors experiments
    Arguments:
        test(boolean, default = False): If true, only the first subject will be read
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
        
        # Take current time to measure computation time
        end = time.time()
        print('Sujeto ' + str(i + 1) + ' procesado. Tiempo empleado: ' + str(end - start))
        
        # Storing subject
        indoor_dic[str(i + 1)] = dic
    
    return indoor_dic
        