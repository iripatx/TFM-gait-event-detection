# -*- coding: utf-8 -*-


import peak_based_detection as pbd
from metrics import *
import io_utils
import TCN_detection, LSTM_detection

#db = io_utils.get_database()

TCN_detection.detect_events()