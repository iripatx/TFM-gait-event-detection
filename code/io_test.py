# -*- coding: utf-8 -*-


import peak_based_detection as pbd
from metrics import *
import io_utils
import LSTM_detection

#db = io_utils.get_database()

LSTM_detection.detect_events()