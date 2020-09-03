# -*- coding: utf-8 -*-


import peak_based_detection as pbd
from metrics import *
import io_utils

#db = io_utils.get_database()
#db = io_utils.read_labels(db)

data = pbd.detect_events()

accuracy = compute_accuracy(data)
precision = compute_precision(data)
recall = compute_recall(data)